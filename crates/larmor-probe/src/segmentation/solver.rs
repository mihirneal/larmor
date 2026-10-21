//! `probe.py::fit_logistic` solved to its optimum, with the data on a Candle device (Metal, CUDA or CPU).
//!
//! The objective is nanobrain's: mean BCE over all `n x 512` outputs plus `alpha ||W||^2 / (n 512)`, the
//! intercept unpenalised. Each of the 512 outputs is an independent L2 logistic regression on shared
//! features.
//!
//! torch's L-BFGS stops at a largest gradient component of 1e-7 on that averaged loss. On realistic data this
//! leaves the probabilities up to ~1e-2 from the optimum, and torch itself moves them by as much when only
//! the row order or thread count changes, so we do not retrace it; we solve the convex problem to the
//! optimum:
//!
//! - preconditioned L-BFGS: the initial inverse Hessian is, per output j, the inverse of `c_j X'X / n + 2
//!   alpha / n` with `c_j = mean(y_j) (1 - mean(y_j))` (the curvature at the starting point), applied through
//!   one eigendecomposition of `X'X / n`. Correlated embedding features make the plain problem badly
//!   conditioned; this removes most of it.
//! - a line search on the slope only: near the optimum the float32 loss no longer resolves the decrease,
//!   while the directional derivative still does. For a convex objective `|phi'(t)| <= 0.9 |phi'(0)|` implies
//!   sufficient decrease.
//! - the gradient is computed on the device in float32 (each row with a trailing 1 so one product covers
//!   weights and intercept), summed over blocks of 2048 rows, then across blocks on the device and across
//!   chunks of rows in float64. At 633k rows that keeps it within 1.4e-9 of a float64 gradient; one float32
//!   sum over 131k-row chunks was off by 8.7e-8, more than the 1e-8 the solver stops at.

use crate::device::{Device, DeviceError};
use crate::matrix::Matrix;
use candle_core::Tensor;
use candle_nn::ops::sigmoid;
use nalgebra::{DMatrix, SymmetricEigen};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cell::Cell;
use std::collections::VecDeque;
use std::time::Instant;

/// Stopping rules and data layout for [`fit_logistic`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverOptions {
    /// Iteration budget; fit_logistic uses 1000.
    pub max_iter: usize,
    /// L-BFGS history; fit_logistic uses 10.
    pub history: usize,
    /// Stop when every output's gradient (of its mean BCE plus `alpha ||w||^2 / n`) is at most this.
    pub gradient_tol: f64,
    /// Stop after this many iterations without the largest gradient component shrinking by 10%: float32
    /// rounding has been reached.
    pub stall_iterations: usize,
    /// Rows per device chunk, bounding the size of each buffer.
    pub chunk_rows: usize,
    /// Rows per float32 partial sum of the gradient (see `DeviceProblem::weighted_row_sum`); `chunk_rows` is
    /// rounded down to a multiple of it.
    pub block_rows: usize,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            history: 10,
            gradient_tol: 1e-8,
            stall_iterations: 30,
            chunk_rows: 1 << 17,
            block_rows: 1 << 11,
        }
    }
}

/// Why the solver stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverStop {
    Converged,
    Stalled,
    MaxIter,
}

/// A fitted head: `probabilities = sigmoid(x @ coef + intercept)` on standardized features.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearHead {
    /// `[n_features, n_outputs]`.
    pub coef: Matrix<f32>,
    pub intercept: Vec<f32>,
}

/// How a fit went.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitStats {
    pub iterations: usize,
    pub evaluations: usize,
    /// Largest per-output gradient component at the solution.
    pub gradient_max: f64,
    /// fit_logistic's loss at the solution.
    pub loss: f64,
    pub stop: SolverStop,
    /// Wall time of the whole fit.
    pub seconds: f64,
    /// Of which: uploading the rows and building the preconditioner.
    pub setup_seconds: f64,
    /// Of which: the iterations.
    pub solve_seconds: f64,
    /// Of the iterations: computing gradients (device work and reading them back); the rest is the L-BFGS
    /// update and the preconditioner on the host.
    pub gradient_seconds: f64,
}

/// `fit_logistic(features, targets, alpha)` solved to the optimum.
///
/// `features` must already be standardized; `targets` holds 0/1 per output. `init` warm-starts from another
/// solution (any start reaches the same optimum); without it the start is fit_logistic's: zero weights and
/// the intercept at the clamped logit of each output's mean.
pub fn fit_logistic(
    features: &Matrix<f32>,
    targets: &Matrix<u8>,
    alpha: f64,
    options: &SolverOptions,
    init: Option<&LinearHead>,
    device: &Device,
) -> Result<(LinearHead, FitStats), DeviceError> {
    let start = Instant::now();
    let problem = DeviceProblem::upload(features, targets, alpha, options, device)?;
    let theta0 = match init {
        Some(head) => Theta::from_head(head),
        None => Theta::initial(features.cols(), targets),
    };
    let preconditioner = Preconditioner::new(&problem, targets, alpha)?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let (theta, mut stats) = minimize(&problem, &preconditioner, theta0, options)?;
    stats.solve_seconds = start.elapsed().as_secs_f64() - setup_seconds;
    stats.setup_seconds = setup_seconds;
    stats.gradient_seconds = problem.gradient_seconds.get();
    stats.loss = problem.loss(&theta)?;
    stats.seconds = start.elapsed().as_secs_f64();
    Ok((theta.to_head(problem.n_features, problem.n_outputs), stats))
}

/// `sigmoid(x @ coef + intercept)` on the device (fit_predict_segmentation's prediction), returned row-major
/// `[n, n_outputs]`.
pub fn predict_on_device(
    head: &LinearHead,
    standardized: &Matrix<f32>,
    device: &Device,
) -> Result<Vec<f32>, DeviceError> {
    if standardized.rows() == 0 {
        return Ok(Vec::new());
    }
    let device = device.as_candle();
    let x = Tensor::from_slice(standardized.data(), (standardized.rows(), standardized.cols()), device)?;
    let coef = Tensor::from_slice(head.coef.data(), (head.coef.rows(), head.coef.cols()), device)?;
    let intercept = Tensor::from_slice(&head.intercept, (1, head.intercept.len()), device)?;
    let probabilities = sigmoid(&x.matmul(&coef)?.broadcast_add(&intercept)?)?;
    download(&probabilities)
}

/// The parameters as one float64 vector: `n_features` rows of weights (one per output), then the intercepts.
#[derive(Clone, Debug)]
struct Theta {
    values: Vec<f64>,
}

impl Theta {
    /// fit_logistic's start: zero weights, intercepts `logit(mean(y_j).clamp(1e-6, 1 - 1e-6))` (float32).
    fn initial(n_features: usize, targets: &Matrix<u8>) -> Self {
        let n = targets.rows() as f32;
        let mut values = vec![0.0; n_features * targets.cols()];
        values.extend(column_counts(targets).iter().map(|&count| {
            let mean = (count as f32 / n).clamp(1e-6, 1.0 - 1e-6);
            f64::from((mean / (1.0 - mean)).ln())
        }));
        Self { values }
    }

    fn from_head(head: &LinearHead) -> Self {
        let mut values: Vec<f64> = head.coef.data().iter().map(|&v| f64::from(v)).collect();
        values.extend(head.intercept.iter().map(|&v| f64::from(v)));
        Self { values }
    }

    fn to_head(&self, n_features: usize, n_outputs: usize) -> LinearHead {
        let split = n_features * n_outputs;
        LinearHead {
            coef: Matrix::new(
                n_features,
                n_outputs,
                self.values[..split].iter().map(|&v| v as f32).collect(),
            ),
            intercept: self.values[split..].iter().map(|&v| v as f32).collect(),
        }
    }
}

/// The training rows on the device, in chunks of rows: `([x, 1], y)` as float32 `[rows, n_features + 1]` and
/// `[rows, n_outputs]`.
struct DeviceProblem {
    chunks: Vec<(Tensor, Tensor)>,
    block_rows: usize,
    n_rows: usize,
    n_features: usize,
    n_outputs: usize,
    alpha: f64,
    device: candle_core::Device,
    /// Wall time spent in [`DeviceProblem::gradient`].
    gradient_seconds: Cell<f64>,
}

impl DeviceProblem {
    /// Uploads the rows in chunks, each feature row with a trailing 1 so that one product gives the logits
    /// (`[x, 1] @ [W; b]`) and one product gives the weight and intercept gradients together.
    fn upload(
        features: &Matrix<f32>,
        targets: &Matrix<u8>,
        alpha: f64,
        options: &SolverOptions,
        device: &Device,
    ) -> Result<Self, DeviceError> {
        assert_eq!(features.rows(), targets.rows(), "one target row per feature row");
        let (n, d, k) = (features.rows(), features.cols(), targets.cols());
        let device = device.as_candle();
        let block_rows = options.block_rows.max(1);
        let chunk_rows = (options.chunk_rows / block_rows).max(1) * block_rows;
        let chunks = chunk_bounds(n, chunk_rows, block_rows)
            .into_iter()
            .map(|(start, end)| {
                let mut augmented = Vec::with_capacity((end - start) * (d + 1));
                for i in start..end {
                    augmented.extend_from_slice(features.row(i));
                    augmented.push(1.0);
                }
                let y: Vec<f32> = targets.data()[start * k..end * k].iter().map(|&v| f32::from(v)).collect();
                Ok((
                    Tensor::from_vec(augmented, (end - start, d + 1), device)?,
                    Tensor::from_vec(y, (end - start, k), device)?,
                ))
            })
            .collect::<Result<_, DeviceError>>()?;
        Ok(Self {
            chunks,
            block_rows,
            n_rows: n,
            n_features: d,
            n_outputs: k,
            alpha,
            device: device.clone(),
            gradient_seconds: Cell::new(0.0),
        })
    }

    /// The gradient of `F = sum_j [mean_i BCE_ij + alpha / n ||w_j||^2]`, i.e. fit_logistic's loss times
    /// `n_outputs`, so each output's block is the gradient of its own problem.
    fn gradient(&self, theta: &Theta) -> Result<Vec<f64>, DeviceError> {
        let start = Instant::now();
        let (d, k, n) = (self.n_features, self.n_outputs, self.n_rows as f64);
        let theta_device = self.upload_theta(theta)?;
        // every chunk's partial sum is queued before the first is read back
        let partials = self
            .chunks
            .iter()
            .map(|(x, y)| {
                let residual = sigmoid(&x.matmul(&theta_device)?)?.sub(y)?;
                self.weighted_row_sum(x, &residual)
            })
            .collect::<candle_core::Result<Vec<_>>>()?;
        let mut grad = vec![0.0f64; (d + 1) * k];
        for partial in &partials {
            for (g, v) in grad.iter_mut().zip(download(partial)?) {
                *g += f64::from(v);
            }
        }
        let penalty = 2.0 * self.alpha / n;
        for (i, g) in grad.iter_mut().enumerate() {
            *g /= n;
            if i < d * k {
                *g += penalty * theta.values[i];
            }
        }
        self.gradient_seconds.set(self.gradient_seconds.get() + start.elapsed().as_secs_f64());
        Ok(grad)
    }

    /// `x' r` for one chunk, `[n_features + 1, n_outputs]`.
    ///
    /// A matrix product accumulates over the rows in float32 one after another, and over 10^5 rows that error
    /// grows past the solver's tolerance (8.7e-8 against 1e-8 at 633k rows). So the rows are summed in blocks
    /// of `block_rows` (one batched product) and the blocks are added after: 1.4e-9 at 2048 rows.
    fn weighted_row_sum(&self, x: &Tensor, residual: &Tensor) -> candle_core::Result<Tensor> {
        let (rows, columns) = x.dims2()?;
        let blocks = rows / self.block_rows;
        if blocks <= 1 || rows % self.block_rows != 0 {
            return x.t()?.matmul(residual);
        }
        let x_blocks = x.reshape((blocks, self.block_rows, columns))?.transpose(1, 2)?;
        let residual_blocks = residual.reshape((blocks, self.block_rows, residual.dim(1)?))?;
        x_blocks.matmul(&residual_blocks)?.sum(0)
    }

    /// fit_logistic's loss: mean BCE over `n x n_outputs` plus `alpha ||W||^2 / (n n_outputs)`.
    fn loss(&self, theta: &Theta) -> Result<f64, DeviceError> {
        let (d, k, n) = (self.n_features, self.n_outputs, self.n_rows as f64);
        let theta_device = self.upload_theta(theta)?;
        let mut total = 0.0f64;
        for (x, y) in &self.chunks {
            let logits = x.matmul(&theta_device)?;
            // softplus(z) = max(z, 0) + ln(1 + exp(-|z|)), which cannot overflow
            let softplus = logits.relu()?.add(&logits.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
            let bce = softplus.sub(&y.mul(&logits)?)?;
            total += f64::from(bce.sum_all()?.to_scalar::<f32>()?);
        }
        let squared_norm: f64 = theta.values[..d * k].iter().map(|v| v * v).sum();
        Ok(total / (n * k as f64) + self.alpha * squared_norm / (n * k as f64))
    }

    /// `X'X / n` in float64 (without the column of ones), for the preconditioner.
    fn gram(&self) -> Result<DMatrix<f64>, DeviceError> {
        let d = self.n_features;
        let mut gram = DMatrix::<f64>::zeros(d, d);
        for (x, _) in &self.chunks {
            let block = download(&x.t()?.matmul(x)?)?;
            for i in 0..d {
                for j in 0..d {
                    gram[(i, j)] += f64::from(block[i * (d + 1) + j]);
                }
            }
        }
        Ok(gram / self.n_rows as f64)
    }

    /// `[W; b]` as one `[n_features + 1, n_outputs]` float32 tensor.
    fn upload_theta(&self, theta: &Theta) -> Result<Tensor, DeviceError> {
        let values: Vec<f32> = theta.values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::from_vec(values, (self.n_features + 1, self.n_outputs), &self.device)?)
    }
}

/// `H0^-1` of the preconditioned L-BFGS (see the module docs).
struct Preconditioner {
    /// The eigenvectors of `X'X / n` as columns, and transposed.
    vectors: DMatrix<f64>,
    vectors_t: DMatrix<f64>,
    /// `[n_features, n_outputs]`: the eigenvalues of each output's approximate Hessian.
    denominators: DMatrix<f64>,
    curvature: Vec<f64>,
    n_features: usize,
    n_outputs: usize,
}

impl Preconditioner {
    fn new(problem: &DeviceProblem, targets: &Matrix<u8>, alpha: f64) -> Result<Self, DeviceError> {
        let (d, k, n) = (problem.n_features, problem.n_outputs, problem.n_rows as f64);
        let eigen = SymmetricEigen::new(problem.gram()?);
        let curvature: Vec<f64> = column_counts(targets)
            .iter()
            .map(|&c| {
                let mean = c as f64 / n;
                (mean * (1.0 - mean)).max(1e-4)
            })
            .collect();
        let denominators =
            DMatrix::from_fn(d, k, |i, j| curvature[j] * eigen.eigenvalues[i].max(0.0) + 2.0 * alpha / n);
        Ok(Self {
            vectors_t: eigen.eigenvectors.transpose(),
            vectors: eigen.eigenvectors,
            denominators,
            curvature,
            n_features: d,
            n_outputs: k,
        })
    }

    fn apply(&self, r: &[f64]) -> Vec<f64> {
        const BLOCK_OUTPUTS: usize = 16;
        let (d, k) = (self.n_features, self.n_outputs);
        // r is row-major [d + 1, k]; nalgebra is column-major, so read it as its transpose
        let weights = DMatrix::from_row_slice(d, k, &r[..d * k]);
        // the outputs are independent, so blocks of them go to different threads
        let starts: Vec<usize> = (0..k).step_by(BLOCK_OUTPUTS).collect();
        let blocks: Vec<DMatrix<f64>> = starts
            .par_iter()
            .map(|&start| {
                let width = BLOCK_OUTPUTS.min(k - start);
                let rotated = &self.vectors_t * weights.columns(start, width);
                &self.vectors * rotated.component_div(&self.denominators.columns(start, width))
            })
            .collect();
        let mut out = Vec::with_capacity((d + 1) * k);
        for i in 0..d {
            for block in &blocks {
                out.extend(block.row(i).iter());
            }
        }
        out.extend(r[d * k..].iter().zip(&self.curvature).map(|(v, c)| v / c));
        out
    }
}

fn minimize(
    problem: &DeviceProblem,
    preconditioner: &Preconditioner,
    theta0: Theta,
    options: &SolverOptions,
) -> Result<(Theta, FitStats), DeviceError> {
    let mut theta = theta0;
    let mut grad = problem.gradient(&theta)?;
    let mut evaluations = 1;
    let mut memory: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(options.history);
    let mut gamma = 1.0;
    let mut best = max_abs(&grad);
    let mut stalled = 0;
    let stats = |iterations, evaluations, gradient_max, stop| FitStats {
        iterations,
        evaluations,
        gradient_max,
        loss: f64::NAN,
        stop,
        seconds: 0.0,
        setup_seconds: 0.0,
        solve_seconds: 0.0,
        gradient_seconds: 0.0,
    };
    if best <= options.gradient_tol {
        return Ok((theta, stats(0, evaluations, best, SolverStop::Converged)));
    }
    for iteration in 1..=options.max_iter {
        let mut direction = two_loop(&memory, gamma, preconditioner, &grad);
        let mut slope0 = dot(&grad, &direction);
        if slope0 >= 0.0 {
            // the memory no longer gives a descent direction: restart from the preconditioned gradient
            memory.clear();
            gamma = 1.0;
            direction = preconditioner.apply(&grad).iter().map(|v| -v).collect();
            slope0 = dot(&grad, &direction);
        }
        let (step, new_grad, trials) = slope_search(problem, &theta, &direction, slope0)?;
        evaluations += trials;
        let s: Vec<f64> = direction.iter().map(|d| step * d).collect();
        let y: Vec<f64> = new_grad.iter().zip(&grad).map(|(a, b)| a - b).collect();
        for (t, si) in theta.values.iter_mut().zip(&s) {
            *t += si;
        }
        grad = new_grad;

        let gradient_max = max_abs(&grad);
        if gradient_max <= options.gradient_tol {
            return Ok((theta, stats(iteration, evaluations, gradient_max, SolverStop::Converged)));
        }
        stalled = if gradient_max > 0.9 * best { stalled + 1 } else { 0 };
        best = best.min(gradient_max);
        if stalled >= options.stall_iterations {
            return Ok((theta, stats(iteration, evaluations, gradient_max, SolverStop::Stalled)));
        }
        let s_dot_y = dot(&s, &y);
        if s_dot_y > 0.0 {
            gamma = s_dot_y / dot(&y, &preconditioner.apply(&y));
            if memory.len() == options.history {
                memory.pop_front();
            }
            memory.push_back((s, y, 1.0 / s_dot_y));
        }
    }
    let gradient_max = max_abs(&grad);
    Ok((theta, stats(options.max_iter, evaluations, gradient_max, SolverStop::MaxIter)))
}

/// `-H g` by the two-loop recursion with `H0 = gamma M^-1`.
fn two_loop(
    memory: &VecDeque<(Vec<f64>, Vec<f64>, f64)>,
    gamma: f64,
    preconditioner: &Preconditioner,
    grad: &[f64],
) -> Vec<f64> {
    let mut q: Vec<f64> = grad.iter().map(|g| -g).collect();
    let mut alphas = vec![0.0; memory.len()];
    for (i, (s, y, rho)) in memory.iter().enumerate().rev() {
        alphas[i] = rho * dot(s, &q);
        axpy(-alphas[i], y, &mut q);
    }
    let mut r: Vec<f64> = preconditioner.apply(&q).into_iter().map(|v| gamma * v).collect();
    for (i, (s, y, rho)) in memory.iter().enumerate() {
        let beta = rho * dot(y, &r);
        axpy(alphas[i] - beta, s, &mut r);
    }
    r
}

/// A step along `direction` where the slope has fallen to at most 0.9 of its initial magnitude, found by
/// expanding and then secant steps on the slope. Returns the step, the gradient there and the evaluations.
fn slope_search(
    problem: &DeviceProblem,
    theta: &Theta,
    direction: &[f64],
    slope0: f64,
) -> Result<(f64, Vec<f64>, usize), DeviceError> {
    const CURVATURE: f64 = 0.9;
    const MAX_TRIALS: usize = 30;
    let (mut low, mut slope_low) = (0.0, slope0);
    let mut high: Option<(f64, f64)> = None;
    let mut step = 1.0;
    let mut trials = 0;
    loop {
        let trial = Theta { values: theta.values.iter().zip(direction).map(|(t, d)| t + step * d).collect() };
        let grad = problem.gradient(&trial)?;
        trials += 1;
        let slope = dot(&grad, direction);
        if slope.abs() <= CURVATURE * -slope0 || trials >= MAX_TRIALS {
            return Ok((step, grad, trials));
        }
        if slope < 0.0 {
            low = step;
            slope_low = slope;
        } else {
            high = Some((step, slope));
        }
        step = match high {
            None => 4.0 * step,
            Some((high, slope_high)) => {
                let secant = low - slope_low * (high - low) / (slope_high - slope_low);
                let width = high - low;
                secant.clamp(low + 0.1 * width, high - 0.1 * width)
            }
        };
    }
}

/// Positives per output column.
fn column_counts(targets: &Matrix<u8>) -> Vec<usize> {
    let mut counts = vec![0usize; targets.cols()];
    for row in targets.iter_rows() {
        for (count, &label) in counts.iter_mut().zip(row) {
            *count += usize::from(label);
        }
    }
    counts
}

/// `(start, end)` row ranges of the device chunks: `chunk_rows` (a multiple of `block_rows`) at a time, the
/// remainder split into whole blocks and a last short chunk.
fn chunk_bounds(n_rows: usize, chunk_rows: usize, block_rows: usize) -> Vec<(usize, usize)> {
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < n_rows {
        let left = n_rows - start;
        let rows = if left >= chunk_rows {
            chunk_rows
        } else if left >= block_rows {
            left / block_rows * block_rows
        } else {
            left
        };
        bounds.push((start, start + rows));
        start += rows;
    }
    bounds
}

/// The gradient [`fit_logistic`]'s solver computes at `head` (weights then intercepts, `[(n_features + 1) *
/// n_outputs]`), for checking its float32 accuracy against a float64 computation.
pub fn solver_gradient(
    features: &Matrix<f32>,
    targets: &Matrix<u8>,
    alpha: f64,
    head: &LinearHead,
    options: &SolverOptions,
    device: &Device,
) -> Result<Vec<f64>, DeviceError> {
    let problem = DeviceProblem::upload(features, targets, alpha, options, device)?;
    problem.gradient(&Theta::from_head(head))
}

/// A float32 tensor's values in row-major order, waiting for the device to finish computing them.
fn download(tensor: &Tensor) -> Result<Vec<f32>, DeviceError> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_rows_in_whole_blocks() {
        assert_eq!(chunk_bounds(10, 4, 2), vec![(0, 4), (4, 8), (8, 10)]);
        assert_eq!(chunk_bounds(11, 8, 4), vec![(0, 8), (8, 11)]);
        assert_eq!(chunk_bounds(13, 8, 2), vec![(0, 8), (8, 12), (12, 13)]);
        assert_eq!(chunk_bounds(3, 8, 4), vec![(0, 3)]);
    }

    /// The analytic gradient, in float64 on the host.
    fn gradient_f64(x: &Matrix<f32>, y: &Matrix<u8>, alpha: f64, head: &LinearHead) -> Vec<f64> {
        let (n, d, k) = (x.rows(), x.cols(), y.cols());
        let mut grad = vec![0.0f64; (d + 1) * k];
        for (i, row) in x.iter_rows().enumerate() {
            for j in 0..k {
                let z = f64::from(head.intercept[j])
                    + row
                        .iter()
                        .enumerate()
                        .map(|(f, &v)| f64::from(v) * f64::from(head.coef.get(f, j)))
                        .sum::<f64>();
                let residual = 1.0 / (1.0 + (-z).exp()) - f64::from(y.get(i, j));
                for (f, &v) in row.iter().enumerate() {
                    grad[f * k + j] += f64::from(v) * residual;
                }
                grad[d * k + j] += residual;
            }
        }
        let n = n as f64;
        grad.iter()
            .enumerate()
            .map(|(i, g)| {
                g / n + if i < d * k { 2.0 * alpha / n * f64::from(head.coef.data()[i]) } else { 0.0 }
            })
            .collect()
    }

    #[test]
    fn device_gradient_matches_float64_in_blocks_and_chunks() {
        let (n, d, k) = (1000usize, 7usize, 5usize);
        let wave = |i: usize, scale: f32| ((i as f32 * 0.618_034).fract() - 0.5) * scale;
        let x = Matrix::new(n, d, (0..n * d).map(|i| wave(i, 3.0)).collect());
        let y = Matrix::new(n, k, (0..n * k).map(|i| u8::from(wave(i * 13 + 5, 1.0) > 0.3)).collect());
        let head = LinearHead {
            coef: Matrix::new(d, k, (0..d * k).map(|i| wave(i * 7 + 1, 0.4)).collect()),
            intercept: (0..k).map(|j| wave(j + 3, 2.0)).collect(),
        };
        let expected = gradient_f64(&x, &y, 10.0, &head);
        for (chunk_rows, block_rows) in [(1 << 17, 1 << 11), (256, 64), (300, 1000)] {
            let options = SolverOptions { chunk_rows, block_rows, ..Default::default() };
            let ours = solver_gradient(&x, &y, 10.0, &head, &options, &Device::cpu()).unwrap();
            let error = ours.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            // float32 rounding of gradient components of order 0.1
            assert!(error < 1e-6, "chunks of {chunk_rows}, blocks of {block_rows}: error {error:e}");
        }
    }
}
