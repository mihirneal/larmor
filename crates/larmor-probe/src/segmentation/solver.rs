//! `probe.py::fit_logistic` solved to its optimum, with the data on a Burn device.
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
//! - the gradient is computed on the device in float32 per chunk of rows (each row with a trailing 1 so one
//!   product covers weights and intercept) and summed across chunks in float64.

use crate::matrix::Matrix;
use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData, activation};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};
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
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self { max_iter: 1000, history: 10, gradient_tol: 1e-8, stall_iterations: 30, chunk_rows: 1 << 17 }
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
) -> (LinearHead, FitStats) {
    let start = Instant::now();
    let problem = DeviceProblem::upload(features, targets, alpha, options.chunk_rows, device);
    let theta0 = match init {
        Some(head) => Theta::from_head(head),
        None => Theta::initial(features.cols(), targets),
    };
    let preconditioner = Preconditioner::new(&problem, targets, alpha);
    let setup_seconds = start.elapsed().as_secs_f64();
    let (theta, mut stats) = minimize(&problem, &preconditioner, theta0, options);
    stats.solve_seconds = start.elapsed().as_secs_f64() - setup_seconds;
    stats.setup_seconds = setup_seconds;
    stats.loss = problem.loss(&theta);
    stats.seconds = start.elapsed().as_secs_f64();
    (theta.to_head(problem.n_features, problem.n_outputs), stats)
}

/// `sigmoid(x @ coef + intercept)` on the device, returned row-major `[n, n_outputs]`.
pub fn predict_on_device(head: &LinearHead, standardized: &Matrix<f32>, device: &Device) -> Vec<f32> {
    if standardized.rows() == 0 {
        return Vec::new();
    }
    let x = upload(standardized.data(), [standardized.rows(), standardized.cols()], device);
    let (w, b) = upload_head(head, device);
    let probabilities = activation::sigmoid(x.matmul(w) + b);
    download(probabilities)
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

/// The training rows on the device, in chunks of rows.
struct DeviceProblem {
    chunks: Vec<(Tensor<2>, Tensor<2>)>,
    n_rows: usize,
    n_features: usize,
    n_outputs: usize,
    alpha: f64,
    device: Device,
}

impl DeviceProblem {
    /// Uploads the rows in chunks, each feature row with a trailing 1 so that one product gives the logits
    /// (`[x, 1] @ [W; b]`) and one product gives the weight and intercept gradients together.
    fn upload(
        features: &Matrix<f32>,
        targets: &Matrix<u8>,
        alpha: f64,
        chunk_rows: usize,
        device: &Device,
    ) -> Self {
        assert_eq!(features.rows(), targets.rows(), "one target row per feature row");
        let (n, d, k) = (features.rows(), features.cols(), targets.cols());
        let chunk_rows = chunk_rows.max(1);
        let chunks = (0..n)
            .step_by(chunk_rows)
            .map(|start| {
                let end = (start + chunk_rows).min(n);
                let mut augmented = Vec::with_capacity((end - start) * (d + 1));
                for i in start..end {
                    augmented.extend_from_slice(features.row(i));
                    augmented.push(1.0);
                }
                let y: Vec<f32> = targets.data()[start * k..end * k].iter().map(|&v| f32::from(v)).collect();
                (upload(&augmented, [end - start, d + 1], device), upload(&y, [end - start, k], device))
            })
            .collect();
        Self { chunks, n_rows: n, n_features: d, n_outputs: k, alpha, device: device.clone() }
    }

    /// The gradient of `F = sum_j [mean_i BCE_ij + alpha / n ||w_j||^2]`, i.e. fit_logistic's loss times
    /// `n_outputs`, so each output's block is the gradient of its own problem.
    fn gradient(&self, theta: &Theta) -> Vec<f64> {
        let (d, k, n) = (self.n_features, self.n_outputs, self.n_rows as f64);
        let theta_device = self.upload_theta(theta);
        let mut grad = vec![0.0f64; (d + 1) * k];
        for (x, y) in &self.chunks {
            let residual = activation::sigmoid(x.clone().matmul(theta_device.clone())) - y.clone();
            let partial = download(x.clone().transpose().matmul(residual));
            for (g, v) in grad.iter_mut().zip(&partial) {
                *g += f64::from(*v);
            }
        }
        let penalty = 2.0 * self.alpha / n;
        for (i, g) in grad.iter_mut().enumerate() {
            *g /= n;
            if i < d * k {
                *g += penalty * theta.values[i];
            }
        }
        grad
    }

    /// fit_logistic's loss: mean BCE over `n x n_outputs` plus `alpha ||W||^2 / (n n_outputs)`.
    fn loss(&self, theta: &Theta) -> f64 {
        let (d, k, n) = (self.n_features, self.n_outputs, self.n_rows as f64);
        let theta_device = self.upload_theta(theta);
        let mut total = 0.0f64;
        for (x, y) in &self.chunks {
            let logits = x.clone().matmul(theta_device.clone());
            let bce = activation::softplus(logits.clone(), 1.0) - y.clone() * logits;
            total += f64::from(download(bce.sum())[0]);
        }
        let squared_norm: f64 = theta.values[..d * k].iter().map(|v| v * v).sum();
        total / (n * k as f64) + self.alpha * squared_norm / (n * k as f64)
    }

    /// `X'X / n` in float64 (without the column of ones), for the preconditioner.
    fn gram(&self) -> DMatrix<f64> {
        let d = self.n_features;
        let mut gram = DMatrix::<f64>::zeros(d, d);
        for (x, _) in &self.chunks {
            let block = download(x.clone().transpose().matmul(x.clone()));
            for i in 0..d {
                for j in 0..d {
                    gram[(i, j)] += f64::from(block[i * (d + 1) + j]);
                }
            }
        }
        gram / self.n_rows as f64
    }

    /// `[W; b]` as one `[n_features + 1, n_outputs]` float32 tensor.
    fn upload_theta(&self, theta: &Theta) -> Tensor<2> {
        let values: Vec<f32> = theta.values.iter().map(|&v| v as f32).collect();
        upload(&values, [self.n_features + 1, self.n_outputs], &self.device)
    }
}

/// `H0^-1` of the preconditioned L-BFGS (see the module docs).
struct Preconditioner {
    vectors: DMatrix<f64>,
    /// `[n_features, n_outputs]`: the eigenvalues of each output's approximate Hessian.
    denominators: DMatrix<f64>,
    curvature: Vec<f64>,
    n_features: usize,
    n_outputs: usize,
}

impl Preconditioner {
    fn new(problem: &DeviceProblem, targets: &Matrix<u8>, alpha: f64) -> Self {
        let (d, k, n) = (problem.n_features, problem.n_outputs, problem.n_rows as f64);
        let eigen = SymmetricEigen::new(problem.gram());
        let curvature: Vec<f64> = column_counts(targets)
            .iter()
            .map(|&c| {
                let mean = c as f64 / n;
                (mean * (1.0 - mean)).max(1e-4)
            })
            .collect();
        let denominators =
            DMatrix::from_fn(d, k, |i, j| curvature[j] * eigen.eigenvalues[i].max(0.0) + 2.0 * alpha / n);
        Self { vectors: eigen.eigenvectors, denominators, curvature, n_features: d, n_outputs: k }
    }

    fn apply(&self, r: &[f64]) -> Vec<f64> {
        let (d, k) = (self.n_features, self.n_outputs);
        // r is row-major [d + 1, k]; nalgebra is column-major, so read it as its transpose
        let weights = DMatrix::from_row_slice(d, k, &r[..d * k]);
        let rotated = self.vectors.transpose() * weights;
        let scaled = rotated.component_div(&self.denominators);
        let back = &self.vectors * scaled;
        let mut out = Vec::with_capacity((d + 1) * k);
        for i in 0..d {
            out.extend((0..k).map(|j| back[(i, j)]));
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
) -> (Theta, FitStats) {
    let mut theta = theta0;
    let mut grad = problem.gradient(&theta);
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
    };
    if best <= options.gradient_tol {
        return (theta, stats(0, evaluations, best, SolverStop::Converged));
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
        let (step, new_grad, trials) = slope_search(problem, &theta, &direction, slope0);
        evaluations += trials;
        let s: Vec<f64> = direction.iter().map(|d| step * d).collect();
        let y: Vec<f64> = new_grad.iter().zip(&grad).map(|(a, b)| a - b).collect();
        for (t, si) in theta.values.iter_mut().zip(&s) {
            *t += si;
        }
        grad = new_grad;

        let gradient_max = max_abs(&grad);
        if gradient_max <= options.gradient_tol {
            return (theta, stats(iteration, evaluations, gradient_max, SolverStop::Converged));
        }
        stalled = if gradient_max > 0.9 * best { stalled + 1 } else { 0 };
        best = best.min(gradient_max);
        if stalled >= options.stall_iterations {
            return (theta, stats(iteration, evaluations, gradient_max, SolverStop::Stalled));
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
    (theta, stats(options.max_iter, evaluations, gradient_max, SolverStop::MaxIter))
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
) -> (f64, Vec<f64>, usize) {
    const CURVATURE: f64 = 0.9;
    const MAX_TRIALS: usize = 30;
    let (mut low, mut slope_low) = (0.0, slope0);
    let mut high: Option<(f64, f64)> = None;
    let mut step = 1.0;
    let mut trials = 0;
    loop {
        let trial = Theta { values: theta.values.iter().zip(direction).map(|(t, d)| t + step * d).collect() };
        let grad = problem.gradient(&trial);
        trials += 1;
        let slope = dot(&grad, direction);
        if slope.abs() <= CURVATURE * -slope0 || trials >= MAX_TRIALS {
            return (step, grad, trials);
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

fn upload(values: &[f32], shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(
        TensorData::new(values.to_vec(), shape),
        TensorCreationOptions::from((device, DType::F32)),
    )
}

fn upload_head(head: &LinearHead, device: &Device) -> (Tensor<2>, Tensor<2>) {
    let (d, k) = (head.coef.rows(), head.coef.cols());
    (upload(head.coef.data(), [d, k], device), upload(&head.intercept, [1, k], device))
}

fn download<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor.into_data().try_into_vec::<f32>().expect("float32 tensor")
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
