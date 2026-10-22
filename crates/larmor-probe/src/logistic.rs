//! `sklearn.linear_model.LogisticRegressionCV` as probe_binary_classification configures it: L2 penalty,
//! `Cs=np.logspace(-4, 4, 9)`, `scoring="neg_log_loss"`, `class_weight="balanced"`, `max_iter=1000`,
//! inner `StratifiedKFold(5)` without shuffling, refit at the best C.

use crate::folds::{FoldError, stratified_kfold};
use crate::lbfgsb::{self, LbfgsbOptions};
use crate::matrix::Matrix;
use crate::numpy;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// The inverse penalties probe_binary_classification searches (`probe.py::LOGISTIC_CS`).
pub fn logistic_cs() -> Vec<f64> {
    numpy::logspace_int(-4, 4)
}

/// How each penalised fit is solved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogisticSolver {
    /// sklearn's own path: scipy L-BFGS-B stopped at `tol`, warm-started along the Cs. Reproduces the eval
    /// framework's coefficients, which are `tol`-accurate rather than optimal.
    #[default]
    SklearnLbfgs,
    /// Newton's method to the optimum (gradient below 1e-10). Differs from the eval framework by sklearn's
    /// stopping error, about 1e-4 in the probabilities.
    Newton,
}

/// Settings of `LogisticRegressionCV`; the defaults are probe_binary_classification's.
#[derive(Clone, Debug)]
pub struct LogisticCvOptions {
    pub cs: Vec<f64>,
    pub n_inner_folds: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub solver: LogisticSolver,
}

impl Default for LogisticCvOptions {
    fn default() -> Self {
        Self {
            cs: logistic_cs(),
            n_inner_folds: 5,
            max_iter: 1000,
            tol: 1e-4,
            solver: LogisticSolver::default(),
        }
    }
}

/// A fitted `LogisticRegressionCV`.
#[derive(Clone, Debug)]
pub struct LogisticFit {
    /// `C_`.
    pub c: f64,
    /// `coef_[0]`, one weight per feature.
    pub coef: Vec<f64>,
    /// `intercept_[0]`.
    pub intercept: f64,
    /// `scores_[fold, 0, c]`: the inner-fold negative log loss for every C.
    pub scores: Vec<Vec<f64>>,
    /// `coefs_paths_[fold, 0, c, 0, :]`: weights then intercept, for every inner fold and C.
    pub coefs_paths: Vec<Vec<Vec<f64>>>,
    /// `n_iter_[fold, 0, c]`, plus the refit's count last.
    pub n_iter: Vec<Vec<usize>>,
}

impl LogisticFit {
    /// `decision_function(x)` for float32 features: a float64 product, as numpy promotes it.
    pub fn decision_function(&self, x: &Matrix<f32>) -> Vec<f64> {
        x.iter_rows()
            .map(|row| {
                row.iter().zip(&self.coef).map(|(&v, w)| f64::from(v) * w).sum::<f64>() + self.intercept
            })
            .collect()
    }

    /// `predict_proba(x)[:, 1]`.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Vec<f64> {
        self.decision_function(x).into_iter().map(expit).collect()
    }
}

/// `LogisticRegressionCV(...).fit(x, labels)` for binary labels (`_logistic.py::LogisticRegressionCV.fit`).
///
/// - The balanced class weights come from all of `labels` (the outer training fold), not per inner fold; they
///   weight the training loss only, while the inner scores are the unweighted log loss.
/// - The objective of each fit is `sum_i s_i loss_i / sum_i s_i + ||w||^2 / (2 C sum_i s_i)`, the intercept
///   unpenalised.
/// - C_ maximises the inner scores summed over folds, the first C on ties. The refit on all of `labels`
///   starts from the mean of the inner folds' coefficients at that C.
pub fn logistic_regression_cv(
    x: &Matrix<f32>,
    labels: &[bool],
    options: &LogisticCvOptions,
) -> Result<LogisticFit, FoldError> {
    // sklearn's check_array: "Input X contains NaN"
    if x.data().iter().any(|v| !v.is_finite()) {
        return Err(FoldError::NonFinite);
    }
    // below n_inner_folds members of a class, the unshuffled inner StratifiedKFold leaves a validation
    // fold with one class (or a training fold with none), and sklearn raises; so does this
    let positives = labels.iter().filter(|&&l| l).count();
    let negatives = labels.len() - positives;
    if negatives.min(positives) < options.n_inner_folds {
        return Err(FoldError::TooFewInClass { negatives, positives, n_splits: options.n_inner_folds });
    }
    let x64 = x.to_f64();
    let class_weight = balanced_class_weights(labels);
    let folds = stratified_kfold(labels, options.n_inner_folds, None)?;

    let mut scores = Vec::with_capacity(folds.len());
    let mut coefs_paths = Vec::with_capacity(folds.len());
    let mut n_iter = Vec::with_capacity(folds.len() + 1);
    for fold in &folds {
        let train = Problem::new(&x64, labels, &fold.train, class_weight);
        let (path, iterations) = logistic_path(&train, &options.cs, None, options)?;
        let test_x = x64.select_rows(&fold.test);
        let test_labels: Vec<bool> = fold.test.iter().map(|&i| labels[i]).collect();
        scores.push(
            path.iter().map(|w| -log_loss(&test_labels, &probabilities(&test_x, w))).collect::<Vec<_>>(),
        );
        coefs_paths.push(path);
        n_iter.push(iterations);
    }

    let best = argmax_first(&sum_over_folds(&scores));
    let c = options.cs[best];
    let n_dof = x.cols() + 1;
    let coef_init: Vec<f64> = (0..n_dof)
        .map(|k| coefs_paths.iter().map(|path| path[best][k]).sum::<f64>() / folds.len() as f64)
        .collect();
    let all: Vec<usize> = (0..labels.len()).collect();
    let problem = Problem::new(&x64, labels, &all, class_weight);
    let (refit, iterations) = logistic_path(&problem, &[c], Some(&coef_init), options)?;
    n_iter.push(iterations);
    let w = &refit[0];
    Ok(LogisticFit { c, coef: w[..x.cols()].to_vec(), intercept: w[x.cols()], scores, coefs_paths, n_iter })
}

/// `compute_class_weight("balanced")`: `n / (2 n_k)` for class k, indexed `[false, true]`.
pub fn balanced_class_weights(labels: &[bool]) -> [f64; 2] {
    let positives = labels.iter().filter(|&&l| l).count() as f64;
    let negatives = labels.len() as f64 - positives;
    let total = negatives + positives;
    [total / (2.0 * negatives), total / (2.0 * positives)]
}

/// The objective one inner fit minimises, with its gradient (`LinearModelLoss.loss_gradient`): `x` are the
/// scaled features of the fit's rows, `w` is weights then intercept.
pub fn penalized_objective(
    x: &Matrix<f32>,
    labels: &[bool],
    class_weight: [f64; 2],
    c: f64,
    w: &[f64],
) -> (f64, Vec<f64>) {
    let rows: Vec<usize> = (0..labels.len()).collect();
    let problem = Problem::new(&x.to_f64(), labels, &rows, class_weight);
    problem.loss_gradient(w, 1.0 / (c * problem.weight_sum))
}

/// One weighted binary problem: the rows of a fold, their 0/1 targets and sample weights.
struct Problem {
    x: Matrix<f64>,
    y: Vec<f64>,
    sample_weight: Vec<f64>,
    weight_sum: f64,
}

impl Problem {
    fn new(x: &Matrix<f64>, labels: &[bool], rows: &[usize], class_weight: [f64; 2]) -> Self {
        let y: Vec<f64> = rows.iter().map(|&i| f64::from(u8::from(labels[i]))).collect();
        let sample_weight: Vec<f64> = rows.iter().map(|&i| class_weight[usize::from(labels[i])]).collect();
        let weight_sum = numpy::sum(&sample_weight);
        Self { x: x.select_rows(rows), y, sample_weight, weight_sum }
    }

    fn n_features(&self) -> usize {
        self.x.cols()
    }

    fn raw_predictions(&self, w: &[f64]) -> Vec<f64> {
        let d = self.n_features();
        self.x
            .iter_rows()
            .map(|row| row.iter().zip(&w[..d]).map(|(a, b)| a * b).sum::<f64>() + w[d])
            .collect()
    }

    /// `LinearModelLoss(HalfBinomialLoss()).loss_gradient(w, X, y, sample_weight, l2_reg_strength)`.
    fn loss_gradient(&self, w: &[f64], l2: f64) -> (f64, Vec<f64>) {
        let d = self.n_features();
        let raw = self.raw_predictions(w);
        let mut losses = Vec::with_capacity(raw.len());
        let mut pointwise = Vec::with_capacity(raw.len());
        for ((&r, &y), &s) in raw.iter().zip(&self.y).zip(&self.sample_weight) {
            let (loss, gradient) = half_binomial_loss_gradient(y, r);
            losses.push(s * loss);
            pointwise.push(s * gradient / self.weight_sum);
        }
        let weights = &w[..d];
        let squared_norm: f64 = weights.iter().map(|v| v * v).sum();
        let loss = numpy::sum(&losses) / self.weight_sum + 0.5 * l2 * squared_norm;
        let mut grad = vec![0.0; d + 1];
        for (row, &p) in self.x.iter_rows().zip(&pointwise) {
            for (g, &v) in grad[..d].iter_mut().zip(row) {
                *g += v * p;
            }
        }
        for (g, &v) in grad[..d].iter_mut().zip(weights) {
            *g += l2 * v;
        }
        grad[d] = numpy::sum(&pointwise);
        (loss, grad)
    }
}

/// `_loss.pyx::closs_grad_half_binomial`: `log(1 + e^r) - y r` and `expit(r) - y`, with sklearn's branches.
fn half_binomial_loss_gradient(y: f64, raw: f64) -> (f64, f64) {
    if raw <= -37.0 {
        let e = raw.exp();
        (e - y * raw, e - y)
    } else if raw <= -2.0 {
        let e = raw.exp();
        (e.ln_1p() - y * raw, ((1.0 - y) * e - y) / (1.0 + e))
    } else if raw <= 18.0 {
        let e = (-raw).exp();
        (e.ln_1p() + (1.0 - y) * raw, ((1.0 - y) - y * e) / (1.0 + e))
    } else {
        let e = (-raw).exp();
        (e + (1.0 - y) * raw, ((1.0 - y) - y * e) / (1.0 + e))
    }
}

/// `_logistic_regression_path` for the given Cs: each fit starts from the previous solution (zeros, or
/// `init`, for the first). Returns weights-then-intercept per C and the iteration counts.
fn logistic_path(
    problem: &Problem,
    cs: &[f64],
    init: Option<&[f64]>,
    options: &LogisticCvOptions,
) -> Result<(Vec<Vec<f64>>, Vec<usize>), FoldError> {
    let mut w = init.map_or_else(|| vec![0.0; problem.n_features() + 1], <[f64]>::to_vec);
    let mut path = Vec::with_capacity(cs.len());
    let mut iterations = Vec::with_capacity(cs.len());
    for &c in cs {
        let l2 = 1.0 / (c * problem.weight_sum);
        let (solution, n_iter) = match options.solver {
            LogisticSolver::SklearnLbfgs => {
                let lbfgs = LbfgsbOptions {
                    memory: 10,
                    gtol: options.tol,
                    ftol: 64.0 * f64::EPSILON,
                    max_iter: options.max_iter,
                    max_line_search: 50,
                };
                let minimum = lbfgsb::minimize(|v| problem.loss_gradient(v, l2), &w, &lbfgs);
                // finite features keep the loss finite; a NaN point must not pass for a solution
                if minimum.stop == lbfgsb::Stop::NonFinite {
                    return Err(FoldError::NonFinite);
                }
                (minimum.x, minimum.n_iter.min(options.max_iter))
            }
            LogisticSolver::Newton => newton(problem, l2, &w),
        };
        if solution.iter().any(|v| !v.is_finite()) {
            return Err(FoldError::NonFinite);
        }
        w = solution;
        path.push(w.clone());
        iterations.push(n_iter);
    }
    Ok((path, iterations))
}

/// Newton's method with backtracking on the same objective, until the gradient is at rounding level.
fn newton(problem: &Problem, l2: f64, w0: &[f64]) -> (Vec<f64>, usize) {
    const GRADIENT_TOL: f64 = 1e-14;
    const MAX_ITER: usize = 100;
    let mut w = w0.to_vec();
    let (mut loss, mut grad) = problem.loss_gradient(&w, l2);
    for iteration in 0..MAX_ITER {
        if max_abs(&grad) <= GRADIENT_TOL {
            return (w, iteration);
        }
        let step = newton_step(problem, l2, &w, &grad);
        // backtrack until the objective does not increase beyond rounding; Newton steps on a logistic loss
        // can overshoot far from the optimum
        let mut scale = 1.0;
        let mut accepted = false;
        while scale > 1e-10 {
            let candidate: Vec<f64> = w.iter().zip(step.iter()).map(|(wi, si)| wi - scale * si).collect();
            let (candidate_loss, candidate_grad) = problem.loss_gradient(&candidate, l2);
            if candidate_loss <= loss + 4.0 * f64::EPSILON * loss.abs() {
                w = candidate;
                loss = candidate_loss;
                grad = candidate_grad;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return (w, iteration + 1);
        }
    }
    (w, MAX_ITER)
}

/// The Newton direction `H^-1 g`, with the Hessian `X~' diag(s p (1 - p)) X~ / sum(s) + l2` (intercept free).
fn newton_step(problem: &Problem, l2: f64, w: &[f64], grad: &[f64]) -> DVector<f64> {
    let d = problem.n_features();
    let raw = problem.raw_predictions(w);
    let mut hessian = DMatrix::<f64>::zeros(d + 1, d + 1);
    for (i, row) in problem.x.iter_rows().enumerate() {
        let p = expit(raw[i]);
        let h = problem.sample_weight[i] * p * (1.0 - p) / problem.weight_sum;
        let augmented: Vec<f64> = row.iter().copied().chain(std::iter::once(1.0)).collect();
        for a in 0..=d {
            let ha = h * augmented[a];
            for b in a..=d {
                hessian[(a, b)] += ha * augmented[b];
            }
        }
    }
    for a in 0..=d {
        if a < d {
            hessian[(a, a)] += l2;
        }
        for b in 0..a {
            hessian[(a, b)] = hessian[(b, a)];
        }
    }
    let gradient = DVector::from_column_slice(grad);
    match hessian.clone().cholesky() {
        Some(cholesky) => cholesky.solve(&gradient),
        // the intercept direction can lose curvature when every probability saturates
        None => hessian.lu().solve(&gradient).unwrap_or(gradient),
    }
}

/// The largest absolute value, NaN if any value is NaN, so a NaN gradient never counts as converged.
fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |m, v| if m.is_nan() || v.is_nan() { f64::NAN } else { m.max(v.abs()) })
}

fn probabilities(x: &Matrix<f64>, w: &[f64]) -> Vec<f64> {
    let d = x.cols();
    x.iter_rows().map(|row| expit(row.iter().zip(&w[..d]).map(|(a, b)| a * b).sum::<f64>() + w[d])).collect()
}

/// `scipy.special.expit`.
pub fn expit(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `sklearn.metrics.log_loss` for binary labels: probabilities clipped to `[eps, 1 - eps]`.
pub fn log_loss(labels: &[bool], probabilities: &[f64]) -> f64 {
    let eps = f64::EPSILON;
    let losses: Vec<f64> = labels
        .iter()
        .zip(probabilities)
        .map(|(&label, &p)| {
            let q = if label { p } else { 1.0 - p };
            -q.clamp(eps, 1.0 - eps).ln()
        })
        .collect();
    numpy::mean(&losses)
}

fn sum_over_folds(scores: &[Vec<f64>]) -> Vec<f64> {
    (0..scores[0].len()).map(|c| scores.iter().fold(0.0, |acc, fold| acc + fold[c])).collect()
}

/// `np.argmax`: the first maximum.
fn argmax_first(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (i, &v)| if v > best.1 { (i, v) } else { best })
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_of_a_class_or_non_finite_features_are_refused_like_sklearn() {
        let mut state = 1u64;
        let mut noise = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let x = Matrix::new(34, 3, (0..34 * 3).map(|_| noise()).collect());
        let options = LogisticCvOptions::default();
        for (positives, ok) in [(4, false), (5, true)] {
            let labels: Vec<bool> = (0..34).map(|i| i % 6 == 0 && i / 6 < positives).collect();
            let result = logistic_regression_cv(&x, &labels, &options);
            match ok {
                true => assert!(result.is_ok()),
                false => assert_eq!(
                    result.unwrap_err(),
                    FoldError::TooFewInClass { negatives: 30, positives: 4, n_splits: 5 }
                ),
            }
        }
        let labels: Vec<bool> = (0..34).map(|i| i % 2 == 0).collect();
        let mut data = x.data().to_vec();
        data[7] = f32::NAN;
        let err = logistic_regression_cv(&Matrix::new(34, 3, data), &labels, &options).unwrap_err();
        assert_eq!(err, FoldError::NonFinite);
        assert!(max_abs(&[0.5, f64::NAN]).is_nan());
    }

    #[test]
    fn balanced_weights_sum_to_n() {
        let labels = [true, false, false, false];
        let [w0, w1] = balanced_class_weights(&labels);
        assert_eq!((w0, w1), (4.0 / 6.0, 2.0));
    }

    #[test]
    fn loss_branches_are_continuous() {
        for &y in &[0.0, 1.0] {
            for &r in &[-37.0, -2.0, 18.0] {
                let (below, _) = half_binomial_loss_gradient(y, r);
                let (above, _) = half_binomial_loss_gradient(y, r + 1e-12);
                assert!((below - above).abs() < 1e-9);
            }
        }
    }
}
