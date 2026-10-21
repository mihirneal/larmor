//! `sklearn.linear_model.RidgeCV` with its default efficient leave-one-out (`_ridge.py::_RidgeGCV`).

use crate::matrix::Matrix;
use crate::numpy;
use nalgebra::{DMatrix, DVector, SVD, SymmetricEigen};

/// The ridge penalties probe_regression searches: `np.logspace(-2, 6, 9)` (`probe.py::RIDGE_ALPHAS`).
pub fn ridge_alphas() -> Vec<f64> {
    numpy::logspace_int(-2, 6)
}

/// How the leave-one-out errors are computed; both give the same numbers up to rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GcvMode {
    /// Eigendecomposition of the Gram matrix `X X^T`, sklearn's choice when `n_samples <= n_features`.
    Eigen,
    /// Thin SVD of `X`, sklearn's choice when `n_samples > n_features`.
    Svd,
}

impl GcvMode {
    /// `_ridge.py::_check_gcv_mode(X, "auto")`.
    pub fn auto(n_samples: usize, n_features: usize) -> Self {
        if n_samples > n_features { GcvMode::Svd } else { GcvMode::Eigen }
    }
}

/// A fitted `RidgeCV`: the chosen penalty and the model refit with it.
#[derive(Clone, Debug, PartialEq)]
pub struct RidgeFit {
    pub alpha: f64,
    /// `coef_`, cast back to float32 because the input features were float32.
    pub coef: Vec<f32>,
    /// `intercept_`, float32 for the same reason.
    pub intercept: f32,
    /// Mean squared leave-one-out error for every candidate alpha; the chosen alpha minimises it.
    pub loo_mse: Vec<f64>,
    pub mode: GcvMode,
}

impl RidgeFit {
    /// `RidgeCV.predict(x)` on float32 features: a float32 matrix-vector product plus the intercept.
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<f64> {
        x.iter_rows()
            .map(|row| {
                // BLAS sums in float32; summing in float64 and rounding once differs by an ulp at most
                let dot: f64 = row.iter().zip(&self.coef).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum();
                f64::from(dot as f32 + self.intercept)
            })
            .collect()
    }
}

/// `RidgeCV(alphas).fit(x, y)` with `cv=None`, `fit_intercept=True`, no sample weights (`_RidgeGCV.fit`).
///
/// `x` is centred (in float64), the intercept is left unpenalised by giving the constant direction an
/// infinite eigenvalue, and each alpha is scored by the mean squared leave-one-out error
/// `(c / diag(G^-1))^2`. The first alpha with the best score wins (a strict `>` in sklearn).
pub fn ridge_cv(x: &Matrix<f32>, y: &[f64], alphas: &[f64]) -> RidgeFit {
    let (n, d) = (x.rows(), x.cols());
    assert_eq!(y.len(), n, "one target per sample");
    let x_offset = column_means(x);
    let centered = DMatrix::from_fn(n, d, |i, j| f64::from(x.get(i, j)) - x_offset[j]);
    let y_offset = numpy::mean(y);
    let y_centered = DVector::from_iterator(n, y.iter().map(|v| v - y_offset));

    let mode = GcvMode::auto(n, d);
    let decomposition = match mode {
        GcvMode::Eigen => Decomposition::eigen_gram(&centered, &y_centered),
        GcvMode::Svd => Decomposition::svd_design(&centered, &y_centered),
    };

    let mut best: Option<(f64, f64, DVector<f64>)> = None;
    let mut loo_mse = Vec::with_capacity(alphas.len());
    for &alpha in alphas {
        let (g_inverse_diag, dual_coef) = decomposition.solve(alpha, &y_centered);
        let squared_errors: Vec<f64> =
            dual_coef.iter().zip(g_inverse_diag.iter()).map(|(c, g)| (c / g).powi(2)).collect();
        let score = -numpy::mean(&squared_errors);
        loo_mse.push(-score);
        if best.as_ref().is_none_or(|(best_score, _, _)| score > *best_score) {
            best = Some((score, alpha, dual_coef));
        }
    }
    let (_, alpha, dual_coef) = best.expect("at least one alpha");

    // coef_ = dual_coef_ @ X (centred), intercept_ = y_offset - X_offset @ coef_, then both cast to float32
    let coef64 = centered.transpose() * &dual_coef;
    let intercept64 = y_offset - coef64.iter().zip(&x_offset).map(|(c, m)| c * m).sum::<f64>();
    RidgeFit {
        alpha,
        coef: coef64.iter().map(|&c| c as f32).collect(),
        intercept: intercept64 as f32,
        loo_mse,
        mode,
    }
}

/// `np.average(X, axis=0)`: an axis-0 reduction adds the rows in order.
fn column_means(x: &Matrix<f32>) -> Vec<f64> {
    let mut sums = vec![0.0f64; x.cols()];
    for row in x.iter_rows() {
        for (sum, &v) in sums.iter_mut().zip(row) {
            *sum += f64::from(v);
        }
    }
    sums.iter().map(|s| s / x.rows() as f64).collect()
}

enum Decomposition {
    /// `_eigen_decompose_gram`: eigenpairs of `X X^T + 1 1^T` and `Q^T y`.
    EigenGram { eigenvalues: DVector<f64>, vectors: DMatrix<f64>, qt_y: DVector<f64>, intercept_dim: usize },
    /// `_svd_decompose_design_matrix`: thin SVD of `[X, 1]`, squared singular values and `U^T y`.
    SvdDesign { singular_sq: DVector<f64>, u: DMatrix<f64>, ut_y: DVector<f64>, intercept_dim: usize },
}

impl Decomposition {
    fn eigen_gram(centered: &DMatrix<f64>, y: &DVector<f64>) -> Self {
        let n = centered.nrows();
        // the column of ones emulates the intercept; centring makes it orthogonal to X's columns
        let gram = centered * centered.transpose() + DMatrix::from_element(n, n, 1.0);
        let eigen = SymmetricEigen::new(gram);
        let qt_y = eigen.eigenvectors.transpose() * y;
        let intercept_dim = smallest_angle_to_ones(&eigen.eigenvectors);
        Decomposition::EigenGram {
            eigenvalues: eigen.eigenvalues,
            vectors: eigen.eigenvectors,
            qt_y,
            intercept_dim,
        }
    }

    fn svd_design(centered: &DMatrix<f64>, y: &DVector<f64>) -> Self {
        let (n, d) = centered.shape();
        let augmented = centered.clone().insert_column(d, 1.0);
        let svd = SVD::new(augmented, true, false);
        let u = svd.u.expect("SVD computed with U");
        let singular_sq = svd.singular_values.map(|s| s * s);
        let ut_y = u.transpose() * y;
        let intercept_dim = smallest_angle_to_ones(&u);
        debug_assert_eq!(u.nrows(), n);
        Decomposition::SvdDesign { singular_sq, u, ut_y, intercept_dim }
    }

    /// `(diag(G^-1), c)` for one alpha (`_solve_eigen_gram` / `_solve_svd_design_matrix`).
    fn solve(&self, alpha: f64, y: &DVector<f64>) -> (DVector<f64>, DVector<f64>) {
        match self {
            Decomposition::EigenGram { eigenvalues, vectors, qt_y, intercept_dim } => {
                let mut w = eigenvalues.map(|e| 1.0 / (e + alpha));
                // no penalty on the intercept: its eigenvector behaves as if its eigenvalue were infinite
                w[*intercept_dim] = 0.0;
                let dual_coef = vectors * w.component_mul(qt_y);
                (weighted_row_squares(vectors, &w), dual_coef)
            }
            Decomposition::SvdDesign { singular_sq, u, ut_y, intercept_dim } => {
                let mut w = singular_sq.map(|s| 1.0 / (s + alpha) - 1.0 / alpha);
                w[*intercept_dim] = -1.0 / alpha;
                let dual_coef = u * w.component_mul(ut_y) + y / alpha;
                (weighted_row_squares(u, &w).add_scalar(1.0 / alpha), dual_coef)
            }
        }
    }
}

/// `_decomp_diag`: `sum_j w_j Q_ij^2` for every row `i`.
fn weighted_row_squares(q: &DMatrix<f64>, w: &DVector<f64>) -> DVector<f64> {
    DVector::from_iterator(
        q.nrows(),
        (0..q.nrows()).map(|i| (0..q.ncols()).map(|j| w[j] * q[(i, j)].powi(2)).sum()),
    )
}

/// `_find_smallest_angle(ones / ||ones||, vectors)`: the column most aligned with the constant vector.
fn smallest_angle_to_ones(vectors: &DMatrix<f64>) -> usize {
    let norm = (vectors.nrows() as f64).sqrt();
    let cosines = (0..vectors.ncols()).map(|j| (vectors.column(j).sum() / norm).abs());
    // np.argmax: the first maximum
    cosines.enumerate().fold((0, f64::NEG_INFINITY), |best, (j, c)| if c > best.1 { (j, c) } else { best }).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) as f32
            })
            .collect()
    }

    #[test]
    fn eigen_and_svd_modes_agree() {
        let (n, d) = (30, 12);
        let x = Matrix::new(n, d, pseudo_random(n * d, 1));
        let y: Vec<f64> = (0..n).map(|i| x.row(i)[0] as f64 * 3.0 + (i as f64 * 0.37).sin()).collect();
        let centered = DMatrix::from_fn(n, d, |i, j| f64::from(x.get(i, j)) - column_means(&x)[j]);
        let y_offset = numpy::mean(&y);
        let yc = DVector::from_iterator(n, y.iter().map(|v| v - y_offset));
        let eigen = Decomposition::eigen_gram(&centered, &yc);
        let svd = Decomposition::svd_design(&centered, &yc);
        for alpha in [0.1, 1.0, 100.0] {
            let (g1, c1) = eigen.solve(alpha, &yc);
            let (g2, c2) = svd.solve(alpha, &yc);
            for i in 0..n {
                assert!((c1[i] / g1[i] - c2[i] / g2[i]).abs() < 1e-8, "alpha {alpha} sample {i}");
            }
        }
    }
}
