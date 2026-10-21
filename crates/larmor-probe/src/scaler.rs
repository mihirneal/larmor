//! `sklearn.preprocessing.StandardScaler` for f32 features.

use crate::matrix::Matrix;

/// A fitted standard scaler: per-feature mean and scale, both float64 as sklearn keeps them.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardScaler {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

impl StandardScaler {
    /// `StandardScaler().fit(x)` (`_data.py::StandardScaler.partial_fit` on dense input).
    ///
    /// The mean and the population variance are accumulated in float64 with the corrected two-pass formula of
    /// `extmath.py::_incremental_mean_and_var`. Features whose variance is within rounding of zero
    /// (`_is_constant_feature`) get scale 1, so they transform to 0 instead of blowing up.
    pub fn fit(x: &Matrix<f32>) -> Self {
        let n = x.rows() as f64;
        let mut sums = vec![0.0f64; x.cols()];
        for row in x.iter_rows() {
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += f64::from(value);
            }
        }
        let mean: Vec<f64> = sums.iter().map(|sum| sum / n).collect();

        let mut correction = vec![0.0f64; x.cols()];
        let mut squares = vec![0.0f64; x.cols()];
        for row in x.iter_rows() {
            for j in 0..x.cols() {
                let centered = f64::from(row[j]) - mean[j];
                correction[j] += centered;
                squares[j] += centered * centered;
            }
        }
        let variance: Vec<f64> =
            squares.iter().zip(&correction).map(|(sq, corr)| (sq - corr * corr / n) / n).collect();

        let scale = variance
            .iter()
            .zip(&mean)
            .map(|(&var, &mu)| if is_constant_feature(var, mu, n) { 1.0 } else { var.sqrt() })
            .collect();
        Self { mean, scale }
    }

    /// `scaler.transform(x)` on f32 input: sklearn subtracts and divides in place in float32, after casting
    /// the float64 mean and scale to float32, so the result is float32 rounded at each step.
    pub fn transform(&self, x: &Matrix<f32>) -> Matrix<f32> {
        assert_eq!(x.cols(), self.mean.len(), "scaler was fitted on a different number of features");
        let mean: Vec<f32> = self.mean.iter().map(|&m| m as f32).collect();
        let scale: Vec<f32> = self.scale.iter().map(|&s| s as f32).collect();
        let mut out = x.clone();
        for i in 0..out.rows() {
            for ((value, &mu), &sigma) in out.row_mut(i).iter_mut().zip(&mean).zip(&scale) {
                *value = (*value - mu) / sigma;
            }
        }
        out
    }
}

/// `_data.py::_is_constant_feature`: the variance is below the error bound of the two-pass algorithm.
fn is_constant_feature(variance: f64, mean: f64, n_samples: f64) -> bool {
    let eps = f64::EPSILON;
    let upper_bound = n_samples * eps * variance + (n_samples * mean * eps).powi(2);
    variance <= upper_bound
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_feature_gets_unit_scale() {
        let x = Matrix::new(3, 2, vec![1.0, 0.25, 2.0, 0.25, 4.0, 0.25]);
        let scaler = StandardScaler::fit(&x);
        assert_eq!(scaler.scale[1], 1.0);
        assert!((scaler.mean[0] - 7.0 / 3.0).abs() < 1e-15);
        let scaled = scaler.transform(&x);
        assert_eq!(scaled.get(0, 1), 0.0);
    }
}
