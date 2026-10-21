//! The regression probe: `probe.py::probe_regression` from the pooled embeddings on.

use crate::bootstrap::{self, take};
use crate::folds::{Fold, FoldError, kfold};
use crate::matrix::Matrix;
use crate::metrics::{mean_absolute_error, pearson_r, r2_score};
use crate::ridge::{RidgeFit, ridge_alphas, ridge_cv};
use crate::scaler::StandardScaler;
use serde::{Deserialize, Serialize};

/// Outer folds and inner fits use this seed (`probe.py::CV_SEED`).
pub const CV_SEED: u32 = 0;

/// The outer folds of probe_regression: `KFold(5, shuffle=True, random_state=0)`.
pub fn regression_folds(n_samples: usize) -> Result<Vec<Fold>, FoldError> {
    kfold(n_samples, 5, Some(CV_SEED))
}

/// `make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))`, fitted.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionModel {
    pub scaler: StandardScaler,
    pub ridge: RidgeFit,
}

impl RegressionModel {
    /// `pipeline.fit(features, targets)`.
    pub fn fit(features: &Matrix<f32>, targets: &[f64]) -> Self {
        let scaler = StandardScaler::fit(features);
        let ridge = ridge_cv(&scaler.transform(features), targets, &ridge_alphas());
        Self { scaler, ridge }
    }

    /// `pipeline.predict(features)`.
    pub fn predict(&self, features: &Matrix<f32>) -> Vec<f64> {
        self.ridge.predict(&self.scaler.transform(features))
    }
}

/// The numbers probe_regression reports (its result dict without the embedding bookkeeping).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionReport {
    pub mae: f64,
    pub mae_ci: [f64; 2],
    pub r: f64,
    pub r_ci: [f64; 2],
    pub r2: f64,
    pub r2_ci: [f64; 2],
    pub fold_alpha: Vec<f64>,
    /// Out-of-fold predictions, one per sample.
    pub predictions: Vec<f64>,
    pub n_samples: usize,
    pub embed_dim: usize,
}

/// The outer cross-validation and the models fitted in each fold.
#[derive(Clone, Debug)]
pub struct RegressionCv {
    pub report: RegressionReport,
    pub fold_models: Vec<RegressionModel>,
}

/// `probe_regression` on given features: a scaler and RidgeCV per outer fold, out-of-fold predictions, then
/// MAE, Pearson r and R^2 with bootstrap CIs.
pub fn probe_regression(features: &Matrix<f32>, targets: &[f64], folds: &[Fold]) -> RegressionCv {
    let n = targets.len();
    assert_eq!(features.rows(), n, "one feature row per target");
    let mut predictions = vec![0.0; n];
    let mut fold_models = Vec::with_capacity(folds.len());
    for fold in folds {
        let model = RegressionModel::fit(&features.select_rows(&fold.train), &take(targets, &fold.train));
        let fold_predictions = model.predict(&features.select_rows(&fold.test));
        for (&i, p) in fold.test.iter().zip(fold_predictions) {
            predictions[i] = p;
        }
        fold_models.push(model);
    }

    let (mut mae, mut r, mut r2) = (Vec::new(), Vec::new(), Vec::new());
    for ids in bootstrap::resamples(n) {
        let (t, p) = (take(targets, &ids), take(&predictions, &ids));
        mae.push(mean_absolute_error(&t, &p));
        r.push(pearson_r(&t, &p));
        r2.push(r2_score(&t, &p));
    }
    let report = RegressionReport {
        mae: mean_absolute_error(targets, &predictions),
        mae_ci: bootstrap::ci(&mae),
        r: pearson_r(targets, &predictions),
        r_ci: bootstrap::ci(&r),
        r2: r2_score(targets, &predictions),
        r2_ci: bootstrap::ci(&r2),
        fold_alpha: fold_models.iter().map(|m| m.ridge.alpha).collect(),
        predictions,
        n_samples: n,
        embed_dim: features.cols(),
    };
    RegressionCv { report, fold_models }
}
