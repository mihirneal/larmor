//! The binary classification probe: `probe.py::probe_binary_classification` from the pooled embeddings on.

use crate::bootstrap::{self, take};
use crate::folds::{Fold, FoldError, stratified_kfold};
use crate::logistic::{LogisticCvOptions, LogisticFit, logistic_regression_cv};
use crate::matrix::Matrix;
use crate::metrics::{balanced_accuracy_score, roc_auc_score};
use crate::regression::CV_SEED;
use crate::scaler::StandardScaler;
use serde::{Deserialize, Serialize};

/// The outer folds of probe_binary_classification: `StratifiedKFold(5, shuffle=True, random_state=0)`.
pub fn classification_folds(labels: &[bool]) -> Result<Vec<Fold>, FoldError> {
    stratified_kfold(labels, 5, Some(CV_SEED))
}

/// `make_pipeline(StandardScaler(), LogisticRegressionCV(...))`, fitted.
#[derive(Clone, Debug)]
pub struct ClassificationModel {
    pub scaler: StandardScaler,
    pub logistic: LogisticFit,
}

impl ClassificationModel {
    /// `pipeline.fit(features, labels)`.
    pub fn fit(
        features: &Matrix<f32>,
        labels: &[bool],
        options: &LogisticCvOptions,
    ) -> Result<Self, FoldError> {
        let scaler = StandardScaler::fit(features);
        let logistic = logistic_regression_cv(&scaler.transform(features), labels, options)?;
        Ok(Self { scaler, logistic })
    }

    /// `pipeline.predict_proba(features)[:, 1]`.
    pub fn predict_proba(&self, features: &Matrix<f32>) -> Vec<f64> {
        self.logistic.predict_proba(&self.scaler.transform(features))
    }
}

/// The numbers probe_binary_classification reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassificationReport {
    pub auroc: f64,
    pub auroc_ci: [f64; 2],
    /// At the 0.5 cutoff, which the balanced class weights make sensible.
    pub balanced_accuracy: f64,
    pub balanced_accuracy_ci: [f64; 2],
    pub fold_c: Vec<f64>,
    /// Out-of-fold probabilities of the positive class.
    pub probabilities: Vec<f64>,
    pub n_samples: usize,
    pub embed_dim: usize,
    /// Bootstrap resamples with both classes (the others are skipped, AUROC being undefined).
    pub n_bootstrap: usize,
}

/// The outer cross-validation and the models fitted in each fold.
#[derive(Clone, Debug)]
pub struct ClassificationCv {
    pub report: ClassificationReport,
    pub fold_models: Vec<ClassificationModel>,
}

/// `probe_binary_classification` on given features.
pub fn probe_binary_classification(
    features: &Matrix<f32>,
    labels: &[bool],
    folds: &[Fold],
    options: &LogisticCvOptions,
) -> Result<ClassificationCv, FoldError> {
    let n = labels.len();
    assert_eq!(features.rows(), n, "one feature row per label");
    let mut probabilities = vec![0.0; n];
    let mut fold_models = Vec::with_capacity(folds.len());
    for fold in folds {
        let model = ClassificationModel::fit(
            &features.select_rows(&fold.train),
            &take(labels, &fold.train),
            options,
        )?;
        for (&i, p) in fold.test.iter().zip(model.predict_proba(&features.select_rows(&fold.test))) {
            probabilities[i] = p;
        }
        fold_models.push(model);
    }
    let predictions: Vec<bool> = probabilities.iter().map(|&p| p > 0.5).collect();

    let (mut auroc, mut balanced) = (Vec::new(), Vec::new());
    for ids in bootstrap::resamples(n) {
        let resampled = take(labels, &ids);
        if resampled.iter().all(|&l| l) || resampled.iter().all(|&l| !l) {
            continue;
        }
        auroc.push(roc_auc_score(&resampled, &take(&probabilities, &ids)));
        balanced.push(balanced_accuracy_score(&resampled, &take(&predictions, &ids)));
    }
    let report = ClassificationReport {
        auroc: roc_auc_score(labels, &probabilities),
        auroc_ci: bootstrap::ci(&auroc),
        balanced_accuracy: balanced_accuracy_score(labels, &predictions),
        balanced_accuracy_ci: bootstrap::ci(&balanced),
        fold_c: fold_models.iter().map(|m| m.logistic.c).collect(),
        probabilities,
        n_samples: n,
        embed_dim: features.cols(),
        n_bootstrap: auroc.len(),
    };
    Ok(ClassificationCv { report, fold_models })
}
