//! `larmor probe fit`: the eval-identical outer cross-validation plus a deployable probe refit on everything.
//!
//! The outer CV is exactly the eval framework's (its numbers go into `probe.toml`). The deployable probe
//! chooses its hyperparameters the same way the eval chooses them inside each outer fold, but on all samples:
//! ridge by leave-one-out on all, logistic by inner CV on all, segmentation by inner CV over all subjects.

use crate::classification::{
    ClassificationCv, ClassificationModel, classification_folds, probe_binary_classification,
};
use crate::device::Device;
use crate::folds::{Fold, folds_from_column};
use crate::logistic::{LogisticCvOptions, LogisticSolver};
use crate::matrix::Matrix;
use crate::probe_file::{Hyperparameters, PROBE_FORMAT, Probe, ProbeMetadata, ProbeWeights};
use crate::regression::{RegressionCv, RegressionModel, probe_regression, regression_folds};
use crate::segmentation::{
    SegmentationCv, SegmentationOptions, SubjectPatches, fit_segmentation_probe, probe_binary_segmentation,
    segmentation_folds,
};
use anyhow::{Result, ensure};
use toml::{Table, Value};

/// Where the training embeddings and labels came from; copied into `probe.toml`.
#[derive(Clone, Debug)]
pub struct Provenance {
    pub model_id: String,
    pub model_sha256: String,
    pub preprocessing_version: String,
    /// "last" for the final block after the final norm.
    pub layer: String,
    /// The label column (and for classification the negative and positive class names).
    pub label_names: Vec<String>,
    /// [`crate::probe_file::sha256_hex`] of the label table's bytes.
    pub label_table_sha256: String,
}

/// The outer folds: the eval framework's, or the label table's `fold` column when it has one.
fn outer_folds(
    fold_column: Option<&[usize]>,
    default: impl FnOnce() -> Result<Vec<Fold>>,
) -> Result<Vec<Fold>> {
    match fold_column {
        Some(column) => Ok(folds_from_column(column)?),
        None => default(),
    }
}

/// Regression probe: `probe_regression`'s CV, then `StandardScaler + RidgeCV` on all samples.
pub fn fit_regression_probe(
    features: &Matrix<f32>,
    targets: &[f64],
    fold_column: Option<&[usize]>,
    provenance: &Provenance,
) -> Result<(Probe, RegressionCv)> {
    ensure!(
        features.rows() == targets.len(),
        "{} feature rows for {} targets",
        features.rows(),
        targets.len()
    );
    ensure!(targets.iter().all(|t| t.is_finite()), "regression targets must be finite");
    let folds = outer_folds(fold_column, || Ok(regression_folds(targets.len())?))?;
    let cv = probe_regression(features, targets, &folds);
    let model = RegressionModel::fit(features, targets);
    let report = &cv.report;
    let mut table = Table::new();
    table.insert("n_folds".into(), Value::Integer(folds.len() as i64));
    insert(&mut table, "mae", report.mae, Some(report.mae_ci));
    insert(&mut table, "r", report.r, Some(report.r_ci));
    insert(&mut table, "r2", report.r2, Some(report.r2_ci));
    table.insert("fold_alpha".into(), floats(&report.fold_alpha));
    let hyperparameters = Hyperparameters {
        alpha: Some(model.ridge.alpha),
        selection: "RidgeCV leave-one-out on all samples".into(),
        ..Default::default()
    };
    let probe =
        new_probe(ProbeWeights::Regression(model), provenance, "mean", targets.len(), hyperparameters, table);
    Ok((probe, cv))
}

/// Classification probe: `probe_binary_classification`'s CV, then `StandardScaler + LogisticRegressionCV` on
/// all.
pub fn fit_classification_probe(
    features: &Matrix<f32>,
    labels: &[bool],
    fold_column: Option<&[usize]>,
    options: &LogisticCvOptions,
    provenance: &Provenance,
) -> Result<(Probe, ClassificationCv)> {
    ensure!(features.rows() == labels.len(), "{} feature rows for {} labels", features.rows(), labels.len());
    ensure!(labels.iter().any(|&l| l) && labels.iter().any(|&l| !l), "classification needs both classes");
    let folds = outer_folds(fold_column, || Ok(classification_folds(labels)?))?;
    let cv = probe_binary_classification(features, labels, &folds, options)?;
    let model = ClassificationModel::fit(features, labels, options)?;
    let report = &cv.report;
    let mut table = Table::new();
    table.insert("n_folds".into(), Value::Integer(folds.len() as i64));
    insert(&mut table, "auroc", report.auroc, Some(report.auroc_ci));
    insert(&mut table, "balanced_accuracy", report.balanced_accuracy, Some(report.balanced_accuracy_ci));
    table.insert("fold_C".into(), floats(&report.fold_c));
    table.insert("n_bootstrap".into(), Value::Integer(report.n_bootstrap as i64));
    let solver = match options.solver {
        LogisticSolver::SklearnLbfgs => "sklearn lbfgs path (scipy L-BFGS-B, tol 1e-4)",
        LogisticSolver::Newton => "newton to the optimum",
    };
    let hyperparameters = Hyperparameters {
        c: Some(model.logistic.c),
        selection: "LogisticRegressionCV inner StratifiedKFold(5) on all samples, neg_log_loss".into(),
        solver: Some(solver.into()),
        ..Default::default()
    };
    let probe = new_probe(
        ProbeWeights::Classification(model),
        provenance,
        "mean",
        labels.len(),
        hyperparameters,
        table,
    );
    Ok((probe, cv))
}

/// Segmentation probe: `probe_binary_segmentation`'s CV, then penalty and threshold by inner CV over all
/// subjects and a fit on all of them.
pub fn fit_segmentation_probe_file(
    subjects: &[SubjectPatches],
    fold_column: Option<&[usize]>,
    options: &SegmentationOptions,
    provenance: &Provenance,
    device: &Device,
) -> Result<(Probe, SegmentationCv)> {
    ensure!(!subjects.is_empty(), "no subjects");
    let folds = outer_folds(fold_column, || Ok(segmentation_folds(subjects.len())?))?;
    let cv = probe_binary_segmentation(subjects, &folds, options, device)?;
    let (model, _, _) = fit_segmentation_probe(subjects, options, device)?;
    let report = &cv.report;
    let mut table = Table::new();
    table.insert("n_folds".into(), Value::Integer(folds.len() as i64));
    insert(&mut table, "dice", report.dice, Some(report.dice_ci));
    insert(&mut table, "average_precision", report.average_precision, Some(report.average_precision_ci));
    insert(&mut table, "voxel_auroc", report.voxel_auroc, Some(report.voxel_auroc_ci));
    table.insert("fold_alpha".into(), floats(&report.fold_alpha));
    let thresholds: Vec<f64> = report.fold_threshold.iter().map(|&t| f64::from(t)).collect();
    table.insert("fold_threshold".into(), floats(&thresholds));
    let hyperparameters = Hyperparameters {
        alpha: Some(model.alpha),
        threshold: model.threshold,
        max_negative_ratio: options.max_negative_ratio,
        selection: "inner KFold(5, shuffle, seed 0) over all subjects, mean Dice".into(),
        solver: Some("preconditioned L-BFGS to the optimum".into()),
        ..Default::default()
    };
    let probe = new_probe(
        ProbeWeights::Segmentation(model),
        provenance,
        "none",
        subjects.len(),
        hyperparameters,
        table,
    );
    Ok((probe, cv))
}

fn new_probe(
    weights: ProbeWeights,
    provenance: &Provenance,
    pooling: &str,
    n_train: usize,
    hyperparameters: Hyperparameters,
    cv: Table,
) -> Probe {
    let metadata = ProbeMetadata {
        format: PROBE_FORMAT,
        kind: weights.kind(),
        model_id: provenance.model_id.clone(),
        model_sha256: provenance.model_sha256.clone(),
        preprocessing_version: provenance.preprocessing_version.clone(),
        layer: provenance.layer.clone(),
        pooling: pooling.into(),
        embed_dim: weights.embed_dim(),
        label_names: provenance.label_names.clone(),
        label_table_sha256: provenance.label_table_sha256.clone(),
        n_train,
        larmor_version: env!("CARGO_PKG_VERSION").into(),
        weights: String::new(),
        weights_sha256: String::new(),
        hyperparameters,
        cv,
    };
    Probe { metadata, weights }
}

fn insert(table: &mut Table, name: &str, value: f64, ci: Option<[f64; 2]>) {
    table.insert(name.into(), Value::Float(value));
    if let Some(ci) = ci {
        table.insert(format!("{name}_ci"), floats(&ci));
    }
}

fn floats(values: &[f64]) -> Value {
    Value::Array(values.iter().map(|&v| Value::Float(v)).collect())
}
