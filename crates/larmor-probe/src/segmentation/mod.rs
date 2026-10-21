//! The dense segmentation probe: `probe.py::probe_binary_segmentation` from the patch embeddings on.
//!
//! Every in-mask patch embedding predicts the voxel labels of its patch with one L2 logistic regression per
//! voxel position (512 outputs on nanobrain's 8 mm patch grid; the code takes the width from the labels). The
//! penalty and the probability threshold are chosen by inner `KFold(5, shuffle=True, random_state=0)` over
//! the training subjects on mean Dice; subjects are scored on Dice at that threshold, voxel average precision
//! and voxel AUROC, with bootstrap CIs over subjects.

pub mod solver;

use crate::bootstrap::{self, take};
use crate::folds::{Fold, FoldError, kfold};
use crate::matrix::Matrix;
use crate::numpy;
use crate::regression::CV_SEED;
use crate::rng::torch_randperm;
use burn::tensor::Device;
use serde::{Deserialize, Serialize};
pub use solver::{FitStats, LinearHead, SolverOptions, SolverStop, fit_logistic, predict_on_device};

/// Voxels per 8 mm patch: the outputs of the probe on nanobrain's eval grid (`probe.py::patchify3d`).
pub const PATCH_VOXELS: usize = 512;

/// The penalties searched (`probe.py::SEGMENTATION_ALPHAS`).
pub const SEGMENTATION_ALPHAS: [f64; 5] = [1e1, 1e2, 1e3, 1e4, 1e5];

/// `torch.logspace(-3, -0.1, 30)` exactly as torch produces it in float32
/// (`probe.py::SEGMENTATION_THRESHOLDS`).
#[allow(clippy::excessive_precision)]
pub const SEGMENTATION_THRESHOLDS: [f32; 30] = [
    0.0010000000474974513,
    0.001258925418369472,
    0.0015848932089284062,
    0.0019952624570578337,
    0.002511886414140463,
    0.003162277862429619,
    0.00398107198998332,
    0.005011872388422489,
    0.006309574004262686,
    0.007943282835185528,
    0.010000000707805157,
    0.012589255347847939,
    0.015848932787775993,
    0.019952625036239624,
    0.025118866935372353,
    0.03162277489900589,
    0.03981071338057518,
    0.050118718296289444,
    0.06309572607278824,
    0.07943281531333923,
    0.09999999403953552,
    0.1258925348520279,
    0.15848931670188904,
    0.19952622056007385,
    0.25118863582611084,
    0.3162277638912201,
    0.3981071710586548,
    0.5011872053146362,
    0.6309573650360107,
    0.7943282127380371,
];

/// One subject's in-mask patches.
#[derive(Clone, Debug)]
pub struct SubjectPatches {
    /// `[n_patches, embed_dim]`, the cached f16 tokens as f32.
    pub features: Matrix<f32>,
    /// `[n_patches, voxels]` labels (0 or 1), voxels in `(px, py, pz)` order within each patch (512 at 8 mm).
    pub labels: Matrix<u8>,
}

/// Settings of the segmentation probe; the defaults are probe_binary_segmentation's.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentationOptions {
    pub alphas: Vec<f64>,
    pub thresholds: Vec<f32>,
    pub n_inner_folds: usize,
    /// Keep every patch with a positive voxel and at most this many negative patches per positive one.
    pub max_negative_ratio: Option<f64>,
    pub solver: SolverOptions,
    /// Start each penalty's fit from the next larger penalty's solution. The optimum is the same; it saves
    /// iterations.
    pub warm_start: bool,
    /// Keep every fit's head and statistics in [`SegmentationCv::fits`] (for checks against the reference).
    pub record_fits: bool,
}

impl Default for SegmentationOptions {
    fn default() -> Self {
        Self {
            alphas: SEGMENTATION_ALPHAS.to_vec(),
            thresholds: SEGMENTATION_THRESHOLDS.to_vec(),
            n_inner_folds: 5,
            max_negative_ratio: None,
            solver: SolverOptions::default(),
            warm_start: true,
            record_fits: false,
        }
    }
}

/// The outer folds of probe_binary_segmentation: `KFold(5, shuffle=True, random_state=0)` over subjects.
pub fn segmentation_folds(n_subjects: usize) -> Result<Vec<Fold>, FoldError> {
    kfold(n_subjects, 5, Some(CV_SEED))
}

/// A fitted segmentation probe: standardization, the logistic head, and the chosen penalty and threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentationModel {
    /// Per-feature mean and standard deviation of the training patches (float32, std floored at 1e-6).
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub head: LinearHead,
    pub alpha: f64,
    /// The Dice-optimal threshold when the model came out of a selection; `None` for a single fit.
    pub threshold: Option<f32>,
}

impl SegmentationModel {
    /// `(features - mean) / std` in float32, as fit_predict_segmentation computes it.
    pub fn standardize(&self, features: &Matrix<f32>) -> Matrix<f32> {
        standardize(features, &self.mean, &self.std)
    }

    /// Voxel probabilities `[n_patches, voxels]`, row-major: `sigmoid(standardize(x) @ coef + intercept)`.
    pub fn predict(&self, features: &Matrix<f32>, device: &Device) -> Vec<f32> {
        predict_on_device(&self.head, &self.standardize(features), device)
    }
}

/// The training patches of a set of subjects, concatenated in subject order and optionally subsampled.
#[derive(Clone, Debug)]
pub struct TrainingSet {
    pub features: Matrix<f32>,
    pub labels: Matrix<u8>,
    /// Rows kept by `subsample_negatives`, as indices into the concatenation; `None` without subsampling.
    pub keep_ids: Option<Vec<usize>>,
}

impl TrainingSet {
    /// The first lines of `fit_predict_segmentation`: concatenate the subjects' patches, then
    /// `subsample_negatives` when a ratio is given.
    pub fn new(subjects: &[SubjectPatches], ids: &[usize], max_negative_ratio: Option<f64>) -> Self {
        let features = concat_rows(ids.iter().map(|&i| &subjects[i].features));
        let labels = concat_rows(ids.iter().map(|&i| &subjects[i].labels));
        match max_negative_ratio {
            None => Self { features, labels, keep_ids: None },
            Some(ratio) => {
                let keep = subsample_negatives(&labels, ratio);
                Self {
                    features: features.select_rows(&keep),
                    labels: labels.select_rows(&keep),
                    keep_ids: Some(keep),
                }
            }
        }
    }
}

fn concat_rows<'a, T: Copy + 'a>(parts: impl Iterator<Item = &'a Matrix<T>>) -> Matrix<T> {
    let mut cols = None;
    let mut rows = 0;
    let mut data = Vec::new();
    for part in parts {
        assert_eq!(*cols.get_or_insert(part.cols()), part.cols(), "subjects differ in width");
        rows += part.rows();
        data.extend_from_slice(part.data());
    }
    Matrix::new(rows, cols.unwrap_or(0), data)
}

/// `probe.py::subsample_negatives`: every patch with a positive voxel, then at most
/// `int(ratio * n_positive)` negative patches in the order of `torch.randperm(n_negative)` seeded with 0.
pub fn subsample_negatives(labels: &Matrix<u8>, max_negative_ratio: f64) -> Vec<usize> {
    let (positive, negative): (Vec<usize>, Vec<usize>) =
        (0..labels.rows()).partition(|&i| labels.row(i).iter().any(|&v| v > 0));
    let n_negative = negative.len().min((max_negative_ratio * positive.len() as f64) as usize);
    let order = torch_randperm(negative.len(), u64::from(CV_SEED));
    positive.into_iter().chain(order[..n_negative].iter().map(|&j| negative[j as usize])).collect()
}

/// The mean and the population standard deviation (floored at 1e-6) per feature, float32 like torch.
pub fn feature_moments(features: &Matrix<f32>) -> (Vec<f32>, Vec<f32>) {
    let n = features.rows() as f64;
    let d = features.cols();
    let mut sums = vec![0.0f64; d];
    for row in features.iter_rows() {
        for (s, &v) in sums.iter_mut().zip(row) {
            *s += f64::from(v);
        }
    }
    let means: Vec<f64> = sums.iter().map(|s| s / n).collect();
    let mut squares = vec![0.0f64; d];
    for row in features.iter_rows() {
        for ((s, &v), m) in squares.iter_mut().zip(row).zip(&means) {
            *s += (f64::from(v) - m).powi(2);
        }
    }
    let std = squares.iter().map(|s| ((s / n).sqrt() as f32).max(1e-6)).collect();
    (means.iter().map(|&m| m as f32).collect(), std)
}

fn standardize(features: &Matrix<f32>, mean: &[f32], std: &[f32]) -> Matrix<f32> {
    let mut out = features.clone();
    for i in 0..out.rows() {
        for ((v, &m), &s) in out.row_mut(i).iter_mut().zip(mean).zip(std) {
            *v = (*v - m) / s;
        }
    }
    out
}

/// `fit_predict_segmentation` without the prediction: standardize the training patches and fit the head.
pub fn fit_segmentation(
    training: &TrainingSet,
    alpha: f64,
    solver: &SolverOptions,
    init: Option<&LinearHead>,
    device: &Device,
) -> (SegmentationModel, FitStats) {
    let (mean, std) = feature_moments(&training.features);
    let standardized = standardize(&training.features, &mean, &std);
    let (head, stats) = fit_logistic(&standardized, &training.labels, alpha, solver, init, device);
    (SegmentationModel { mean, std, head, alpha, threshold: None }, stats)
}

/// `probe.py::dice_by_threshold`: Dice at each threshold (`p >= t`), 1 when prediction and label are empty.
pub fn dice_by_threshold(probabilities: &[f32], labels: &[u8], thresholds: &[f32]) -> Vec<f32> {
    let n_label: usize = labels.iter().map(|&l| usize::from(l)).sum();
    thresholds
        .iter()
        .map(|&threshold| {
            let (mut predicted, mut overlap) = (0usize, 0usize);
            for (&p, &l) in probabilities.iter().zip(labels) {
                if p >= threshold {
                    predicted += 1;
                    overlap += usize::from(l > 0);
                }
            }
            let denominator = predicted + n_label;
            // torch divides the integer counts in float32
            if denominator > 0 { (2 * overlap) as f32 / denominator as f32 } else { 1.0 }
        })
        .collect()
}

/// `probe.py::average_precision`: voxel average precision, NaN without positives. Ties are not merged; they
/// keep their voxel order (torch's unstable sort may order them differently).
pub fn average_precision(probabilities: &[f32], labels: &[u8]) -> f64 {
    let n_positive: usize = labels.iter().map(|&l| usize::from(l > 0)).sum();
    if n_positive == 0 {
        return f64::NAN;
    }
    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    order.sort_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
    let mut hits = 0.0f32;
    let mut total = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        if labels[i] > 0 {
            hits += 1.0;
            // precision at this rank, a float32 division as in torch
            total += f64::from(hits / (rank + 1) as f32);
        }
    }
    total / n_positive as f64
}

/// `probe.py::voxel_auroc`: AUROC from ranks, NaN without positives (or without negatives). Ties are not
/// merged.
pub fn voxel_auroc(probabilities: &[f32], labels: &[u8]) -> f64 {
    let n_positive = labels.iter().filter(|&&l| l > 0).count();
    if n_positive == 0 {
        return f64::NAN;
    }
    let n_negative = labels.len() - n_positive;
    let mut order: Vec<usize> = (0..probabilities.len()).collect();
    order.sort_by(|&a, &b| probabilities[a].total_cmp(&probabilities[b]));
    let rank_sum: f64 =
        order.iter().enumerate().filter(|&(_, &i)| labels[i] > 0).map(|(rank, _)| (rank + 1) as f64).sum();
    let (p, q) = (n_positive as f64, n_negative as f64);
    (rank_sum - p * (p + 1.0) / 2.0) / (p * q)
}

/// The penalty and threshold chosen by inner CV, with the grid they were chosen on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub alpha_id: usize,
    pub threshold_id: usize,
    /// `[alpha][threshold]`: Dice averaged over the inner validation subjects.
    pub mean_inner_dice: Vec<Vec<f32>>,
}

/// Where in the cross-validation a fit happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitStage {
    /// Inner split `inner` of outer fold `fold` (`fold` is `None` for the deployable probe's selection).
    Inner { fold: Option<usize>, inner: usize },
    /// The refit on an outer fold's training subjects.
    Outer { fold: usize },
    /// The deployable refit on all subjects.
    Final,
}

/// One recorded fit (see [`SegmentationOptions::record_fits`]).
#[derive(Clone, Debug)]
pub struct FitRecord {
    pub stage: FitStage,
    pub alpha_id: usize,
    pub train_ids: Vec<usize>,
    pub keep_ids: Option<Vec<usize>>,
    pub model: SegmentationModel,
    pub stats: FitStats,
}

/// `fit_predict_segmentation` over the inner splits of `train_ids` for every penalty, then the
/// `np.unravel_index(argmax)` of the mean inner Dice grid (the first maximum on ties).
pub fn select_alpha_threshold(
    subjects: &[SubjectPatches],
    train_ids: &[usize],
    fold: Option<usize>,
    options: &SegmentationOptions,
    device: &Device,
    records: &mut Vec<FitRecord>,
) -> Result<Selection, FoldError> {
    let (n_alphas, n_thresholds) = (options.alphas.len(), options.thresholds.len());
    // inner_dice[alpha][subject][threshold], filled for the subjects of train_ids
    let mut inner_dice = vec![vec![vec![0.0f32; n_thresholds]; subjects.len()]; n_alphas];
    for (inner, split) in kfold(train_ids.len(), options.n_inner_folds, Some(CV_SEED))?.iter().enumerate() {
        let inner_train: Vec<usize> = split.train.iter().map(|&i| train_ids[i]).collect();
        let inner_val: Vec<usize> = split.test.iter().map(|&i| train_ids[i]).collect();
        let training = TrainingSet::new(subjects, &inner_train, options.max_negative_ratio);
        let fits = fit_all_alphas(&training, options, device);
        for (alpha_id, (model, stats)) in fits.into_iter().enumerate() {
            for &subject in &inner_val {
                let probabilities = model.predict(&subjects[subject].features, device);
                inner_dice[alpha_id][subject] =
                    dice_by_threshold(&probabilities, subjects[subject].labels.data(), &options.thresholds);
            }
            if options.record_fits {
                records.push(FitRecord {
                    stage: FitStage::Inner { fold, inner },
                    alpha_id,
                    train_ids: inner_train.clone(),
                    keep_ids: training.keep_ids.clone(),
                    model,
                    stats,
                });
            }
        }
    }
    let mean_inner_dice: Vec<Vec<f32>> = (0..n_alphas)
        .map(|a| {
            (0..n_thresholds)
                .map(|t| {
                    let column: Vec<f32> = train_ids.iter().map(|&s| inner_dice[a][s][t]).collect();
                    torch_sum_dim1(&column, t, n_thresholds) / train_ids.len() as f32
                })
                .collect()
        })
        .collect();
    let mut best = (0, 0);
    for (a, row) in mean_inner_dice.iter().enumerate() {
        for (t, &value) in row.iter().enumerate() {
            if value > mean_inner_dice[best.0][best.1] {
                best = (a, t);
            }
        }
    }
    Ok(Selection { alpha_id: best.0, threshold_id: best.1, mean_inner_dice })
}

/// One fit per penalty on the same training set, largest penalty first when warm-starting.
fn fit_all_alphas(
    training: &TrainingSet,
    options: &SegmentationOptions,
    device: &Device,
) -> Vec<(SegmentationModel, FitStats)> {
    let mut fits: Vec<Option<(SegmentationModel, FitStats)>> = vec![None; options.alphas.len()];
    let mut order: Vec<usize> = (0..options.alphas.len()).collect();
    if options.warm_start {
        order.sort_by(|&a, &b| options.alphas[b].total_cmp(&options.alphas[a]));
    }
    let mut previous: Option<LinearHead> = None;
    for alpha_id in order {
        let init = if options.warm_start { previous.as_ref() } else { None };
        let (model, stats) =
            fit_segmentation(training, options.alphas[alpha_id], &options.solver, init, device);
        previous = Some(model.head.clone());
        fits[alpha_id] = Some((model, stats));
    }
    fits.into_iter().map(|fit| fit.expect("every alpha fitted")).collect()
}

/// `tensor.sum(dim=1)` of a float32 `[n_alphas, n_subjects, n_thresholds]` tensor on an aarch64 CPU, for one
/// output column: torch sums the first `16 * (n_columns / 16)` columns with `multi_row_sum` and the rest with
/// `row_sum` (four interleaved partial sums), each in float32. The order matters only for exact ties in the
/// argmax over the grid.
fn torch_sum_dim1(values: &[f32], column: usize, n_columns: usize) -> f32 {
    // Vectorized<float> has 4 lanes on aarch64 and vectorized_outer_sum takes 4 vectors at a time
    const BLOCK_COLUMNS: usize = 16;
    if column < BLOCK_COLUMNS * (n_columns / BLOCK_COLUMNS) {
        multi_row_sum(values.iter().copied())
    } else {
        let n_ilp = values.len() / 4;
        let mut partial: Vec<f32> =
            (0..4).map(|k| multi_row_sum(values[..4 * n_ilp].iter().skip(k).step_by(4).copied())).collect();
        for &v in &values[4 * n_ilp..] {
            partial[0] += v;
        }
        partial[1..].iter().fold(partial[0], |acc, &p| acc + p)
    }
}

/// ATen `SumKernel.cpp::multi_row_sum` for one column: a four-level cascade flushed every 16 rows.
fn multi_row_sum(values: impl ExactSizeIterator<Item = f32>) -> f32 {
    const LEVELS: usize = 4;
    let size = values.len();
    let ceil_log2 = if size <= 2 { 1 } else { usize::BITS - (size - 1).leading_zeros() } as usize;
    let level_power = (ceil_log2 / LEVELS).max(4);
    let level_step = 1usize << level_power;
    let level_mask = level_step - 1;
    let mut acc = [0.0f32; LEVELS];
    let values: Vec<f32> = values.collect();
    let mut i = 0;
    while i + level_step <= size {
        for _ in 0..level_step {
            acc[0] += values[i];
            i += 1;
        }
        for j in 1..LEVELS {
            acc[j] += acc[j - 1];
            acc[j - 1] = 0.0;
            if i & (level_mask << (j * level_power)) != 0 {
                break;
            }
        }
    }
    for &v in &values[i..] {
        acc[0] += v;
    }
    acc[1..].iter().fold(acc[0], |total, &a| total + a)
}

/// The numbers probe_binary_segmentation reports.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentationReport {
    pub dice: f64,
    pub dice_ci: [f64; 2],
    pub average_precision: f64,
    pub average_precision_ci: [f64; 2],
    pub voxel_auroc: f64,
    pub voxel_auroc_ci: [f64; 2],
    pub fold_alpha: Vec<f64>,
    pub fold_threshold: Vec<f32>,
    pub subject_dice: Vec<f64>,
    pub subject_average_precision: Vec<f64>,
    pub subject_voxel_auroc: Vec<f64>,
    pub n_samples: usize,
    pub embed_dim: usize,
}

/// The outer cross-validation with what it chose and predicted.
#[derive(Clone, Debug)]
pub struct SegmentationCv {
    pub report: SegmentationReport,
    pub selections: Vec<Selection>,
    /// Out-of-fold voxel probabilities per subject, `[n_patches * voxels]` row-major.
    pub probabilities: Vec<Vec<f32>>,
    pub subject_fold: Vec<usize>,
    pub fits: Vec<FitRecord>,
}

/// `probe_binary_segmentation` on given patch embeddings.
pub fn probe_binary_segmentation(
    subjects: &[SubjectPatches],
    folds: &[Fold],
    options: &SegmentationOptions,
    device: &Device,
) -> Result<SegmentationCv, FoldError> {
    let n = subjects.len();
    let mut subject_dice = vec![0.0f64; n];
    let mut subject_ap = vec![0.0f64; n];
    let mut subject_auroc = vec![0.0f64; n];
    let mut probabilities = vec![Vec::new(); n];
    let mut subject_fold = vec![0usize; n];
    let mut selections = Vec::with_capacity(folds.len());
    let mut fits = Vec::new();
    for (fold_index, fold) in folds.iter().enumerate() {
        let selection =
            select_alpha_threshold(subjects, &fold.train, Some(fold_index), options, device, &mut fits)?;
        let alpha = options.alphas[selection.alpha_id];
        let training = TrainingSet::new(subjects, &fold.train, options.max_negative_ratio);
        let (mut model, stats) = fit_segmentation(&training, alpha, &options.solver, None, device);
        model.threshold = Some(options.thresholds[selection.threshold_id]);
        for &subject in &fold.test {
            let p = model.predict(&subjects[subject].features, device);
            let labels = subjects[subject].labels.data();
            subject_dice[subject] =
                f64::from(dice_by_threshold(&p, labels, &options.thresholds)[selection.threshold_id]);
            subject_ap[subject] = average_precision(&p, labels);
            subject_auroc[subject] = voxel_auroc(&p, labels);
            subject_fold[subject] = fold_index;
            probabilities[subject] = p;
        }
        if options.record_fits {
            fits.push(FitRecord {
                stage: FitStage::Outer { fold: fold_index },
                alpha_id: selection.alpha_id,
                train_ids: fold.train.clone(),
                keep_ids: training.keep_ids.clone(),
                model,
                stats,
            });
        }
        selections.push(selection);
    }

    let (mut dice, mut ap, mut auroc) = (Vec::new(), Vec::new(), Vec::new());
    for ids in bootstrap::resamples(n) {
        dice.push(numpy::mean(&take(&subject_dice, &ids)));
        ap.push(numpy::nanmean(&take(&subject_ap, &ids)));
        auroc.push(numpy::nanmean(&take(&subject_auroc, &ids)));
    }
    let report = SegmentationReport {
        dice: numpy::mean(&subject_dice),
        dice_ci: bootstrap::ci(&dice),
        average_precision: numpy::nanmean(&subject_ap),
        average_precision_ci: bootstrap::nan_ci(&ap),
        voxel_auroc: numpy::nanmean(&subject_auroc),
        voxel_auroc_ci: bootstrap::nan_ci(&auroc),
        fold_alpha: selections.iter().map(|s| options.alphas[s.alpha_id]).collect(),
        fold_threshold: selections.iter().map(|s| options.thresholds[s.threshold_id]).collect(),
        subject_dice,
        subject_average_precision: subject_ap,
        subject_voxel_auroc: subject_auroc,
        n_samples: n,
        embed_dim: subjects.first().map_or(0, |s| s.features.cols()),
    };
    Ok(SegmentationCv { report, selections, probabilities, subject_fold, fits })
}

/// The deployable segmentation probe: penalty and threshold by inner CV over all subjects, then a fit on all.
pub fn fit_segmentation_probe(
    subjects: &[SubjectPatches],
    options: &SegmentationOptions,
    device: &Device,
) -> Result<(SegmentationModel, Selection, Vec<FitRecord>), FoldError> {
    let all: Vec<usize> = (0..subjects.len()).collect();
    let mut records = Vec::new();
    let selection = select_alpha_threshold(subjects, &all, None, options, device, &mut records)?;
    let training = TrainingSet::new(subjects, &all, options.max_negative_ratio);
    let (mut model, stats) =
        fit_segmentation(&training, options.alphas[selection.alpha_id], &options.solver, None, device);
    model.threshold = Some(options.thresholds[selection.threshold_id]);
    if options.record_fits {
        records.push(FitRecord {
            stage: FitStage::Final,
            alpha_id: selection.alpha_id,
            train_ids: all,
            keep_ids: training.keep_ids,
            model: model.clone(),
            stats,
        });
    }
    Ok((model, selection, records))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dice_counts_empty_prediction_and_label_as_one() {
        let dice = dice_by_threshold(&[0.0005, 0.0002], &[0, 0], &[0.001]);
        assert_eq!(dice, vec![1.0]);
        let dice = dice_by_threshold(&[0.9, 0.2, 0.6], &[1, 0, 0], &[0.5]);
        assert_eq!(dice, vec![2.0 / 3.0]);
    }

    #[test]
    fn auroc_and_average_precision_of_a_perfect_ranking() {
        let p = [0.1, 0.9, 0.2, 0.8];
        let y = [0, 1, 0, 1];
        assert_eq!(voxel_auroc(&p, &y), 1.0);
        assert_eq!(average_precision(&p, &y), 1.0);
        assert!(voxel_auroc(&p, &[0, 0, 0, 0]).is_nan());
    }

    #[test]
    fn subsample_keeps_positives_first() {
        let labels = Matrix::new(5, 2, vec![0, 0, 1, 0, 0, 0, 0, 1, 0, 0]);
        let keep = subsample_negatives(&labels, 1.0);
        assert_eq!(&keep[..2], &[1, 3]);
        assert_eq!(keep.len(), 4);
    }

    #[test]
    fn cascade_sum_is_sequential_below_sixteen_rows() {
        let values: Vec<f32> = (0..15).map(|i| 0.1 * i as f32).collect();
        let sequential = values.iter().fold(0.0f32, |a, &b| a + b);
        assert_eq!(multi_row_sum(values.iter().copied()), sequential);
    }
}
