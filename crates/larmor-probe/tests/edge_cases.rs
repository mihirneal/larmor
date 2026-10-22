//! Inputs the eval framework refuses: too few samples of a class, NaN or infinite embeddings,
//! segmentation training sets with nothing to learn. Larmor fails where scikit-learn and torch fail
//! instead of writing a probe with numbers the eval framework cannot produce.

mod common;

use common::{Reference, json_f64s, test_data, test_device};
use larmor_probe::Matrix;
use larmor_probe::fit::{
    Provenance, fit_classification_probe, fit_regression_probe, fit_segmentation_probe_file,
};
use larmor_probe::logistic::LogisticCvOptions;
use larmor_probe::probe_file::{ProbeError, sha256_hex};
use larmor_probe::segmentation::{
    PATCH_VOXELS, SegmentationError, SegmentationOptions, SubjectPatches, TrainingSet, fit_segmentation,
};

fn provenance() -> Provenance {
    Provenance {
        model_id: "m".into(),
        model_sha256: "a".repeat(64),
        encoder_key: "e".repeat(64),
        precision: "f32".into(),
        preprocessing_version: "b".repeat(64),
        layer: "last".into(),
        label_names: vec!["label".into()],
        label_table_sha256: sha256_hex(b""),
    }
}

/// Deterministic features in [-0.5, 0.5).
fn noise(rows: usize, cols: usize, seed: u64) -> Matrix<f32> {
    let mut state = seed;
    let data = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    Matrix::new(rows, cols, data)
}

#[test]
fn a_small_minority_class_fails_where_sklearn_raises() {
    let reference = Reference::load(&test_data("probe_minority"));
    let options = LogisticCvOptions::default();
    for n_pos in [1, 6, 7] {
        let name = format!("pos{n_pos}");
        let shape = reference.shape(&format!("{name}.features"));
        let x = Matrix::new(shape[0], shape[1], reference.f32s(&format!("{name}.features")));
        let labels: Vec<bool> = reference.i64s(&format!("{name}.labels")).iter().map(|&l| l == 1).collect();
        let expected = &reference.info["cases"][&name];
        let result = fit_classification_probe(&x, &labels, None, &options, &provenance());
        if let Some(message) = expected["raises"].as_str() {
            let err =
                result.err().unwrap_or_else(|| panic!("{name}: sklearn raises {message:?}, larmor fitted"));
            let text = err.to_string();
            assert!(
                text.contains("outer fold") && text.contains("at least 5 of each class"),
                "{name}: {text}"
            );
            continue;
        }
        let (_, cv) = result.unwrap_or_else(|e| panic!("{name}: {e:#}"));
        assert_eq!(cv.report.fold_c, json_f64s(&expected["fold_C"]), "{name}: fold C");
        let auroc = expected["auroc"].as_f64().unwrap();
        assert!((cv.report.auroc - auroc).abs() < 1e-12, "{name}: AUROC {} vs {auroc}", cv.report.auroc);
        let balanced = expected["balanced_accuracy"].as_f64().unwrap();
        assert!((cv.report.balanced_accuracy - balanced).abs() < 1e-12, "{name}: balanced accuracy");
    }
}

#[test]
fn a_fold_column_that_starves_a_training_fold_of_a_class_names_the_fold() {
    let x = noise(24, 6, 1);
    let labels: Vec<bool> = (0..24).map(|i| i % 2 == 1).collect();
    // each class in its own fold: fold 0 trains on positives only
    let folds: Vec<usize> = (0..24).map(|i| i % 2).collect();
    let err =
        fit_classification_probe(&x, &labels, Some(&folds), &LogisticCvOptions::default(), &provenance())
            .unwrap_err()
            .to_string();
    assert!(err.contains("outer fold 0 trains on 0 negative and 12 positive"), "{err}");
}

#[test]
fn too_few_scans_say_what_the_cross_validation_needs() {
    let x = noise(4, 6, 2);
    let err = fit_regression_probe(&x, &[1.0, 2.0, 3.0, 4.0], None, &provenance()).unwrap_err().to_string();
    assert!(err.contains("needs at least 5 labelled samples; found 4"), "{err}");
}

#[test]
fn nan_and_infinite_embeddings_are_refused() {
    let targets: Vec<f64> = (0..60).map(|i| f64::from(i % 7)).collect();
    let labels: Vec<bool> = (0..60).map(|i| i % 3 == 0).collect();
    for bad in [f32::NAN, f32::INFINITY] {
        let mut data = noise(60, 8, 3).data().to_vec();
        data[7 * 8 + 5] = bad;
        let x = Matrix::new(60, 8, data);
        let err = fit_regression_probe(&x, &targets, None, &provenance()).unwrap_err().to_string();
        assert!(err.contains("row 7") && err.contains("non-finite"), "{err}");
        let options = LogisticCvOptions::default();
        let err =
            fit_classification_probe(&x, &labels, None, &options, &provenance()).unwrap_err().to_string();
        assert!(err.contains("row 7"), "{err}");

        // a fitted probe refuses them too
        let clean = noise(60, 8, 3);
        let (probe, _) = fit_regression_probe(&clean, &targets, None, &provenance()).unwrap();
        assert!(matches!(probe.predict_scans(&x), Err(ProbeError::NonFinite { row: 7, .. })));
    }
}

/// `n` subjects of 20 patches each, of which only those in `positive` have labelled voxels.
fn subjects(n: usize, positive: &[usize]) -> Vec<SubjectPatches> {
    (0..n)
        .map(|s| {
            let features = noise(20, 6, 10 + s as u64);
            let labels = (0..20 * PATCH_VOXELS)
                .map(|i| u8::from(positive.contains(&s) && i / PATCH_VOXELS < 3 && i % 3 == 0))
                .collect();
            SubjectPatches { features, labels: Matrix::new(20, PATCH_VOXELS, labels) }
        })
        .collect()
}

#[test]
fn an_all_negative_training_set_with_a_ratio_is_an_error_not_a_nan_probe() {
    let device = test_device();
    let all_negative = subjects(1, &[]);
    // nanobrain's subsample_negatives keeps int(10 * 0) = 0 rows and fit_logistic raises
    let err = TrainingSet::new(&all_negative, &[0], Some(10.0)).unwrap_err();
    assert!(matches!(err, SegmentationError::EmptyTrainingSet), "{err}");
    // without a ratio nanobrain fits it (and predicts nothing); so does larmor, with finite weights
    let training = TrainingSet::new(&all_negative, &[0], None).unwrap();
    let (model, _) = fit_segmentation(&training, 100.0, &Default::default(), None, &device).unwrap();
    assert!(model.head.intercept.iter().chain(&model.mean).all(|v| v.is_finite()));

    // non-finite features never pass for a converged fit
    let mut bad = subjects(1, &[0]);
    let mut data = bad[0].features.data().to_vec();
    data[3] = f32::NAN;
    bad[0].features = Matrix::new(20, 6, data);
    let training = TrainingSet::new(&bad, &[0], None).unwrap();
    let err = fit_segmentation(&training, 100.0, &Default::default(), None, &device).unwrap_err();
    assert!(matches!(err, SegmentationError::NonFinite), "{err}");
}

#[test]
fn segmentation_probes_need_labels_and_enough_subjects() {
    let device = test_device();
    let options = SegmentationOptions { alphas: vec![1e2], ..Default::default() };
    let err = fit_segmentation_probe_file(&subjects(8, &[]), None, &options, &provenance(), &device)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no subject has a labelled voxel"), "{err}");

    // 6 subjects: an outer fold trains on 4, too few for the inner KFold(5)
    let err = fit_segmentation_probe_file(
        &subjects(6, &[0, 1, 2, 3, 4, 5]),
        None,
        &options,
        &provenance(),
        &device,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("trains on 4 subjects") && err.contains("KFold(5)"), "{err}");

    // two positive subjects among ten with a ratio: some inner training sets have no positive patch
    let ratio = SegmentationOptions { max_negative_ratio: Some(10.0), ..options };
    let err = fit_segmentation_probe_file(&subjects(10, &[0, 1]), None, &ratio, &provenance(), &device)
        .unwrap_err()
        .to_string();
    assert!(err.contains("keeps no rows"), "{err}");
}
