//! Probe files: write, read back, predict the same, and refuse embeddings from another model or
//! preprocessing.

mod common;

use common::{Reference, test_data};
use larmor_probe::Matrix;
use larmor_probe::fit::{
    Provenance, fit_classification_probe, fit_regression_probe, fit_segmentation_probe_file,
};
use larmor_probe::logistic::LogisticCvOptions;
use larmor_probe::probe_file::{EmbeddingSource, Probe, ProbeError, ProbeKind, sha256_hex};
use larmor_probe::segmentation::{PATCH_VOXELS, SegmentationOptions, SubjectPatches};
use std::path::PathBuf;

fn provenance(label: &str) -> Provenance {
    Provenance {
        model_id: "nanobrain-vits-p8-ref".into(),
        model_sha256: "a".repeat(64),
        preprocessing_version: "b".repeat(64),
        layer: "last".into(),
        label_names: vec![label.into()],
        label_table_sha256: sha256_hex(b"scan\tage\n"),
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("larmor-probe-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("probe.toml")
}

fn features(reference: &Reference) -> Matrix<f32> {
    let shape = reference.shape("features");
    Matrix::new(shape[0], shape[1], reference.f32s("features"))
}

#[test]
fn regression_probe_round_trips_and_refuses_other_embeddings() {
    let reference = Reference::load(&test_data("probe_regression"));
    let x = features(&reference);
    let targets = reference.f64s("targets");
    let (mut probe, cv) = fit_regression_probe(&x, &targets, None, &provenance("age")).unwrap();
    let path = scratch("regression");
    probe.write(&path).unwrap();
    let read = Probe::read(&path).unwrap();
    assert_eq!(read.metadata, probe.metadata);
    assert_eq!(read.metadata.kind, ProbeKind::Regression);
    assert_eq!(read.metadata.hyperparameters.alpha, Some(reference.info["final"]["alpha"].as_f64().unwrap()));
    assert_eq!(read.metadata.cv["mae"].as_float(), Some(cv.report.mae));
    assert_eq!(read.predict_scans(&x).unwrap(), probe.predict_scans(&x).unwrap());

    let source = EmbeddingSource {
        model_sha256: "a".repeat(64),
        preprocessing_version: "b".repeat(64),
        embed_dim: 384,
    };
    read.check_embeddings(&source).unwrap();
    let other_model = EmbeddingSource { model_sha256: "c".repeat(64), ..source.clone() };
    assert!(matches!(read.check_embeddings(&other_model), Err(ProbeError::ModelMismatch { .. })));
    let other_prep = EmbeddingSource { preprocessing_version: "d".repeat(64), ..source.clone() };
    assert!(matches!(read.check_embeddings(&other_prep), Err(ProbeError::PreprocessingMismatch { .. })));
    let other_width = EmbeddingSource { embed_dim: 768, ..source };
    assert!(matches!(read.check_embeddings(&other_width), Err(ProbeError::DimensionMismatch { .. })));

    // a weights file edited after the fact is refused
    let weights = path.with_extension("safetensors");
    let mut bytes = std::fs::read(&weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&weights, bytes).unwrap();
    assert!(matches!(Probe::read(&path), Err(ProbeError::WeightsChanged { .. })));
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
fn classification_probe_round_trips() {
    let reference = Reference::load(&test_data("probe_classification"));
    let x = features(&reference);
    let labels: Vec<bool> = reference.i64s("labels").iter().map(|&l| l == 1).collect();
    let (mut probe, _) =
        fit_classification_probe(&x, &labels, None, &LogisticCvOptions::default(), &provenance("label"))
            .unwrap();
    let path = scratch("classification");
    probe.write(&path).unwrap();
    let read = Probe::read(&path).unwrap();
    assert_eq!(read.metadata.hyperparameters.c, Some(reference.info["final"]["C"].as_f64().unwrap()));
    let probabilities = read.predict_scans(&x).unwrap();
    let expected = reference.f64s("final.probabilities");
    let diff = probabilities.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(diff < 1e-9, "deployable probabilities differ from sklearn by {diff}");
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}

#[test]
#[allow(deprecated)]
fn segmentation_probe_round_trips() {
    let reference = Reference::load(&test_data("segmentation_fit"));
    let d = reference.shape("features")[1];
    let rows = reference.shape("features")[0];
    let features = reference.f32s("features");
    let labels = reference.u8s("labels");
    // six small subjects cut from the fit fixture
    let cut: Vec<usize> = (0..=6).map(|i| i * rows / 6).collect();
    let subjects: Vec<SubjectPatches> = cut
        .windows(2)
        .map(|w| SubjectPatches {
            features: Matrix::new(w[1] - w[0], d, features[w[0] * d..w[1] * d].to_vec()),
            labels: Matrix::new(
                w[1] - w[0],
                PATCH_VOXELS,
                labels[w[0] * PATCH_VOXELS..w[1] * PATCH_VOXELS].to_vec(),
            ),
        })
        .collect();
    let device = burn::tensor::Device::ndarray();
    let options = SegmentationOptions { alphas: vec![1e3, 1e5], n_inner_folds: 3, ..Default::default() };
    let fold_column = [0, 0, 1, 1, 2, 2];
    let (mut probe, _) =
        fit_segmentation_probe_file(&subjects, Some(&fold_column), &options, &provenance("lesion"), &device)
            .unwrap();
    let path = scratch("segmentation");
    probe.write(&path).unwrap();
    let read = Probe::read(&path).unwrap();
    assert_eq!(read.metadata.kind, ProbeKind::Segmentation);
    let (ours, threshold) = read.predict_patches(&subjects[0].features, &device).unwrap();
    let (expected, expected_threshold) = probe.predict_patches(&subjects[0].features, &device).unwrap();
    assert_eq!((ours, threshold), (expected, expected_threshold));
    assert!(matches!(read.predict_scans(&subjects[0].features), Err(ProbeError::WrongKind { .. })));
    std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
}
