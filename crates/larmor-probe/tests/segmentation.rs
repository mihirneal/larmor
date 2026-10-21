//! The segmentation probe against nanobrain's probe_binary_segmentation on the same patch features
//! (`tools/reference/probe_reference.py fit|segmentation`).
//!
//! The reference runs the probe twice: as is (torch's L-BFGS, which stops ~1e-3 short of the optimum in
//! probability) and with fit_logistic solved to the optimum. Ours is held tightly to the second and reported
//! against the first.
//!
//! The fixture tests are `#[ignore]`d because their data lives in `fixtures/` (about 200 MB):
//! `cargo test -p larmor-probe --release --test segmentation -- --ignored --nocapture`.
//! Set `LARMOR_TEST_DEVICE=gpu` to run them on the default GPU device instead of ndarray.

mod common;

use burn::tensor::Device;
use common::{Reference, check, fixture, json_f64s, max_abs_diff, report, test_data};
use larmor_probe::Matrix;
use larmor_probe::folds::fold_assignment;
use larmor_probe::segmentation::{
    FitStage, PATCH_VOXELS, SegmentationOptions, SubjectPatches, TrainingSet, dice_by_threshold,
    feature_moments, fit_segmentation, fit_segmentation_probe, probe_binary_segmentation, segmentation_folds,
};
use serde_json::Value;
use std::time::Instant;

/// ndarray by default (burn 0.22 deprecates it in favour of flex, but it is the reference CPU backend here).
#[allow(deprecated)]
fn device() -> Device {
    match std::env::var("LARMOR_TEST_DEVICE").as_deref() {
        Ok("gpu") => Device::default(),
        _ => Device::ndarray(),
    }
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn sigmoid_rows(features: &Matrix<f32>, coef: &[f32], intercept: &[f32]) -> Vec<f64> {
    let k = intercept.len();
    let mut out = Vec::with_capacity(features.rows() * k);
    for row in features.iter_rows() {
        for j in 0..k {
            let z: f64 =
                row.iter().enumerate().map(|(i, &x)| f64::from(x) * f64::from(coef[i * k + j])).sum::<f64>()
                    + f64::from(intercept[j]);
            out.push(1.0 / (1.0 + (-z).exp()));
        }
    }
    out
}

#[test]
fn single_fits_reach_the_optimum() {
    let reference = Reference::load(&test_data("segmentation_fit"));
    let d = reference.shape("features")[1];
    let features = Matrix::new(reference.shape("features")[0], d, reference.f32s("features"));
    let labels = Matrix::new(features.rows(), PATCH_VOXELS, reference.u8s("labels"));
    let test_features = Matrix::new(reference.shape("test_features")[0], d, reference.f32s("test_features"));
    let subjects = vec![SubjectPatches { features, labels }];
    let device = device();
    for fit in reference.info["fits"].as_array().unwrap() {
        let name = fit["name"].as_str().unwrap();
        let tag = name.split('.').next().unwrap();
        let ratio = fit["max_negative_ratio"].as_f64();
        let training = TrainingSet::new(&subjects, &[0], ratio);
        if let Some(keep) = &training.keep_ids {
            let expected: Vec<usize> =
                reference.i64s(&format!("{tag}.keep_ids")).iter().map(|&i| i as usize).collect();
            assert_eq!(keep, &expected, "{name}: subsample_negatives rows");
        }
        let (mean, std) = feature_moments(&training.features);
        check(
            &format!("fit {name} standardization mean"),
            max_abs_diff(&widen(&mean), &reference.f64s(&format!("{tag}.mean"))),
            1e-6,
        );
        check(
            &format!("fit {name} standardization std"),
            max_abs_diff(&widen(&std), &reference.f64s(&format!("{tag}.std"))),
            1e-6,
        );

        let alpha = fit["alpha"].as_f64().unwrap();
        let start = Instant::now();
        let (model, stats) = fit_segmentation(&training, alpha, &Default::default(), None, &device);
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "fit {name}: {stats:?} ({seconds:.2}s here, torch {:.2}s; timings preliminary, shared machine)",
            fit["torch_seconds"].as_f64().unwrap()
        );

        let test = model.standardize(&test_features);
        let ours = sigmoid_rows(&test, model.head.coef.data(), &model.head.intercept);
        let exact = sigmoid_rows(
            &test,
            &reference.f32s(&format!("{name}.exact_coef")),
            &reference.f32s(&format!("{name}.exact_intercept")),
        );
        let torch = sigmoid_rows(
            &test,
            &reference.f32s(&format!("{name}.torch_coef")),
            &reference.f32s(&format!("{name}.torch_intercept")),
        );
        check(&format!("fit {name} test probabilities vs optimum"), max_abs_diff(&ours, &exact), 2e-5);
        report(
            &format!("fit {name} test probabilities vs torch fit_logistic"),
            max_abs_diff(&ours, &torch),
            1e-4,
        );
        let expected_dice = json_f64s(&fit["exact_test_dice"]);
        let probabilities: Vec<f32> = ours.iter().map(|&p| p as f32).collect();
        let labels = reference.u8s("test_labels");
        let dice = widen(&dice_by_threshold(
            &probabilities,
            &labels,
            &larmor_probe::segmentation::SEGMENTATION_THRESHOLDS,
        ));
        check(
            &format!("fit {name} test Dice by threshold vs optimum"),
            max_abs_diff(&dice, &expected_dice),
            1e-3,
        );
    }
}

fn load_subjects(reference: &Reference) -> Vec<SubjectPatches> {
    let counts: Vec<usize> = reference.i64s("patch_counts").iter().map(|&c| c as usize).collect();
    let d = reference.shape("features")[1];
    let features = reference.f32s("features");
    let labels = reference.u8s("labels");
    let mut start = 0;
    counts
        .iter()
        .map(|&n| {
            let subject = SubjectPatches {
                features: Matrix::new(n, d, features[start * d..(start + n) * d].to_vec()),
                labels: Matrix::new(
                    n,
                    PATCH_VOXELS,
                    labels[start * PATCH_VOXELS..(start + n) * PATCH_VOXELS].to_vec(),
                ),
            };
            start += n;
            subject
        })
        .collect()
}

/// The reference's index of a fit: per outer fold, (inner split x alpha) then the outer fit; then the
/// deployable probe's (inner split x alpha) and its final fit.
fn reference_index(stage: FitStage, alpha_id: usize) -> usize {
    match stage {
        FitStage::Inner { fold: Some(fold), inner } => fold * 26 + inner * 5 + alpha_id,
        FitStage::Outer { fold } => fold * 26 + 25,
        FitStage::Inner { fold: None, inner } => 130 + inner * 5 + alpha_id,
        FitStage::Final => 155,
    }
}

fn result_f64s(result: &Value, key: &str) -> Vec<f64> {
    json_f64s(&result[key])
}

fn check_segmentation_fixture(stem: &str) {
    let reference = Reference::load(&fixture(stem));
    let subjects = load_subjects(&reference);
    let ratio = reference.info["max_negative_ratio"].as_f64();
    let device = device();
    let options = SegmentationOptions { max_negative_ratio: ratio, record_fits: true, ..Default::default() };

    let folds = segmentation_folds(subjects.len()).unwrap();
    let fold_of: Vec<i64> = fold_assignment(&folds, subjects.len()).iter().map(|&f| f as i64).collect();
    assert_eq!(fold_of, reference.i64s("exact.subject_fold"), "{stem}: outer KFold over subjects");

    let start = Instant::now();
    let cv = probe_binary_segmentation(&subjects, &folds, &options, &device).unwrap();
    let cv_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (final_model, final_selection, final_fits) =
        fit_segmentation_probe(&subjects, &options, &device).unwrap();
    let final_seconds = start.elapsed().as_secs_f64();
    println!(
        "{stem}: outer CV {cv_seconds:.1}s, deployable probe {final_seconds:.1}s on {device:?} (preliminary, shared machine)"
    );

    // every fit: the same training rows, and coefficients at the optimum
    let fits_info = &reference.info["exact"]["fits"];
    let (mut coef_exact, mut coef_torch, mut intercept_exact, mut iterations) =
        (0.0f64, 0.0f64, 0.0f64, Vec::new());
    for record in cv.fits.iter().chain(&final_fits) {
        let index = reference_index(record.stage, record.alpha_id);
        let info = &fits_info[index];
        let expected_train: Vec<usize> =
            info["train_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        assert_eq!(record.train_ids, expected_train, "{stem}: fit {index} training subjects");
        assert_eq!(
            record.alpha_id,
            info["alpha_id"].as_u64().map_or(record.alpha_id, |a| a as usize),
            "{stem}: fit {index} alpha"
        );
        if let Some(keep) = &record.keep_ids {
            let expected: Vec<usize> =
                reference.i64s(&format!("fit{index:03}.keep_ids")).iter().map(|&i| i as usize).collect();
            assert_eq!(keep, &expected, "{stem}: fit {index} subsample_negatives rows");
        }
        let coef = widen(record.model.head.coef.data());
        coef_exact =
            coef_exact.max(max_abs_diff(&coef, &reference.f64s(&format!("exact.fit{index:03}.coef"))));
        coef_torch =
            coef_torch.max(max_abs_diff(&coef, &reference.f64s(&format!("torch.fit{index:03}.coef"))));
        intercept_exact = intercept_exact.max(max_abs_diff(
            &widen(&record.model.head.intercept),
            &reference.f64s(&format!("exact.fit{index:03}.intercept")),
        ));
        iterations.push(record.stats.iterations);
    }
    assert_eq!(cv.fits.len() + final_fits.len(), 156, "{stem}: number of fits");
    check(&format!("{stem} coef, all 156 fits, vs optimum"), coef_exact, 2e-3);
    check(&format!("{stem} intercept, all 156 fits, vs optimum"), intercept_exact, 2e-2);
    report(&format!("{stem} coef, all 156 fits, vs torch"), coef_torch, 1e-4);
    iterations.sort_unstable();
    println!(
        "{stem}: solver iterations per fit: median {} max {}",
        iterations[iterations.len() / 2],
        iterations[iterations.len() - 1]
    );

    // the inner Dice grids and what they chose
    for (tag, tolerance) in [("exact", 1e-4), ("torch", 1e-2)] {
        let grids = reference.f64s(&format!("{tag}.mean_inner_dice"));
        let ours: Vec<f64> = cv
            .selections
            .iter()
            .flat_map(|s| s.mean_inner_dice.iter().flatten().map(|&v| f64::from(v)))
            .collect();
        let diff = max_abs_diff(&ours, &grids);
        if tag == "exact" {
            check(
                &format!("{stem} mean inner Dice grids (5 folds x 5 alphas x 30 thresholds) vs {tag}"),
                diff,
                tolerance,
            )
        } else {
            report(&format!("{stem} mean inner Dice grids vs {tag}"), diff, tolerance)
        }
        let final_grid: Vec<f64> =
            final_selection.mean_inner_dice.iter().flatten().map(|&v| f64::from(v)).collect();
        let diff = max_abs_diff(&final_grid, &reference.f64s(&format!("{tag}.final.mean_inner_dice")));
        if tag == "exact" {
            check(&format!("{stem} deployable mean inner Dice grid vs {tag}"), diff, tolerance)
        } else {
            report(&format!("{stem} deployable mean inner Dice grid vs {tag}"), diff, tolerance)
        }
    }

    for tag in ["exact", "torch"] {
        let result = &reference.info[tag]["result"];
        let report_ = &cv.report;
        let same_alphas = report_.fold_alpha == result_f64s(result, "fold_alpha");
        let same_thresholds = widen(&report_.fold_threshold) == result_f64s(result, "fold_threshold");
        println!(
            "PARITY | {stem} chosen alphas / thresholds vs {tag}: identical {same_alphas} / {same_thresholds}"
        );
        let final_info = &reference.info[tag]["final"];
        let same_final = final_model.alpha == final_info["alpha"].as_f64().unwrap()
            && f64::from(final_model.threshold.unwrap()) == final_info["threshold"].as_f64().unwrap();
        println!("PARITY | {stem} deployable alpha and threshold vs {tag}: identical {same_final}");
        if tag == "exact" {
            assert!(
                same_alphas && same_thresholds && same_final,
                "{stem}: selections differ from the optimum run"
            );
        }
        let tolerance = if tag == "exact" { 1e-4 } else { 1e-3 };
        let compare = |name: &str, ours: f64, key: &str| {
            let diff = (ours - result[key].as_f64().unwrap_or(f64::NAN)).abs();
            if tag == "exact" {
                check(&format!("{stem} {name} vs {tag}"), diff, tolerance)
            } else {
                report(&format!("{stem} {name} vs {tag}"), diff, tolerance)
            }
        };
        compare("dice", report_.dice, "dice");
        compare("average precision", report_.average_precision, "average_precision");
        compare("voxel AUROC", report_.voxel_auroc, "voxel_auroc");
        let compare_many = |name: &str, ours: &[f64], key: &str| {
            let diff = max_abs_diff(ours, &result_f64s(result, key));
            if tag == "exact" {
                check(&format!("{stem} {name} vs {tag}"), diff, tolerance)
            } else {
                report(&format!("{stem} {name} vs {tag}"), diff, tolerance)
            }
        };
        compare_many("dice CI", &report_.dice_ci, "dice_ci");
        compare_many("average precision CI", &report_.average_precision_ci, "average_precision_ci");
        compare_many("voxel AUROC CI", &report_.voxel_auroc_ci, "voxel_auroc_ci");
        compare_many("subject Dice", &report_.subject_dice, "subject_dice");
        compare_many(
            "subject average precision",
            &report_.subject_average_precision,
            "subject_average_precision",
        );
        compare_many("subject voxel AUROC", &report_.subject_voxel_auroc, "subject_voxel_auroc");
        let oof = reference.f64s(&format!("{tag}.oof_probabilities"));
        let ours: Vec<f64> = cv.probabilities.iter().flat_map(|p| p.iter().map(|&v| f64::from(v))).collect();
        let diff = max_abs_diff(&ours, &oof);
        if tag == "exact" {
            check(&format!("{stem} out-of-fold probabilities vs {tag}"), diff, 2e-4)
        } else {
            report(&format!("{stem} out-of-fold probabilities vs {tag}"), diff, 1e-4)
        }
    }
}

#[test]
#[ignore = "needs fixtures/reference/probes (tools/reference/probe_reference.py segmentation --ratio 10)"]
fn segmentation_with_negative_subsampling_matches_reference() {
    check_segmentation_fixture("segmentation_ratio10");
}

#[test]
#[ignore = "needs fixtures/reference/probes (tools/reference/probe_reference.py segmentation --ratio none)"]
fn segmentation_without_subsampling_matches_reference() {
    check_segmentation_fixture("segmentation_rationone");
}
