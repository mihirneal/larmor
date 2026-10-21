//! The scan-level probes against nanobrain's probe_regression and probe_binary_classification run on the same
//! features (`tools/reference/probe_reference.py regression|classification`).

mod common;

use common::{Reference, check, json_f64s, max_abs_diff, test_data};
use larmor_probe::Matrix;
use larmor_probe::bootstrap::{self, take};
use larmor_probe::classification::{ClassificationModel, classification_folds, probe_binary_classification};
use larmor_probe::folds::fold_assignment;
use larmor_probe::logistic::{LogisticCvOptions, LogisticSolver};
use larmor_probe::metrics::{
    balanced_accuracy_score, mean_absolute_error, pearson_r, r2_score, roc_auc_score,
};
use larmor_probe::regression::{RegressionModel, probe_regression, regression_folds};

fn features(reference: &Reference) -> Matrix<f32> {
    let shape = reference.shape("features");
    Matrix::new(shape[0], shape[1], reference.f32s("features"))
}

fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}

fn check_regression(name: &str) {
    let reference = Reference::load(&test_data(name));
    let x = features(&reference);
    let targets = reference.f64s("targets");
    let result = &reference.info["result"];

    let folds = regression_folds(targets.len()).unwrap();
    let fold_of: Vec<i64> = fold_assignment(&folds, targets.len()).iter().map(|&f| f as i64).collect();
    assert_eq!(fold_of, reference.i64s("fold"), "{name}: outer KFold assignment");

    let cv = probe_regression(&x, &targets, &folds);
    let report = &cv.report;
    for (fold, model) in cv.fold_models.iter().enumerate() {
        let expected = &reference.info["folds"][fold];
        assert_eq!(model.ridge.alpha, expected["alpha"].as_f64().unwrap(), "{name}: fold {fold} alpha");
        check(
            &format!("{name} fold {fold} scaler mean"),
            max_abs_diff(&model.scaler.mean, &reference.f64s(&format!("fold{fold}.scaler_mean"))),
            1e-12,
        );
        check(
            &format!("{name} fold {fold} scaler scale"),
            max_abs_diff(&model.scaler.scale, &reference.f64s(&format!("fold{fold}.scaler_scale"))),
            1e-12,
        );
        // sklearn keeps cv_results_ in the input dtype (float32); best_score_ is float64
        check(
            &format!("{name} fold {fold} LOO MSE per alpha (relative, f32-stored)"),
            relative_diff(&model.ridge.loo_mse, &reference.f64s(&format!("fold{fold}.loo_mse"))),
            1e-6,
        );
        let best = model.ridge.loo_mse.iter().copied().fold(f64::INFINITY, f64::min);
        check(
            &format!("{name} fold {fold} best LOO MSE (relative)"),
            (best + expected["best_score"].as_f64().unwrap()).abs() / best,
            1e-10,
        );
        check(
            &format!("{name} fold {fold} ridge coef"),
            max_abs_diff(&widen(&model.ridge.coef), &reference.f64s(&format!("fold{fold}.coef"))),
            1e-6,
        );
        check(
            &format!("{name} fold {fold} ridge intercept"),
            (f64::from(model.ridge.intercept) - expected["intercept"].as_f64().unwrap()).abs(),
            1e-4,
        );
    }
    assert_eq!(report.fold_alpha, json_f64s(&result["fold_alpha"]), "{name}: fold alphas");
    // predictions are float32 in sklearn (float32 features, coef_ cast back to float32): 1-2 ulps apart
    check(
        &format!("{name} out-of-fold predictions"),
        max_abs_diff(&report.predictions, &json_f64s(&result["predictions"])),
        3e-5,
    );
    for (key, ours) in [("mae", report.mae), ("r", report.r), ("r2", report.r2)] {
        check(&format!("{name} {key}"), (ours - result[key].as_f64().unwrap()).abs(), 1e-4);
    }
    for (key, ours) in [("mae_ci", report.mae_ci), ("r_ci", report.r_ci), ("r2_ci", report.r2_ci)] {
        check(&format!("{name} {key}"), max_abs_diff(&ours, &json_f64s(&result[key])), 1e-4);
    }

    // the same resamples, and each resample's metrics
    let draws = bootstrap::resamples(targets.len());
    let expected_ids = reference.i64s("bootstrap_ids");
    let ours_ids: Vec<i64> = draws.iter().flatten().map(|&i| i as i64).collect();
    assert_eq!(ours_ids, expected_ids, "{name}: bootstrap resamples");
    let per_draw = |metric: fn(&[f64], &[f64]) -> f64| -> Vec<f64> {
        draws.iter().map(|ids| metric(&take(&targets, ids), &take(&report.predictions, ids))).collect()
    };
    check(
        &format!("{name} bootstrap MAE per resample"),
        max_abs_diff(&per_draw(mean_absolute_error), &reference.f64s("bootstrap_mae")),
        1e-4,
    );
    check(
        &format!("{name} bootstrap r per resample"),
        max_abs_diff(&per_draw(pearson_r), &reference.f64s("bootstrap_r")),
        1e-5,
    );
    check(
        &format!("{name} bootstrap R2 per resample"),
        max_abs_diff(&per_draw(r2_score), &reference.f64s("bootstrap_r2")),
        1e-5,
    );

    // the deployable probe: same pipeline on all samples
    let final_model = RegressionModel::fit(&x, &targets);
    assert_eq!(
        final_model.ridge.alpha,
        reference.info["final"]["alpha"].as_f64().unwrap(),
        "{name}: final alpha"
    );
    check(
        &format!("{name} final coef"),
        max_abs_diff(&widen(&final_model.ridge.coef), &reference.f64s("final.coef")),
        1e-6,
    );
    check(
        &format!("{name} final predictions (in-sample)"),
        max_abs_diff(&final_model.predict(&x), &reference.f64s("final.predictions")),
        1e-4,
    );
}

fn relative_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / y.abs().max(1e-300)).fold(0.0, f64::max)
}

#[test]
fn regression_matches_probe_regression_eigen_mode() {
    check_regression("probe_regression");
}

#[test]
fn regression_matches_probe_regression_svd_mode() {
    check_regression("probe_regression_svd");
}

fn labels(reference: &Reference) -> Vec<bool> {
    reference.i64s("labels").iter().map(|&l| l == 1).collect()
}

#[test]
fn classification_matches_probe_binary_classification() {
    let reference = Reference::load(&test_data("probe_classification"));
    let x = features(&reference);
    let labels = labels(&reference);
    let result = &reference.info["result"];

    let folds = classification_folds(&labels).unwrap();
    let fold_of: Vec<i64> = fold_assignment(&folds, labels.len()).iter().map(|&f| f as i64).collect();
    assert_eq!(fold_of, reference.i64s("fold"), "outer StratifiedKFold assignment");

    let cv = probe_binary_classification(&x, &labels, &folds, &LogisticCvOptions::default()).unwrap();
    let report = &cv.report;
    for (fold, model) in cv.fold_models.iter().enumerate() {
        let expected = &reference.info["folds"][fold];
        let fit = &model.logistic;
        assert_eq!(fit.c, expected["C"].as_f64().unwrap(), "fold {fold} C");
        check(
            &format!("classification fold {fold} scaler mean"),
            max_abs_diff(&model.scaler.mean, &reference.f64s(&format!("fold{fold}.scaler_mean"))),
            1e-12,
        );
        check(
            &format!("classification fold {fold} scaler scale"),
            max_abs_diff(&model.scaler.scale, &reference.f64s(&format!("fold{fold}.scaler_scale"))),
            1e-12,
        );
        let scores: Vec<f64> = fit.scores.iter().flatten().copied().collect();
        check(
            &format!("classification fold {fold} inner scores (5 folds x 9 Cs)"),
            max_abs_diff(&scores, &reference.f64s(&format!("fold{fold}.scores"))),
            1e-6,
        );
        let paths: Vec<f64> = fit.coefs_paths.iter().flatten().flatten().copied().collect();
        check(
            &format!("classification fold {fold} coefs_paths, C <= 1"),
            max_abs_diff_where(
                &paths,
                &reference.f64s(&format!("fold{fold}.coefs_paths")),
                x.cols() + 1,
                9,
                5,
            ),
            1e-6,
        );
        check(
            &format!("classification fold {fold} refit coef"),
            max_abs_diff(&fit.coef, &reference.f64s(&format!("fold{fold}.coef"))),
            1e-6,
        );
        check(
            &format!("classification fold {fold} refit intercept"),
            (fit.intercept - expected["intercept"].as_f64().unwrap()).abs(),
            1e-6,
        );
    }
    assert_eq!(report.fold_c, json_f64s(&result["fold_C"]), "fold Cs");
    check(
        "classification out-of-fold probabilities",
        max_abs_diff(&report.probabilities, &json_f64s(&result["probabilities"])),
        1e-6,
    );
    for (key, ours) in [("auroc", report.auroc), ("balanced_accuracy", report.balanced_accuracy)] {
        check(&format!("classification {key}"), (ours - result[key].as_f64().unwrap()).abs(), 1e-9);
    }
    for (key, ours) in [("auroc_ci", report.auroc_ci), ("balanced_accuracy_ci", report.balanced_accuracy_ci)]
    {
        check(&format!("classification {key}"), max_abs_diff(&ours, &json_f64s(&result[key])), 1e-9);
    }

    let draws = bootstrap::resamples(labels.len());
    let kept: Vec<bool> = draws
        .iter()
        .map(|ids| take(&labels, ids).iter().any(|&l| l) && take(&labels, ids).iter().any(|&l| !l))
        .collect();
    assert_eq!(
        kept,
        reference.u8s("bootstrap_kept").iter().map(|&k| k == 1).collect::<Vec<_>>(),
        "skipped resamples"
    );
    assert_eq!(report.n_bootstrap, kept.iter().filter(|&&k| k).count());
    let predictions: Vec<bool> = report.probabilities.iter().map(|&p| p > 0.5).collect();
    let kept_draws: Vec<&Vec<usize>> = draws.iter().zip(&kept).filter(|(_, k)| **k).map(|(d, _)| d).collect();
    let auroc: Vec<f64> = kept_draws
        .iter()
        .map(|ids| roc_auc_score(&take(&labels, ids), &take(&report.probabilities, ids)))
        .collect();
    let balanced: Vec<f64> = kept_draws
        .iter()
        .map(|ids| balanced_accuracy_score(&take(&labels, ids), &take(&predictions, ids)))
        .collect();
    check(
        "classification bootstrap AUROC per resample",
        max_abs_diff(&auroc, &reference.f64s("bootstrap_auroc")),
        1e-12,
    );
    check(
        "classification bootstrap balanced accuracy per resample",
        max_abs_diff(&balanced, &reference.f64s("bootstrap_balanced_accuracy")),
        1e-12,
    );

    let final_model = ClassificationModel::fit(&x, &labels, &LogisticCvOptions::default()).unwrap();
    assert_eq!(final_model.logistic.c, reference.info["final"]["C"].as_f64().unwrap(), "final C");
    check(
        "classification final coef",
        max_abs_diff(&final_model.logistic.coef, &reference.f64s("final.coef")),
        1e-6,
    );
    check(
        "classification final probabilities (in-sample)",
        max_abs_diff(&final_model.predict_proba(&x), &reference.f64s("final.probabilities")),
        1e-6,
    );
}

/// Max difference over the path entries for the Cs where sklearn's L-BFGS still moves (C <= 1); above that
/// the warm-started fits stop after zero to a few iterations and both stay at the C = 1 solution's
/// neighbourhood.
fn max_abs_diff_where(ours: &[f64], expected: &[f64], n_dof: usize, n_cs: usize, n_folds: usize) -> f64 {
    let mut worst = 0.0f64;
    for fold in 0..n_folds {
        for c in 0..n_cs {
            let start = (fold * n_cs + c) * n_dof;
            let diff = max_abs_diff(&ours[start..start + n_dof], &expected[start..start + n_dof]);
            if c <= 4 {
                worst = worst.max(diff);
            }
        }
    }
    worst
}

#[test]
fn newton_solver_matches_the_exact_optimum() {
    let reference = Reference::load(&test_data("probe_classification"));
    let x = features(&reference);
    let labels = labels(&reference);
    let folds = classification_folds(&labels).unwrap();
    let options = LogisticCvOptions { solver: LogisticSolver::Newton, ..LogisticCvOptions::default() };
    let cv = probe_binary_classification(&x, &labels, &folds, &options).unwrap();
    for (fold, model) in cv.fold_models.iter().enumerate() {
        let fit = &model.logistic;
        assert_eq!(fit.c, reference.info["folds"][fold]["exact_C"].as_f64().unwrap(), "fold {fold} exact C");
        let scores: Vec<f64> = fit.scores.iter().flatten().copied().collect();
        check(
            &format!("newton fold {fold} inner scores vs sklearn newton-cholesky"),
            max_abs_diff(&scores, &reference.f64s(&format!("fold{fold}.exact_scores"))),
            1e-8,
        );
        let paths: Vec<f64> = fit.coefs_paths.iter().flatten().flatten().copied().collect();
        check(
            &format!("newton fold {fold} coefs_paths vs sklearn newton-cholesky"),
            max_abs_diff(&paths, &reference.f64s(&format!("fold{fold}.exact_coefs_paths"))),
            1e-6,
        );
        check(
            &format!("newton fold {fold} refit coef vs sklearn newton-cholesky"),
            max_abs_diff(&fit.coef, &reference.f64s(&format!("fold{fold}.exact_coef"))),
            1e-8,
        );
    }
    check(
        "newton out-of-fold probabilities vs sklearn newton-cholesky",
        max_abs_diff(&cv.report.probabilities, &reference.f64s("exact_probabilities")),
        1e-8,
    );
    check(
        "newton out-of-fold probabilities vs the eval framework (sklearn lbfgs)",
        max_abs_diff(&cv.report.probabilities, &json_f64s(&reference.info["result"]["probabilities"])),
        1e-3,
    );
}
