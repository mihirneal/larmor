//! The sklearn metrics the scan-level probes report.

use crate::numpy;

/// `sklearn.metrics.mean_absolute_error(y_true, y_pred)`.
pub fn mean_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let errors: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| (p - t).abs()).collect();
    numpy::mean(&errors)
}

/// `sklearn.metrics.r2_score(y_true, y_pred)` with `force_finite=True`: a constant target scores 1 when the
/// prediction is perfect and 0 otherwise.
pub fn r2_score(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = numpy::mean(y_true);
    let residual: Vec<f64> = y_true.iter().zip(y_pred).map(|(t, p)| (t - p).powi(2)).collect();
    let total: Vec<f64> = y_true.iter().map(|t| (t - mean).powi(2)).collect();
    let (numerator, denominator) = (numpy::sum(&residual), numpy::sum(&total));
    if denominator != 0.0 {
        1.0 - numerator / denominator
    } else if numerator == 0.0 {
        1.0
    } else {
        0.0
    }
}

/// Pearson correlation as probe_regression computes it: `np.corrcoef(targets, predictions)[0, 1]`.
pub fn pearson_r(y_true: &[f64], y_pred: &[f64]) -> f64 {
    numpy::corrcoef(y_true, y_pred)
}

/// `sklearn.metrics.roc_auc_score(labels, scores)` for binary labels.
///
/// Mirrors `_ranking.py::_binary_clf_curve` (scores sorted descending with a stable sort, one ROC point per
/// distinct score, so ties count half), `roc_curve(drop_intermediate=True)` and the trapezoidal `auc`.
pub fn roc_auc_score(labels: &[bool], scores: &[f64]) -> f64 {
    assert_eq!(labels.len(), scores.len());
    let n = labels.len();
    // np.argsort(kind="mergesort")[::-1]: stable ascending, then reversed
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    order.reverse();

    let mut tps = Vec::new();
    let mut fps = Vec::new();
    let mut true_positives = 0.0f64;
    for (rank, &i) in order.iter().enumerate() {
        true_positives += f64::from(u8::from(labels[i]));
        let last_of_value = rank + 1 == n || scores[order[rank + 1]] != scores[i];
        if last_of_value {
            tps.push(true_positives);
            fps.push(1.0 + rank as f64 - true_positives);
        }
    }
    let (tps, fps) = drop_collinear(&tps, &fps);
    let mut tpr = vec![0.0];
    let mut fpr = vec![0.0];
    let (tp_total, fp_total) = (*tps.last().unwrap(), *fps.last().unwrap());
    tpr.extend(tps.iter().map(|t| t / tp_total));
    fpr.extend(fps.iter().map(|f| f / fp_total));
    trapezoid(&tpr, &fpr)
}

/// `roc_curve`'s `drop_intermediate`: keep the end points and every point where the curve bends.
fn drop_collinear(tps: &[f64], fps: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = tps.len();
    if n <= 2 {
        return (tps.to_vec(), fps.to_vec());
    }
    let second_diff = |v: &[f64], i: usize| v[i + 1] - 2.0 * v[i] + v[i - 1];
    let keep: Vec<usize> = (0..n)
        .filter(|&i| i == 0 || i == n - 1 || second_diff(fps, i) != 0.0 || second_diff(tps, i) != 0.0)
        .collect();
    (keep.iter().map(|&i| tps[i]).collect(), keep.iter().map(|&i| fps[i]).collect())
}

/// `np.trapezoid(y, x)`.
fn trapezoid(y: &[f64], x: &[f64]) -> f64 {
    let areas: Vec<f64> = (1..x.len()).map(|i| (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0).collect();
    numpy::sum(&areas)
}

/// `sklearn.metrics.balanced_accuracy_score(labels, predictions)`: the mean recall over the classes present
/// in `labels`.
pub fn balanced_accuracy_score(labels: &[bool], predictions: &[bool]) -> f64 {
    let recalls: Vec<f64> = [false, true]
        .iter()
        .filter_map(|&class| {
            let members = labels.iter().filter(|&&l| l == class).count();
            let hits = labels.iter().zip(predictions).filter(|&(&l, &p)| l == class && p == class).count();
            (members > 0).then(|| hits as f64 / members as f64)
        })
        .collect();
    numpy::mean(&recalls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auroc_counts_ties_as_half() {
        let labels = [false, true, false, true];
        assert_eq!(roc_auc_score(&labels, &[0.1, 0.3, 0.35, 0.8]), 0.75);
        assert_eq!(roc_auc_score(&labels, &[0.5, 0.5, 0.5, 0.5]), 0.5);
    }

    #[test]
    fn balanced_accuracy_averages_recalls() {
        let labels = [false, false, false, true];
        let predictions = [false, true, false, true];
        assert!((balanced_accuracy_score(&labels, &predictions) - (2.0 / 3.0 + 1.0) / 2.0).abs() < 1e-15);
    }
}
