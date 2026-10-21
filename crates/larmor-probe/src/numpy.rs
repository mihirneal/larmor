//! numpy reductions with numpy's own summation order, so bootstrap statistics round the same way.

/// `np.sum` of a contiguous float64 array: zero plus numpy's pairwise sum (`pairwise_sum_DOUBLE`).
pub fn sum(values: &[f64]) -> f64 {
    0.0 + pairwise_sum(values)
}

fn pairwise_sum(values: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = values.len();
    if n < 8 {
        values.iter().fold(0.0, |acc, &v| acc + v)
    } else if n <= BLOCK {
        // eight interleaved accumulators, combined as a tree, then the tail in order
        let mut partial = [0.0f64; 8];
        partial.copy_from_slice(&values[..8]);
        let unrolled_end = n - n % 8;
        for chunk in values[8..unrolled_end].as_chunks::<8>().0 {
            for (acc, &v) in partial.iter_mut().zip(chunk) {
                *acc += v;
            }
        }
        let mut total = ((partial[0] + partial[1]) + (partial[2] + partial[3]))
            + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
        for &v in &values[unrolled_end..] {
            total += v;
        }
        total
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&values[..half]) + pairwise_sum(&values[half..])
    }
}

/// `np.mean`.
pub fn mean(values: &[f64]) -> f64 {
    sum(values) / values.len() as f64
}

/// `np.nanmean`: NaNs are replaced by zero for the sum and left out of the count.
pub fn nanmean(values: &[f64]) -> f64 {
    let replaced: Vec<f64> = values.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
    let count = values.iter().filter(|v| !v.is_nan()).count();
    sum(&replaced) / count as f64
}

/// `np.percentile(values, q)` with the default "linear" method; NaN if any value is NaN.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    linear_quantile(&sorted, q / 100.0)
}

/// `np.nanpercentile(values, q)`: the linear percentile of the non-NaN values.
pub fn nanpercentile(values: &[f64], q: f64) -> f64 {
    let mut kept: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    if kept.is_empty() {
        return f64::NAN;
    }
    kept.sort_by(f64::total_cmp);
    linear_quantile(&kept, q / 100.0)
}

/// numpy `_quantile` for method "linear": virtual index `(n - 1) * q`, then `_lerp`.
fn linear_quantile(sorted: &[f64], quantile: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = (n as f64 - 1.0) * quantile;
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let below = sorted[(previous as usize).min(n - 1)];
    let above = sorted[(previous as usize + 1).min(n - 1)];
    lerp(below, above, gamma)
}

/// numpy `_lerp`: interpolate from the nearer end so `t = 1` returns `b` exactly.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff = b - a;
    if t >= 0.5 { b - diff * (1.0 - t) } else { a + diff * t }
}

/// `np.corrcoef(x, y)[0, 1]`.
///
/// Mirrors `np.cov` then the normalisation in `np.corrcoef`: centre each row by its mean, take the dot
/// products divided by `n - 1`, divide the off-diagonal by each standard deviation in turn, clip to [-1, 1].
pub fn corrcoef(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    let (mean_x, mean_y) = (mean(x), mean(y));
    let centered_x: Vec<f64> = x.iter().map(|v| v - mean_x).collect();
    let centered_y: Vec<f64> = y.iter().map(|v| v - mean_y).collect();
    let factor = 1.0 / (n as f64 - 1.0);
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let c_xx = dot(&centered_x, &centered_x) * factor;
    let c_yy = dot(&centered_y, &centered_y) * factor;
    let c_xy = dot(&centered_x, &centered_y) * factor;
    (c_xy / c_xx.sqrt() / c_yy.sqrt()).clamp(-1.0, 1.0)
}

/// `np.logspace(start, stop, num)` for integer exponents, where every value is an exact power of ten.
pub fn logspace_int(start: i32, stop: i32) -> Vec<f64> {
    (start..=stop).map(|exponent| 10f64.powi(exponent)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_percentile_matches_numpy_examples() {
        // np.percentile([1, 2, 3, 4], [2.5, 97.5]) = [1.075, 3.925]
        let values = [4.0, 1.0, 3.0, 2.0];
        assert!((percentile(&values, 2.5) - 1.075).abs() < 1e-15);
        assert!((percentile(&values, 97.5) - 3.925).abs() < 1e-15);
        assert_eq!(percentile(&values, 100.0), 4.0);
        assert!(percentile(&[1.0, f64::NAN], 50.0).is_nan());
        assert_eq!(nanpercentile(&[1.0, f64::NAN, 3.0], 50.0), 2.0);
    }

    #[test]
    fn nanmean_skips_nans() {
        assert_eq!(nanmean(&[1.0, f64::NAN, 3.0]), 2.0);
    }

    #[test]
    fn powers_of_ten_are_exact() {
        let alphas = logspace_int(-2, 6);
        assert_eq!(alphas.len(), 9);
        assert_eq!(alphas[0], 0.01);
        assert_eq!(alphas[8], 1e6);
    }
}
