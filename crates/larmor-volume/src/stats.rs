//! Reductions and order statistics with numpy's exact rounding.
//!
//! A threshold like `data > data.mean()` flips voxels if the mean is off by one float32 ulp, so the
//! mean is summed in exactly numpy's pairwise order rather than in f64.

/// numpy's pairwise-summation block size (`PW_BLOCKSIZE`).
const BLOCK: usize = 128;

macro_rules! pairwise_sum {
    ($name:ident, $t:ty, $python:literal) => {
        #[doc = concat!("Sum of a contiguous ", $python, " array in numpy's pairwise order.")]
        ///
        /// Mirrors `np.add.reduce` on a contiguous 1-D array (`pairwise_sum` in numpy's
        /// `loops_utils.h.src`): blocks of up to 128 values use eight interleaved accumulators,
        /// longer runs split in two at a multiple of eight. The accumulator keeps the input dtype.
        pub fn $name(values: &[$t]) -> $t {
            let n = values.len();
            if n < 8 {
                values.iter().fold(0.0, |acc, &v| acc + v)
            } else if n <= BLOCK {
                let mut r: [$t; 8] = values[..8].try_into().unwrap();
                let whole = n - n % 8;
                for chunk in values[8..whole].chunks_exact(8) {
                    for (acc, &v) in r.iter_mut().zip(chunk) {
                        *acc += v;
                    }
                }
                let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                for &v in &values[whole..] {
                    res += v;
                }
                res
            } else {
                let mut half = n / 2;
                half -= half % 8;
                let (left, right) = values.split_at(half);
                // the split points are fixed, so summing the halves in parallel changes nothing
                if n > 1 << 16 {
                    let (a, b) = rayon::join(|| $name(left), || $name(right));
                    a + b
                } else {
                    $name(left) + $name(right)
                }
            }
        }
    };
}

pairwise_sum!(pairwise_sum_f32, f32, "float32");
pairwise_sum!(pairwise_sum_f64, f64, "float64");

/// Mean of float32 values as numpy computes `arr.mean()` for a C-contiguous float32 array.
///
/// Mirrors `numpy._core._methods._mean`: a float32 pairwise sum, divided by the count (an `intp`,
/// so the division happens in float64) and rounded back to float32. Returns NaN for no values.
pub fn mean_f32(values: &[f32]) -> f32 {
    (pairwise_sum_f32(values) as f64 / values.len() as f64) as f32
}

/// Python's built-in `round(x)` for a float: nearest integer, ties to even.
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Quantiles of float32 values with numpy's default `"linear"` method, returned as float64.
///
/// Mirrors `np.quantile(values, qs)` for a float32 array and a sequence of quantiles (a float64
/// array, so the result is float64): the virtual index is `(n - 1) * q`, and the interpolation is
/// numpy's `_lerp`, which works from the upper neighbour when the weight is at least 0.5. The
/// difference of the two neighbours is taken in float32, as numpy does. `values` is reordered.
/// Returns NaN for every quantile if any value is NaN, and `None` if there are no values or a
/// quantile is outside `[0, 1]`.
pub fn quantiles_linear(values: &mut [f32], qs: &[f64]) -> Option<Vec<f64>> {
    let n = values.len();
    if n == 0 || qs.iter().any(|q| !(0.0..=1.0).contains(q)) {
        return None;
    }
    if values.iter().any(|v| v.is_nan()) {
        return Some(vec![f64::NAN; qs.len()]);
    }

    let neighbours: Vec<(usize, usize, f64)> = qs
        .iter()
        .map(|&q| {
            let virtual_index = (n - 1) as f64 * q;
            let previous = virtual_index.floor();
            if virtual_index >= (n - 1) as f64 {
                (n - 1, n - 1, 0.0)
            } else {
                (previous as usize, previous as usize + 1, virtual_index - previous)
            }
        })
        .collect();

    let mut wanted: Vec<usize> = neighbours.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
    wanted.sort_unstable();
    wanted.dedup();
    let sorted = order_statistics(values, &wanted);
    let value_at = |index: usize| sorted[wanted.binary_search(&index).unwrap()];

    let result = neighbours
        .iter()
        .map(|&(lo, hi, gamma)| {
            let (a, b) = (value_at(lo), value_at(hi));
            let diff = (b - a) as f64;
            if gamma >= 0.5 { b as f64 - diff * (1.0 - gamma) } else { a as f64 + diff * gamma }
        })
        .collect();
    Some(result)
}

/// The values that would sit at each of the ascending positions `ranks` if `values` were sorted.
fn order_statistics(values: &mut [f32], ranks: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(ranks.len());
    let mut offset = 0;
    let mut rest = values;
    for &rank in ranks {
        let (_, nth, right) = rest.select_nth_unstable_by(rank - offset, |a, b| a.partial_cmp(b).unwrap());
        out.push(*nth);
        offset = rank + 1;
        rest = right;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantile_of_small_arrays() {
        let mut v = vec![3.0f32, 1.0, 2.0];
        assert_eq!(quantiles_linear(&mut v, &[0.0, 0.5, 1.0]).unwrap(), vec![1.0, 2.0, 3.0]);
        let mut one = vec![7.5f32];
        assert_eq!(quantiles_linear(&mut one, &[0.005, 0.995]).unwrap(), vec![7.5, 7.5]);
        assert!(quantiles_linear(&mut [], &[0.5]).is_none());
    }

    #[test]
    fn round_ties_to_even_like_python() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(round_half_even(-0.5), -0.0);
    }
}
