//! Bootstrap confidence intervals exactly as the eval probes draw them.

use crate::numpy;
use crate::rng::Generator;

/// Resamples per probe (`probe.py::N_BOOTSTRAP`).
pub const N_BOOTSTRAP: usize = 1000;

/// The bootstrap generator's seed: every probe calls `np.random.default_rng(0)`.
pub const BOOTSTRAP_SEED: u128 = 0;

/// The probes' resamples: `rng.integers(0, n, n)` from `default_rng(0)`, `N_BOOTSTRAP` times.
pub fn resamples(n_samples: usize) -> Vec<Vec<usize>> {
    let mut rng = Generator::default_rng(BOOTSTRAP_SEED);
    (0..N_BOOTSTRAP)
        .map(|_| rng.integers(0, n_samples as i64, n_samples).into_iter().map(|i| i as usize).collect())
        .collect()
}

/// `np.percentile(values, [2.5, 97.5])`.
pub fn ci(values: &[f64]) -> [f64; 2] {
    [numpy::percentile(values, 2.5), numpy::percentile(values, 97.5)]
}

/// `np.nanpercentile(values, [2.5, 97.5])`.
pub fn nan_ci(values: &[f64]) -> [f64; 2] {
    [numpy::nanpercentile(values, 2.5), numpy::nanpercentile(values, 97.5)]
}

/// `values[ids]`.
pub fn take<T: Copy>(values: &[T], ids: &[usize]) -> Vec<T> {
    ids.iter().map(|&i| values[i]).collect()
}
