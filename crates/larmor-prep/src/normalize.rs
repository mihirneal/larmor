//! Step 7: the model-input normalisation of `ViTMAE3D.dense_embed`.

use larmor_volume::pairwise_sum_f64;
use ndarray::{Array3, Zip};
use rayon::prelude::*;

/// Voxels per partial sum. Fixed chunks keep the parallel sums deterministic.
const CHUNK: usize = 1 << 16;

/// In-mask values divided by their maximum, then z-scored; zero outside the mask.
///
/// Mirrors the first lines of `model.py::ViTMAE3D.dense_embed` (PyTorch, float32):
///
/// ```python
/// values = images[ii][image_mask]
/// values = values / values.max()
/// values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
/// ```
///
/// The division and the z-score are float32, as in torch. torch's float32 mean comes from its
/// cascade sum, whose rounding depends on the CPU thread count; here the sum is taken in float64
/// and rounded once, which can put the mean one float32 ulp away from torch's. The standard
/// deviation follows torch's CPU two-pass algorithm: float64 squared deviations about the
/// float32 mean, rounded to float32 at the end.
pub fn max_zscore(quantized: &Array3<u16>, mask: &Array3<bool>, std_floor: f64) -> Array3<f32> {
    let quantized = quantized.as_standard_layout();
    let mask = mask.as_standard_layout();
    let (q, m) = (quantized.as_slice().unwrap(), mask.as_slice().unwrap());
    let in_mask = |f: &(dyn Fn(u16) -> f64 + Sync)| -> f64 {
        let partials: Vec<f64> = q
            .par_chunks(CHUNK)
            .zip(m.par_chunks(CHUNK))
            .map(|(qs, ms)| qs.iter().zip(ms).filter(|&(_, &inside)| inside).map(|(&v, _)| f(v)).sum())
            .collect();
        pairwise_sum_f64(&partials)
    };

    let count = m.par_iter().filter(|&&inside| inside).count();
    if count == 0 {
        return Array3::zeros(quantized.raw_dim());
    }
    let max = q.par_iter().zip(m).filter(|&(_, &inside)| inside).map(|(&v, _)| v).max().unwrap() as f32;
    let mean = in_mask(&|v| (v as f32 / max) as f64) as f32 / count as f32;
    let variance = in_mask(&|v| {
        let d = (v as f32 / max) as f64 - mean as f64;
        d * d
    }) / count as f64;
    let std = (variance.sqrt() as f32).max(std_floor as f32);

    Zip::from(&quantized)
        .and(&mask)
        .par_map_collect(|&v, &inside| if inside { (v as f32 / max - mean) / std } else { 0.0 })
}
