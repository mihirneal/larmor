//! Step 6: clip to in-mask quantiles and scale to uint16.

use larmor_volume::quantiles_linear;
use ndarray::{Array3, Zip};

use crate::error::PrepError;

/// The quantized image and the intensity window it used.
#[derive(Clone, Debug)]
pub struct Quantized {
    /// `uint16` image, zero outside the mask.
    pub data: Array3<u16>,
    /// Lower clip value (the `low` quantile of in-mask voxels).
    pub vmin: f64,
    /// Upper clip value (the `high` quantile).
    pub vmax: f64,
}

/// Scales in-mask voxels to `[0, 65535]` between two quantiles; zero elsewhere.
///
/// Mirrors `eval/prepare.py::truncate_image(img, mask_img, (low, high))`: the quantiles of the
/// float32 in-mask values are float64 (`np.quantile` with a tuple), so `values - vmin` promotes
/// to float64; the clipped product with 65535 is truncated to uint16 by `astype`.
pub fn truncate_image(
    image: &Array3<f32>,
    mask: &Array3<bool>,
    low: f64,
    high: f64,
) -> Result<Quantized, PrepError> {
    let mut values: Vec<f32> =
        image.iter().zip(mask.iter()).filter(|&(_, &inside)| inside).map(|(&v, _)| v).collect();
    let window = quantiles_linear(&mut values, &[low, high]).ok_or(PrepError::EmptyMask)?;
    let (vmin, vmax) = (window[0], window[1]);
    // nanobrain's `assert vmax > vmin`, which NaN also fails
    if vmax.partial_cmp(&vmin) != Some(std::cmp::Ordering::Greater) {
        return Err(PrepError::DegenerateRange { vmin, vmax });
    }
    let range = vmax - vmin;
    let data = Zip::from(image).and(mask).par_map_collect(|&v, &inside| {
        if inside { (((v as f64 - vmin) / range).clamp(0.0, 1.0) * 65535.0) as u16 } else { 0 }
    });
    Ok(Quantized { data, vmin, vmax })
}
