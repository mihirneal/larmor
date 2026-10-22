//! Step 1: the head mask, in the scan's native voxel space.

use larmor_volume::{
    Affine, ExtendMode, Interpolation, Nifti, Sigma, Volume, binary_fill_holes, gaussian_filter,
    largest_component, mean_f32, resample_from_to, vox2out_vox,
};

use crate::error::PrepError;
use crate::recipe::ThresholdMask;
use crate::scan::Scan;

/// Smoothing truncation, scipy's default `truncate=4.0`.
const TRUNCATE: f64 = 4.0;

/// The most voxels the mask grid may have (a 2 mm grid spanning 2 m on every side has 10^9). More
/// means the header's geometry is corrupt; nibabel would fail allocating it.
pub(crate) const MAX_GRID_VOXELS: usize = 1 << 31;

/// The head mask of a scan, in its native voxel space (1 inside the head).
///
/// Mirrors `utils/preprocessing.py::threshold_mask`: resample to `resolution` mm with
/// `resample_to_output(order=1)` (which keeps the stored dtype, so integer scans are rounded),
/// smooth with a Gaussian of `sigma` mm (`mode="reflect"`), threshold at the float32 mean, keep
/// the largest component, fill holes, and resample back with nearest neighbour.
pub fn threshold_mask(nifti: &Nifti, config: &ThresholdMask) -> Result<Volume<u8>, PrepError> {
    let scan = Scan::of(nifti)?;
    let (low, sigmas) = resample_for_mask(nifti, &scan, config)?;
    let smoothed = gaussian_filter(&low.data, sigmas, ExtendMode::Reflect, TRUNCATE)?;
    let threshold = mean_f32(smoothed.as_slice().expect("gaussian_filter returns C order"));
    let mut mask = smoothed.mapv(|v| v > threshold);
    if config.remove_islands {
        mask = largest_component(mask.view());
    }
    if config.fill_holes {
        mask = binary_fill_holes(mask.view());
    }
    if !mask.iter().any(|&v| v) {
        return Err(PrepError::EmptyMask);
    }

    let mask = mask.mapv(u8::from);
    let native = resample_from_to(mask.view(), &low.affine, scan.shape, &scan.affine, Interpolation::Nearest)
        .ok_or(PrepError::SingularAffine)?;
    Ok(Volume { data: native, affine: scan.affine, zooms: scan.zooms })
}

/// The scan resampled to `resolution` mm world-aligned voxels, as float32, and the smoothing
/// sigmas (voxels) its header zooms give.
///
/// Mirrors `resample_to_output(img, voxel_sizes=3 * (resolution,), order=1)` followed by
/// `get_fdata(dtype=np.float32)` and `sigma / np.array(img.header.get_zooms()[:3])`.
fn resample_for_mask(
    nifti: &Nifti,
    scan: &Scan,
    config: &ThresholdMask,
) -> Result<(Volume<f32>, [Sigma; 3]), PrepError> {
    let (shape, affine) = vox2out_vox(scan.shape, &scan.affine, [config.resolution; 3]);
    check_grid_size(shape, &affine)?;
    let data = nifti.resample_to_f32(shape, &affine, Interpolation::Linear)?;
    // nibabel keeps the source header's zooms when the new affine is allclose to the old one. The
    // result is a Nifti1Image (resample_to_output's out_class) even for a NIfTI-2 scan, whose
    // float64 zooms are then rounded to float32 like any NIfTI-1 header's
    let zooms = if affine.allclose(&scan.affine) { scan.zooms } else { affine.header_zooms() };
    // `sigma / spacing` divides a Python float by float32 zooms, so the sigmas are float32
    let sigmas = zooms.map(|zoom| Sigma::F32(config.sigma as f32 / zoom));
    Ok((Volume { data, affine, zooms }, sigmas))
}

/// Refuses a mask grid whose size does not fit (from a corrupt header: voxel sizes of 1e30 mm,
/// say) instead of trying to allocate it.
fn check_grid_size(shape: [usize; 3], affine: &Affine) -> Result<(), PrepError> {
    let finite = affine.0.iter().flatten().all(|v| v.is_finite());
    let voxels = shape.iter().try_fold(1usize, |acc, &n| acc.checked_mul(n));
    match voxels {
        Some(n) if finite && n <= MAX_GRID_VOXELS => Ok(()),
        _ => Err(PrepError::GridTooLarge { shape }),
    }
}
