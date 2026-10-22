//! Separable Gaussian smoothing, as `scipy.ndimage.gaussian_filter` computes it.

use ndarray::{Array3, Axis, Zip};
use rayon::prelude::*;
use thiserror::Error;

use crate::stats::pairwise_sum_f64;

/// How a filter sees samples beyond the array edge (scipy's `mode=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtendMode {
    /// `"reflect"`: `d c b a | a b c d | d c b a`, repeating for short lines.
    Reflect,
    /// `"nearest"`: `a a a a | a b c d | d d d d`.
    Nearest,
}

impl ExtendMode {
    /// The in-range index that sample `i` of a line of length `n` reads.
    fn source_index(self, i: isize, n: usize) -> usize {
        let n = n as isize;
        match self {
            ExtendMode::Reflect => {
                let m = i.rem_euclid(2 * n);
                (if m < n { m } else { 2 * n - 1 - m }) as usize
            }
            ExtendMode::Nearest => i.clamp(0, n - 1) as usize,
        }
    }
}

/// A Gaussian width in voxels, tagged with the precision of the numpy scalar it reaches scipy as.
///
/// scipy builds the kernel as `exp(-0.5 / sigma**2 * x**2)`; with a float32 sigma (nanobrain's
/// `sigma / spacing`, where `spacing` comes from float32 header zooms) the factor `-0.5 / sigma**2`
/// is rounded to float32 first, which changes the kernel in the last bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sigma {
    /// A float32 sigma, as `np.float32`.
    F32(f32),
    /// A float64 sigma.
    F64(f64),
}

impl Sigma {
    /// The sigma as a Python float (`float(sigma)`).
    pub fn value(self) -> f64 {
        match self {
            Sigma::F32(s) => s as f64,
            Sigma::F64(s) => s,
        }
    }

    /// Whether scipy filters along this axis at all (`sigma > 1e-15`, compared in its precision).
    pub fn is_active(self) -> bool {
        match self {
            Sigma::F32(s) => s > 1e-15f32,
            Sigma::F64(s) => s > 1e-15,
        }
    }

    /// Kernel radius: `int(truncate * sigma + 0.5)`, as in `gaussian_filter1d` (saturating for
    /// absurd sigmas; [`Sigma::checked_radius`] refuses those).
    pub fn radius(self, truncate: f64) -> usize {
        (truncate * self.value() + 0.5) as usize
    }

    /// [`Sigma::radius`], or an error when the kernel would have more than
    /// [`MAX_KERNEL_RADIUS`] taps on each side (or the radius is not finite).
    ///
    /// scipy fails there too, with `ValueError: Maximum allowed size exceeded` or a
    /// `MemoryError` once the radius stops fitting in memory; a sigma that large comes from a
    /// corrupt header (voxel sizes like 1e-30 mm), never from an MRI scan.
    pub fn checked_radius(self, truncate: f64) -> Result<usize, KernelTooLarge> {
        let radius = truncate * self.value() + 0.5;
        if radius.is_finite() && radius < (MAX_KERNEL_RADIUS + 1) as f64 {
            Ok(radius as usize)
        } else {
            Err(KernelTooLarge { sigma: self.value(), radius })
        }
    }
}

/// The largest Gaussian kernel radius (taps on each side of the centre) [`gaussian_filter`] builds:
/// a sigma of 16384 voxels at scipy's `truncate=4`.
pub const MAX_KERNEL_RADIUS: usize = 1 << 16;

/// A Gaussian too wide to filter with (see [`Sigma::checked_radius`]).
#[derive(Clone, Debug, Error, PartialEq)]
#[error(
    "a Gaussian of sigma {sigma:e} voxels needs a kernel radius of {radius:e} voxels, more than the \
     {MAX_KERNEL_RADIUS} supported (are the header's voxel sizes right?)"
)]
pub struct KernelTooLarge {
    /// The sigma, in voxels.
    pub sigma: f64,
    /// The radius it would need.
    pub radius: f64,
}

/// The normalised 1-D Gaussian kernel of `2 * radius + 1` taps.
///
/// Mirrors `scipy.ndimage._filters._gaussian_kernel1d(sigma, 0, radius)`, including numpy's pairwise
/// sum for the normalisation.
pub fn gaussian_kernel1d(sigma: Sigma, radius: usize) -> Vec<f64> {
    let factor = match sigma {
        Sigma::F32(s) => (-0.5f32 / (s * s)) as f64,
        Sigma::F64(s) => -0.5 / (s * s),
    };
    let r = radius as i64;
    let phi: Vec<f64> = (-r..=r).map(|x| (factor * (x * x) as f64).exp()).collect();
    let total = pairwise_sum_f64(&phi);
    phi.into_iter().map(|p| p / total).collect()
}

/// Correlates every line along `axis` with a symmetric kernel, in place.
///
/// Mirrors `NI_Correlate1D` for symmetric weights: each sample is widened to float64, the sum
/// starts at the centre tap and adds the pairs from the outside in, each pair as one fused
/// multiply-add, and the result is cast back to float32. Samples beyond the ends come from `mode`.
pub fn correlate1d_symmetric(data: &mut Array3<f32>, axis: usize, weights: &[f64], mode: ExtendMode) {
    assert!(weights.len() % 2 == 1, "a symmetric kernel has an odd number of taps");
    if axis < 2 && data.is_standard_layout() {
        correlate_rows(data, axis, weights, mode);
    } else {
        correlate_lanes(data, axis, weights, mode);
    }
}

/// One line at a time; used for the contiguous last axis.
fn correlate_lanes(data: &mut Array3<f32>, axis: usize, weights: &[f64], mode: ExtendMode) {
    let radius = weights.len() / 2;
    Zip::from(data.lanes_mut(Axis(axis))).par_for_each(|mut lane| {
        let n = lane.len();
        let padded: Vec<f64> = (-(radius as isize)..(n + radius) as isize)
            .map(|i| lane[mode.source_index(i, n)] as f64)
            .collect();
        for (ll, out) in lane.iter_mut().enumerate() {
            let mid = ll + radius;
            let mut acc = padded[mid] * weights[radius];
            for j in (1..=radius).rev() {
                // scipy's `oline += (left + right) * w` is compiled to a fused multiply-add
                acc = (padded[mid - j] + padded[mid + j]).mul_add(weights[radius - j], acc);
            }
            *out = acc as f32;
        }
    });
}

/// Whole contiguous rows at a time, for axes 0 and 1 of a C-order array: output row `i` along the
/// axis combines input rows `i ± j`, element by element in the same order as [`correlate_lanes`],
/// so the result is identical but memory is read sequentially.
fn correlate_rows(data: &mut Array3<f32>, axis: usize, weights: &[f64], mode: ExtendMode) {
    const CHUNK: usize = 256;
    let shape = data.shape().to_vec();
    let n = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    if n == 0 || inner == 0 {
        return;
    }
    let radius = weights.len() / 2;
    let source = data.as_slice().expect("standard layout").to_vec();
    let target = data.as_slice_mut().expect("standard layout");
    target.par_chunks_mut(inner).enumerate().for_each(|(row, out)| {
        let (block, i) = (row / n, (row % n) as isize);
        let row_at = |k: isize| {
            let start = (block * n + mode.source_index(k, n)) * inner;
            &source[start..start + inner]
        };
        let centre = row_at(i);
        for start in (0..inner).step_by(CHUNK) {
            let end = (start + CHUNK).min(inner);
            let mut acc = [0.0f64; CHUNK];
            for e in start..end {
                acc[e - start] = centre[e] as f64 * weights[radius];
            }
            for j in (1..=radius).rev() {
                let (left, right) = (row_at(i - j as isize), row_at(i + j as isize));
                let w = weights[radius - j];
                for e in start..end {
                    acc[e - start] = (left[e] as f64 + right[e] as f64).mul_add(w, acc[e - start]);
                }
            }
            for e in start..end {
                out[e] = acc[e - start] as f32;
            }
        }
    });
}

/// Smooths a float32 volume with a separable Gaussian, one axis at a time.
///
/// Mirrors `scipy.ndimage.gaussian_filter(input, sigma, mode=mode, truncate=truncate)` for a float32
/// array: axes whose sigma is not above 1e-15 are skipped, and each pass is rounded to float32
/// before the next one reads it. Fails, before filtering anything, when a kernel would be larger
/// than [`MAX_KERNEL_RADIUS`].
pub fn gaussian_filter(
    input: &Array3<f32>,
    sigmas: [Sigma; 3],
    mode: ExtendMode,
    truncate: f64,
) -> Result<Array3<f32>, KernelTooLarge> {
    let mut radii = [0; 3];
    for (radius, sigma) in radii.iter_mut().zip(sigmas) {
        if sigma.is_active() {
            *radius = sigma.checked_radius(truncate)?;
        }
    }
    let mut output = input.as_standard_layout().into_owned();
    for (axis, (sigma, radius)) in sigmas.into_iter().zip(radii).enumerate() {
        if sigma.is_active() {
            let weights = gaussian_kernel1d(sigma, radius);
            correlate1d_symmetric(&mut output, axis, &weights, mode);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_and_nearest_indices() {
        let reflect: Vec<usize> = (-5..8).map(|i| ExtendMode::Reflect.source_index(i, 3)).collect();
        assert_eq!(reflect, [1, 2, 2, 1, 0, 0, 1, 2, 2, 1, 0, 0, 1]);
        let nearest: Vec<usize> = (-2..5).map(|i| ExtendMode::Nearest.source_index(i, 3)).collect();
        assert_eq!(nearest, [0, 0, 0, 1, 2, 2, 2]);
    }

    #[test]
    fn kernel_sums_to_one_and_is_symmetric() {
        let k = gaussian_kernel1d(Sigma::F32(2.0), Sigma::F32(2.0).radius(4.0));
        assert_eq!(k.len(), 17);
        assert!((k.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!(k.iter().zip(k.iter().rev()).all(|(a, b)| a == b));
    }

    #[test]
    fn absurd_sigmas_are_refused_instead_of_building_an_empty_kernel() {
        // a header voxel size of 1e-30 mm makes the mask's 4 mm sigma 4e30 voxels
        let x = Array3::<f32>::ones([3, 3, 3]);
        let tiny_voxels = [Sigma::F32(4e30), Sigma::F32(2.0), Sigma::F32(2.0)];
        let err = gaussian_filter(&x, tiny_voxels, ExtendMode::Reflect, 4.0).unwrap_err();
        assert!(err.to_string().contains("kernel radius"), "{err}");
        assert!(Sigma::F64(f64::INFINITY).checked_radius(4.0).is_err());
        assert_eq!(Sigma::F32(2.0).checked_radius(4.0), Ok(8));
        let largest = MAX_KERNEL_RADIUS as f64 / 4.0;
        assert_eq!(Sigma::F64(largest).checked_radius(4.0), Ok(MAX_KERNEL_RADIUS));
        assert!(Sigma::F64(largest + 0.25).checked_radius(4.0).is_err());
        // inactive axes are not checked, as scipy skips them
        let skipped = [Sigma::F64(0.0), Sigma::F64(f64::NAN), Sigma::F64(1.0)];
        assert!(gaussian_filter(&x, skipped, ExtendMode::Nearest, 4.0).is_ok());
    }
}
