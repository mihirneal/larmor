//! Steps 2–5: reorient to RAS+, anti-alias, and resample onto the fixed model grid.

use larmor_volume::{
    Affine, ExtendMode, Interpolation, Nifti, Orientation, Sigma, Volume, gaussian_filter, io_orientation,
    resample_from_to, round_half_even,
};
use ndarray::{ArrayView3, Axis};
use rayon::prelude::*;

use crate::error::PrepError;
use crate::recipe::Conform;
use crate::scan::Scan;

/// `nibabel.processing.SIGMA2FWHM`, `np.sqrt(8 * np.log(2))`, as numpy computes it.
const SIGMA2FWHM: f64 = 2.3548200450309493;

/// The conformed image and how it was placed.
#[derive(Clone, Debug)]
pub struct Conformed {
    /// Float32 image on the model grid (before quantization), with its affine.
    pub image: Volume<f32>,
    /// Voxel index in the RAS+ scan that the grid centre (`(shape - 1) // 2`) maps to.
    pub centroid: [i64; 3],
    /// Anti-aliasing FWHM (mm) per RAS+ axis; all zero when no axis was finer than the grid.
    pub fwhm: [f64; 3],
    /// Affine of the scan after reorientation to RAS+.
    pub canonical_affine: Affine,
}

/// Reorients, anti-aliases and resamples a scan onto the fixed grid, centred on its head mask.
///
/// Mirrors `eval/preprocessing.py::conform_image(img, mask_img, min_voxel_size, max_fov,
/// top_margin, order, fixed_grid=True)`: `as_closest_canonical` and a float32 cast; Gaussian
/// smoothing (`smooth_image`, `mode="nearest"`) with FWHM `sqrt(new² - old²)` on axes finer than
/// the grid; x/y centred on the mask's bounding box, z anchored `top_margin` mm above the mask
/// top; `rescale_affine`; and `resample_from_to(order)`.
pub fn conform_image(nifti: &Nifti, mask: &Volume<u8>, config: &Conform) -> Result<Conformed, PrepError> {
    let scan = Scan::of(nifti)?;
    if mask.shape() != scan.shape {
        return Err(PrepError::MaskShape { mask: mask.shape(), scan: scan.shape });
    }
    let ornt = io_orientation(&scan.affine)?;
    let canonical = canonical_f32(nifti, &scan, &ornt)?;
    let canonical_affine = canonical.affine;
    let voxel_sizes = canonical.zooms;
    let canonical_shape = canonical.shape();

    let (canonical, fwhm) = anti_alias(canonical, config.min_voxel_size)?;
    let centroid = grid_centroid(ornt.apply(mask.data.view()), voxel_sizes, canonical_shape[2], config)?;

    let new_shape = config.fov.map(|w| (w / config.min_voxel_size).ceil() as usize);
    let new_affine = rescale_affine(&canonical_affine, [config.min_voxel_size; 3], new_shape, centroid);
    let order = Interpolation::from_order(config.order).expect("order validated by the recipe");
    let data = resample_from_to(canonical.data.view(), &canonical_affine, new_shape, &new_affine, order)
        .ok_or(PrepError::SingularAffine)?;
    Ok(Conformed { image: Volume::new(data, new_affine), centroid, fwhm, canonical_affine })
}

/// The scan as float32, reoriented to RAS+, in C order, with the zooms of a fresh header.
///
/// Mirrors `img = nib.as_closest_canonical(img)` then
/// `nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine)`.
fn canonical_f32(nifti: &Nifti, scan: &Scan, ornt: &Orientation) -> Result<Volume<f32>, PrepError> {
    let data = nifti.fdata_f32_oriented(ornt)?;
    let affine = if *ornt == Orientation::RAS {
        scan.affine
    } else {
        ornt.reoriented_affine(&scan.affine, scan.shape)
    };
    Ok(Volume::new(data, affine))
}

/// Gaussian anti-aliasing for axes finer than the target voxel size.
///
/// Mirrors the `smooth before downsampling` block of `conform_image` and
/// `nibabel.processing.smooth_image(img, fwhm, mode="nearest")`. The FWHM arithmetic follows
/// numpy's promotion: `sz_**2 - sz**2` is a Python float minus a float32, so it is float32.
fn anti_alias(image: Volume<f32>, min_voxel_size: f64) -> Result<(Volume<f32>, [f64; 3]), PrepError> {
    let target = min_voxel_size as f32;
    if !image.zooms.iter().any(|&size| target > size) {
        return Ok((image, [0.0; 3]));
    }
    let target_sq = (min_voxel_size * min_voxel_size) as f32;
    let fwhm = image.zooms.map(|size| {
        let diff = target_sq - size * size;
        // Python's max(diff, 0.0) keeps diff unless 0.0 is strictly larger
        let diff = if 0.0 > diff { 0.0 } else { diff };
        (diff as f64).sqrt()
    });
    let vox = image.affine.voxel_sizes();
    let sigmas = std::array::from_fn(|axis| Sigma::F64(fwhm[axis] / vox[axis] / SIGMA2FWHM));
    let data = gaussian_filter(&image.data, sigmas, ExtendMode::Nearest, 4.0)?;
    Ok((Volume { data, ..image }, fwhm))
}

/// The RAS+ voxel the output grid is centred on.
///
/// Mirrors `conform_image`: x and y are the midpoints of the mask's bounding box, rounded half to
/// even like Python's `round`; z is `round(top - fov_z / zoom_z / 2)` with
/// `top = min(mask_z_max + top_margin / zoom_z, n_z)`, in the precisions numpy uses.
fn grid_centroid(
    mask: ArrayView3<'_, u8>,
    zooms: [f32; 3],
    n_z: usize,
    config: &Conform,
) -> Result<[i64; 3], PrepError> {
    let [lo, hi] = bounding_box(mask).ok_or(PrepError::EmptyMask)?;
    let midpoint = |axis: usize| round_half_even((lo[axis] + hi[axis]) as f64 / 2.0) as i64;

    let zoom_z = zooms[2];
    let top = hi[2] as f64 + (config.top_margin as f32 / zoom_z) as f64;
    let half_height = config.fov[2] as f32 / zoom_z / 2.0;
    let z = if (n_z as f64) < top {
        // the capped `top` is a Python int, so the subtraction happens in float32
        round_half_even((n_z as f32 - half_height) as f64)
    } else {
        round_half_even(top - half_height as f64)
    };
    Ok([midpoint(0), midpoint(1), z as i64])
}

/// Per-axis `[min, max]` indices of nonzero voxels, or `None` for an empty mask.
fn bounding_box(mask: ArrayView3<'_, u8>) -> Option<[[usize; 3]; 2]> {
    let planes: Vec<Option<[[usize; 3]; 2]>> = mask
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(x, plane)| {
            let mut bbox: Option<[[usize; 3]; 2]> = None;
            for ((y, z), &v) in plane.indexed_iter() {
                if v > 0 {
                    let b = bbox.get_or_insert([[x, y, z], [x, y, z]]);
                    b[0][1] = b[0][1].min(y);
                    b[0][2] = b[0][2].min(z);
                    b[1][1] = b[1][1].max(y);
                    b[1][2] = b[1][2].max(z);
                }
            }
            bbox
        })
        .collect();
    planes.into_iter().flatten().reduce(|a, b| {
        [std::array::from_fn(|i| a[0][i].min(b[0][i])), std::array::from_fn(|i| a[1][i].max(b[1][i]))]
    })
}

/// An affine with new voxel sizes whose grid centre lands on `centroid`.
///
/// Mirrors `eval/preprocessing.py::rescale_affine(affine, shape, zooms, new_shape, centroid)`:
/// `rzs_out = affine[:3, :3] * zooms / voxel_sizes(affine)` and
/// `t_out = apply_affine(affine, centroid) - rzs_out @ ((new_shape - 1) // 2)`.
pub fn rescale_affine(affine: &Affine, zooms: [f64; 3], new_shape: [usize; 3], centroid: [i64; 3]) -> Affine {
    let sizes = affine.voxel_sizes();
    let rzs = affine.rzs();
    let rzs_out: [[f64; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|j| rzs[i][j] * zooms[j] / sizes[j]));
    let world = affine.apply(centroid.map(|c| c as f64));
    let half = new_shape.map(|n| ((n - 1) / 2) as f64);
    // numpy's (3, 3) @ (3,) product sums left to right without BLAS
    let t_out = std::array::from_fn(|i| {
        world[i] - (rzs_out[i][0] * half[0] + rzs_out[i][1] * half[1] + rzs_out[i][2] * half[2])
    });
    Affine::from_matvec(rzs_out, t_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescale_affine_centres_the_grid() {
        let affine =
            Affine::from_matvec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [-91.0, -126.0, -72.0]);
        let out = rescale_affine(&affine, [1.0; 3], [192, 240, 192], [90, 110, 79]);
        assert_eq!(out.translation(), [-96.0, -135.0, -88.0]);
    }

    #[test]
    fn centroid_rounds_half_to_even_and_caps_the_top() {
        let mut mask = ndarray::Array3::<u8>::zeros([10, 10, 10]);
        mask[[1, 1, 2]] = 1;
        mask[[4, 6, 7]] = 1;
        let config = Conform { fov: [8.0, 8.0, 8.0], top_margin: 5.0, ..Conform::default() };
        // x: 2.5 -> 2, y: 3.5 -> 4; top 7 + 5 = 12 is capped at n_z = 10, so z = 10 - 4
        assert_eq!(grid_centroid(mask.view(), [1.0; 3], 10, &config).unwrap(), [2, 4, 6]);
        // uncapped, z = 12 - 4
        assert_eq!(grid_centroid(mask.view(), [1.0; 3], 20, &config).unwrap()[2], 8);
        // 12 - 5.5 = 6.5 rounds to even
        let odd = Conform { fov: [8.0, 8.0, 11.0], ..config };
        assert_eq!(grid_centroid(mask.view(), [1.0; 3], 20, &odd).unwrap()[2], 6);
        let empty = ndarray::Array3::<u8>::zeros([3, 3, 3]);
        assert!(matches!(grid_centroid(empty.view(), [1.0; 3], 3, &config), Err(PrepError::EmptyMask)));
    }

    #[test]
    fn bounding_box_of_view() {
        let mut mask = ndarray::Array3::<u8>::zeros([4, 5, 6]);
        mask[[1, 2, 3]] = 1;
        mask[[2, 4, 1]] = 1;
        assert_eq!(bounding_box(mask.view()), Some([[1, 2, 1], [2, 4, 3]]));
        assert_eq!(bounding_box(ndarray::Array3::<u8>::zeros([2, 2, 2]).view()), None);
    }
}
