//! Running the whole recipe on a scan.

use std::time::{Duration, Instant};

use larmor_volume::{Affine, Interpolation, Nifti, Volume, Voxel, io_orientation, resample_from_to};
use ndarray::{Array3, ArrayView3};

use crate::conform::{conform_image, place_grid};
use crate::error::PrepError;
use crate::mask::threshold_mask;
use crate::normalize::max_zscore;
use crate::quantize::truncate_image;
use crate::recipe::Recipe;
use crate::scan::Scan;
use crate::tokens::{Tokens, tokens};

/// Wall-clock time of each stage of [`Recipe::run`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Head mask (step 1).
    pub mask: Duration,
    /// Reorient, anti-alias, conform and the fit-grid mask (steps 2–5).
    pub conform: Duration,
    /// Quantize (step 6).
    pub quantize: Duration,
    /// Normalize (step 7).
    pub normalize: Duration,
}

impl Timings {
    /// Sum of the stages.
    pub fn total(&self) -> Duration {
        self.mask + self.conform + self.quantize + self.normalize
    }
}

/// A preprocessed scan: everything the encoder needs, plus the intermediate stages that parity
/// checks and outputs use.
#[derive(Clone, Debug)]
pub struct Prepared {
    /// Normalised model input on the grid (step 7), zero outside `mask`.
    pub image: Array3<f32>,
    /// Quantized uint16 image on the grid (step 6), what nanobrain's eval datasets store.
    pub quantized: Array3<u16>,
    /// Float32 conformed image on the grid before quantization (step 4).
    pub conformed: Array3<f32>,
    /// Head mask on the grid (step 5).
    pub mask: Array3<bool>,
    /// Head mask on the scan's own voxel grid (step 1); with a reference scan, the reference's
    /// mask (the two share a voxel grid).
    pub native_mask: Array3<bool>,
    /// Affine of the grid: of `mask` and of labels mapped onto it (the reference's grid when there
    /// is one, as `prepare_images` stores it).
    pub fit_affine: Affine,
    /// Affine the image was conformed on. Equal to `fit_affine` for a scan on its own; with a
    /// reference scan it is the target's own placement, which `prepare_images` requires to agree
    /// with the reference's to 1e-3.
    pub image_affine: Affine,
    /// Affine of the scan.
    pub native_affine: Affine,
    /// Shape of the scan.
    pub native_shape: [usize; 3],
    /// RAS+ voxel the grid is centred on (see [`crate::conform::Conformed::centroid`]).
    pub centroid: [i64; 3],
    /// Anti-aliasing FWHM (mm) per RAS+ axis, zero when none was applied.
    pub fwhm: [f64; 3],
    /// Lower quantization window bound.
    pub vmin: f64,
    /// Upper quantization window bound.
    pub vmax: f64,
    /// Volume of the native head mask in millilitres.
    pub mask_volume_ml: f64,
    /// Time spent in each stage.
    pub timings: Timings,
}

/// Absolute tolerance `prepare_images` allows between a modality's conformed grid and the
/// reference's (`np.allclose(fit_img.affine, affine, atol=1e-3)`, numpy's default `rtol=1e-5`).
const GRID_ATOL: f64 = 1e-3;
const GRID_RTOL: f64 = 1e-5;

impl Recipe {
    /// Preprocesses one scan.
    ///
    /// Mirrors `eval/prepare.py::prepare_images({"image": img}, {}, reference="image")` (head
    /// mask, conform, fit-grid mask, quantize) followed by the normalisation in
    /// `ViTMAE3D.dense_embed`.
    pub fn run(&self, nifti: &Nifti) -> Result<Prepared, PrepError> {
        self.validate()?;
        let start = Instant::now();
        let native_mask = threshold_mask(nifti, &self.mask)?;
        let mask_time = start.elapsed();
        self.finish(nifti, nifti, native_mask, mask_time)
    }

    /// Preprocesses `target` on the head mask and grid of `reference`, another modality of the same
    /// subject on the same voxel grid.
    ///
    /// Mirrors `eval/prepare.py::prepare_images({..., "ref": reference, "other": target}, segs,
    /// reference="ref")` for the target: the head mask comes from the reference alone, the target
    /// is conformed with that mask, its grid must agree with the reference's to 1e-3 (the
    /// function's assertion), and it is quantized and normalised inside the reference's grid mask.
    /// The eval framework prepares FOMO tasks 1 and 2 this way (ADC and DWI on the FLAIR's mask).
    ///
    /// `run_with_reference(x, x)` equals `run(x)`. A target on a different voxel grid (shape,
    /// orientation, or an affine whose grid lands elsewhere) is an error.
    pub fn run_with_reference(&self, target: &Nifti, reference: &Nifti) -> Result<Prepared, PrepError> {
        self.validate()?;
        let (target_scan, reference_scan) = (Scan::of(target)?, Scan::of(reference)?);
        if target_scan.shape != reference_scan.shape {
            return Err(PrepError::ReferenceGrid(format!(
                "the scan is {:?} voxels but the reference is {:?}",
                target_scan.shape, reference_scan.shape
            )));
        }
        let (target_ornt, reference_ornt) =
            (io_orientation(&target_scan.affine)?, io_orientation(&reference_scan.affine)?);
        if target_ornt != reference_ornt {
            return Err(PrepError::ReferenceGrid(
                "the scan and the reference store their voxel axes in different orientations".into(),
            ));
        }
        let start = Instant::now();
        let native_mask = threshold_mask(reference, &self.mask)?;
        let mask_time = start.elapsed();
        self.finish(target, reference, native_mask, mask_time)
    }

    /// Steps 2–7 of `target` with the head mask `native_mask` of `reference` (the same image for
    /// a scan on its own).
    fn finish(
        &self,
        target: &Nifti,
        reference: &Nifti,
        native_mask: Volume<u8>,
        mask_time: std::time::Duration,
    ) -> Result<Prepared, PrepError> {
        let scan = Scan::of(target)?;
        let mut timings = Timings { mask: mask_time, ..Timings::default() };

        let start = Instant::now();
        let conformed = conform_image(target, &native_mask, &self.conform)?;
        let image_affine = conformed.image.affine;
        let grid = conformed.image.shape();
        let (mask_affine, fit_affine) = if std::ptr::eq(target, reference) {
            (scan.affine, image_affine)
        } else {
            // prepare_images: the grid is the reference's conform_image placement
            let reference_scan = Scan::of(reference)?;
            let placement = place_grid(&reference_scan, &native_mask, &self.conform)?;
            check_grid_agrees(&image_affine, &placement.affine)?;
            (reference_scan.affine, placement.affine)
        };
        // prepare_images: resample_from_to(mask_img, reference_img, order=0) > 0
        let mask = resample_from_to(
            native_mask.data.view(),
            &mask_affine,
            grid,
            &fit_affine,
            Interpolation::Nearest,
        )
        .ok_or(PrepError::SingularAffine)?
        .mapv(|v| v > 0);
        timings.conform = start.elapsed();

        let start = Instant::now();
        let quantized = truncate_image(&conformed.image.data, &mask, self.quantize.low, self.quantize.high)?;
        timings.quantize = start.elapsed();

        let start = Instant::now();
        let image = max_zscore(&quantized.data, &mask, self.normalize.std_floor);
        timings.normalize = start.elapsed();

        let native_mask = native_mask.data.mapv(|v| v > 0);
        let voxel_ml = scan.affine.determinant3().abs() / 1000.0;
        let mask_volume_ml = native_mask.iter().filter(|&&inside| inside).count() as f64 * voxel_ml;
        Ok(Prepared {
            image,
            quantized: quantized.data,
            conformed: conformed.image.data,
            mask,
            native_mask,
            fit_affine,
            image_affine,
            native_affine: scan.affine,
            native_shape: scan.shape,
            centroid: conformed.centroid,
            fwhm: conformed.fwhm,
            vmin: quantized.vmin,
            vmax: quantized.vmax,
            mask_volume_ml,
            timings,
        })
    }
}

/// `prepare_images`' `assert np.allclose(fit_img.affine, affine, atol=1e-3)`, as an error.
fn check_grid_agrees(image_affine: &Affine, reference_affine: &Affine) -> Result<(), PrepError> {
    let worst = image_affine
        .0
        .iter()
        .flatten()
        .zip(reference_affine.0.iter().flatten())
        .map(|(a, b)| (a - b).abs() - (GRID_ATOL + GRID_RTOL * b.abs()))
        .fold(f64::NEG_INFINITY, f64::max);
    // NaN (a comparison numpy's allclose also fails) counts as a disagreement
    if worst <= 0.0 {
        Ok(())
    } else {
        Err(PrepError::ReferenceGrid(format!(
            "conformed on the reference's head mask, the scan's grid lands {:.3} mm (or more) away from \
             the reference's; the two are not on the same voxel grid",
            worst + GRID_ATOL
        )))
    }
}

impl Prepared {
    /// The grid shape.
    pub fn grid_shape(&self) -> [usize; 3] {
        let s = self.image.shape();
        [s[0], s[1], s[2]]
    }

    /// The encoder tokens for `patch`-voxel patches.
    ///
    /// Mirrors `patchify3d` and the in-mask selection of `ViTMAE3D.dense_embed`.
    pub fn tokens(&self, patch: usize) -> Result<Tokens, PrepError> {
        tokens(&self.image, &self.mask, patch)
    }

    /// Maps a volume on the model grid (a segmentation, say) back to the scan's voxel grid with
    /// nearest neighbour; voxels outside the grid get 0.
    ///
    /// Mirrors `nibabel.processing.resample_from_to(grid_img, native_img, order=0)` with
    /// `grid_img` carrying [`Prepared::fit_affine`]. Integer labels stay exact.
    pub fn to_native<T: Voxel>(&self, grid: ArrayView3<'_, T>) -> Result<Array3<T>, PrepError> {
        if grid.shape() != self.grid_shape() {
            return Err(PrepError::GridShape { expected: self.grid_shape(), got: grid.shape().to_vec() });
        }
        resample_from_to(
            grid,
            &self.fit_affine,
            self.native_shape,
            &self.native_affine,
            Interpolation::Nearest,
        )
        .ok_or(PrepError::SingularAffine)
    }
}
