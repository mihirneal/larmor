//! Errors from preprocessing.

use larmor_volume::{NiftiError, OrientationError};
use thiserror::Error;

use crate::recipe::RecipeError;

/// Why a scan could not be preprocessed.
#[derive(Debug, Error)]
pub enum PrepError {
    /// The recipe is invalid.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// The scan's header could not be interpreted (for example a bad qform quaternion).
    #[error(transparent)]
    Nifti(#[from] NiftiError),
    /// The scan's affine has no voxel orientation.
    #[error(transparent)]
    Orientation(#[from] OrientationError),
    /// The recipe needs a 3-D image.
    #[error("expected a 3-D image, got shape {0:?}")]
    NotThreeD(Vec<usize>),
    /// An affine could not be inverted.
    #[error("the scan's affine is singular")]
    SingularAffine,
    /// Thresholding found no head (an empty or constant image, or NaNs).
    #[error("the head mask is empty (is the image blank, constant or NaN?)")]
    EmptyMask,
    /// The in-mask intensity window collapsed, as nanobrain's `assert vmax > vmin` checks.
    #[error("degenerate in-mask intensity range: vmin {vmin}, vmax {vmax}")]
    DegenerateRange {
        /// Lower quantile.
        vmin: f64,
        /// Upper quantile.
        vmax: f64,
    },
    /// The grid does not divide into patches.
    #[error("patch size {patch} does not divide the grid {grid:?}")]
    PatchSize {
        /// Requested patch size.
        patch: usize,
        /// Grid shape.
        grid: [usize; 3],
    },
    /// A head mask passed to conform does not have the scan's shape.
    #[error("head mask shape {mask:?} does not match the scan shape {scan:?}")]
    MaskShape {
        /// The mask's shape.
        mask: [usize; 3],
        /// The scan's shape.
        scan: [usize; 3],
    },
    /// A volume passed back to native space does not have the grid's shape.
    #[error("expected a volume of shape {expected:?} on the model grid, got {got:?}")]
    GridShape {
        /// The model grid shape.
        expected: [usize; 3],
        /// The shape passed in.
        got: Vec<usize>,
    },
}
