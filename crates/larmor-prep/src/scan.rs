//! The geometry of the input scan, checked once.

use larmor_volume::{Affine, Nifti};

use crate::error::PrepError;

/// Shape, affine and header zooms of a 3-D input scan.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scan {
    /// Array shape `[nx, ny, nz]`.
    pub(crate) shape: [usize; 3],
    /// Voxel-to-world affine (`img.affine`).
    pub(crate) affine: Affine,
    /// Header voxel sizes (`img.header.get_zooms()`).
    pub(crate) zooms: [f32; 3],
}

impl Scan {
    /// The geometry of `nifti`, which must be 3-D, like nanobrain's `assert img.ndim == 3`.
    pub(crate) fn of(nifti: &Nifti) -> Result<Scan, PrepError> {
        let shape: [usize; 3] =
            nifti.shape().try_into().map_err(|_| PrepError::NotThreeD(nifti.shape().to_vec()))?;
        let zooms = nifti.zooms();
        Ok(Scan { shape, affine: nifti.affine()?, zooms: [zooms[0], zooms[1], zooms[2]] })
    }
}
