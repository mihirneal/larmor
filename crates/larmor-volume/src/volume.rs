//! A 3-D image in memory: voxel array, affine and header zooms.

use ndarray::Array3;

use crate::affine::Affine;

/// A 3-D image: a C-order voxel array indexed `[x, y, z]`, its voxel-to-world affine and the
/// voxel sizes its header would carry.
///
/// This is the in-memory `nib.Nifti1Image(data, affine, header)` that nanobrain passes between
/// steps. `zooms` is kept separately from the affine because nibabel does too: `get_zooms()`
/// reads the header's float32 `pixdim`, which nibabel only refreshes from the affine when the
/// affine changed (see [`Volume::zooms_for`]).
#[derive(Clone, Debug)]
pub struct Volume<T> {
    /// Voxel values, `data[[x, y, z]]`.
    pub data: Array3<T>,
    /// Voxel-to-world (RAS mm) affine.
    pub affine: Affine,
    /// Header voxel sizes (`img.header.get_zooms()`), float32 like NIfTI's `pixdim`.
    pub zooms: [f32; 3],
}

impl<T> Volume<T> {
    /// A volume with a fresh header, like `nib.Nifti1Image(data, affine)`: the zooms are the
    /// affine's column norms rounded to float32.
    pub fn new(data: Array3<T>, affine: Affine) -> Volume<T> {
        let zooms = affine.header_zooms();
        Volume { data, affine, zooms }
    }

    /// The array shape `[nx, ny, nz]`.
    pub fn shape(&self) -> [usize; 3] {
        let s = self.data.shape();
        [s[0], s[1], s[2]]
    }

    /// The zooms a new image gets when it is built with `affine` and this volume's header, like
    /// `nib.Nifti1Image(new_data, affine, self.header)`.
    ///
    /// Mirrors `SpatialImage.update_header`: the header (and so its zooms) is left alone when the
    /// new affine is `np.allclose` to the header's, otherwise the zooms come from the new affine.
    pub fn zooms_for(&self, affine: &Affine) -> [f32; 3] {
        if affine.allclose(&self.affine) { self.zooms } else { affine.header_zooms() }
    }
}
