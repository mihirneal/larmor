//! Segmentation labels and predictions on the model grid, cut into (and assembled from) patches.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use larmor_probe::Matrix;
use larmor_volume::ndarray::{Array3, Ix3};
use larmor_volume::{Affine, Interpolation, Nifti, round_half_even};

/// A binary label image in the scan's own space, resampled onto the model grid.
///
/// Mirrors `eval/prepare.py::prepare_images` for segs: `resample_from_to(seg, reference_img,
/// order=0)` then `np.asarray(dataobj).round().astype(np.uint8)`; `probe_binary_segmentation` then
/// requires labels of 0 and 1.
pub fn labels_on_grid(path: &Path, grid_shape: [usize; 3], fit_affine: &Affine) -> Result<Array3<u8>> {
    let nifti = Nifti::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        nifti.shape().len() == 3,
        "{}: expected a 3-D label image, got shape {:?}",
        path.display(),
        nifti.shape()
    );
    let affine = nifti.affine()?;
    let resampled = nifti
        .dataobj()
        .resample_from_to(&affine, grid_shape, fit_affine, Interpolation::Nearest)
        .with_context(|| format!("{}: cannot resample (singular affine?)", path.display()))?;
    let values = resampled.to_f64().into_dimensionality::<Ix3>()?;
    let mut labels = Array3::zeros(values.raw_dim());
    for (label, &value) in labels.iter_mut().zip(values.iter()) {
        let rounded = round_half_even(value);
        if rounded != 0.0 && rounded != 1.0 {
            bail!("{}: segmentation labels must be 0 or 1, found {value}", path.display());
        }
        *label = rounded as u8;
    }
    Ok(labels)
}

/// The voxels of each patch at `coords`, `[N, patch³]` in `(px, py, pz)` order.
///
/// Mirrors `probe.py::patchify3d` on the label volume, rows selected like the in-mask patches.
pub fn patch_rows<T: Copy>(volume: &Array3<T>, coords: &[[u16; 3]], patch: usize) -> Matrix<T> {
    let mut data = Vec::with_capacity(coords.len() * patch.pow(3));
    for c in coords {
        let [x0, y0, z0] = c.map(|g| usize::from(g) * patch);
        for px in 0..patch {
            for py in 0..patch {
                for pz in 0..patch {
                    data.push(volume[[x0 + px, y0 + py, z0 + pz]]);
                }
            }
        }
    }
    Matrix::new(coords.len(), patch.pow(3), data)
}

/// The inverse of [`patch_rows`]: a grid volume with each patch's values in place, `fill` elsewhere.
pub fn assemble_patches<T: Copy>(
    rows: &[T],
    coords: &[[u16; 3]],
    patch: usize,
    shape: [usize; 3],
    fill: T,
) -> Array3<T> {
    let mut volume = Array3::from_elem(shape, fill);
    let mut values = rows.iter();
    for c in coords {
        let [x0, y0, z0] = c.map(|g| usize::from(g) * patch);
        for px in 0..patch {
            for py in 0..patch {
                for pz in 0..patch {
                    volume[[x0 + px, y0 + py, z0 + pz]] = *values.next().expect("one row per coordinate");
                }
            }
        }
    }
    volume
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patches_round_trip() {
        let volume = Array3::from_shape_fn([4, 4, 2], |(x, y, z)| (x * 100 + y * 10 + z) as u16);
        let coords = [[1u16, 0, 0], [0, 1, 0]];
        let rows = patch_rows(&volume, &coords, 2);
        assert_eq!(rows.row(0), &[200, 201, 210, 211, 300, 301, 310, 311]);
        let back = assemble_patches(rows.data(), &coords, 2, [4, 4, 2], 0);
        assert_eq!(back[[3, 1, 1]], 311);
        assert_eq!(back[[0, 2, 0]], 20);
        assert_eq!(back[[0, 0, 0]], 0);
    }
}
