//! Step 8: cut the normalised image into patches and keep the ones inside the mask.

use ndarray::{Array3, s};
use rayon::prelude::*;

use crate::error::PrepError;

/// The encoder's input: in-mask patches and their patch-grid coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Tokens {
    /// Patch edge length in voxels.
    pub patch: usize,
    /// Patch grid shape `[gx, gy, gz]`.
    pub grid: [usize; 3],
    /// Row-major `[N, patch³]`: patch `n` is `patches[n * patch³..(n + 1) * patch³]`, its voxels
    /// in `(px, py, pz)` order.
    pub patches: Vec<f32>,
    /// `[gx, gy, gz]` of each patch, in the order of `patches`.
    pub coords: Vec<[u16; 3]>,
}

impl Tokens {
    /// Number of tokens (in-mask patches).
    pub fn len(&self) -> usize {
        self.coords.len()
    }

    /// True when no patch touches the mask.
    pub fn is_empty(&self) -> bool {
        self.coords.is_empty()
    }

    /// Values per patch, `patch³`.
    pub fn patch_len(&self) -> usize {
        self.patch.pow(3)
    }

    /// The values of token `n`.
    pub fn patch_values(&self, n: usize) -> &[f32] {
        let p = self.patch_len();
        &self.patches[n * p..(n + 1) * p]
    }
}

/// Patchifies `image` and keeps every patch with at least one voxel in `mask`.
///
/// Mirrors `model.py::patchify3d` (einops `b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)`)
/// and the selection in `ViTMAE3D.dense_embed`
/// (`patchify3d(image_mask).any(dim=2)`, coordinates from `make_coord_grid`, `ij` order).
pub fn tokens(image: &Array3<f32>, mask: &Array3<bool>, patch: usize) -> Result<Tokens, PrepError> {
    let shape = image.shape();
    if patch == 0 || shape.iter().any(|&n| n % patch != 0) {
        return Err(PrepError::PatchSize { patch, grid: [shape[0], shape[1], shape[2]] });
    }
    let grid = [shape[0] / patch, shape[1] / patch, shape[2] / patch];
    let per_slab: Vec<(Vec<[u16; 3]>, Vec<f32>)> = (0..grid[0])
        .into_par_iter()
        .map(|gx| {
            let mut coords = Vec::new();
            let mut values = Vec::new();
            for gy in 0..grid[1] {
                for gz in 0..grid[2] {
                    let block = s![
                        gx * patch..(gx + 1) * patch,
                        gy * patch..(gy + 1) * patch,
                        gz * patch..(gz + 1) * patch
                    ];
                    if mask.slice(block).iter().any(|&inside| inside) {
                        coords.push([gx as u16, gy as u16, gz as u16]);
                        values.extend(image.slice(block).iter());
                    }
                }
            }
            (coords, values)
        })
        .collect();
    let mut out = Tokens { patch, grid, patches: Vec::new(), coords: Vec::new() };
    for (coords, values) in per_slab {
        out.coords.extend(coords);
        out.patches.extend(values);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_order_matches_einops() {
        let image = Array3::from_shape_fn([4, 2, 2], |(x, y, z)| (x * 100 + y * 10 + z) as f32);
        let mut mask = Array3::from_elem([4, 2, 2], false);
        mask[[3, 1, 0]] = true;
        let t = tokens(&image, &mask, 2).unwrap();
        assert_eq!(t.coords, vec![[1, 0, 0]]);
        assert_eq!(t.patch_values(0), &[200.0, 201.0, 210.0, 211.0, 300.0, 301.0, 310.0, 311.0]);
        assert!(tokens(&image, &mask, 3).is_err());
    }
}
