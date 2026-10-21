//! Patch tokens from a normalized image and its mask.

use anyhow::{Result, ensure};

/// In-mask patches of an image, in patch-grid order.
#[derive(Debug, Clone, PartialEq)]
pub struct Patches {
    /// `[N, patch^3]` row-major, voxels of each patch in `(px, py, pz)` order.
    pub values: Vec<f32>,
    /// `(gx, gy, gz)` of each patch.
    pub coords: Vec<[u32; 3]>,
}

/// Patchify an `[X, Y, Z]` C-order image and keep the patches with any mask voxel set.
///
/// Mirrors `nanobrain/model.py::patchify3d` (einops
/// `b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)`) plus the selection in
/// `ViTMAE3D.dense_embed`: `in_mask = patchify3d(mask).any(dim=2)`, coordinates from
/// `make_coord_grid` (`ij` meshgrid, so gx varies slowest).
pub fn in_mask_patches(image: &[f32], mask: &[u8], shape: [usize; 3], patch: usize) -> Result<Patches> {
    let [nx, ny, nz] = shape;
    ensure!(image.len() == nx * ny * nz && mask.len() == image.len(), "image and mask must be {shape:?}");
    ensure!(
        shape.iter().all(|s| s.is_multiple_of(patch)),
        "shape {shape:?} is not divisible by patch {patch}"
    );
    let voxel = |x: usize, y: usize, z: usize| (x * ny + y) * nz + z;
    let mut patches = Patches { values: Vec::new(), coords: Vec::new() };
    for gx in 0..nx / patch {
        for gy in 0..ny / patch {
            for gz in 0..nz / patch {
                let mut values = Vec::with_capacity(patch.pow(3));
                let mut any_in_mask = false;
                for px in 0..patch {
                    for py in 0..patch {
                        for pz in 0..patch {
                            let i = voxel(gx * patch + px, gy * patch + py, gz * patch + pz);
                            values.push(image[i]);
                            any_in_mask |= mask[i] > 0;
                        }
                    }
                }
                if any_in_mask {
                    patches.values.extend(values);
                    patches.coords.push([gx as u32, gy as u32, gz as u32]);
                }
            }
        }
    }
    Ok(patches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_order_matches_einops() {
        // 4x2x2 image, patch 2: grid 2x1x1, voxel value = its flat index.
        let shape = [4, 2, 2];
        let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let mut mask = vec![0u8; 16];
        mask[13] = 1; // x=3, y=0, z=1: only the second patch is in the mask
        let patches = in_mask_patches(&image, &mask, shape, 2).unwrap();
        assert_eq!(patches.coords, vec![[1, 0, 0]]);
        // (px, py, pz) order over x in 2..4, y in 0..2, z in 0..2
        assert_eq!(patches.values, vec![8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
    }
}
