//! A synthetic head and random-weight model, so `larmor bench` runs without fixtures or weights.

use std::path::Path;

use anyhow::Result;
use larmor_model::{Manifest, synthetic};
use larmor_volume::ndarray::Array3;
use larmor_volume::{Affine, Nifti, VoxelData};

use crate::hashing::sha256_hex;

/// A T1-like head, int16 at 1 mm in a `[176, 208, 176]` RAS+ volume (a typical MPRAGE size):
/// white matter, cortex with a folding pattern, CSF, scalp, and noise.
pub fn head_phantom() -> Nifti {
    let shape = [176, 208, 176];
    let center = shape.map(|n| n as f64 / 2.0);
    let radii = [72.0, 92.0, 78.0];
    let mut rng = SplitMix64(7);
    let data = Array3::from_shape_fn(shape, |(x, y, z)| {
        let p = [x as f64 - center[0], y as f64 - center[1], z as f64 - center[2]];
        let r = (0..3).map(|i| (p[i] / radii[i]).powi(2)).sum::<f64>().sqrt();
        let folds = 80.0 * ((p[0] * 0.21).sin() * (p[1] * 0.17).cos() + (p[2] * 0.19).sin());
        let tissue = if r < 0.6 {
            900.0
        } else if r < 0.84 {
            600.0 + folds
        } else if r < 0.9 {
            200.0
        } else if r < 1.0 {
            450.0
        } else {
            0.0
        };
        (tissue + 25.0 * rng.uniform_pm1()).max(0.0) as i16
    });
    let affine = Affine::from_matvec([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], center.map(|c| -c));
    Nifti::new(VoxelData::I16(data.into_dyn()), &affine)
}

/// Writes `nanobrain ViT-S`-shaped random weights and their `model.toml` into `dir`; returns the
/// manifest path. Timing only: the embeddings mean nothing.
pub fn random_vit_small(dir: &Path, patch: usize) -> Result<std::path::PathBuf> {
    let id = format!("random-vits-p{patch}");
    let text = |sha: &str| {
        format!(
            r#"id = "{id}"
family = "vit3d"
description = "random weights in the shape of nanobrain ViT-S (larmor bench)"

[architecture]
grid = [192, 240, 192]
patch = {patch}
width = 384
depth = 12
heads = 6
mlp_ratio = 4.0
class_tokens = 1
pos_embed = "separable"
qkv = "separate"
qkv_bias = true
proj_bias = true
norm_eps = 1e-6
activation = "gelu"

[readout]
tokens = "last"
pool = "mean"

[weights]
file = "{id}.safetensors"
sha256 = "{sha}"
naming = "nanobrain"

[[preprocessing.steps]]
op = "threshold_mask"

[[preprocessing.steps]]
op = "conform"

[[preprocessing.steps]]
op = "quantize"

[[preprocessing.steps]]
op = "normalize"
"#
        )
    };
    let draft = Manifest::parse(&text(&"0".repeat(64)), dir)?;
    let weights = synthetic::random_safetensors(&draft.architecture, 0);
    std::fs::write(dir.join(format!("{id}.safetensors")), &weights)?;
    let path = dir.join(format!("{id}.toml"));
    std::fs::write(&path, text(&sha256_hex(&weights)))?;
    Ok(path)
}

/// Steele et al.'s SplitMix64, enough for phantom noise.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn uniform_pm1(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}
