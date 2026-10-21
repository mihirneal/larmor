//! The `[architecture]` section of a Larmor model manifest (`model.toml`).
//!
//! The encoder is built from these values alone, so any manifest with supported values runs
//! without code changes (ViT-S, ViT-B, ViT-L, any patch size).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

/// Encoder hyperparameters, as written by `tools/make_models.py::write_model`.
///
/// Mirrors the constructor arguments of `nanobrain/model.py::ViTMAE3D.__init__` that the
/// encoder uses (the MAE decoder is never exported).
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct Architecture {
    pub grid: [usize; 3],
    pub patch: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub mlp_ratio: f64,
    pub class_tokens: usize,
    pub pos_embed: String,
    pub qkv: String,
    pub qkv_bias: bool,
    pub proj_bias: bool,
    pub norm_eps: f64,
    pub activation: String,
}

impl Architecture {
    /// A nanobrain-style ViT with the given shape, for benchmarking with random weights.
    pub fn vit(width: usize, depth: usize, heads: usize, patch: usize) -> Self {
        Self {
            grid: [192, 240, 192],
            patch,
            width,
            depth,
            heads,
            mlp_ratio: 4.0,
            class_tokens: 1,
            pos_embed: "separable".into(),
            qkv: "separate".into(),
            qkv_bias: true,
            proj_bias: true,
            norm_eps: 1e-6,
            activation: "gelu".into(),
        }
    }

    /// Patch-grid size per axis: `tuple(size // patch_size for size in grid_size)` in
    /// `model.py::ViTMAE3D.__init__`.
    pub fn patch_grid(&self) -> [usize; 3] {
        self.grid.map(|size| size / self.patch)
    }

    /// Voxels per patch, the input width of `patch_embed`.
    pub fn patch_dim(&self) -> usize {
        self.patch.pow(3)
    }

    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    /// MLP hidden width: `int(dim * mlp_ratio)` in `modules.py::Mlp.__init__` (truncates).
    pub fn mlp_hidden(&self) -> usize {
        (self.width as f64 * self.mlp_ratio) as usize
    }

    /// Fails on values the encoder does not implement, so an unsupported manifest is rejected
    /// up front instead of producing wrong embeddings.
    pub fn validate(&self) -> Result<()> {
        ensure!(self.pos_embed == "separable", "pos_embed {:?} is not supported", self.pos_embed);
        ensure!(
            self.qkv == "separate",
            "qkv {:?} is not supported (only nanobrain's separate q/k/v)",
            self.qkv
        );
        ensure!(self.activation == "gelu", "activation {:?} is not supported", self.activation);
        ensure!(
            self.width.is_multiple_of(self.heads),
            "width {} is not divisible by heads {}",
            self.width,
            self.heads
        );
        for size in self.grid {
            ensure!(
                size.is_multiple_of(self.patch),
                "grid {:?} is not divisible by patch {}",
                self.grid,
                self.patch
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct ManifestFile {
    id: String,
    family: String,
    architecture: Architecture,
    weights: WeightsEntry,
}

#[derive(Debug, Deserialize)]
struct WeightsEntry {
    file: String,
    naming: String,
}

/// A parsed `model.toml`: the model id, its architecture and where its weights live.
#[derive(Debug, Clone)]
pub struct Manifest {
    pub id: String,
    pub architecture: Architecture,
    pub weights_path: PathBuf,
}

impl Manifest {
    /// Reads and validates a manifest; the weights path is resolved relative to it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let file: ManifestFile =
            toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        if file.family != "vit3d" {
            bail!("model family {:?} is not supported", file.family);
        }
        if file.weights.naming != "nanobrain" {
            bail!("weight naming {:?} is not supported", file.weights.naming);
        }
        file.architecture.validate()?;
        let dir = path.parent().unwrap_or(Path::new("."));
        Ok(Self { id: file.id, architecture: file.architecture, weights_path: dir.join(file.weights.file) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_sizes_match_nanobrain() {
        let arch = Architecture::vit(384, 12, 6, 8);
        assert_eq!(arch.patch_grid(), [24, 30, 24]);
        assert_eq!(arch.patch_dim(), 512);
        assert_eq!(arch.head_dim(), 64);
        assert_eq!(arch.mlp_hidden(), 1536);
        arch.validate().unwrap();
    }

    #[test]
    fn rejects_fused_qkv() {
        let arch = Architecture { qkv: "fused".into(), ..Architecture::vit(384, 12, 6, 8) };
        assert!(arch.validate().is_err());
    }

    #[test]
    fn parses_fixture_manifest_when_present() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/models/nanobrain-vits-p8-ref.toml");
        if !path.exists() {
            return;
        }
        let manifest = Manifest::load(&path).unwrap();
        assert_eq!(manifest.id, "nanobrain-vits-p8-ref");
        assert_eq!(manifest.architecture, Architecture::vit(384, 12, 6, 8));
        assert!(manifest.weights_path.ends_with("nanobrain-vits-p8-ref.safetensors"));
    }
}
