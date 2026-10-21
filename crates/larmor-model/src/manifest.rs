//! `model.toml` manifests: what a model is, where its weights live and how scans are prepared for it.
//!
//! The format is specified in `docs/DESIGN.md` ("Model manifest"). Parsing is strict: unknown keys
//! and architecture values this crate cannot run are errors, because silently ignoring a model
//! property would produce embeddings that look fine and are wrong.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::ManifestError;

/// A parsed and validated `model.toml`.
///
/// Mirrors the manifests written by `tools/make_models.py::write_model`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Model id, what `--model` takes (e.g. `nanobrain-vits-p8`).
    pub id: String,
    /// Architecture family; only `vit3d` exists today.
    pub family: String,
    /// Free-form description.
    #[serde(default)]
    pub description: String,
    /// Encoder shape and layer options.
    pub architecture: Architecture,
    /// Which encoder output is the embedding.
    pub readout: Readout,
    /// The weights file and its checksum.
    pub weights: Weights,
    /// Preprocessing recipe, kept as raw tables for `larmor-prep` to interpret.
    #[serde(default)]
    pub preprocessing: Preprocessing,
    /// Provenance, free-form.
    #[serde(default)]
    pub source: toml::Table,
    /// Directory of the manifest file; `weights.file` is relative to it.
    #[serde(skip)]
    pub dir: PathBuf,
}

/// `[architecture]`: the ViT hyperparameters, as in `nanobrain.model.ViTMAE3D.__init__`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Architecture {
    /// Input grid in voxels, `[X, Y, Z]`.
    pub grid: [usize; 3],
    /// Patch edge length in voxels (cubic patches).
    pub patch: usize,
    /// Embedding width `D`.
    pub width: usize,
    /// Number of transformer blocks.
    pub depth: usize,
    /// Attention heads per block.
    pub heads: usize,
    /// MLP hidden width as a multiple of `width`.
    pub mlp_ratio: f64,
    /// Number of class tokens prepended to the sequence.
    pub class_tokens: usize,
    /// Position embedding kind; only `separable` (`SeparablePosEmbed3D`).
    pub pos_embed: String,
    /// `separate` q/k/v linears (nanobrain) or `fused` (timm); only `separate` is implemented.
    pub qkv: String,
    /// Whether the q/k/v linears have biases.
    pub qkv_bias: bool,
    /// Whether the attention projection and MLP linears have biases.
    pub proj_bias: bool,
    /// LayerNorm epsilon.
    pub norm_eps: f64,
    /// MLP activation; only `gelu` (exact, erf-based).
    pub activation: String,
}

/// `[readout]`: which tokens are the embedding and how they are pooled.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Readout {
    /// `last`: the last block's output after the final norm.
    pub tokens: String,
    /// `mean`: the mean over in-mask tokens.
    pub pool: String,
}

/// `[weights]`: the safetensors file holding the encoder.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weights {
    /// File name, relative to the manifest.
    pub file: String,
    /// Lower-case hex SHA-256 of the file.
    pub sha256: String,
    /// Optional download location.
    #[serde(default)]
    pub url: String,
    /// Tensor naming scheme; only `nanobrain`.
    pub naming: String,
}

/// `[preprocessing]`: the ordered recipe steps.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Preprocessing {
    /// Each `[[preprocessing.steps]]` table as written; `larmor-prep` parses them.
    #[serde(default)]
    pub steps: Vec<toml::Table>,
}

impl Manifest {
    /// Read, parse and validate a `model.toml`. Does not touch the weights file.
    ///
    /// Mirrors `tools/make_reference.py::load_model`, which reads the same fields with `tomllib`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ManifestError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|source| ManifestError::Read { path: path.to_path_buf(), source })?;
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Self::parse(&text, dir).map_err(|err| err.in_file(path))
    }

    /// Parse and validate manifest text. `dir` is where relative weight paths resolve from.
    pub fn parse(text: &str, dir: impl Into<PathBuf>) -> Result<Self, ManifestError> {
        let mut manifest: Manifest =
            toml::from_str(text).map_err(|err| ManifestError::Parse(Box::new(err)))?;
        manifest.dir = dir.into();
        manifest.validate()?;
        Ok(manifest)
    }

    /// Absolute (or manifest-relative) path of the weights file.
    pub fn weights_path(&self) -> PathBuf {
        self.dir.join(&self.weights.file)
    }

    /// Read the weights file and check its SHA-256 against the manifest. Returns the file bytes so
    /// callers that go on to load the tensors read the file only once.
    pub fn read_verified_weights(&self) -> Result<Vec<u8>, ManifestError> {
        let path = self.weights_path();
        let bytes = fs::read(&path).map_err(|source| ManifestError::Read { path: path.clone(), source })?;
        let actual = sha256_hex(&bytes);
        if !actual.eq_ignore_ascii_case(&self.weights.sha256) {
            return Err(ManifestError::Checksum { path, expected: self.weights.sha256.clone(), actual });
        }
        Ok(bytes)
    }

    /// Check the weights file's SHA-256 without keeping its contents.
    pub fn verify_weights(&self) -> Result<(), ManifestError> {
        self.read_verified_weights().map(drop)
    }

    fn validate(&self) -> Result<(), ManifestError> {
        expect_one_of("family", &self.family, &["vit3d"])?;
        self.architecture.validate()?;
        expect_one_of("readout.tokens", &self.readout.tokens, &["last"])?;
        expect_one_of("readout.pool", &self.readout.pool, &["mean"])?;
        expect_one_of("weights.naming", &self.weights.naming, &["nanobrain"])?;
        let sha = &self.weights.sha256;
        if sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(invalid("weights.sha256", format!("{sha:?} is not a 64-digit hex SHA-256")));
        }
        if self.weights.file.is_empty() {
            return Err(invalid("weights.file", "is empty".to_string()));
        }
        for (i, step) in self.preprocessing.steps.iter().enumerate() {
            if !matches!(step.get("op"), Some(toml::Value::String(_))) {
                return Err(invalid(
                    &format!("preprocessing.steps[{i}].op"),
                    "missing or not a string".into(),
                ));
            }
        }
        Ok(())
    }
}

impl Architecture {
    /// Patch grid `[X/p, Y/p, Z/p]`, as `ViTMAE3D.__init__` computes `patch_grid_size`.
    pub fn patch_grid(&self) -> [usize; 3] {
        self.grid.map(|size| size / self.patch)
    }

    /// Values per patch, `p³`.
    pub fn patch_dim(&self) -> usize {
        self.patch.pow(3)
    }

    /// Per-head width, `width / heads`.
    pub fn head_dim(&self) -> usize {
        self.width / self.heads
    }

    /// MLP hidden width. `modules.py::Mlp` uses `int(dim * mlp_ratio)`, which truncates.
    pub fn mlp_hidden(&self) -> usize {
        (self.width as f64 * self.mlp_ratio) as usize
    }

    fn validate(&self) -> Result<(), ManifestError> {
        expect_one_of("architecture.pos_embed", &self.pos_embed, &["separable"])?;
        expect_one_of("architecture.qkv", &self.qkv, &["separate"])?;
        expect_one_of("architecture.activation", &self.activation, &["gelu"])?;
        let positive = [
            ("architecture.patch", self.patch),
            ("architecture.width", self.width),
            ("architecture.depth", self.depth),
            ("architecture.heads", self.heads),
        ];
        for (field, value) in positive {
            if value == 0 {
                return Err(invalid(field, "must be at least 1".into()));
            }
        }
        if let Some(axis) = self.grid.iter().position(|&size| size == 0 || !size.is_multiple_of(self.patch)) {
            return Err(invalid(
                "architecture.grid",
                format!(
                    "axis {axis} ({}) is not a positive multiple of patch {}",
                    self.grid[axis], self.patch
                ),
            ));
        }
        if !self.width.is_multiple_of(self.heads) {
            return Err(invalid(
                "architecture.heads",
                format!("width {} is not divisible by {} heads", self.width, self.heads),
            ));
        }
        if !(self.mlp_ratio.is_finite() && self.mlp_ratio > 0.0) || self.mlp_hidden() == 0 {
            return Err(invalid(
                "architecture.mlp_ratio",
                format!("{} gives no hidden units", self.mlp_ratio),
            ));
        }
        if !(self.norm_eps.is_finite() && self.norm_eps > 0.0) {
            return Err(invalid("architecture.norm_eps", format!("{} must be positive", self.norm_eps)));
        }
        if self.patch_grid().iter().any(|&n| n > usize::from(u16::MAX)) {
            return Err(invalid("architecture.grid", "patch grid does not fit u16 coordinates".into()));
        }
        Ok(())
    }
}

/// Lower-case hex SHA-256, the format `tools/larmor_ref.py::sha256_file` writes.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn expect_one_of(field: &str, value: &str, supported: &[&str]) -> Result<(), ManifestError> {
    if supported.contains(&value) {
        Ok(())
    } else {
        Err(ManifestError::Unsupported {
            field: field.to_string(),
            value: value.to_string(),
            supported: supported.join(", "),
        })
    }
}

fn invalid(field: &str, reason: String) -> ManifestError {
    ManifestError::Invalid { field: field.to_string(), reason }
}
