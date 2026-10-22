//! The embedding cache: one safetensors file per (scan, reference scan, preprocessing, encoder,
//! precision).
//!
//! ```text
//! <cache>/<model_id>-<encoder_key8>/<prep_version8>/<precision>/<scan_sha256>[_ref-<reference_sha256>].safetensors
//!     tokens f16 [N, D], coords u16 [N, 3], pooled f32 [D]; metadata: the key and the run.json fields
//! ```
//!
//! The encoder key covers the weights and every manifest value that changes the forward pass
//! (`heads`, `norm_eps`...), so correcting a manifest never serves the old embeddings. Changing the
//! scan bytes, the reference scan, the preprocessing, the encoder or the precision changes the
//! path, so a stale entry is never read; the metadata repeats the key and a read checks it too.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use half::f16;
use larmor_model::Precision;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::model::Model;
use crate::run_info::RunInfo;

/// Bumped when the layout of cache entries changes.
pub const CACHE_FORMAT: &str = "2";

/// What, besides the model, identifies one cached embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EntryKey<'a> {
    /// SHA-256 of the scan file's bytes.
    pub scan_sha256: &'a str,
    /// SHA-256 of the reference scan (`--reference`), when its head mask and grid were used.
    pub reference_sha256: Option<&'a str>,
    /// The precision the encoder ran in.
    pub precision: Precision,
}

/// The cache file for one scan embedded by `model`.
pub fn entry_path(root: &Path, model: &Model, key: &EntryKey) -> PathBuf {
    let name = match key.reference_sha256 {
        None => format!("{}.safetensors", key.scan_sha256),
        Some(reference) => format!("{}_ref-{reference}.safetensors", key.scan_sha256),
    };
    root.join(format!("{}-{}", model.id(), &model.encoder_key[..8]))
        .join(&model.prep_version[..8])
        .join(key.precision.to_string())
        .join(name)
}

/// Tokens as a cache entry stores them (float16) and gives them back (widened to float32), so a
/// freshly encoded scan hands the probes exactly the features a cache hit would.
pub fn as_stored(tokens: &[f32]) -> Vec<f32> {
    tokens.iter().map(|&v| f16::from_f32(v).to_f32()).collect()
}

/// A cached embedding.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// `[N, D]` row-major patch tokens, widened from the stored f16; empty when read without tokens.
    pub tokens: Vec<f32>,
    /// `[N]` patch grid coordinates `(gx, gy, gz)`.
    pub coords: Vec<[u16; 3]>,
    /// `[D]` mean of the f32 tokens.
    pub pooled: Vec<f32>,
    pub info: RunInfo,
}

impl CacheEntry {
    /// Writes an entry atomically (to a synced temporary file, then renamed), creating folders as
    /// needed.
    pub fn write(
        path: &Path,
        key: &EntryKey,
        tokens: &[f32],
        coords: &[[u16; 3]],
        pooled: &[f32],
        info: &RunInfo,
    ) -> Result<()> {
        let width = pooled.len();
        ensure!(width > 0 && tokens.len() == coords.len() * width, "tokens do not match coords and pooled");
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
        let coord_bytes: Vec<u8> = coords.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        let pooled_bytes: Vec<u8> = pooled.iter().flat_map(|v| v.to_le_bytes()).collect();
        let n = coords.len();
        let views = vec![
            ("tokens", TensorView::new(Dtype::F16, vec![n, width], &token_bytes)?),
            ("coords", TensorView::new(Dtype::U16, vec![n, 3], &coord_bytes)?),
            ("pooled", TensorView::new(Dtype::F32, vec![width], &pooled_bytes)?),
        ];
        let mut metadata = HashMap::from([
            ("format".to_string(), CACHE_FORMAT.to_string()),
            ("model_id".to_string(), info.model.id.clone()),
            ("model_sha256".to_string(), info.model.sha256.clone()),
            ("encoder_key".to_string(), info.model.encoder_key.clone()),
            ("precision".to_string(), key.precision.to_string()),
            ("preprocessing_version".to_string(), info.preprocessing.version.clone()),
            ("scan_sha256".to_string(), key.scan_sha256.to_string()),
            ("run".to_string(), serde_json::to_string(info)?),
        ]);
        if let Some(reference) = key.reference_sha256 {
            metadata.insert("reference_sha256".to_string(), reference.to_string());
        }
        let bytes = safetensors::serialize(views, Some(metadata))?;
        larmor_volume::write_atomic(path, &bytes).with_context(|| format!("writing {}", path.display()))
    }

    /// Reads the entry for `model` and `key`; the tokens only when `with_tokens` (scan-level probes
    /// need just `pooled`). An entry whose metadata names another key, or that holds non-finite
    /// values, is an error, which callers treat like a missing entry.
    pub fn read(path: &Path, model: &Model, key: &EntryKey, with_tokens: bool) -> Result<CacheEntry> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes, model, key, with_tokens)
            .with_context(|| format!("cache entry {}", path.display()))
    }

    fn parse(bytes: &[u8], model: &Model, key: &EntryKey, with_tokens: bool) -> Result<CacheEntry> {
        let (_, header) = SafeTensors::read_metadata(bytes)?;
        let metadata = header.metadata().clone().unwrap_or_default();
        let format = metadata.get("format").map(String::as_str).unwrap_or("?");
        if format != CACHE_FORMAT {
            bail!("cache format {format}, this larmor reads {CACHE_FORMAT}");
        }
        let precision = key.precision.to_string();
        let expected = [
            ("encoder_key", Some(model.encoder_key.as_str())),
            ("preprocessing_version", Some(model.prep_version.as_str())),
            ("precision", Some(precision.as_str())),
            ("scan_sha256", Some(key.scan_sha256)),
            ("reference_sha256", key.reference_sha256),
        ];
        for (field, value) in expected {
            let found = metadata.get(field).map(String::as_str);
            ensure!(found == value, "{field} is {found:?}, expected {value:?}");
        }
        let info: RunInfo = serde_json::from_str(metadata.get("run").context("no run metadata")?)?;
        let file = SafeTensors::deserialize(bytes)?;
        let pooled: Vec<f32> = tensor(&file, "pooled", Dtype::F32)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect();
        let coords: Vec<[u16; 3]> = tensor(&file, "coords", Dtype::U16)?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| u16::from_le_bytes(c))
            .collect::<Vec<u16>>()
            .as_chunks::<3>()
            .0
            .to_vec();
        let tokens = if with_tokens {
            let raw = tensor(&file, "tokens", Dtype::F16)?;
            raw.as_chunks::<2>().0.iter().map(|&c| f16::from_le_bytes(c).to_f32()).collect()
        } else {
            Vec::new()
        };
        ensure!(
            pooled.len() == info.width && coords.len() == info.tokens,
            "tensor shapes disagree with the metadata"
        );
        ensure!(pooled.iter().chain(&tokens).all(|v| v.is_finite()), "the embedding holds non-finite values");
        Ok(CacheEntry { tokens, coords, pooled, info })
    }

    /// Embedding width `D`.
    pub fn width(&self) -> usize {
        self.pooled.len()
    }
}

fn tensor<'a>(file: &'a SafeTensors<'a>, name: &str, dtype: Dtype) -> Result<&'a [u8]> {
    let view = file.tensor(name).with_context(|| format!("no {name} tensor"))?;
    ensure!(view.dtype() == dtype, "{name} is {:?}, expected {dtype:?}", view.dtype());
    Ok(view.data())
}

#[cfg(test)]
mod tests {
    use super::*;
    use larmor_model::Manifest;

    fn model(text: &str) -> Model {
        let manifest = Manifest::parse(text, "models").unwrap();
        Model::new(manifest, PathBuf::from("models/m.toml")).unwrap()
    }

    /// nanobrain ViT-S, patch 8, with the nanobrain-eval recipe (the steps take their defaults).
    fn manifest_text() -> String {
        let steps = ["threshold_mask", "conform", "quantize", "normalize"]
            .map(|op| format!("[[preprocessing.steps]]\nop = \"{op}\"\n"))
            .join("\n");
        format!(
            r#"id = "vits"
family = "vit3d"
description = "ViT-S"

[architecture]
grid = [192, 240, 192]
patch = 8
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
file = "vits.safetensors"
sha256 = "{}"
naming = "nanobrain"

{steps}"#,
            "9".repeat(64)
        )
    }

    #[test]
    fn the_key_separates_encoders_precisions_and_references() {
        let text = manifest_text();
        let base = model(&text);
        let key = EntryKey { scan_sha256: "ab", reference_sha256: None, precision: Precision::F32 };
        let root = Path::new("cache");
        let path = entry_path(root, &base, &key);
        assert!(
            path.ends_with(format!("{}/f32/ab.safetensors", &base.prep_version[..8])),
            "{}",
            path.display()
        );

        // a manifest differing only in a value that changes the forward pass is another entry
        for (from, to) in [("heads = 6", "heads = 12"), ("norm_eps = 1e-6", "norm_eps = 1e-5")] {
            assert!(text.contains(from), "{from}");
            assert_ne!(entry_path(root, &model(&text.replace(from, to)), &key), path, "{to}");
        }
        // the description does not change the numbers, so it shares the entry
        assert_eq!(
            entry_path(root, &model(&text.replacen("description = \"", "description = \"x", 1)), &key),
            path
        );
        let bf16 = EntryKey { precision: Precision::Bf16, ..key };
        assert_ne!(entry_path(root, &base, &bf16), path);
        let referenced = EntryKey { reference_sha256: Some("cd"), ..key };
        assert!(entry_path(root, &base, &referenced).ends_with("f32/ab_ref-cd.safetensors"));
    }
}
