//! The embedding cache: one safetensors file per (scan, preprocessing, model weights).
//!
//! ```text
//! <cache>/<model_id>-<model_sha8>/<prep_version8>/<scan_sha256>.safetensors
//!     tokens f16 [N, D], coords u16 [N, 3], pooled f32 [D]; metadata: the run.json fields
//! ```
//!
//! Changing the scan bytes, the preprocessing or the weights changes the path, so a stale entry is
//! never read.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use half::f16;
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};

use crate::model::Model;
use crate::run_info::RunInfo;

/// Bumped when the layout of cache entries changes.
pub const CACHE_FORMAT: &str = "1";

/// The cache file for a scan with this SHA-256, embedded by `model`.
pub fn entry_path(root: &Path, model: &Model, scan_sha256: &str) -> PathBuf {
    let sha = model.weights_sha256();
    root.join(format!("{}-{}", model.id(), &sha[..8]))
        .join(&model.prep_version[..8])
        .join(format!("{scan_sha256}.safetensors"))
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
    /// Writes an entry atomically (to a temporary file, then renamed), creating folders as needed.
    pub fn write(
        path: &Path,
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
        let metadata = HashMap::from([
            ("format".to_string(), CACHE_FORMAT.to_string()),
            ("model_id".to_string(), info.model.id.clone()),
            ("model_sha256".to_string(), info.model.sha256.clone()),
            ("preprocessing_version".to_string(), info.preprocessing.version.clone()),
            ("scan_sha256".to_string(), info.scan.sha256.clone()),
            ("run".to_string(), serde_json::to_string(info)?),
        ]);
        let bytes = safetensors::serialize(views, Some(metadata))?;
        let dir = path.parent().context("cache path has no folder")?;
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    /// Reads an entry; the tokens only when `with_tokens` (scan-level probes need just `pooled`).
    pub fn read(path: &Path, with_tokens: bool) -> Result<CacheEntry> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&bytes, with_tokens).with_context(|| format!("cache entry {}", path.display()))
    }

    fn parse(bytes: &[u8], with_tokens: bool) -> Result<CacheEntry> {
        let (_, header) = SafeTensors::read_metadata(bytes)?;
        let metadata = header.metadata().clone().unwrap_or_default();
        let format = metadata.get("format").map(String::as_str).unwrap_or("?");
        if format != CACHE_FORMAT {
            bail!("cache format {format}, this larmor reads {CACHE_FORMAT}");
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
