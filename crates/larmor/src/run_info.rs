//! What one encode did: the contents of `<stem>_model-<id>_run.json`, also kept in the cache entry's
//! metadata so a cache hit can report how its embedding was made.

use std::path::Path;

use anyhow::Result;
use larmor_prep::{IMPL_VERSION, Prepared};
use larmor_volume::{Nifti, aff2axcodes};
use serde::{Deserialize, Serialize};

use crate::device::{DeviceArg, PrecisionArg};
use crate::model::Model;

/// One scan's encode record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunInfo {
    pub larmor_version: String,
    pub model: ModelInfo,
    pub preprocessing: PrepInfo,
    pub scan: ScanInfo,
    pub grid: GridInfo,
    /// Lower quantization window bound (the in-mask 0.5% quantile by default).
    pub vmin: f64,
    /// Upper quantization window bound.
    pub vmax: f64,
    /// Volume of the native-space head mask, in millilitres.
    pub head_volume_ml: f64,
    /// Number of in-mask patch tokens.
    pub tokens: usize,
    /// Embedding width.
    pub width: usize,
    /// Device the encoder ran on (`metal`, `cpu`, `cuda:0`).
    pub device: String,
    /// Encoder matmul precision (`f32`, `f16`, `bf16`).
    pub precision: String,
    pub seconds: StageSeconds,
    /// RFC 3339 local time the scan was read.
    pub started: String,
    /// RFC 3339 local time the cache entry was written.
    pub finished: String,
    /// Where the embedding is cached and whether this run found it there (run.json only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheInfo>,
}

/// The encoder that made the embedding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// SHA-256 of the weights file.
    pub sha256: String,
    /// The `model.toml` used.
    pub manifest: String,
    pub family: String,
    pub patch: usize,
    pub width: usize,
    pub depth: usize,
}

/// The preprocessing recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrepInfo {
    /// `Recipe::version_hash`: SHA-256 of the canonical steps plus the implementation version.
    pub version: String,
    /// `larmor_prep::IMPL_VERSION`.
    pub impl_version: String,
    /// Every step with every value, as `[[preprocessing.steps]]` would list them.
    pub steps: serde_json::Value,
}

/// The input scan, as its header describes it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScanInfo {
    /// Absolute path when encoded.
    pub path: String,
    /// SHA-256 of the file bytes.
    pub sha256: String,
    pub shape: Vec<usize>,
    /// Header voxel sizes (mm).
    pub zooms: Vec<f32>,
    /// Axis codes of the voxel axes (`RAS`, `LPS`, `PIL`...).
    pub orientation: String,
    /// Stored data type (`int16`, `float32`...).
    pub dtype: String,
    /// Voxel-to-world affine of the scan.
    pub affine: [[f64; 4]; 4],
}

/// The model grid the scan was resampled onto.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridInfo {
    /// Grid shape in voxels (`[192, 240, 192]`).
    pub shape: [usize; 3],
    /// Voxel-to-world affine of the grid; maps labels from native space onto the grid.
    pub fit_affine: [[f64; 4]; 4],
    /// RAS+ voxel of the scan at the grid centre.
    pub centroid: [i64; 3],
    /// Anti-aliasing FWHM (mm) per RAS+ axis, zero when none was applied.
    pub fwhm: [f64; 3],
    /// Patch edge in voxels.
    pub patch: usize,
}

/// Wall-clock seconds per stage.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StageSeconds {
    /// Reading and decompressing the file.
    pub read: f64,
    /// SHA-256 of the file bytes.
    pub hash: f64,
    pub mask: f64,
    pub conform: f64,
    pub quantize: f64,
    pub normalize: f64,
    pub tokens: f64,
    pub encode: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// The cache entry a run.json belongs to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CacheInfo {
    pub path: String,
    /// True when this run found the embedding in the cache instead of encoding.
    pub hit: bool,
}

/// How an embedding was computed, for its [`RunInfo`].
pub struct Encoding {
    pub device: DeviceArg,
    pub precision: PrecisionArg,
    pub seconds: StageSeconds,
    /// When the scan was read.
    pub started: String,
}

impl RunInfo {
    /// The record of one encode, from the scan, its preprocessing and the model.
    pub fn new(
        model: &Model,
        encoding: Encoding,
        scan: &Path,
        sha: &str,
        nifti: &Nifti,
        prepared: &Prepared,
        n_tokens: usize,
    ) -> Result<RunInfo> {
        let arch = &model.manifest.architecture;
        let affine = nifti.affine()?;
        Ok(RunInfo {
            larmor_version: env!("CARGO_PKG_VERSION").to_string(),
            model: ModelInfo {
                id: model.id().to_string(),
                sha256: model.weights_sha256(),
                manifest: absolute(&model.path),
                family: model.manifest.family.clone(),
                patch: arch.patch,
                width: arch.width,
                depth: arch.depth,
            },
            preprocessing: PrepInfo {
                version: model.prep_version.clone(),
                impl_version: IMPL_VERSION.to_string(),
                steps: serde_json::to_value(model.recipe.steps())?,
            },
            scan: ScanInfo {
                path: absolute(scan),
                sha256: sha.to_string(),
                shape: nifti.shape().to_vec(),
                zooms: nifti.zooms(),
                orientation: aff2axcodes(&affine).unwrap_or_else(|_| "?".into()),
                dtype: nifti.data.dtype().name().to_string(),
                affine: affine.0,
            },
            grid: GridInfo {
                shape: prepared.grid_shape(),
                fit_affine: prepared.fit_affine.0,
                centroid: prepared.centroid,
                fwhm: prepared.fwhm,
                patch: arch.patch,
            },
            vmin: prepared.vmin,
            vmax: prepared.vmax,
            head_volume_ml: prepared.mask_volume_ml,
            tokens: n_tokens,
            width: arch.width,
            device: encoding.device.to_string(),
            precision: encoding.precision.to_string(),
            seconds: encoding.seconds,
            started: encoding.started,
            finished: now(),
            cache: None,
        })
    }
}

/// `path` made absolute (without resolving symlinks), as text.
pub fn absolute(path: &Path) -> String {
    std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf()).display().to_string()
}

/// Local time in RFC 3339, to the second.
pub fn now() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, false)
}
