//! What one encode did: the contents of `<stem>_model-<id>_run.json`, also kept in the cache entry's
//! metadata so a cache hit can report how its embedding was made.

use std::path::Path;

use anyhow::Result;
use larmor_dicom::SeriesInfo;
use larmor_model::{DeviceChoice, Encoder, Precision};
use larmor_prep::{IMPL_VERSION, Prepared};
use larmor_volume::{Nifti, aff2axcodes};
use serde::{Deserialize, Serialize};

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
    /// The scan whose head mask and grid were used (`--reference`), when not the scan itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<ReferenceInfo>,
    /// Where the embedding is cached and whether this run found it there (run.json only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheInfo>,
    /// On a cache hit, the run that computed the embedding: its scan path and manifest (another file
    /// with the same bytes, or another manifest with the same encoder, can have made it) and its
    /// times (run.json only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoded_from: Option<EncodedFrom>,
}

/// A reference scan: another modality of the same subject whose head mask and grid the scan was
/// preprocessed on.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceInfo {
    /// Absolute path when encoded.
    pub path: String,
    /// SHA-256 of the file bytes (of a DICOM series, as for the scan).
    pub sha256: String,
}

/// The run that encoded a cached embedding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncodedFrom {
    /// The scan file that run read.
    pub scan_path: String,
    /// The `model.toml` that run used.
    pub manifest: String,
    pub started: String,
    pub finished: String,
}

/// The encoder that made the embedding.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    /// SHA-256 of the weights file.
    pub sha256: String,
    /// `Manifest::encoder_key`: the weights, every architecture and readout value and the encoder
    /// implementation version, hashed (part of the cache key).
    #[serde(default)]
    pub encoder_key: String,
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
    /// Absolute path when encoded: the NIfTI file, or the DICOM series' folder.
    pub path: String,
    /// SHA-256 of the file bytes; for a DICOM series, `larmor_dicom::series_sha256` (its files'
    /// SHA-256s, sorted, with the conversion's version).
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
    /// For a DICOM series: what it is and how it was converted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dicom: Option<DicomInfo>,
}

/// A DICOM series, converted to the image dcm2niix writes for it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DicomInfo {
    pub series_instance_uid: String,
    pub series_number: Option<i32>,
    pub series_description: String,
    pub protocol_name: String,
    pub modality: String,
    pub manufacturer: String,
    /// Slices (files) in the volume.
    pub slices: usize,
    pub transfer_syntaxes: Vec<String>,
    /// `larmor-dicom <IMPL_VERSION> (dcm2niix <version>)`: the conversion and what it reproduces.
    pub conversion: String,
}

impl DicomInfo {
    pub fn of(series: &SeriesInfo) -> DicomInfo {
        DicomInfo {
            series_instance_uid: series.series_instance_uid.clone(),
            series_number: series.series_number,
            series_description: series.description.clone(),
            protocol_name: series.protocol_name.clone(),
            modality: series.modality.clone(),
            manufacturer: series.manufacturer.clone(),
            slices: series.slices,
            transfer_syntaxes: series.transfer_syntaxes.clone(),
            conversion: format!(
                "larmor-dicom {} (dcm2niix {})",
                larmor_dicom::IMPL_VERSION,
                larmor_dicom::DCM2NIIX_VERSION
            ),
        }
    }
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
    pub device: DeviceChoice,
    pub precision: Precision,
    pub seconds: StageSeconds,
    /// When the scan was read.
    pub started: String,
}

impl Encoding {
    /// The device and precision `encoder` actually runs with, and the stage timings.
    pub fn new(encoder: &Encoder, seconds: StageSeconds, started: String) -> Encoding {
        let options = encoder.options();
        Encoding { device: options.device, precision: options.precision, seconds, started }
    }
}

impl RunInfo {
    /// The record of one encode, from the scan, its preprocessing and the model.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: &Model,
        encoding: Encoding,
        scan: &Path,
        sha: &str,
        nifti: &Nifti,
        dicom: Option<&SeriesInfo>,
        prepared: &Prepared,
        n_tokens: usize,
        reference: Option<ReferenceInfo>,
    ) -> Result<RunInfo> {
        let arch = &model.manifest.architecture;
        let affine = nifti.affine()?;
        Ok(RunInfo {
            larmor_version: env!("CARGO_PKG_VERSION").to_string(),
            model: ModelInfo {
                id: model.id().to_string(),
                sha256: model.weights_sha256(),
                encoder_key: model.encoder_key.clone(),
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
                dicom: dicom.map(DicomInfo::of),
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
            reference,
            cache: None,
            encoded_from: None,
        })
    }

    /// This record as the run.json of `scan` encoded with `model` in this run: on a cache hit the
    /// scan path and manifest become this run's, and the cached run's go to `encoded_from`.
    pub fn for_run(&self, model: &Model, scan: &Path, cache: CacheInfo) -> RunInfo {
        let mut info = self.clone();
        if cache.hit {
            info.encoded_from = Some(EncodedFrom {
                scan_path: info.scan.path.clone(),
                manifest: info.model.manifest.clone(),
                started: info.started.clone(),
                finished: info.finished.clone(),
            });
            info.scan.path = absolute(scan);
            info.model.manifest = absolute(&model.path);
        }
        info.cache = Some(cache);
        info
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
