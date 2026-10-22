//! Probe files: `probe.toml` (what the probe is, what it was trained on, how well it did) next to
//! `probe.safetensors` (the weights).
//!
//! A probe is only valid on embeddings from the same model weights and the same preprocessing, so
//! [`Probe::check_embeddings`] refuses any other.

use crate::classification::ClassificationModel;
use crate::device::{Device, DeviceError};
use crate::logistic::LogisticFit;
use crate::matrix::Matrix;
use crate::regression::RegressionModel;
use crate::ridge::{GcvMode, RidgeFit};
use crate::scaler::StandardScaler;
use crate::segmentation::{LinearHead, SegmentationModel};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Bumped whenever the layout of probe files changes.
pub const PROBE_FORMAT: u32 = 1;

/// What a probe predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeKind {
    /// One value per scan from the pooled embedding (ridge).
    Regression,
    /// The probability of the positive class per scan from the pooled embedding (logistic).
    Classification,
    /// Voxel probabilities per in-mask patch from the patch tokens (one logistic per voxel of the patch).
    Segmentation,
}

/// The chosen hyperparameters. Only the ones that apply to the probe's kind are set.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hyperparameters {
    /// Ridge or segmentation penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f64>,
    /// Logistic inverse penalty `C`.
    #[serde(rename = "C", skip_serializing_if = "Option::is_none")]
    pub c: Option<f64>,
    /// Segmentation probability threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_negative_ratio: Option<f64>,
    /// How they were chosen, e.g. "leave-one-out on all samples".
    pub selection: String,
    /// Solver details worth knowing when comparing with other implementations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver: Option<String>,
}

/// The contents of `probe.toml`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeMetadata {
    pub format: u32,
    pub kind: ProbeKind,
    /// The encoder whose embeddings the probe reads (`--model`).
    pub model_id: String,
    /// SHA-256 of the encoder weights; embeddings from other weights are refused.
    pub model_sha256: String,
    /// The model's encoder key (`larmor_model::Manifest::encoder_key`: weights, every
    /// `[architecture]` and `[readout]` value and the encoder implementation version). Embeddings
    /// from an encoder with another key are refused even when the weights are the same (a
    /// corrected `heads` or `norm_eps`, say). Absent in probes written before larmor recorded it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_key: Option<String>,
    /// Precision of the training embeddings (`f32`, `f16` or `bf16`). Absent in older probes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<String>,
    /// Preprocessing version of the embeddings it was trained on; others are refused.
    pub preprocessing_version: String,
    /// Which encoder output: "last" is the last block after the final norm.
    pub layer: String,
    /// "mean" over in-mask tokens for scan-level probes, "none" (per token) for segmentation.
    pub pooling: String,
    pub embed_dim: usize,
    /// The label column(s) of the training table; for classification, then the negative and positive class.
    pub label_names: Vec<String>,
    /// SHA-256 of the training label table (`labels.tsv`).
    pub label_table_sha256: String,
    pub n_train: usize,
    pub larmor_version: String,
    /// The weights file, relative to `probe.toml`.
    pub weights: String,
    pub weights_sha256: String,
    pub hyperparameters: Hyperparameters,
    /// The eval-identical outer cross-validation: metrics, CIs and per-fold choices.
    pub cv: toml::Table,
}

/// The weights of a deployable probe.
#[derive(Clone, Debug)]
pub enum ProbeWeights {
    Regression(RegressionModel),
    Classification(ClassificationModel),
    Segmentation(SegmentationModel),
}

impl ProbeWeights {
    /// The kind of probe these weights make.
    pub fn kind(&self) -> ProbeKind {
        match self {
            ProbeWeights::Regression(_) => ProbeKind::Regression,
            ProbeWeights::Classification(_) => ProbeKind::Classification,
            ProbeWeights::Segmentation(_) => ProbeKind::Segmentation,
        }
    }

    /// The embedding width the weights expect.
    pub fn embed_dim(&self) -> usize {
        match self {
            ProbeWeights::Regression(m) => m.scaler.mean.len(),
            ProbeWeights::Classification(m) => m.scaler.mean.len(),
            ProbeWeights::Segmentation(m) => m.mean.len(),
        }
    }
}

/// Where a set of embeddings came from, as recorded with them in the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddingSource {
    pub model_sha256: String,
    /// The encoder key (see [`ProbeMetadata::encoder_key`]).
    pub encoder_key: String,
    pub preprocessing_version: String,
    pub embed_dim: usize,
}

/// Why a probe cannot be read or used.
#[derive(Debug, Error)]
pub enum ProbeError {
    #[error("probe was trained on model weights {probe}, but the embeddings come from {embeddings}")]
    ModelMismatch { probe: String, embeddings: String },
    #[error(
        "probe was trained on embeddings of encoder {probe}, but this model's encoder is {embeddings} (same \
         weights, but another architecture, readout or encoder version: a manifest value such as heads or \
         norm_eps changed); fit the probe again"
    )]
    EncoderMismatch { probe: String, embeddings: String },
    #[error("probe was trained with preprocessing {probe}, but the embeddings were made with {embeddings}")]
    PreprocessingMismatch { probe: String, embeddings: String },
    #[error("probe expects {probe}-dimensional embeddings, got {embeddings}")]
    DimensionMismatch { probe: usize, embeddings: usize },
    #[error(
        "{what} {row} has a non-finite value; the embeddings are damaged (encoder overflow or a bad cache entry)"
    )]
    NonFinite { what: &'static str, row: usize },
    #[error("probe kind is {probe:?}; this prediction needs {wanted:?}")]
    WrongKind { probe: ProbeKind, wanted: ProbeKind },
    #[error("weights file {path} has sha256 {actual}, probe.toml expects {expected}")]
    WeightsChanged { path: PathBuf, expected: String, actual: String },
    #[error("unsupported probe format {0} (this larmor reads format {PROBE_FORMAT})")]
    Format(u32),
    #[error("probe weights: {0}")]
    Weights(String),
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("probe.toml: {0}")]
    Toml(String),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

/// A probe file: metadata and weights.
#[derive(Clone, Debug)]
pub struct Probe {
    pub metadata: ProbeMetadata,
    pub weights: ProbeWeights,
}

impl Probe {
    /// Writes `toml_path` and the weights file next to it (`probe.safetensors` for `probe.toml`), filling in
    /// the metadata's `weights`, `weights_sha256`, `kind`, `embed_dim` and `format`.
    ///
    /// Each file is written atomically (a synced temporary file renamed into place), the weights
    /// first: an interrupted write leaves the previous probe, or new weights that the old
    /// `probe.toml` refuses by their checksum, never a truncated file.
    pub fn write(&mut self, toml_path: &Path) -> Result<(), ProbeError> {
        let weights_path = toml_path.with_extension("safetensors");
        let bytes = serialize_weights(&self.weights)?;
        write_file(&weights_path, &bytes)?;
        let metadata = &mut self.metadata;
        metadata.format = PROBE_FORMAT;
        metadata.kind = self.weights.kind();
        metadata.embed_dim = self.weights.embed_dim();
        metadata.weights = weights_path.file_name().unwrap().to_string_lossy().into_owned();
        metadata.weights_sha256 = sha256_hex(&bytes);
        let text = toml::to_string(metadata).map_err(|e| ProbeError::Toml(e.to_string()))?;
        write_file(toml_path, text.as_bytes())
    }

    /// Reads `probe.toml` and its weights, refusing a weights file that changed since it was written.
    pub fn read(toml_path: &Path) -> Result<Self, ProbeError> {
        let text = read_file(toml_path)?;
        let metadata: ProbeMetadata =
            toml::from_str(&String::from_utf8_lossy(&text)).map_err(|e| ProbeError::Toml(e.to_string()))?;
        if metadata.format != PROBE_FORMAT {
            return Err(ProbeError::Format(metadata.format));
        }
        let weights_path = toml_path.with_file_name(&metadata.weights);
        let bytes = read_file(&weights_path)?;
        let actual = sha256_hex(&bytes);
        if actual != metadata.weights_sha256 {
            return Err(ProbeError::WeightsChanged {
                path: weights_path,
                expected: metadata.weights_sha256.clone(),
                actual,
            });
        }
        let weights = deserialize_weights(metadata.kind, &bytes)?;
        Ok(Self { metadata, weights })
    }

    /// Refuses embeddings from other model weights, other preprocessing or of another width.
    pub fn check_embeddings(&self, source: &EmbeddingSource) -> Result<(), ProbeError> {
        let metadata = &self.metadata;
        if metadata.model_sha256 != source.model_sha256 {
            return Err(ProbeError::ModelMismatch {
                probe: metadata.model_sha256.clone(),
                embeddings: source.model_sha256.clone(),
            });
        }
        if let Some(key) = &metadata.encoder_key
            && *key != source.encoder_key
        {
            return Err(ProbeError::EncoderMismatch {
                probe: key.clone(),
                embeddings: source.encoder_key.clone(),
            });
        }
        if metadata.preprocessing_version != source.preprocessing_version {
            return Err(ProbeError::PreprocessingMismatch {
                probe: metadata.preprocessing_version.clone(),
                embeddings: source.preprocessing_version.clone(),
            });
        }
        if self.weights.embed_dim() != source.embed_dim {
            return Err(ProbeError::DimensionMismatch {
                probe: self.weights.embed_dim(),
                embeddings: source.embed_dim,
            });
        }
        Ok(())
    }

    /// Scan-level predictions from pooled embeddings `[n_scans, embed_dim]`: the regression value, or the
    /// probability of the positive class.
    pub fn predict_scans(&self, pooled: &Matrix<f32>) -> Result<Vec<f64>, ProbeError> {
        self.check_width(pooled.cols())?;
        check_finite(pooled, "pooled embedding")?;
        match &self.weights {
            ProbeWeights::Regression(model) => Ok(model.predict(pooled)),
            ProbeWeights::Classification(model) => Ok(model.predict_proba(pooled)),
            ProbeWeights::Segmentation(_) => {
                Err(ProbeError::WrongKind { probe: ProbeKind::Segmentation, wanted: ProbeKind::Regression })
            }
        }
    }

    /// Voxel probabilities `[n_patches, 512]` (row-major) from one scan's in-mask patch tokens, and the
    /// probe's threshold for turning them into a mask.
    pub fn predict_patches(
        &self,
        tokens: &Matrix<f32>,
        device: &Device,
    ) -> Result<(Vec<f32>, f32), ProbeError> {
        self.check_width(tokens.cols())?;
        check_finite(tokens, "patch token")?;
        match &self.weights {
            ProbeWeights::Segmentation(model) => {
                let threshold = model.threshold.ok_or_else(|| ProbeError::Weights("no threshold".into()))?;
                Ok((model.predict(tokens, device)?, threshold))
            }
            other => Err(ProbeError::WrongKind { probe: other.kind(), wanted: ProbeKind::Segmentation }),
        }
    }

    fn check_width(&self, cols: usize) -> Result<(), ProbeError> {
        if cols != self.weights.embed_dim() {
            return Err(ProbeError::DimensionMismatch { probe: self.weights.embed_dim(), embeddings: cols });
        }
        Ok(())
    }
}

/// Lower-case hex SHA-256, for label tables and weights files.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// Fails on the first row with a NaN or infinite value.
fn check_finite(rows: &Matrix<f32>, what: &'static str) -> Result<(), ProbeError> {
    match rows.iter_rows().position(|row| row.iter().any(|v| !v.is_finite())) {
        Some(row) => Err(ProbeError::NonFinite { what, row }),
        None => Ok(()),
    }
}

/// Writes through a synced, uniquely named temporary file in the same folder and a rename, so the
/// file is either the old one or the complete new one.
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), ProbeError> {
    let io_err = |source| ProbeError::Io { path: path.to_path_buf(), source };
    let dir = match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    let mut tmp =
        tempfile::Builder::new().prefix(".probe.").suffix(".tmp").tempfile_in(dir).map_err(io_err)?;
    tmp.write_all(bytes).map_err(io_err)?;
    tmp.as_file().sync_all().map_err(io_err)?;
    tmp.persist(path).map_err(|e| io_err(e.error))?;
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<u8>, ProbeError> {
    std::fs::read(path).map_err(|source| ProbeError::Io { path: path.to_path_buf(), source })
}

/// Named little-endian tensors on their way into a safetensors file.
#[derive(Default)]
struct TensorSet {
    tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
}

impl TensorSet {
    fn f64(&mut self, name: &str, values: &[f64]) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensors.push((name.into(), Dtype::F64, vec![values.len()], bytes));
    }

    fn f32(&mut self, name: &str, shape: Vec<usize>, values: &[f32]) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensors.push((name.into(), Dtype::F32, shape, bytes));
    }

    fn serialize(&self, kind: ProbeKind, extra: &[(&str, String)]) -> Result<Vec<u8>, ProbeError> {
        let views: Vec<(String, TensorView<'_>)> = self
            .tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.clone(), view))
            })
            .collect::<Result<_, _>>()
            .map_err(|e| ProbeError::Weights(e.to_string()))?;
        let kind = serde_json::to_string(&kind).unwrap_or_default();
        let mut metadata = HashMap::from([("kind".to_string(), kind.trim_matches('"').to_string())]);
        metadata.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));
        safetensors::serialize(views, Some(metadata)).map_err(|e| ProbeError::Weights(e.to_string()))
    }
}

fn serialize_weights(weights: &ProbeWeights) -> Result<Vec<u8>, ProbeError> {
    let mut set = TensorSet::default();
    let mut extra = Vec::new();
    match weights {
        ProbeWeights::Regression(model) => {
            set.f64("scaler_mean", &model.scaler.mean);
            set.f64("scaler_scale", &model.scaler.scale);
            set.f32("coef", vec![model.ridge.coef.len()], &model.ridge.coef);
            set.f32("intercept", vec![1], &[model.ridge.intercept]);
            set.f64("alpha", &[model.ridge.alpha]);
            set.f64("loo_mse", &model.ridge.loo_mse);
            let mode = match model.ridge.mode {
                GcvMode::Eigen => "eigen",
                GcvMode::Svd => "svd",
            };
            extra.push(("gcv_mode", mode.to_string()));
        }
        ProbeWeights::Classification(model) => {
            set.f64("scaler_mean", &model.scaler.mean);
            set.f64("scaler_scale", &model.scaler.scale);
            set.f64("coef", &model.logistic.coef);
            set.f64("intercept", &[model.logistic.intercept]);
            set.f64("C", &[model.logistic.c]);
        }
        ProbeWeights::Segmentation(model) => {
            let (d, k) = (model.head.coef.rows(), model.head.coef.cols());
            set.f32("scaler_mean", vec![d], &model.mean);
            set.f32("scaler_scale", vec![d], &model.std);
            set.f32("coef", vec![d, k], model.head.coef.data());
            set.f32("intercept", vec![k], &model.head.intercept);
            set.f64("alpha", &[model.alpha]);
            if let Some(threshold) = model.threshold {
                set.f32("threshold", vec![1], &[threshold]);
            }
        }
    }
    set.serialize(weights.kind(), &extra)
}

fn deserialize_weights(kind: ProbeKind, bytes: &[u8]) -> Result<ProbeWeights, ProbeError> {
    let tensors = SafeTensors::deserialize(bytes).map_err(|e| ProbeError::Weights(e.to_string()))?;
    let (_, header) = SafeTensors::read_metadata(bytes).map_err(|e| ProbeError::Weights(e.to_string()))?;
    let extra = header.metadata().clone().unwrap_or_default();
    let read = Reader { tensors: &tensors };
    Ok(match kind {
        ProbeKind::Regression => {
            let scaler = StandardScaler { mean: read.f64("scaler_mean")?, scale: read.f64("scaler_scale")? };
            let mode = if extra.get("gcv_mode").map(String::as_str) == Some("svd") {
                GcvMode::Svd
            } else {
                GcvMode::Eigen
            };
            let ridge = RidgeFit {
                alpha: read.scalar_f64("alpha")?,
                intercept: read.scalar_f32("intercept")?,
                coef: read.f32("coef")?,
                loo_mse: read.f64("loo_mse")?,
                mode,
            };
            ProbeWeights::Regression(RegressionModel { scaler, ridge })
        }
        ProbeKind::Classification => {
            let scaler = StandardScaler { mean: read.f64("scaler_mean")?, scale: read.f64("scaler_scale")? };
            let logistic = LogisticFit {
                c: read.scalar_f64("C")?,
                coef: read.f64("coef")?,
                intercept: read.scalar_f64("intercept")?,
                scores: Vec::new(),
                coefs_paths: Vec::new(),
                n_iter: Vec::new(),
            };
            ProbeWeights::Classification(ClassificationModel { scaler, logistic })
        }
        ProbeKind::Segmentation => {
            let mean = read.f32("scaler_mean")?;
            let coef = read.f32("coef")?;
            let intercept = read.f32("intercept")?;
            if mean.is_empty() || coef.len() != mean.len() * intercept.len() {
                return Err(ProbeError::Weights(
                    "segmentation coef does not match its mean and intercept".into(),
                ));
            }
            ProbeWeights::Segmentation(SegmentationModel {
                std: read.f32("scaler_scale")?,
                head: LinearHead { coef: Matrix::new(mean.len(), intercept.len(), coef), intercept },
                alpha: read.scalar_f64("alpha")?,
                threshold: read.f32("threshold").ok().and_then(|t| t.first().copied()),
                mean,
            })
        }
    })
}

struct Reader<'a> {
    tensors: &'a SafeTensors<'a>,
}

impl Reader<'_> {
    fn bytes(&self, name: &str, dtype: Dtype) -> Result<Vec<u8>, ProbeError> {
        let view = self.tensors.tensor(name).map_err(|e| ProbeError::Weights(format!("{name}: {e}")))?;
        if view.dtype() != dtype {
            return Err(ProbeError::Weights(format!("{name} is {:?}, expected {dtype:?}", view.dtype())));
        }
        Ok(view.data().to_vec())
    }

    fn f64(&self, name: &str) -> Result<Vec<f64>, ProbeError> {
        let bytes = self.bytes(name, Dtype::F64)?;
        Ok(bytes.as_chunks::<8>().0.iter().map(|&c| f64::from_le_bytes(c)).collect())
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>, ProbeError> {
        let bytes = self.bytes(name, Dtype::F32)?;
        Ok(bytes.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect())
    }

    fn scalar_f32(&self, name: &str) -> Result<f32, ProbeError> {
        self.f32(name)?.first().copied().ok_or_else(|| ProbeError::Weights(format!("{name} is empty")))
    }

    fn scalar_f64(&self, name: &str) -> Result<f64, ProbeError> {
        self.f64(name)?.first().copied().ok_or_else(|| ProbeError::Weights(format!("{name} is empty")))
    }
}
