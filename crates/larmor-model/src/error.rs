//! Error types. Library code returns these; the CLI wraps them in `anyhow`.

use std::path::{Path, PathBuf};

/// A manifest that cannot be read, parsed, or run by this crate.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    /// The manifest or weights file could not be read.
    #[error("cannot read {path}: {source}")]
    Read {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The TOML is malformed or has unknown or missing keys.
    #[error("invalid model.toml: {0}")]
    Parse(#[source] Box<toml::de::Error>),
    /// A value is well-formed but this crate does not implement it.
    #[error("{field} = {value:?} is not supported by larmor-model (supported: {supported})")]
    Unsupported {
        /// Dotted key, e.g. `architecture.qkv`.
        field: String,
        /// The value found.
        value: String,
        /// Comma-separated supported values.
        supported: String,
    },
    /// A value is out of range or inconsistent with another one.
    #[error("{field} {reason}")]
    Invalid {
        /// Dotted key.
        field: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The weights file does not match `weights.sha256`.
    #[error("{path}: sha256 {actual} does not match the manifest's {expected}")]
    Checksum {
        /// Weights file.
        path: PathBuf,
        /// From the manifest.
        expected: String,
        /// Computed from the file.
        actual: String,
    },
    /// Any of the above, with the manifest path that caused it.
    #[error("{path}: {source}")]
    InFile {
        /// Manifest file.
        path: PathBuf,
        /// The error.
        source: Box<ManifestError>,
    },
}

impl ManifestError {
    pub(crate) fn in_file(self, path: &Path) -> Self {
        match self {
            // these already name a file
            Self::Read { .. } | Self::Checksum { .. } | Self::InFile { .. } => self,
            other => Self::InFile { path: path.to_path_buf(), source: Box::new(other) },
        }
    }
}

/// Failures while loading weights or running the encoder.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// The manifest or its weights file is unusable.
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    /// A tensor file could not be read.
    #[error("cannot read {path}: {source}")]
    Read {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// The safetensors file is malformed.
    #[error("cannot parse safetensors: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// A tensor the architecture needs is absent.
    #[error("weights are missing tensor {0:?}")]
    MissingTensor(String),
    /// Tensors present in the file that the architecture does not use.
    #[error("weights have tensors this architecture does not use: {0:?}")]
    UnexpectedTensors(Vec<String>),
    /// A tensor has the wrong shape or dtype.
    #[error("tensor {name:?}: {reason}")]
    BadTensor {
        /// Tensor name.
        name: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The requested device is not compiled in or not present.
    #[error("device unavailable: {0}")]
    Device(String),
    /// The input patches or coordinates do not fit the model.
    #[error("invalid input: {0}")]
    Input(String),
    /// The backend failed while running or reading back results.
    #[error("backend error: {0}")]
    Backend(String),
}
