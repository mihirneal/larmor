//! Errors from reading and converting DICOM.

use std::path::Path;

use thiserror::Error;

/// Why a DICOM file or series could not be read or converted.
#[derive(Debug, Error)]
pub enum DicomError {
    /// A file could not be read or parsed as DICOM.
    #[error("{path}: {message}")]
    Read {
        /// The file.
        path: String,
        /// What went wrong.
        message: String,
    },
    /// A file lacks an attribute the conversion needs.
    #[error("{path}: no {what}")]
    Missing {
        /// The file.
        path: String,
        /// The attribute.
        what: &'static str,
    },
    /// The series is not one 3-D volume Larmor can convert (a localizer, a 4-D series, varying slice
    /// spacing...), or uses a format it does not decode. The message says which and what to do.
    #[error("{0}")]
    Unsupported(String),
}

impl DicomError {
    pub(crate) fn read(path: &Path, error: impl std::fmt::Display) -> DicomError {
        DicomError::Read { path: path.display().to_string(), message: error.to_string() }
    }

    pub(crate) fn missing(path: &Path, what: &'static str) -> DicomError {
        DicomError::Missing { path: path.display().to_string(), what }
    }
}
