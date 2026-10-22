//! DICOM series to the NIfTI image dcm2niix writes for them, so a scan's DICOM export goes through
//! Larmor's preprocessing exactly as dcm2niix's NIfTI of it would.
//!
//! ```no_run
//! use larmor_dicom::{SliceHeader, group_series};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut headers = Vec::new();
//! for entry in std::fs::read_dir("dicom/T1_MPRAGE")? {
//!     let path = entry?.path();
//!     if larmor_dicom::is_dicom_file(&path) {
//!         headers.push(SliceHeader::read(&path)?);
//!     }
//! }
//! for series in group_series(headers) {
//!     let converted = series.convert()?;
//!     println!("{}: {:?}", series.label(), converted.nifti.shape());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! The target is dcm2niix v1.0.20260724 with its default options, loaded by nibabel: the same
//! voxel array and dtype, sform (`img.affine`), header zooms and scale factors. dcm2niix computes
//! the sform in float32 and is built by clang, which fuses multiply-adds; [`geometry`] reproduces
//! both, so the affines are bit-identical on the gate's series (`larmor check-parity`).
//!
//! What is read: classic DICOM (one slice per file) of one 3-D volume in any transfer syntax
//! dicom-rs decodes in pure Rust (uncompressed in either byte order, deflate, RLE, JPEG
//! baseline, extended and lossless, JPEG 2000). What is refused, with the reason: enhanced
//! multi-frame files, Siemens mosaics, colour images, JPEG-LS, localizers and other series whose
//! slices differ in orientation or size, 4-D series (repeated slice positions), several echoes,
//! slice spacings that vary (missing slices), gantry tilt, and Philips series whose scaling
//! varies between slices.

#![warn(missing_docs)]

mod convert;
mod error;
mod geometry;
mod header;
mod pixels;
mod series;
pub mod synthetic;

use std::path::PathBuf;

use sha2::{Digest, Sha256};

pub use convert::{Converted, SeriesInfo, convert_files, convert_paths, decode_pixels};
pub use error::DicomError;
pub use header::{Manufacturer, SliceHeader, is_dicom_bytes, is_dicom_file};

/// Bumped whenever a change could move any converted image: it is part of a series'
/// [`series_sha256`], so cached embeddings of DICOM series are recomputed after such a change.
pub const IMPL_VERSION: &str = "1";

/// The dcm2niix version whose output the conversion reproduces.
pub const DCM2NIIX_VERSION: &str = "v1.0.20260724";

/// The files of one series, found from their headers alone.
#[derive(Clone, Debug)]
pub struct Series {
    /// SeriesInstanceUID (0020,000E).
    pub uid: String,
    /// The headers of its files, sorted by path.
    pub headers: Vec<SliceHeader>,
}

impl Series {
    /// The files, sorted by path.
    pub fn files(&self) -> Vec<PathBuf> {
        self.headers.iter().map(|h| h.path.clone()).collect()
    }

    /// The deepest folder holding every file of the series.
    pub fn folder(&self) -> PathBuf {
        convert::common_folder(&self.files())
    }

    /// `series 7 "T1_MPRAGE"`.
    pub fn label(&self) -> String {
        series::describe(&self.headers[0])
    }

    /// SeriesNumber (0020,0011) of the first file.
    pub fn number(&self) -> Option<i32> {
        self.headers[0].series_number
    }

    /// SeriesDescription (0008,103E), else ProtocolName (0018,1030), of the first file.
    pub fn description(&self) -> &str {
        let h = &self.headers[0];
        if h.series_description.is_empty() { &h.protocol_name } else { &h.series_description }
    }

    /// Checks from the headers alone that the series is one 3-D volume Larmor converts; the error
    /// says why not (a localizer, a 4-D series...).
    pub fn check(&self) -> Result<(), DicomError> {
        series::plan(&self.headers).map(|_| ())
    }

    /// Reads and converts the series; see [`convert_files`].
    pub fn convert(&self) -> Result<Converted, DicomError> {
        convert_paths(&self.files())
    }
}

/// Headers grouped by SeriesInstanceUID, each series' files sorted by path, the series ordered by
/// their first file's path.
pub fn group_series(headers: Vec<SliceHeader>) -> Vec<Series> {
    let mut by_uid: std::collections::BTreeMap<String, Vec<SliceHeader>> = Default::default();
    for header in headers {
        by_uid.entry(header.series_instance_uid.clone()).or_default().push(header);
    }
    let mut series: Vec<Series> = by_uid
        .into_iter()
        .map(|(uid, mut headers)| {
            headers.sort_by(|a, b| a.path.cmp(&b.path));
            Series { uid, headers }
        })
        .collect();
    series.sort_by(|a, b| a.headers[0].path.cmp(&b.headers[0].path));
    series
}

/// The identity of a series' content: the SHA-256 of [`IMPL_VERSION`] and its files' SHA-256s,
/// sorted (so neither file names nor their order matter). It stands in for a NIfTI file's SHA-256
/// in cache keys.
pub fn series_sha256<'a>(files: impl IntoIterator<Item = &'a [u8]>) -> String {
    let mut digests: Vec<String> = files.into_iter().map(|bytes| hex(&Sha256::digest(bytes))).collect();
    digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(format!("larmor-dicom series v{IMPL_VERSION}\n"));
    for digest in &digests {
        hasher.update(digest.as_bytes());
        hasher.update(b"\n");
    }
    hex(&hasher.finalize())
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}
