//! A scan: a NIfTI file, or a DICOM series that Larmor converts to the image dcm2niix would write.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use larmor_dicom::{Series, SeriesInfo};
use larmor_volume::{Nifti, NiftiHeader};
use rayon::prelude::*;

use crate::hashing::{sha256_file, sha256_hex};
use crate::locate;

/// One scan to encode.
#[derive(Clone, Debug)]
pub enum Scan {
    /// A `.nii` or `.nii.gz` file.
    Nifti(PathBuf),
    /// A DICOM series (classic, one slice per file).
    Dicom(Arc<DicomScan>),
}

/// A DICOM series found by its headers.
#[derive(Debug)]
pub struct DicomScan {
    pub series: Series,
    /// The deepest folder holding its files: what outputs mirror and `run.json` names.
    pub folder: PathBuf,
    /// The output file stem: the folder's name, with `_series-<n>` when other series share it.
    pub stem: String,
    /// Other series share its folder.
    pub shares_folder: bool,
}

impl Scan {
    /// A DICOM series. `in_folder` names it among other series in the same folder (its
    /// SeriesNumber, say); `None` when it has the folder to itself.
    pub fn dicom(series: Series, in_folder: Option<String>) -> Scan {
        let folder = series.folder();
        let name =
            folder.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "dicom".into());
        let shares_folder = in_folder.is_some();
        let stem = match in_folder {
            Some(id) => format!("{name}_series-{id}"),
            None => name,
        };
        Scan::Dicom(Arc::new(DicomScan { series, folder, stem, shares_folder }))
    }

    /// The NIfTI file, or the series' folder.
    pub fn path(&self) -> &Path {
        match self {
            Scan::Nifti(path) => path,
            Scan::Dicom(dicom) => &dicom.folder,
        }
    }

    /// Where the scan sits for output paths, whose folder its outputs mirror: the NIfTI file; a DICOM
    /// series' folder (as if it were a file in its parent); for series that share a folder, a file
    /// in it (so their outputs go in a mirror of that folder, like NIfTI files in it).
    pub fn location(&self) -> PathBuf {
        match self {
            Scan::Nifti(path) => path.clone(),
            Scan::Dicom(dicom) if dicom.shares_folder => dicom.folder.join(&dicom.stem),
            Scan::Dicom(dicom) => dicom.folder.clone(),
        }
    }

    /// The stem of its output files: `sub-01_T1w` for `sub-01_T1w.nii.gz`, a series' folder name.
    pub fn stem(&self) -> String {
        match self {
            Scan::Nifti(path) => locate::scan_stem(path),
            Scan::Dicom(dicom) => dicom.stem.clone(),
        }
    }

    /// For messages: the file, or the folder with the series.
    pub fn display(&self) -> String {
        match self {
            Scan::Nifti(path) => path.display().to_string(),
            Scan::Dicom(dicom) => format!("{} ({})", dicom.folder.display(), dicom.series.label()),
        }
    }

    /// The file contents, read into memory.
    pub fn read_content(&self) -> Result<Content> {
        match self {
            Scan::Nifti(path) => Ok(Content::Nifti(
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
            )),
            Scan::Dicom(dicom) => {
                let files = dicom
                    .series
                    .files()
                    .into_par_iter()
                    .map(|path| {
                        let bytes =
                            std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                        Ok((path, bytes))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Content::Dicom(files))
            }
        }
    }

    /// Reads and hashes the scan.
    pub fn read(&self) -> Result<ReadScan> {
        Ok(ReadScan::new(self.clone(), self.read_content()?))
    }

    /// The scan's SHA-256 (see [`ReadScan::sha256`]), reading the files without keeping them.
    pub fn sha256(&self) -> Result<String> {
        match self {
            Scan::Nifti(path) => sha256_file(path).with_context(|| format!("reading {}", path.display())),
            Scan::Dicom(_) => Ok(self.read()?.sha256),
        }
    }

    /// The one scan at `path`: a NIfTI file, or a folder holding exactly one scan (a DICOM series'
    /// folder, say). Used for `--reference` and the `scan` and `reference` columns of `labels.tsv`.
    pub fn resolve(path: &Path) -> Result<Scan> {
        if !path.exists() {
            bail!("{} does not exist", path.display());
        }
        let found = locate::find_scans(std::slice::from_ref(&path.to_path_buf()), &[])?;
        match found.scans.as_slice() {
            [scan] => Ok(scan.clone()),
            [] => {
                bail!("{} holds no scan Larmor reads (a .nii/.nii.gz file or a DICOM series)", path.display())
            }
            scans => {
                let names: Vec<String> = scans.iter().take(5).map(Scan::display).collect();
                bail!(
                    "{} holds {} scans ({}{}); name one of them (a DICOM series' own folder, or the file)",
                    path.display(),
                    scans.len(),
                    names.join(", "),
                    if scans.len() > 5 { ", …" } else { "" }
                )
            }
        }
    }
}

/// A scan's file contents.
pub enum Content {
    Nifti(Vec<u8>),
    /// `(path, bytes)` of every file of the series.
    Dicom(Vec<(PathBuf, Vec<u8>)>),
}

/// A scan read into memory and hashed.
pub struct ReadScan {
    pub scan: Scan,
    /// SHA-256 of a NIfTI file's bytes, or of a DICOM series (`larmor_dicom::series_sha256`: its
    /// files' SHA-256s, sorted, with the DICOM conversion's version).
    pub sha256: String,
    content: Content,
}

/// A scan's image, and for a DICOM series what it was converted from.
pub struct Parsed {
    pub nifti: Nifti,
    pub dicom: Option<SeriesInfo>,
    /// What the DICOM conversion noted (duplicate files ignored, instance order not spatial).
    pub notes: Vec<String>,
}

impl ReadScan {
    /// Hashes `content`.
    pub fn new(scan: Scan, content: Content) -> ReadScan {
        let sha256 = match &content {
            Content::Nifti(bytes) => sha256_hex(bytes),
            Content::Dicom(files) => larmor_dicom::series_sha256(files.iter().map(|(_, b)| b.as_slice())),
        };
        ReadScan { scan, sha256, content }
    }

    /// The image. A NIfTI header is checked to be 3-D before its voxels are decoded (so a 4-D
    /// series is refused cheaply); a DICOM series is converted as dcm2niix would.
    pub fn parse(&self) -> Result<Parsed> {
        match &self.content {
            Content::Nifti(bytes) => {
                let path = self.scan.path();
                let header = NiftiHeader::from_file_bytes(bytes)
                    .with_context(|| format!("parsing {}", path.display()))?;
                let shape = header.shape();
                if shape.len() != 3 {
                    bail!(
                        "{}: expected a 3-D image, got shape {shape:?} (a 4-D series such as fMRI or DWI is not an \
                         anatomical scan; pass a 3-D volume)",
                        path.display()
                    );
                }
                let nifti =
                    Nifti::from_bytes(bytes).with_context(|| format!("parsing {}", path.display()))?;
                Ok(Parsed { nifti, dicom: None, notes: Vec::new() })
            }
            Content::Dicom(files) => {
                let converted = larmor_dicom::convert_files(files).with_context(|| {
                    format!("converting the DICOM series in {}", self.scan.path().display())
                })?;
                Ok(Parsed { nifti: converted.nifti, dicom: Some(converted.series), notes: converted.notes })
            }
        }
    }
}
