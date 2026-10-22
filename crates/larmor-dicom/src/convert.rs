//! A series' files, read into memory, to the NIfTI image dcm2niix writes for them.

use std::path::{Path, PathBuf};

use dicom_object::OpenFileOptions;
use larmor_volume::ndarray::{Array3, ShapeBuilder};
use larmor_volume::{Affine, Nifti, VoxelData};
use rayon::prelude::*;

use crate::error::DicomError;
use crate::header::{SliceHeader, is_dicom_bytes};
use crate::pixels::{Samples, sample_bytes, samples};
use crate::series::{Plan, Reorient, Scaling, Storage, describe, plan};

/// A converted series: the image and where it came from.
#[derive(Clone, Debug)]
pub struct Converted {
    /// The image as nibabel loads dcm2niix's NIfTI: same voxels, dtype, sform, zooms and scaling.
    pub nifti: Nifti,
    /// The series.
    pub series: SeriesInfo,
    /// Things worth telling the user, such as duplicate files that were ignored.
    pub notes: Vec<String>,
}

/// What identifies a series, for logs and `run.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct SeriesInfo {
    /// SeriesInstanceUID (0020,000E).
    pub series_instance_uid: String,
    /// SeriesNumber (0020,0011).
    pub series_number: Option<i32>,
    /// SeriesDescription (0008,103E).
    pub description: String,
    /// ProtocolName (0018,1030).
    pub protocol_name: String,
    /// Modality (0008,0060).
    pub modality: String,
    /// Manufacturer (0008,0070).
    pub manufacturer: String,
    /// Slices (files) in the volume.
    pub slices: usize,
    /// The transfer syntaxes of its files, sorted, without repeats.
    pub transfer_syntaxes: Vec<String>,
}

impl SeriesInfo {
    pub(crate) fn of(headers: &[SliceHeader], slices: usize) -> SeriesInfo {
        let first = &headers[0];
        let mut transfer_syntaxes: Vec<String> = headers.iter().map(|h| h.transfer_syntax.clone()).collect();
        transfer_syntaxes.sort();
        transfer_syntaxes.dedup();
        SeriesInfo {
            series_instance_uid: first.series_instance_uid.clone(),
            series_number: first.series_number,
            description: first.series_description.clone(),
            protocol_name: first.protocol_name.clone(),
            modality: first.modality.clone(),
            manufacturer: first.manufacturer_name.clone(),
            slices,
            transfer_syntaxes,
        }
    }
}

/// Converts the files of one series, given as `(path, bytes)`, to the NIfTI image dcm2niix
/// v1.0.20260724 writes for them (with its default options) as nibabel loads it.
///
/// Every file must belong to one series (one SeriesInstanceUID) and the series must be one 3-D
/// volume of classic DICOM; otherwise the error says why it is not (a localizer, a 4-D series,
/// varying slice spacing, an unsupported transfer syntax...).
pub fn convert_files(files: &[(PathBuf, Vec<u8>)]) -> Result<Converted, DicomError> {
    if files.is_empty() {
        return Err(DicomError::Unsupported("no DICOM files".into()));
    }
    let objects: Vec<_> = files
        .par_iter()
        .map(|(path, bytes)| {
            if !is_dicom_bytes(bytes) {
                return Err(DicomError::read(path, "not a DICOM file (no DICM marker)"));
            }
            let object =
                OpenFileOptions::new().from_reader(&bytes[..]).map_err(|e| DicomError::read(path, e))?;
            let header = SliceHeader::from_object(path, &object)?;
            Ok((object, header))
        })
        .collect::<Result<_, _>>()?;
    let headers: Vec<SliceHeader> = objects.iter().map(|(_, h)| h.clone()).collect();
    let uid = &headers[0].series_instance_uid;
    if let Some(other) = headers.iter().find(|h| h.series_instance_uid != *uid) {
        return Err(DicomError::Unsupported(format!(
            "{} and {} belong to different series ({} and {})",
            headers[0].path.display(),
            other.path.display(),
            describe(&headers[0]),
            describe(other)
        )));
    }
    let plan = plan(&headers)?;
    // per-slice scaling makes float voxels before dcm2niix's 12-bit masking, which then skips them
    let storage = match (plan.scaling, plan.storage) {
        (Scaling::PerSlice, Storage::I16 { .. }) => Storage::I16 { mask12: None },
        (_, storage) => storage,
    };
    let slices: Vec<Samples> = plan
        .order
        .par_iter()
        .map(|&i| sample_bytes(&objects[i].0, &headers[i]).map(|bytes| samples(&bytes, storage)))
        .collect::<Result<_, _>>()?;
    let data = match plan.scaling {
        Scaling::PerSlice => {
            let scaled: Vec<Vec<f32>> = slices
                .iter()
                .zip(&plan.slice_scaling)
                .map(|(s, &(slope, intercept))| s.scaled(slope, intercept))
                .collect();
            VoxelData::F32(arrange(&scaled, &plan).into_dyn())
        }
        Scaling::Header { .. } => stack(slices, &plan),
    };
    let nifti = nifti(data, &plan);
    Ok(Converted { nifti, series: SeriesInfo::of(&headers, plan.order.len()), notes: plan.notes })
}

/// The stored samples of the first frame of one DICOM file, decompressed, as little-endian bytes
/// (for checking decoders against each other).
pub fn decode_pixels(path: &Path, bytes: &[u8]) -> Result<Vec<u8>, DicomError> {
    let object = OpenFileOptions::new().from_reader(bytes).map_err(|e| DicomError::read(path, e))?;
    let header = SliceHeader::from_object(path, &object)?;
    if let Some(problem) = crate::pixels::transfer_syntax_problem(&header.transfer_syntax) {
        return Err(DicomError::Unsupported(format!("{}: {problem}", path.display())));
    }
    sample_bytes(&object, &header)
}

/// Reads the files and converts them; see [`convert_files`].
pub fn convert_paths(paths: &[PathBuf]) -> Result<Converted, DicomError> {
    let files = paths
        .iter()
        .map(|path| Ok((path.clone(), std::fs::read(path).map_err(|e| DicomError::read(path, e))?)))
        .collect::<Result<Vec<_>, DicomError>>()?;
    convert_files(&files)
}

/// The slices as one array of their storage type.
fn stack(slices: Vec<Samples>, plan: &Plan) -> VoxelData {
    // every slice of a series has the storage type of its plan
    macro_rules! stack_as {
        ($variant:ident) => {{
            let typed: Vec<_> = slices
                .into_iter()
                .map(|s| match s {
                    Samples::$variant(v) => v,
                    _ => unreachable!("one storage type per series"),
                })
                .collect();
            VoxelData::$variant(arrange(&typed, plan).into_dyn())
        }};
    }
    match plan.storage {
        Storage::U8 => stack_as!(U8),
        Storage::I16 { .. } => stack_as!(I16),
        Storage::U16 => stack_as!(U16),
        Storage::I32 => stack_as!(I32),
    }
}

/// The voxel array: slices stacked in plan order, reversed when `flip_z`, then rows reversed or
/// axes reoriented, as the plan says.
fn arrange<T: Copy>(slices: &[Vec<T>], plan: &Plan) -> Array3<T> {
    let (nx, ny, nz) = (plan.columns, plan.rows, slices.len());
    let source_shape = [nx, ny, nz];
    let source = |out: [usize; 3]| -> [usize; 3] {
        match plan.reorient {
            Reorient::FlipY => [out[0], ny - 1 - out[1], out[2]],
            Reorient::None => out,
            Reorient::Ortho(axes) => {
                let mut src = [0; 3];
                for (i, &axis) in axes.iter().enumerate() {
                    let r = axis.unsigned_abs() as usize - 1;
                    src[r] = if axis < 0 { source_shape[r] - 1 - out[i] } else { out[i] };
                }
                src
            }
        }
    };
    let [s0, s1, s2] = plan.shape;
    Array3::from_shape_fn((s0, s1, s2).f(), |(i, j, k)| {
        let [x, y, z] = source([i, j, k]);
        let z = if plan.flip_z { nz - 1 - z } else { z };
        slices[z][y * nx + x]
    })
}

/// The NIfTI-1 image dcm2niix writes: sform and qform with code 1 (scanner), its pixdim, its
/// scale factors, units mm and s.
fn nifti(data: VoxelData, plan: &Plan) -> Nifti {
    let s = plan.sform;
    let affine = Affine(std::array::from_fn(|i| std::array::from_fn(|j| f64::from(s[i][j]))));
    let mut nifti = Nifti::new(data, &affine);
    let header = &mut nifti.header;
    header.set_qform(&affine, 1);
    header.set_sform(&affine, 1);
    for (axis, &size) in plan.pixdim.iter().enumerate() {
        header.pixdim[axis + 1] = f64::from(size);
    }
    let (slope, intercept) = match plan.scaling {
        Scaling::Header { slope, intercept } => (slope, intercept),
        Scaling::PerSlice => (1.0, 0.0),
    };
    header.scl_slope = f64::from(slope);
    header.scl_inter = f64::from(intercept);
    // NIFTI_UNITS_MM + NIFTI_UNITS_SEC
    header.xyzt_units = 10;
    nifti
}

/// The deepest folder holding every one of `paths`.
pub(crate) fn common_folder(paths: &[PathBuf]) -> PathBuf {
    let mut folder: Option<PathBuf> = None;
    for path in paths {
        let parent = path.parent().unwrap_or(Path::new("")).to_path_buf();
        folder = Some(match folder {
            None => parent,
            Some(current) => {
                let common: PathBuf = current
                    .components()
                    .zip(parent.components())
                    .take_while(|(a, b)| a == b)
                    .map(|(a, _)| a)
                    .collect();
                common
            }
        });
    }
    folder.unwrap_or_default()
}
