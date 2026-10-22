//! NIfTI-1 and NIfTI-2 single-file images (`.nii`, `.nii.gz`), read and written with nibabel's
//! semantics.
//!
//! The affine, zooms and data scaling follow nibabel 5.4 exactly, including the fixes nibabel
//! applies silently on load (`Nifti1Header.check_fix`): `qfac` other than ±1 becomes 1, zero
//! `pixdim` entries become 1, negative ones their absolute value, invalid xform codes 0.
//!
//! A NIfTI-2 header (540 bytes: 64-bit dims and offsets, float64 `pixdim`, quaternion and sform) is
//! read into the same [`NiftiHeader`], whose fields are wide enough for both versions; a NIfTI-1
//! header's float32 values widen exactly. Two-file images (`.hdr`/`.img`), Analyze 7.5 headers and
//! complex or RGB data are refused with an error that says how to convert them.
//!
//! Files are read header first: only `vox_offset` plus the bytes the header's shape needs are
//! decompressed, so trailing bytes (gzip padding, a second gzip member) are never read, and a header
//! whose sizes overflow is an error rather than an allocation.

use std::borrow::Cow;
use std::io::{BufRead, Read, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use ndarray::{Array3, ArrayD, Axis, Ix3, IxDyn, ShapeBuilder, Zip};
use rayon::prelude::*;
use thiserror::Error;

use crate::affine::Affine;
use crate::orientation::Orientation;
use crate::resample::{Interpolation, Resampling, Voxel, resample_from_to};

/// Size of a NIfTI-1 header in bytes.
const HEADER_SIZE: usize = 348;
/// Offset of the voxel data in the NIfTI-1 files this crate writes (header plus the 4-byte
/// extension flag).
const DATA_OFFSET: usize = 352;
/// Size of a NIfTI-2 header in bytes.
const HEADER2_SIZE: usize = 540;
/// Offset of the voxel data in single-file NIfTI-2 (`Nifti2Header.single_vox_offset`).
const DATA2_OFFSET: usize = 544;
/// NIfTI-2's `eol_check` bytes, which catch files mangled by text-mode line-ending conversion.
const EOL_CHECK: [u8; 4] = [13, 10, 26, 10];
/// Bytes reserved up front for a gzip-compressed image's voxels; the buffer grows past this only as
/// data actually arrives, so a header claiming terabytes cannot trigger a huge allocation.
const MAX_UPFRONT_RESERVE: usize = 1 << 30;

/// Errors from reading or writing NIfTI files.
#[derive(Debug, Error)]
pub enum NiftiError {
    /// The file could not be read or written.
    #[error("{path}: {source}")]
    Io {
        /// The file involved.
        path: String,
        /// The underlying error.
        source: std::io::Error,
    },
    /// The bytes are not a NIfTI image this crate can read.
    #[error("not a readable NIfTI image: {0}")]
    Format(String),
    /// The voxel datatype is valid NIfTI but not supported here (complex, RGB, float128).
    #[error("{}", unsupported_datatype_message(*.0))]
    UnsupportedDataType(i16),
    /// An operation that needs a 3-D image got another shape.
    #[error("expected a 3-D image, got shape {0:?}")]
    NotThreeD(Vec<usize>),
    /// The image affine cannot be inverted.
    #[error("the image affine is singular")]
    SingularAffine,
}

/// Byte order of a header and its data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    /// Least significant byte first (what this crate writes).
    Little,
    /// Most significant byte first.
    Big,
}

/// The voxel storage types this crate reads and writes, named like their numpy dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataType {
    /// `uint8`, NIfTI code 2.
    U8,
    /// `int8`, NIfTI code 256.
    I8,
    /// `int16`, NIfTI code 4.
    I16,
    /// `uint16`, NIfTI code 512.
    U16,
    /// `int32`, NIfTI code 8.
    I32,
    /// `uint32`, NIfTI code 768.
    U32,
    /// `int64`, NIfTI code 1024.
    I64,
    /// `uint64`, NIfTI code 1280.
    U64,
    /// `float32`, NIfTI code 16.
    F32,
    /// `float64`, NIfTI code 64.
    F64,
}

impl DataType {
    const ALL: [DataType; 10] = [
        DataType::U8,
        DataType::I8,
        DataType::I16,
        DataType::U16,
        DataType::I32,
        DataType::U32,
        DataType::I64,
        DataType::U64,
        DataType::F32,
        DataType::F64,
    ];

    /// The NIfTI `datatype` code.
    pub fn code(self) -> i16 {
        match self {
            DataType::U8 => 2,
            DataType::I8 => 256,
            DataType::I16 => 4,
            DataType::U16 => 512,
            DataType::I32 => 8,
            DataType::U32 => 768,
            DataType::I64 => 1024,
            DataType::U64 => 1280,
            DataType::F32 => 16,
            DataType::F64 => 64,
        }
    }

    /// The datatype for a NIfTI code, if supported.
    pub fn from_code(code: i16) -> Option<DataType> {
        DataType::ALL.into_iter().find(|t| t.code() == code)
    }

    /// Bytes per voxel.
    pub fn size(self) -> usize {
        match self {
            DataType::U8 | DataType::I8 => 1,
            DataType::I16 | DataType::U16 => 2,
            DataType::I32 | DataType::U32 | DataType::F32 => 4,
            DataType::I64 | DataType::U64 | DataType::F64 => 8,
        }
    }

    /// The numpy dtype name, such as `"int16"`.
    pub fn name(self) -> &'static str {
        match self {
            DataType::U8 => "uint8",
            DataType::I8 => "int8",
            DataType::I16 => "int16",
            DataType::U16 => "uint16",
            DataType::I32 => "int32",
            DataType::U32 => "uint32",
            DataType::I64 => "int64",
            DataType::U64 => "uint64",
            DataType::F32 => "float32",
            DataType::F64 => "float64",
        }
    }
}

/// A voxel array in one of the supported dtypes, indexed `[x, y, z, ...]` like a nibabel array.
///
/// Arrays read from disk keep NIfTI's Fortran (x-fastest) memory order, as nibabel's do; ndarray
/// indexing is the same either way.
#[derive(Clone, Debug, PartialEq)]
pub enum VoxelData {
    /// `uint8` voxels.
    U8(ArrayD<u8>),
    /// `int8` voxels.
    I8(ArrayD<i8>),
    /// `int16` voxels.
    I16(ArrayD<i16>),
    /// `uint16` voxels.
    U16(ArrayD<u16>),
    /// `int32` voxels.
    I32(ArrayD<i32>),
    /// `uint32` voxels.
    U32(ArrayD<u32>),
    /// `int64` voxels.
    I64(ArrayD<i64>),
    /// `uint64` voxels.
    U64(ArrayD<u64>),
    /// `float32` voxels.
    F32(ArrayD<f32>),
    /// `float64` voxels.
    F64(ArrayD<f64>),
}

/// Runs `$body` with `$arr` bound to the typed array inside any `VoxelData` variant.
macro_rules! with_array {
    ($data:expr, $arr:ident => $body:expr) => {
        match $data {
            VoxelData::U8($arr) => $body,
            VoxelData::I8($arr) => $body,
            VoxelData::I16($arr) => $body,
            VoxelData::U16($arr) => $body,
            VoxelData::I32($arr) => $body,
            VoxelData::U32($arr) => $body,
            VoxelData::I64($arr) => $body,
            VoxelData::U64($arr) => $body,
            VoxelData::F32($arr) => $body,
            VoxelData::F64($arr) => $body,
        }
    };
}

/// Like [`with_array!`], but wraps the (same-typed) result back into the same variant.
macro_rules! map_array {
    ($data:expr, $arr:ident => $body:expr) => {
        match $data {
            VoxelData::U8($arr) => VoxelData::U8($body),
            VoxelData::I8($arr) => VoxelData::I8($body),
            VoxelData::I16($arr) => VoxelData::I16($body),
            VoxelData::U16($arr) => VoxelData::U16($body),
            VoxelData::I32($arr) => VoxelData::I32($body),
            VoxelData::U32($arr) => VoxelData::U32($body),
            VoxelData::I64($arr) => VoxelData::I64($body),
            VoxelData::U64($arr) => VoxelData::U64($body),
            VoxelData::F32($arr) => VoxelData::F32($body),
            VoxelData::F64($arr) => VoxelData::F64($body),
        }
    };
}

impl VoxelData {
    /// The storage dtype.
    pub fn dtype(&self) -> DataType {
        match self {
            VoxelData::U8(_) => DataType::U8,
            VoxelData::I8(_) => DataType::I8,
            VoxelData::I16(_) => DataType::I16,
            VoxelData::U16(_) => DataType::U16,
            VoxelData::I32(_) => DataType::I32,
            VoxelData::U32(_) => DataType::U32,
            VoxelData::I64(_) => DataType::I64,
            VoxelData::U64(_) => DataType::U64,
            VoxelData::F32(_) => DataType::F32,
            VoxelData::F64(_) => DataType::F64,
        }
    }

    /// The array shape.
    pub fn shape(&self) -> &[usize] {
        with_array!(self, a => a.shape())
    }

    /// The values cast to float32, like `np.asarray(data, dtype=np.float32)`.
    pub fn to_f32(&self) -> ArrayD<f32> {
        with_array!(self, a => a.mapv(|v| v.to_f64() as f32))
    }

    /// The values cast to float64.
    pub fn to_f64(&self) -> ArrayD<f64> {
        with_array!(self, a => a.mapv(|v| v.to_f64()))
    }

    /// Applies NIfTI scaling, `value * slope + inter`, in float64.
    ///
    /// Mirrors `nibabel.volumeutils.apply_read_scaling` for nibabel's float64 scale factors: the
    /// multiply is skipped for a slope of 1 and the add for an intercept of 0.
    pub fn scaled(&self, slope: f64, inter: f64) -> ArrayD<f64> {
        with_array!(self, a => a.mapv(|v| scale(v.to_f64(), slope, inter)))
    }

    /// A 3-D array resampled onto the grid `to_shape` / `to_affine`, keeping its dtype.
    ///
    /// Mirrors `nibabel.processing.resample_from_to(img, (to_shape, to_affine), order)` on this
    /// data as `img.dataobj`. Returns `None` if the data are not 3-D or `from_affine` is singular.
    pub fn resample_from_to(
        &self,
        from_affine: &Affine,
        to_shape: [usize; 3],
        to_affine: &Affine,
        order: Interpolation,
    ) -> Option<VoxelData> {
        Some(map_array!(self, a => {
            let view = a.view().into_dimensionality::<Ix3>().ok()?;
            resample_from_to(view, from_affine, to_shape, to_affine, order)?.into_dyn()
        }))
    }

    /// Decodes `bytes`, which must hold exactly `shape`'s voxels of `dtype` in Fortran order.
    fn from_bytes(
        dtype: DataType,
        bytes: &[u8],
        shape: &[usize],
        endian: Endian,
    ) -> Result<VoxelData, NiftiError> {
        let shape_error = |e: ndarray::ShapeError| {
            NiftiError::Format(format!(
                "{} bytes do not hold a {shape:?} {} array: {e}",
                bytes.len(),
                dtype.name()
            ))
        };
        macro_rules! decode {
            ($variant:ident, $t:ty) => {{
                const N: usize = std::mem::size_of::<$t>();
                let values: Vec<$t> = bytes
                    .as_chunks::<N>()
                    .0
                    .iter()
                    .map(|&raw| match endian {
                        Endian::Little => <$t>::from_le_bytes(raw),
                        Endian::Big => <$t>::from_be_bytes(raw),
                    })
                    .collect();
                VoxelData::$variant(ArrayD::from_shape_vec(IxDyn(shape).f(), values).map_err(shape_error)?)
            }};
        }
        Ok(match dtype {
            DataType::U8 => decode!(U8, u8),
            DataType::I8 => decode!(I8, i8),
            DataType::I16 => decode!(I16, i16),
            DataType::U16 => decode!(U16, u16),
            DataType::I32 => decode!(I32, i32),
            DataType::U32 => decode!(U32, u32),
            DataType::I64 => decode!(I64, i64),
            DataType::U64 => decode!(U64, u64),
            DataType::F32 => decode!(F32, f32),
            DataType::F64 => decode!(F64, f64),
        })
    }

    /// Little-endian bytes in NIfTI (Fortran) order.
    fn to_le_bytes(&self) -> Vec<u8> {
        let size = self.dtype().size();
        with_array!(self, a => {
            let mut out = Vec::with_capacity(a.len() * size);
            // reversed axes iterate the original array x-fastest
            for v in a.t().iter() {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        })
    }
}

/// `value * slope + inter` with nibabel's shortcuts for a unit slope and zero intercept.
fn scale(value: f64, slope: f64, inter: f64) -> f64 {
    let mut x = value;
    if slope != 1.0 {
        x *= slope;
    }
    if inter != 0.0 {
        x += inter;
    }
    x
}

/// [`Nifti::resample_to_f32`] for one stored dtype.
fn resample_typed_f32<T: Voxel>(
    data: &ArrayD<T>,
    grid: &Resampling,
    scaling: Option<(f64, f64)>,
) -> Option<Array3<f32>> {
    let view = data.view().into_dimensionality::<Ix3>().ok()?;
    Some(match scaling {
        // a scaled dataobj is float64, so only the final float32 cast rounds
        Some((slope, inter)) => grid.run(view, |v| scale(v.to_f64(), slope, inter), |t| t as f32),
        None => grid.run(view, T::to_f64, |t| T::from_interpolated(t).to_f64() as f32),
    })
}

/// [`Nifti::fdata_f32_oriented`] for one stored dtype.
fn oriented_typed_f32<T: Voxel>(
    data: &ArrayD<T>,
    ornt: &Orientation,
    scaling: Option<(f64, f64)>,
) -> Option<Array3<f32>> {
    let view = ornt.apply(data.view().into_dimensionality::<Ix3>().ok()?);
    let convert = |v: T| match scaling {
        Some((slope, inter)) => scale(v.to_f64(), slope, inter) as f32,
        None => v.to_f64() as f32,
    };
    let mut out = Array3::zeros(view.raw_dim());
    out.axis_iter_mut(Axis(0)).into_par_iter().zip(view.axis_iter(Axis(0)).into_par_iter()).for_each(
        |(mut dst, src)| {
            Zip::from(&mut dst).and(&src).for_each(|d, &s| *d = convert(s));
        },
    );
    Some(out)
}

/// Which NIfTI format a header came from (or is written as).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NiftiVersion {
    /// NIfTI-1: a 348-byte header with 16-bit dims and float32 geometry (`n+1`).
    #[default]
    One,
    /// NIfTI-2: a 540-byte header with 64-bit dims and float64 geometry (`n+2`).
    Two,
}

/// A NIfTI header, field for field (nibabel's `Nifti1Header` / `Nifti2Header` structured array).
///
/// The fields have NIfTI-2's widths, so both versions fit: a NIfTI-1 header's int16 and float32
/// values widen exactly, and the setters round to float32 again for a NIfTI-1 header, so its
/// values stay what the file would store.
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq)]
pub struct NiftiHeader {
    pub version: NiftiVersion,
    pub dim_info: u8,
    pub dim: [i64; 8],
    pub intent_p1: f64,
    pub intent_p2: f64,
    pub intent_p3: f64,
    pub intent_code: i32,
    pub datatype: i16,
    pub bitpix: i16,
    pub slice_start: i64,
    pub pixdim: [f64; 8],
    /// Byte offset of the voxel data (float32 in NIfTI-1, int64 in NIfTI-2).
    pub vox_offset: f64,
    pub scl_slope: f64,
    pub scl_inter: f64,
    pub slice_end: i64,
    pub slice_code: i32,
    pub xyzt_units: i32,
    pub cal_max: f64,
    pub cal_min: f64,
    pub slice_duration: f64,
    pub toffset: f64,
    pub descrip: [u8; 80],
    pub aux_file: [u8; 24],
    pub qform_code: i32,
    pub sform_code: i32,
    pub quatern_b: f64,
    pub quatern_c: f64,
    pub quatern_d: f64,
    pub qoffset_x: f64,
    pub qoffset_y: f64,
    pub qoffset_z: f64,
    pub srow_x: [f64; 4],
    pub srow_y: [f64; 4],
    pub srow_z: [f64; 4],
    pub intent_name: [u8; 16],
    /// `n+1\0` or `n+2\0` (`ni1`/`ni2` for two-file images, which are refused).
    pub magic: [u8; 4],
}

/// Reads fixed-offset fields from a header block in a given byte order.
struct FieldReader<'a> {
    bytes: &'a [u8],
    endian: Endian,
}

macro_rules! field_readers {
    ($($name:ident: $t:ty),*) => {
        $(
            fn $name(&self, offset: usize) -> $t {
                let raw = self.array(offset);
                match self.endian {
                    Endian::Little => <$t>::from_le_bytes(raw),
                    Endian::Big => <$t>::from_be_bytes(raw),
                }
            }
        )*
    };
}

impl FieldReader<'_> {
    fn array<const N: usize>(&self, offset: usize) -> [u8; N] {
        self.bytes[offset..offset + N].try_into().unwrap()
    }
    field_readers!(i16: i16, i32: i32, i64: i64, f32: f32, f64: f64);
    /// `N` int16 values widened to int64.
    fn i16s<const N: usize>(&self, offset: usize) -> [i64; N] {
        std::array::from_fn(|i| i64::from(self.i16(offset + 2 * i)))
    }
    /// `N` float32 values widened (exactly) to float64.
    fn f32s<const N: usize>(&self, offset: usize) -> [f64; N] {
        std::array::from_fn(|i| f64::from(self.f32(offset + 4 * i)))
    }
    fn i64s<const N: usize>(&self, offset: usize) -> [i64; N] {
        std::array::from_fn(|i| self.i64(offset + 8 * i))
    }
    fn f64s<const N: usize>(&self, offset: usize) -> [f64; N] {
        std::array::from_fn(|i| self.f64(offset + 8 * i))
    }
}

/// The message for a datatype code this crate does not read.
fn unsupported_datatype_message(code: i16) -> String {
    let what = match code {
        32 => Some("complex64"),
        1792 => Some("complex128"),
        2048 => Some("complex256"),
        128 => Some("RGB24"),
        2304 => Some("RGBA32"),
        1536 => Some("float128"),
        _ => None,
    };
    match (code, what) {
        (32 | 1792 | 2048, Some(name)) => format!(
            "complex-valued data (NIfTI datatype {code}, {name}) is not supported: convert it to a \
             real magnitude image first (for example np.abs); nibabel would silently keep only the \
             real part"
        ),
        (_, Some(name)) => format!(
            "NIfTI datatype {code} ({name}) is not supported (supported: integer types, float32, float64)"
        ),
        _ => format!("unsupported NIfTI datatype code {code} (supported: integer types, float32, float64)"),
    }
}

impl NiftiHeader {
    /// Parses a NIfTI-1 (348-byte) or NIfTI-2 (540-byte) header in either byte order, then applies
    /// nibabel's load-time fixes.
    ///
    /// Mirrors `Nifti1Header.from_fileobj(..., check=True)` and `Nifti2Header`'s; problems nibabel
    /// raises on (bad magic, unknown datatype, a data offset inside the header) are errors here
    /// too, as are two-file images, which this crate does not read.
    pub fn parse(bytes: &[u8]) -> Result<(NiftiHeader, Endian), NiftiError> {
        if bytes.len() < HEADER_SIZE {
            return Err(NiftiError::Format(format!("{} bytes is too short for a header", bytes.len())));
        }
        let sizeof_hdr: [u8; 4] = bytes[..4].try_into().unwrap();
        let is = |size: usize| {
            if i32::from_le_bytes(sizeof_hdr) == size as i32 {
                Some(Endian::Little)
            } else if i32::from_be_bytes(sizeof_hdr) == size as i32 {
                Some(Endian::Big)
            } else {
                None
            }
        };
        let mut header = if let Some(endian) = is(HEADER_SIZE) {
            (Self::parse_v1(&FieldReader { bytes, endian }), endian)
        } else if let Some(endian) = is(HEADER2_SIZE) {
            if bytes.len() < HEADER2_SIZE {
                return Err(NiftiError::Format(format!(
                    "{} bytes is too short for a NIfTI-2 header",
                    bytes.len()
                )));
            }
            let reader = FieldReader { bytes, endian };
            (Self::parse_v2(&reader)?, endian)
        } else {
            return Err(NiftiError::Format("sizeof_hdr is neither 348 (NIfTI-1) nor 540 (NIfTI-2)".into()));
        };
        header.0.check_fix()?;
        Ok(header)
    }

    /// The fields of a 348-byte NIfTI-1 header.
    fn parse_v1(r: &FieldReader) -> NiftiHeader {
        let bytes = r.bytes;
        NiftiHeader {
            version: NiftiVersion::One,
            dim_info: bytes[39],
            dim: r.i16s(40),
            intent_p1: r.f32(56).into(),
            intent_p2: r.f32(60).into(),
            intent_p3: r.f32(64).into(),
            intent_code: r.i16(68).into(),
            datatype: r.i16(70),
            bitpix: r.i16(72),
            slice_start: r.i16(74).into(),
            pixdim: r.f32s(76),
            vox_offset: r.f32(108).into(),
            scl_slope: r.f32(112).into(),
            scl_inter: r.f32(116).into(),
            slice_end: r.i16(120).into(),
            slice_code: bytes[122].into(),
            xyzt_units: bytes[123].into(),
            cal_max: r.f32(124).into(),
            cal_min: r.f32(128).into(),
            slice_duration: r.f32(132).into(),
            toffset: r.f32(136).into(),
            descrip: r.array(148),
            aux_file: r.array(228),
            qform_code: r.i16(252).into(),
            sform_code: r.i16(254).into(),
            quatern_b: r.f32(256).into(),
            quatern_c: r.f32(260).into(),
            quatern_d: r.f32(264).into(),
            qoffset_x: r.f32(268).into(),
            qoffset_y: r.f32(272).into(),
            qoffset_z: r.f32(276).into(),
            srow_x: r.f32s(280),
            srow_y: r.f32s(296),
            srow_z: r.f32s(312),
            intent_name: r.array(328),
            magic: r.array(344),
        }
    }

    /// The fields of a 540-byte NIfTI-2 header (layout of nibabel's `nifti2.header_dtype`).
    fn parse_v2(r: &FieldReader) -> Result<NiftiHeader, NiftiError> {
        // Nifti2Header._chk_eol_check: all zero is fixed silently, anything else is an error
        let eol: [u8; 4] = r.array(8);
        if eol != EOL_CHECK && eol != [0; 4] {
            return Err(NiftiError::Format(
                "NIfTI-2 EOL check bytes are wrong; the file may have been corrupted by line-ending conversion"
                    .into(),
            ));
        }
        let vox_offset = r.i64(168);
        Ok(NiftiHeader {
            version: NiftiVersion::Two,
            dim_info: r.bytes[524],
            dim: r.i64s(16),
            intent_p1: r.f64(80),
            intent_p2: r.f64(88),
            intent_p3: r.f64(96),
            intent_code: r.i32(504),
            datatype: r.i16(12),
            bitpix: r.i16(14),
            slice_start: r.i64(224),
            pixdim: r.f64s(104),
            // offsets beyond 2^53 bytes do not exist; the conversion keeps every real one exact
            vox_offset: vox_offset as f64,
            scl_slope: r.f64(176),
            scl_inter: r.f64(184),
            slice_end: r.i64(232),
            slice_code: r.i32(496),
            xyzt_units: r.i32(500),
            cal_max: r.f64(192),
            cal_min: r.f64(200),
            slice_duration: r.f64(208),
            toffset: r.f64(216),
            descrip: r.array(240),
            aux_file: r.array(320),
            qform_code: r.i32(344),
            sform_code: r.i32(348),
            quatern_b: r.f64(352),
            quatern_c: r.f64(360),
            quatern_d: r.f64(368),
            qoffset_x: r.f64(376),
            qoffset_y: r.f64(384),
            qoffset_z: r.f64(392),
            srow_x: r.f64s(400),
            srow_y: r.f64s(432),
            srow_z: r.f64s(464),
            intent_name: r.array(508),
            magic: r.array(4),
        })
    }

    /// nibabel's `Nifti1Header` / `Nifti2Header` check battery with `fix=True` and its default
    /// error level.
    fn check_fix(&mut self) -> Result<(), NiftiError> {
        if DataType::from_code(self.datatype).is_none() {
            return Err(NiftiError::UnsupportedDataType(self.datatype));
        }
        self.bitpix = 8 * DataType::from_code(self.datatype).unwrap().size() as i16;
        for z in &mut self.pixdim[1..4] {
            if *z == 0.0 {
                *z = 1.0;
            }
            *z = z.abs();
        }
        if self.pixdim[0] != 1.0 && self.pixdim[0] != -1.0 {
            self.pixdim[0] = 1.0;
        }
        let (single, pair, min_offset) = match self.version {
            NiftiVersion::One => (b"n+1\0", b"ni1\0", DATA_OFFSET),
            NiftiVersion::Two => (b"n+2\0", b"ni2\0", DATA2_OFFSET),
        };
        if &self.magic == pair {
            return Err(NiftiError::Format(
                "two-file NIfTI (.hdr/.img pair) is not supported; convert it to a single .nii.gz \
                 (for example nib.save(nib.load('scan.hdr'), 'scan.nii.gz'))"
                    .into(),
            ));
        }
        if &self.magic != single {
            return Err(NiftiError::Format(match self.version {
                NiftiVersion::One => "bad magic, not a NIfTI-1 header (an Analyze 7.5 .hdr/.img image? \
                                      convert it to a single .nii.gz)"
                    .into(),
                NiftiVersion::Two => "bad magic, not a NIfTI-2 header".into(),
            }));
        }
        // NaN fails this test as well
        if self.vox_offset.partial_cmp(&(min_offset as f64)).is_none_or(|order| order.is_lt()) {
            return Err(NiftiError::Format(format!("vox_offset {} is inside the header", self.vox_offset)));
        }
        let slope_is_valid = self.scl_slope != 0.0 && self.scl_slope.is_finite();
        if slope_is_valid && !self.scl_inter.is_finite() {
            return Err(NiftiError::Format(format!("valid slope but invalid intercept {}", self.scl_inter)));
        }
        for code in [&mut self.qform_code, &mut self.sform_code] {
            if !(0..=5).contains(code) {
                *code = 0;
            }
        }
        let ndim = self.dim[0];
        if !(1..=7).contains(&ndim) {
            return Err(NiftiError::Format(format!("dim[0] = {ndim} is not between 1 and 7")));
        }
        if let Some(bad) = self.dim[1..=ndim as usize].iter().find(|&&d| d < 1) {
            return Err(NiftiError::Format(format!("dimension of size {bad}")));
        }
        Ok(())
    }

    /// Bytes needed to parse the header at the start of `bytes`: 348 for NIfTI-1, 540 for NIfTI-2.
    fn size_of(bytes: &[u8]) -> usize {
        let sizeof_hdr: Option<[u8; 4]> = bytes.get(..4).and_then(|b| b.try_into().ok());
        match sizeof_hdr {
            Some(b) if i32::from_le_bytes(b) == 540 || i32::from_be_bytes(b) == 540 => HEADER2_SIZE,
            _ => HEADER_SIZE,
        }
    }

    /// Reads just the header of a `.nii` or `.nii.gz` file (gzip detected from the content),
    /// decompressing no more than the header itself.
    pub fn read(path: impl AsRef<Path>) -> Result<NiftiHeader, NiftiError> {
        let path = path.as_ref();
        let io_err = |source| NiftiError::Io { path: path.display().to_string(), source };
        let file = std::fs::File::open(path).map_err(io_err)?;
        let mut reader = std::io::BufReader::new(file);
        let gzip = reader.fill_buf().map_err(io_err)?.starts_with(&[0x1f, 0x8b]);
        let head = if gzip {
            read_header_block(&mut MultiGzDecoder::new(reader))
        } else {
            read_header_block(&mut reader)
        };
        let head = head.map_err(io_err)?;
        Ok(NiftiHeader::parse(&head)?.0)
    }

    /// Parses the header at the start of a `.nii` or `.nii.gz` file's bytes (gzip detected from
    /// the content), decompressing no more than the header.
    pub fn from_file_bytes(bytes: &[u8]) -> Result<NiftiHeader, NiftiError> {
        let head = if bytes.starts_with(&[0x1f, 0x8b]) {
            read_header_block(&mut MultiGzDecoder::new(bytes))
                .map_err(|e| NiftiError::Format(format!("reading the header: {e}")))?
        } else {
            bytes[..bytes.len().min(HEADER2_SIZE)].to_vec()
        };
        Ok(NiftiHeader::parse(&head)?.0)
    }

    /// The array shape (`get_data_shape()`).
    pub fn shape(&self) -> Vec<usize> {
        self.dim[1..=self.dim[0] as usize].iter().map(|&d| d as usize).collect()
    }

    /// The voxel datatype.
    pub fn data_type(&self) -> DataType {
        DataType::from_code(self.datatype).expect("datatype checked on construction")
    }

    /// Where the voxel data are in the file and how many bytes they take: `(offset, length)`.
    ///
    /// Fails, rather than wrapping around, when the shape's byte count does not fit in memory
    /// (nibabel raises `OverflowError` there).
    pub fn data_range(&self) -> Result<(usize, usize), NiftiError> {
        let shape = self.shape();
        let dtype = self.data_type();
        let too_big = || {
            NiftiError::Format(format!(
                "shape {shape:?} of {} is too large to address (the header is corrupt)",
                dtype.name()
            ))
        };
        let len = shape
            .iter()
            .try_fold(dtype.size(), |acc, &n| acc.checked_mul(n))
            .filter(|&len| len <= isize::MAX as usize)
            .ok_or_else(too_big)?;
        // int(vox_offset), as nibabel takes it; validated as at least the header size
        let offset = self.vox_offset as usize;
        offset.checked_add(len).ok_or_else(too_big)?;
        Ok((offset, len))
    }

    /// Voxel sizes (and the time step for 4-D) from `pixdim`, as float32.
    ///
    /// Mirrors `get_zooms()` of a NIfTI-1 header. A NIfTI-2 header's float64 zooms are rounded;
    /// [`NiftiHeader::zooms_f64`] has them exactly.
    pub fn zooms(&self) -> Vec<f32> {
        self.zooms_f64().into_iter().map(|z| z as f32).collect()
    }

    /// Voxel sizes from `pixdim` in the header's own precision (float32 values widened for
    /// NIfTI-1, float64 for NIfTI-2), as `get_zooms()` returns them.
    pub fn zooms_f64(&self) -> Vec<f64> {
        self.pixdim[1..=self.dim[0] as usize].to_vec()
    }

    /// The scale factors `(slope, intercept)`, or `None` when the data are used as stored.
    ///
    /// Mirrors `get_slope_inter()` and the `(1, 0)` shortcut of `apply_read_scaling`: a slope
    /// of 0 or non-finite means no scaling; the factors are the header's values (float32 widened
    /// to float64 for NIfTI-1). (A valid slope with a non-finite intercept is rejected when the
    /// header is parsed.)
    pub fn scaling(&self) -> Option<(f64, f64)> {
        let (slope, inter) = (self.scl_slope, self.scl_inter);
        if slope == 0.0 || !slope.is_finite() {
            return None;
        }
        if slope == 1.0 && inter == 0.0 { None } else { Some((slope, inter)) }
    }

    /// The sform matrix from `srow_x/y/z`.
    ///
    /// Mirrors `get_sform()`.
    pub fn sform(&self) -> Affine {
        let rows = [self.srow_x, self.srow_y, self.srow_z];
        let mut out = Affine::IDENTITY;
        for (row, srow) in out.0.iter_mut().zip(rows) {
            *row = srow;
        }
        out
    }

    /// The qform matrix from the quaternion, `pixdim` and `qoffset`.
    ///
    /// Mirrors `get_qform()`: `w` is recovered by `fillpositive` (set to 0 when `1 - |bcd|²` is
    /// within three epsilons of the header's float type of 0), the rotation is `quat2mat`, and
    /// `qfac` flips the third column. Fails if the quaternion's `b, c, d` have norm above 1.
    pub fn qform(&self) -> Result<Affine, NiftiError> {
        let [x, y, z] = [self.quatern_b, self.quatern_c, self.quatern_d];
        let w2 = 1.0 - ((x * x + y * y) + z * z);
        // Nifti1Header / Nifti2Header.quaternion_threshold
        let threshold = match self.version {
            NiftiVersion::One => (f32::EPSILON * 3.0) as f64,
            NiftiVersion::Two => f64::EPSILON * 3.0,
        };
        let w = if w2.abs() < threshold {
            0.0
        } else if w2 < 0.0 {
            return Err(NiftiError::Format(format!("qform quaternion has |b, c, d| > 1 (w² = {w2:e})")));
        } else {
            w2.sqrt()
        };
        let rotation = quat2mat([w, x, y, z]);
        // multiplying by qfac (±1) is exact in either precision
        let mut zooms = [self.pixdim[1], self.pixdim[2], self.pixdim[3]];
        zooms[2] *= self.pixdim[0];
        let rzs: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| rotation[i][j] * zooms[j]));
        Ok(Affine::from_matvec(rzs, [self.qoffset_x, self.qoffset_y, self.qoffset_z]))
    }

    /// The affine nibabel falls back to with neither form set: centred, x flipped.
    ///
    /// Mirrors `get_base_affine()` (`shape_zoom_affine(shape, zooms, x_flip=True)`).
    pub fn base_affine(&self) -> Affine {
        let ndim = self.dim[0] as usize;
        let mut shape = [1.0f64; 3];
        let mut zooms = [1.0f64; 3];
        for axis in 0..ndim.min(3) {
            shape[axis] = self.dim[axis + 1] as f64;
            zooms[axis] = self.pixdim[axis + 1];
        }
        zooms[0] = -zooms[0];
        let mut out = Affine::IDENTITY;
        for axis in 0..3 {
            let zoom = zooms[axis];
            out.0[axis][axis] = zoom;
            out.0[axis][3] = -((shape[axis] - 1.0) / 2.0) * zoom;
        }
        out
    }

    /// The image affine: the sform if its code is set, else the qform, else the base affine.
    ///
    /// Mirrors `get_best_affine()`, which is what `img.affine` is for a loaded image.
    pub fn best_affine(&self) -> Result<Affine, NiftiError> {
        if self.sform_code != 0 {
            Ok(self.sform())
        } else if self.qform_code != 0 {
            self.qform()
        } else {
            Ok(self.base_affine())
        }
    }

    /// A float value as this header's format stores it: rounded to float32 for NIfTI-1.
    fn stored(&self, value: f64) -> f64 {
        match self.version {
            NiftiVersion::One => value as f32 as f64,
            NiftiVersion::Two => value,
        }
    }

    /// Writes `affine` into the sform with `code`.
    ///
    /// Mirrors `set_sform(affine, code)`.
    pub fn set_sform(&mut self, affine: &Affine, code: i16) {
        let row = |r: usize| affine.0[r].map(|v| self.stored(v));
        (self.srow_x, self.srow_y, self.srow_z) = (row(0), row(1), row(2));
        self.sform_code = code.into();
    }

    /// Writes `affine` into the qform (quaternion, `qfac`, `pixdim`, offsets) with `code`.
    ///
    /// Mirrors `set_qform(affine, code)`: the zooms are the column norms, `qfac` makes the
    /// rotation proper, and shears are dropped by taking the closest rotation.
    pub fn set_qform(&mut self, affine: &Affine, code: i16) {
        let rzs = affine.rzs();
        let zooms = affine.voxel_sizes();
        let mut r: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| rzs[i][j] / zooms[j]));
        let det = Affine::from_matvec(r, [0.0; 3]).determinant3();
        let qfac = if det > 0.0 { 1.0 } else { -1.0 };
        if qfac < 0.0 {
            for row in &mut r {
                row[2] = -row[2];
            }
        }
        let q = mat2quat(closest_rotation(r));
        let t = affine.translation();
        self.qoffset_x = self.stored(t[0]);
        self.qoffset_y = self.stored(t[1]);
        self.qoffset_z = self.stored(t[2]);
        self.pixdim[0] = qfac;
        for axis in 0..3 {
            self.pixdim[axis + 1] = self.stored(zooms[axis]);
        }
        self.quatern_b = self.stored(q[1]);
        self.quatern_c = self.stored(q[2]);
        self.quatern_d = self.stored(q[3]);
        self.qform_code = code.into();
    }

    /// A NIfTI-1 header for `data` with no geometry yet: nibabel's `Nifti1Header()` defaults.
    fn for_data(data: &VoxelData) -> NiftiHeader {
        let mut dim = [1i64; 8];
        dim[0] = data.shape().len() as i64;
        for (d, &n) in dim[1..].iter_mut().zip(data.shape()) {
            *d = n as i64;
        }
        NiftiHeader {
            version: NiftiVersion::One,
            dim_info: 0,
            dim,
            intent_p1: 0.0,
            intent_p2: 0.0,
            intent_p3: 0.0,
            intent_code: 0,
            datatype: data.dtype().code(),
            bitpix: 8 * data.dtype().size() as i16,
            slice_start: 0,
            pixdim: [1.0; 8],
            vox_offset: DATA_OFFSET as f64,
            scl_slope: 1.0,
            scl_inter: 0.0,
            slice_end: 0,
            slice_code: 0,
            xyzt_units: 0,
            cal_max: 0.0,
            cal_min: 0.0,
            slice_duration: 0.0,
            toffset: 0.0,
            descrip: [0; 80],
            aux_file: [0; 24],
            qform_code: 0,
            sform_code: 0,
            quatern_b: 0.0,
            quatern_c: 0.0,
            quatern_d: 0.0,
            qoffset_x: 0.0,
            qoffset_y: 0.0,
            qoffset_z: 0.0,
            srow_x: [0.0; 4],
            srow_y: [0.0; 4],
            srow_z: [0.0; 4],
            intent_name: [0; 16],
            magic: *b"n+1\0",
        }
    }

    /// Offset of the voxel data in the files this crate writes for this header's version.
    fn data_offset(&self) -> usize {
        match self.version {
            NiftiVersion::One => DATA_OFFSET,
            NiftiVersion::Two => DATA2_OFFSET,
        }
    }

    /// The header bytes, little-endian: 348 for NIfTI-1, 540 for NIfTI-2.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self.version {
            NiftiVersion::One => self.to_bytes_v1(),
            NiftiVersion::Two => self.to_bytes_v2(),
        }
    }

    fn to_bytes_v1(&self) -> Vec<u8> {
        let mut b = vec![0u8; HEADER_SIZE];
        let mut put = |offset: usize, bytes: &[u8]| b[offset..offset + bytes.len()].copy_from_slice(bytes);
        let f32le = |v: f64| (v as f32).to_le_bytes();
        let i16le = |v: i64| (v as i16).to_le_bytes();
        put(0, &(HEADER_SIZE as i32).to_le_bytes());
        put(39, &[self.dim_info]);
        for (i, &d) in self.dim.iter().enumerate() {
            put(40 + 2 * i, &i16le(d));
        }
        put(56, &f32le(self.intent_p1));
        put(60, &f32le(self.intent_p2));
        put(64, &f32le(self.intent_p3));
        put(68, &i16le(self.intent_code.into()));
        put(70, &self.datatype.to_le_bytes());
        put(72, &self.bitpix.to_le_bytes());
        put(74, &i16le(self.slice_start));
        for (i, &p) in self.pixdim.iter().enumerate() {
            put(76 + 4 * i, &f32le(p));
        }
        put(108, &f32le(self.vox_offset));
        put(112, &f32le(self.scl_slope));
        put(116, &f32le(self.scl_inter));
        put(120, &i16le(self.slice_end));
        put(122, &[self.slice_code as u8, self.xyzt_units as u8]);
        put(124, &f32le(self.cal_max));
        put(128, &f32le(self.cal_min));
        put(132, &f32le(self.slice_duration));
        put(136, &f32le(self.toffset));
        put(148, &self.descrip);
        put(228, &self.aux_file);
        put(252, &i16le(self.qform_code.into()));
        put(254, &i16le(self.sform_code.into()));
        put(256, &f32le(self.quatern_b));
        put(260, &f32le(self.quatern_c));
        put(264, &f32le(self.quatern_d));
        put(268, &f32le(self.qoffset_x));
        put(272, &f32le(self.qoffset_y));
        put(276, &f32le(self.qoffset_z));
        for (row, offset) in [(&self.srow_x, 280), (&self.srow_y, 296), (&self.srow_z, 312)] {
            for (i, &v) in row.iter().enumerate() {
                put(offset + 4 * i, &f32le(v));
            }
        }
        put(328, &self.intent_name);
        put(344, &self.magic);
        b
    }

    fn to_bytes_v2(&self) -> Vec<u8> {
        let mut b = vec![0u8; HEADER2_SIZE];
        let mut put = |offset: usize, bytes: &[u8]| b[offset..offset + bytes.len()].copy_from_slice(bytes);
        put(0, &(HEADER2_SIZE as i32).to_le_bytes());
        put(4, &self.magic);
        put(8, &EOL_CHECK);
        put(12, &self.datatype.to_le_bytes());
        put(14, &self.bitpix.to_le_bytes());
        for (i, d) in self.dim.iter().enumerate() {
            put(16 + 8 * i, &d.to_le_bytes());
        }
        put(80, &self.intent_p1.to_le_bytes());
        put(88, &self.intent_p2.to_le_bytes());
        put(96, &self.intent_p3.to_le_bytes());
        for (i, p) in self.pixdim.iter().enumerate() {
            put(104 + 8 * i, &p.to_le_bytes());
        }
        put(168, &(self.vox_offset as i64).to_le_bytes());
        put(176, &self.scl_slope.to_le_bytes());
        put(184, &self.scl_inter.to_le_bytes());
        put(192, &self.cal_max.to_le_bytes());
        put(200, &self.cal_min.to_le_bytes());
        put(208, &self.slice_duration.to_le_bytes());
        put(216, &self.toffset.to_le_bytes());
        put(224, &self.slice_start.to_le_bytes());
        put(232, &self.slice_end.to_le_bytes());
        put(240, &self.descrip);
        put(320, &self.aux_file);
        put(344, &self.qform_code.to_le_bytes());
        put(348, &self.sform_code.to_le_bytes());
        put(352, &self.quatern_b.to_le_bytes());
        put(360, &self.quatern_c.to_le_bytes());
        put(368, &self.quatern_d.to_le_bytes());
        put(376, &self.qoffset_x.to_le_bytes());
        put(384, &self.qoffset_y.to_le_bytes());
        put(392, &self.qoffset_z.to_le_bytes());
        for (row, offset) in [(&self.srow_x, 400), (&self.srow_y, 432), (&self.srow_z, 464)] {
            for (i, v) in row.iter().enumerate() {
                put(offset + 8 * i, &v.to_le_bytes());
            }
        }
        put(496, &self.slice_code.to_le_bytes());
        put(500, &self.xyzt_units.to_le_bytes());
        put(504, &self.intent_code.to_le_bytes());
        put(508, &self.intent_name);
        put(524, &[self.dim_info]);
        b
    }
}

/// Reads the first bytes of a stream, enough for either header version (fewer only if the stream
/// ends first).
fn read_header_block(reader: &mut impl Read) -> std::io::Result<Vec<u8>> {
    let mut head = Vec::with_capacity(HEADER2_SIZE);
    reader.take(HEADER_SIZE as u64).read_to_end(&mut head)?;
    let wanted = NiftiHeader::size_of(&head);
    if wanted > head.len() {
        reader.take((wanted - head.len()) as u64).read_to_end(&mut head)?;
    }
    Ok(head)
}

/// A NIfTI image: header plus voxel data.
#[derive(Clone, Debug)]
pub struct Nifti {
    /// The header as nibabel sees it after loading.
    pub header: NiftiHeader,
    /// The stored (unscaled) voxel values.
    pub data: VoxelData,
}

impl Nifti {
    /// Reads a `.nii` or `.nii.gz` file (gzip is detected from the content, not the name),
    /// decompressing only the header and the voxel bytes it describes.
    ///
    /// Mirrors `nibabel.load(path)`.
    pub fn read(path: impl AsRef<Path>) -> Result<Nifti, NiftiError> {
        let path = path.as_ref();
        let io_err = |source| NiftiError::Io { path: path.display().to_string(), source };
        let file = std::fs::File::open(path).map_err(io_err)?;
        let mut reader = std::io::BufReader::new(file);
        if reader.fill_buf().map_err(io_err)?.starts_with(&[0x1f, 0x8b]) {
            Nifti::from_stream(MultiGzDecoder::new(reader))
        } else {
            Nifti::from_stream(reader)
        }
    }

    /// Parses an image from the bytes of a `.nii` or `.nii.gz` file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Nifti, NiftiError> {
        if bytes.starts_with(&[0x1f, 0x8b]) {
            return Nifti::from_stream(MultiGzDecoder::new(bytes));
        }
        let (header, endian) = NiftiHeader::parse(bytes)?;
        let (offset, len) = header.data_range()?;
        let Some(voxels) = bytes.get(offset..offset + len) else {
            return Err(missing_data(bytes.len().saturating_sub(offset), &header, len));
        };
        let data = VoxelData::from_bytes(header.data_type(), voxels, &header.shape(), endian)?;
        Ok(Nifti { header, data })
    }

    /// Parses an image from a stream (a decompressor, say), reading the header, then exactly the
    /// bytes up to the end of the voxel data, and nothing after them.
    fn from_stream(mut reader: impl Read) -> Result<Nifti, NiftiError> {
        let stream_err = |e: std::io::Error| NiftiError::Format(format!("reading the image: {e}"));
        let head = read_header_block(&mut reader).map_err(stream_err)?;
        let (header, endian) = NiftiHeader::parse(&head)?;
        let (offset, len) = header.data_range()?;
        let mut voxels = Vec::with_capacity(len.min(MAX_UPFRONT_RESERVE));
        if offset < head.len() {
            voxels.extend_from_slice(&head[offset..head.len().min(offset + len)]);
        } else {
            let skip = (offset - head.len()) as u64;
            let skipped =
                std::io::copy(&mut (&mut reader).take(skip), &mut std::io::sink()).map_err(stream_err)?;
            if skipped < skip {
                return Err(missing_data(0, &header, len));
            }
        }
        let rest = (len - voxels.len()) as u64;
        reader.take(rest).read_to_end(&mut voxels).map_err(stream_err)?;
        if voxels.len() != len {
            return Err(missing_data(voxels.len(), &header, len));
        }
        let data = VoxelData::from_bytes(header.data_type(), &voxels, &header.shape(), endian)?;
        Ok(Nifti { header, data })
    }

    /// A new image of `data` with `affine`, like `nib.Nifti1Image(data, affine)`.
    ///
    /// The header gets the affine as an "aligned" sform (code 2) and as a qform with code 0, and
    /// its zooms from the affine, as nibabel's `Nifti1Image._affine2header` does.
    pub fn new(data: VoxelData, affine: &Affine) -> Nifti {
        let mut header = NiftiHeader::for_data(&data);
        header.set_sform(affine, 2);
        header.set_qform(affine, 0);
        Nifti { header, data }
    }

    /// A new image of `data` on this image's grid, keeping its header geometry (sform, qform,
    /// codes, zooms, units) and its NIfTI version.
    ///
    /// Like `nib.Nifti1Image(data, img.affine, img.header)`: the dtype follows `data` and the
    /// scale factors are reset. `data` must have this image's shape.
    pub fn with_data(&self, data: VoxelData) -> Nifti {
        assert_eq!(data.shape(), self.shape(), "with_data needs data of the image's shape");
        let mut header = self.header.clone();
        header.datatype = data.dtype().code();
        header.bitpix = 8 * data.dtype().size() as i16;
        header.scl_slope = 1.0;
        header.scl_inter = 0.0;
        header.vox_offset = header.data_offset() as f64;
        Nifti { header, data }
    }

    /// The array shape.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// The voxel-to-world affine (`img.affine`).
    pub fn affine(&self) -> Result<Affine, NiftiError> {
        self.header.best_affine()
    }

    /// Header voxel sizes (`img.header.get_zooms()`), as float32.
    pub fn zooms(&self) -> Vec<f32> {
        self.header.zooms()
    }

    /// The data as nibabel's `np.asanyarray(img.dataobj)` returns it: the stored array when
    /// there is no scaling, otherwise the scaled values in float64.
    pub fn dataobj(&self) -> Cow<'_, VoxelData> {
        match self.header.scaling() {
            None => Cow::Borrowed(&self.data),
            Some((slope, inter)) => Cow::Owned(VoxelData::F64(self.data.scaled(slope, inter))),
        }
    }

    /// The scaled data as float32, like `img.get_fdata(dtype=np.float32)`.
    ///
    /// With scale factors nibabel scales in float64 and then rounds to float32; this does too.
    pub fn fdata_f32(&self) -> ArrayD<f32> {
        match self.header.scaling() {
            None => self.data.to_f32(),
            Some((slope, inter)) => self.data.scaled(slope, inter).mapv(|v| v as f32),
        }
    }

    /// This image resampled onto the grid `to_shape` / `to_affine`, as float32.
    ///
    /// Mirrors `resample_from_to(img, (to_shape, to_affine), order).get_fdata(dtype=np.float32)`:
    /// scipy interpolates the values of `img.dataobj` (float64 when the header has scale
    /// factors) and writes them back in that array's dtype, so unscaled integer data are rounded
    /// before the float32 cast. Scale factors are applied while sampling, which gives the same
    /// values as resampling a scaled float64 copy without making one.
    pub fn resample_to_f32(
        &self,
        to_shape: [usize; 3],
        to_affine: &Affine,
        order: Interpolation,
    ) -> Result<Array3<f32>, NiftiError> {
        let grid = Resampling::between(&self.affine()?, to_shape, to_affine, order)
            .ok_or(NiftiError::SingularAffine)?;
        let scaling = self.header.scaling();
        with_array!(&self.data, a => resample_typed_f32(a, &grid, scaling))
            .ok_or_else(|| NiftiError::NotThreeD(self.shape().to_vec()))
    }

    /// The data as float32 with axes flipped and transposed by `ornt`, as a C-order array.
    ///
    /// Mirrors `apply_orientation(img.get_fdata(dtype=np.float32), ornt)`: the data half of
    /// `nib.as_closest_canonical(img)` followed by a float32 cast, done in one parallel pass.
    pub fn fdata_f32_oriented(&self, ornt: &Orientation) -> Result<Array3<f32>, NiftiError> {
        let scaling = self.header.scaling();
        with_array!(&self.data, a => oriented_typed_f32(a, ornt, scaling))
            .ok_or_else(|| NiftiError::NotThreeD(self.shape().to_vec()))
    }

    /// The file bytes of this image as a `.nii`, in its header's NIfTI version.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header = self.header.clone();
        let offset = header.data_offset();
        header.vox_offset = offset as f64;
        header.datatype = self.data.dtype().code();
        header.bitpix = 8 * self.data.dtype().size() as i16;
        let mut out = header.to_bytes();
        out.resize(offset, 0);
        out.extend_from_slice(&self.data.to_le_bytes());
        out
    }

    /// The bytes [`Nifti::write`] puts in a file at `path`: gzip-compressed if `path` ends in
    /// `.gz`.
    pub fn file_bytes(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, NiftiError> {
        let path = path.as_ref();
        let bytes = self.to_bytes();
        if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gz")) {
            return Ok(bytes);
        }
        let io_err = |source| NiftiError::Io { path: path.display().to_string(), source };
        let mut encoder = GzEncoder::new(Vec::with_capacity(bytes.len() / 2), Compression::new(6));
        encoder.write_all(&bytes).map_err(io_err)?;
        encoder.finish().map_err(io_err)
    }

    /// Writes the image, gzip-compressed if `path` ends in `.gz`, atomically: the bytes go to a
    /// temporary file in the same folder, which is synced and then renamed over `path`, so an
    /// interrupted write never leaves a truncated image.
    ///
    /// Like `nib.save(img, path)`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), NiftiError> {
        let path = path.as_ref();
        let bytes = self.file_bytes(path)?;
        crate::io::write_atomic(path, &bytes)
            .map_err(|source| NiftiError::Io { path: path.display().to_string(), source })
    }
}

/// The error for a file that ends before its voxel data does.
fn missing_data(have: usize, header: &NiftiHeader, len: usize) -> NiftiError {
    NiftiError::Format(format!(
        "file holds {have} data bytes, shape {:?} of {} needs {len}",
        header.shape(),
        header.data_type().name()
    ))
}

/// Rotation matrix of a quaternion `[w, x, y, z]`.
///
/// Mirrors `nibabel.quaternions.quat2mat` (scalar float64 arithmetic, no fused operations).
fn quat2mat([w, x, y, z]: [f64; 4]) -> [[f64; 3]; 3] {
    let nq = w * w + x * x + y * y + z * z;
    if nq < f64::EPSILON {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let s = 2.0 / nq;
    let (xs, ys, zs) = (x * s, y * s, z * s);
    let (wx, wy, wz) = (w * xs, w * ys, w * zs);
    let (xx, xy, xz) = (x * xs, x * ys, x * zs);
    let (yy, yz, zz) = (y * ys, y * zs, z * zs);
    [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
}

/// Quaternion `[w, x, y, z]` (with `w >= 0`) of a rotation matrix, by Shepperd's method.
///
/// Plays the role of `nibabel.quaternions.mat2quat` when writing a qform.
fn mat2quat(m: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s]
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s]
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s]
    };
    if q[0] < 0.0 { q.map(|v| -v) } else { q }
}

/// The rotation closest to a nearly orthogonal matrix (shears removed).
fn closest_rotation(r: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    crate::orientation::polar_factor(r).unwrap_or(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_data_and_affine() {
        let data = ArrayD::from_shape_fn(IxDyn(&[3, 4, 5]), |i| (i[0] * 100 + i[1] * 10 + i[2]) as i16 - 200);
        let affine =
            Affine::from_matvec([[0.0, 0.0, -1.5], [0.9, 0.0, 0.0], [0.0, 1.2, 0.0]], [10.0, -20.0, 30.0]);
        let img = Nifti::new(VoxelData::I16(data.clone()), &affine);
        let back = Nifti::from_bytes(&img.to_bytes()).unwrap();
        assert_eq!(back.data, VoxelData::I16(data));
        assert_eq!(back.affine().unwrap(), Affine(affine.0.map(|row| row.map(|v| v as f32 as f64))));
        assert!(back.header.qform().unwrap().allclose(&affine));
        assert_eq!(back.zooms(), vec![0.9, 1.2, 1.5]);
    }

    fn small_image() -> Nifti {
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 4, 4]), |i| (i[0] + 4 * i[1] + 16 * i[2]) as f32);
        Nifti::new(VoxelData::F32(data), &Affine::IDENTITY)
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap()
    }

    /// Counts the bytes read through it.
    struct Counting<R> {
        inner: R,
        count: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl<R: Read> Read for Counting<R> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.count.set(self.count.get() + n);
            Ok(n)
        }
    }

    #[test]
    fn unsupported_files_say_what_to_do() {
        let bytes = small_image().to_bytes();
        let with = |offset: usize, patch: &[u8]| {
            let mut b = bytes.clone();
            b[offset..offset + patch.len()].copy_from_slice(patch);
            Nifti::from_bytes(&b).unwrap_err().to_string()
        };
        let complex = with(70, &32i16.to_le_bytes());
        assert!(complex.contains("complex") && complex.contains("magnitude"), "{complex}");
        assert!(with(70, &1792i16.to_le_bytes()).contains("complex128"));
        assert!(with(70, &128i16.to_le_bytes()).contains("RGB24"));
        let pair = with(344, b"ni1\0");
        assert!(pair.contains("two-file") && pair.contains(".nii.gz"), "{pair}");
        let analyze = with(344, b"\0\0\0\0");
        assert!(analyze.contains("Analyze"), "{analyze}");
        assert!(with(0, &123i32.to_le_bytes()).contains("neither 348"));
    }

    #[test]
    fn overflowing_dims_are_an_error_not_a_panic() {
        // 16384^5 voxels is 2^70: the byte count wraps to 0 without checked arithmetic
        let mut bytes = small_image().to_bytes();
        let dims: [i16; 8] = [5, 16384, 16384, 16384, 16384, 16384, 1, 1];
        for (i, d) in dims.iter().enumerate() {
            bytes[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        bytes.truncate(DATA_OFFSET + 16);
        let err = Nifti::from_bytes(&bytes).unwrap_err().to_string();
        assert!(err.contains("too large"), "{err}");
        let err = Nifti::from_bytes(&gzip(&bytes)).unwrap_err().to_string();
        assert!(err.contains("too large"), "{err}");

        // a plausible shape with too few bytes behind it
        let mut short = small_image().to_bytes();
        short.truncate(short.len() - 10);
        assert!(Nifti::from_bytes(&short).unwrap_err().to_string().contains("data bytes"));
        assert!(Nifti::from_bytes(&gzip(&short)).unwrap_err().to_string().contains("data bytes"));
    }

    #[test]
    fn gzip_padding_and_extra_members_after_the_data_are_not_read() {
        let image = small_image();
        let bytes = image.to_bytes();
        // zero padding after the gzip member (Python's gzip skips it)
        let mut padded = gzip(&bytes);
        padded.extend_from_slice(&[0; 512]);
        assert_eq!(Nifti::from_bytes(&padded).unwrap().data, image.data);
        // a stream split into two members still parses
        let mut two = gzip(&bytes[..200]);
        two.extend_from_slice(&gzip(&bytes[200..]));
        assert_eq!(Nifti::from_bytes(&two).unwrap().data, image.data);
        // bytes after the voxels in an uncompressed file are ignored as well
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[7; 100]);
        assert_eq!(Nifti::from_bytes(&longer).unwrap().data, image.data);
    }

    #[test]
    fn a_small_header_with_a_huge_compressed_tail_reads_only_what_it_needs() {
        let image = small_image();
        let bytes = image.to_bytes();
        let needed = bytes.len();
        // the image, then 256 MiB of zeros in the same gzip member (about 256 KiB compressed)
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(6));
        encoder.write_all(&bytes).unwrap();
        let zeros = vec![0u8; 1 << 20];
        for _ in 0..256 {
            encoder.write_all(&zeros).unwrap();
        }
        let compressed = encoder.finish().unwrap();
        let consumed = std::rc::Rc::new(std::cell::Cell::new(0));
        let produced = std::rc::Rc::new(std::cell::Cell::new(0));
        let input = Counting { inner: compressed.as_slice(), count: consumed.clone() };
        let decoder = Counting { inner: MultiGzDecoder::new(input), count: produced.clone() };
        let back = Nifti::from_stream(decoder).unwrap();
        assert_eq!(back.data, image.data);
        assert_eq!(produced.get(), needed, "decompressed exactly the header and the voxels");
        assert!(
            consumed.get() < compressed.len() / 4,
            "read {} of {} compressed bytes",
            consumed.get(),
            compressed.len()
        );
    }

    #[test]
    fn header_only_reads_and_nifti2_files_round_trip() {
        let dir = std::env::temp_dir().join(format!("larmor-nifti-header-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut image = small_image();
        image.header.version = NiftiVersion::Two;
        image.header.magic = *b"n+2\0";
        image.header.srow_x[3] = 0.1000000000001;
        for name in ["v2.nii", "v2.nii.gz"] {
            let path = dir.join(name);
            image.write(&path).unwrap();
            let header = NiftiHeader::read(&path).unwrap();
            assert_eq!((header.version, header.shape()), (NiftiVersion::Two, vec![4, 4, 4]), "{name}");
            let back = Nifti::read(&path).unwrap();
            assert_eq!(back.header.srow_x[3], 0.1000000000001, "{name}: float64 kept");
            assert_eq!(back.data, image.data, "{name}");
            assert_eq!(
                back.with_data(VoxelData::U8(back.data.to_f32().mapv(|v| v as u8))).header.version,
                NiftiVersion::Two
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn quaternion_matrix_round_trip() {
        let q = [0.9f64.sqrt(), 0.1f64.sqrt() * 0.6, 0.1f64.sqrt() * 0.8, 0.0];
        let back = mat2quat(quat2mat(q));
        for (a, b) in q.iter().zip(back) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
