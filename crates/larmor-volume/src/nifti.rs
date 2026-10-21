//! NIfTI-1 single-file images (`.nii`, `.nii.gz`), read and written with nibabel's semantics.
//!
//! The affine, zooms and data scaling follow nibabel 5.4 exactly, including the fixes nibabel
//! applies silently on load (`Nifti1Header.check_fix`): `qfac` other than ±1 becomes 1, zero
//! `pixdim` entries become 1, negative ones their absolute value, invalid xform codes 0.

use std::borrow::Cow;
use std::io::{Read, Write};
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
/// Offset of the voxel data in the files this crate writes (header plus the 4-byte extension flag).
const DATA_OFFSET: usize = 352;

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
    /// The bytes are not a NIfTI-1 image this crate can read.
    #[error("not a readable NIfTI-1 image: {0}")]
    Format(String),
    /// The voxel datatype is valid NIfTI but not supported here (complex, RGB, float128).
    #[error("unsupported NIfTI datatype code {0} (supported: integer types, float32, float64)")]
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

    fn from_bytes(dtype: DataType, bytes: &[u8], shape: &[usize], endian: Endian) -> VoxelData {
        macro_rules! decode {
            ($variant:ident, $t:ty) => {{
                const N: usize = std::mem::size_of::<$t>();
                let values: Vec<$t> = bytes
                    .chunks_exact(N)
                    .map(|c| {
                        let raw: [u8; N] = c.try_into().unwrap();
                        match endian {
                            Endian::Little => <$t>::from_le_bytes(raw),
                            Endian::Big => <$t>::from_be_bytes(raw),
                        }
                    })
                    .collect();
                VoxelData::$variant(ArrayD::from_shape_vec(IxDyn(shape).f(), values).unwrap())
            }};
        }
        match dtype {
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
        }
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

/// A NIfTI-1 header, field for field (nibabel's `Nifti1Header` structured array).
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq)]
pub struct NiftiHeader {
    pub dim_info: u8,
    pub dim: [i16; 8],
    pub intent_p1: f32,
    pub intent_p2: f32,
    pub intent_p3: f32,
    pub intent_code: i16,
    pub datatype: i16,
    pub bitpix: i16,
    pub slice_start: i16,
    pub pixdim: [f32; 8],
    pub vox_offset: f32,
    pub scl_slope: f32,
    pub scl_inter: f32,
    pub slice_end: i16,
    pub slice_code: u8,
    pub xyzt_units: u8,
    pub cal_max: f32,
    pub cal_min: f32,
    pub slice_duration: f32,
    pub toffset: f32,
    pub descrip: [u8; 80],
    pub aux_file: [u8; 24],
    pub qform_code: i16,
    pub sform_code: i16,
    pub quatern_b: f32,
    pub quatern_c: f32,
    pub quatern_d: f32,
    pub qoffset_x: f32,
    pub qoffset_y: f32,
    pub qoffset_z: f32,
    pub srow_x: [f32; 4],
    pub srow_y: [f32; 4],
    pub srow_z: [f32; 4],
    pub intent_name: [u8; 16],
    pub magic: [u8; 4],
}

/// Reads fixed-offset fields from a header block in a given byte order.
struct FieldReader<'a> {
    bytes: &'a [u8],
    endian: Endian,
}

impl FieldReader<'_> {
    fn array<const N: usize>(&self, offset: usize) -> [u8; N] {
        self.bytes[offset..offset + N].try_into().unwrap()
    }
    fn i16(&self, offset: usize) -> i16 {
        match self.endian {
            Endian::Little => i16::from_le_bytes(self.array(offset)),
            Endian::Big => i16::from_be_bytes(self.array(offset)),
        }
    }
    fn f32(&self, offset: usize) -> f32 {
        match self.endian {
            Endian::Little => f32::from_le_bytes(self.array(offset)),
            Endian::Big => f32::from_be_bytes(self.array(offset)),
        }
    }
    fn i16s<const N: usize>(&self, offset: usize) -> [i16; N] {
        std::array::from_fn(|i| self.i16(offset + 2 * i))
    }
    fn f32s<const N: usize>(&self, offset: usize) -> [f32; N] {
        std::array::from_fn(|i| self.f32(offset + 4 * i))
    }
}

impl NiftiHeader {
    /// Parses a 348-byte header (either byte order), then applies nibabel's load-time fixes.
    ///
    /// Mirrors `Nifti1Header.from_fileobj(..., check=True)`; problems nibabel raises on (bad
    /// magic, unknown datatype, a data offset inside the header) are errors here too.
    pub fn parse(bytes: &[u8]) -> Result<(NiftiHeader, Endian), NiftiError> {
        if bytes.len() < HEADER_SIZE {
            return Err(NiftiError::Format(format!("{} bytes is too short for a header", bytes.len())));
        }
        let sizeof_hdr: [u8; 4] = bytes[..4].try_into().unwrap();
        let endian = if i32::from_le_bytes(sizeof_hdr) == HEADER_SIZE as i32 {
            Endian::Little
        } else if i32::from_be_bytes(sizeof_hdr) == HEADER_SIZE as i32 {
            Endian::Big
        } else if i32::from_le_bytes(sizeof_hdr) == 540 || i32::from_be_bytes(sizeof_hdr) == 540 {
            return Err(NiftiError::Format("NIfTI-2 files are not supported".into()));
        } else {
            return Err(NiftiError::Format("sizeof_hdr is not 348".into()));
        };
        let r = FieldReader { bytes, endian };
        let mut header = NiftiHeader {
            dim_info: bytes[39],
            dim: r.i16s(40),
            intent_p1: r.f32(56),
            intent_p2: r.f32(60),
            intent_p3: r.f32(64),
            intent_code: r.i16(68),
            datatype: r.i16(70),
            bitpix: r.i16(72),
            slice_start: r.i16(74),
            pixdim: r.f32s(76),
            vox_offset: r.f32(108),
            scl_slope: r.f32(112),
            scl_inter: r.f32(116),
            slice_end: r.i16(120),
            slice_code: bytes[122],
            xyzt_units: bytes[123],
            cal_max: r.f32(124),
            cal_min: r.f32(128),
            slice_duration: r.f32(132),
            toffset: r.f32(136),
            descrip: r.array(148),
            aux_file: r.array(228),
            qform_code: r.i16(252),
            sform_code: r.i16(254),
            quatern_b: r.f32(256),
            quatern_c: r.f32(260),
            quatern_d: r.f32(264),
            qoffset_x: r.f32(268),
            qoffset_y: r.f32(272),
            qoffset_z: r.f32(276),
            srow_x: r.f32s(280),
            srow_y: r.f32s(296),
            srow_z: r.f32s(312),
            intent_name: r.array(328),
            magic: r.array(344),
        };
        header.check_fix()?;
        Ok((header, endian))
    }

    /// nibabel's `Nifti1Header` check battery with `fix=True` and its default error level.
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
        match &self.magic {
            b"n+1\0" => {}
            b"ni1\0" => return Err(NiftiError::Format("two-file NIfTI (.hdr/.img) is not supported".into())),
            _ => return Err(NiftiError::Format("bad magic, not a NIfTI-1 header".into())),
        }
        if self.vox_offset < DATA_OFFSET as f32 {
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

    /// The array shape (`get_data_shape()`).
    pub fn shape(&self) -> Vec<usize> {
        self.dim[1..=self.dim[0] as usize].iter().map(|&d| d as usize).collect()
    }

    /// The voxel datatype.
    pub fn data_type(&self) -> DataType {
        DataType::from_code(self.datatype).expect("datatype checked on construction")
    }

    /// Voxel sizes (and the time step for 4-D) from `pixdim`, as float32.
    ///
    /// Mirrors `get_zooms()`.
    pub fn zooms(&self) -> Vec<f32> {
        self.pixdim[1..=self.dim[0] as usize].to_vec()
    }

    /// The scale factors `(slope, intercept)`, or `None` when the data are used as stored.
    ///
    /// Mirrors `get_slope_inter()` and the `(1, 0)` shortcut of `apply_read_scaling`: a slope
    /// of 0 or non-finite means no scaling; the factors are float32 values widened to float64.
    /// (A valid slope with a non-finite intercept is rejected when the header is parsed.)
    pub fn scaling(&self) -> Option<(f64, f64)> {
        let (slope, inter) = (self.scl_slope as f64, self.scl_inter as f64);
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
            *row = srow.map(|v| v as f64);
        }
        out
    }

    /// The qform matrix from the quaternion, `pixdim` and `qoffset`.
    ///
    /// Mirrors `get_qform()`: `w` is recovered by `fillpositive` (set to 0 when `1 - |bcd|²` is
    /// within three float32 epsilons of 0), the rotation is `quat2mat`, and `qfac` flips the
    /// third column. Fails if the quaternion's `b, c, d` have norm above 1.
    pub fn qform(&self) -> Result<Affine, NiftiError> {
        let [x, y, z] = [self.quatern_b, self.quatern_c, self.quatern_d].map(|v| v as f64);
        let w2 = 1.0 - ((x * x + y * y) + z * z);
        let threshold = (f32::EPSILON * 3.0) as f64;
        let w = if w2.abs() < threshold {
            0.0
        } else if w2 < 0.0 {
            return Err(NiftiError::Format(format!("qform quaternion has |b, c, d| > 1 (w² = {w2:e})")));
        } else {
            w2.sqrt()
        };
        let rotation = quat2mat([w, x, y, z]);
        let mut zooms = [self.pixdim[1], self.pixdim[2], self.pixdim[3]];
        zooms[2] *= self.pixdim[0];
        let rzs: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| rotation[i][j] * zooms[j] as f64));
        Ok(Affine::from_matvec(rzs, [self.qoffset_x, self.qoffset_y, self.qoffset_z].map(|v| v as f64)))
    }

    /// The affine nibabel falls back to with neither form set: centred, x flipped.
    ///
    /// Mirrors `get_base_affine()` (`shape_zoom_affine(shape, zooms, x_flip=True)`).
    pub fn base_affine(&self) -> Affine {
        let ndim = self.dim[0] as usize;
        let mut shape = [1.0f64; 3];
        let mut zooms = [1.0f32; 3];
        for axis in 0..ndim.min(3) {
            shape[axis] = self.dim[axis + 1] as f64;
            zooms[axis] = self.pixdim[axis + 1];
        }
        zooms[0] = -zooms[0];
        let mut out = Affine::IDENTITY;
        for axis in 0..3 {
            let zoom = zooms[axis] as f64;
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

    /// Writes `affine` into the sform with `code`.
    ///
    /// Mirrors `set_sform(affine, code)`.
    pub fn set_sform(&mut self, affine: &Affine, code: i16) {
        let row = |r: usize| affine.0[r].map(|v| v as f32);
        self.srow_x = row(0);
        self.srow_y = row(1);
        self.srow_z = row(2);
        self.sform_code = code;
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
        self.qoffset_x = t[0] as f32;
        self.qoffset_y = t[1] as f32;
        self.qoffset_z = t[2] as f32;
        self.pixdim[0] = qfac;
        for axis in 0..3 {
            self.pixdim[axis + 1] = zooms[axis] as f32;
        }
        self.quatern_b = q[1] as f32;
        self.quatern_c = q[2] as f32;
        self.quatern_d = q[3] as f32;
        self.qform_code = code;
    }

    /// A header for `data` with no geometry yet: nibabel's `Nifti1Header()` defaults.
    fn for_data(data: &VoxelData) -> NiftiHeader {
        let mut dim = [1i16; 8];
        dim[0] = data.shape().len() as i16;
        for (d, &n) in dim[1..].iter_mut().zip(data.shape()) {
            *d = n as i16;
        }
        NiftiHeader {
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
            vox_offset: DATA_OFFSET as f32,
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

    /// The 348 header bytes, little-endian.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        let mut put = |offset: usize, bytes: &[u8]| b[offset..offset + bytes.len()].copy_from_slice(bytes);
        put(0, &(HEADER_SIZE as i32).to_le_bytes());
        put(39, &[self.dim_info]);
        for (i, d) in self.dim.iter().enumerate() {
            put(40 + 2 * i, &d.to_le_bytes());
        }
        put(56, &self.intent_p1.to_le_bytes());
        put(60, &self.intent_p2.to_le_bytes());
        put(64, &self.intent_p3.to_le_bytes());
        put(68, &self.intent_code.to_le_bytes());
        put(70, &self.datatype.to_le_bytes());
        put(72, &self.bitpix.to_le_bytes());
        put(74, &self.slice_start.to_le_bytes());
        for (i, p) in self.pixdim.iter().enumerate() {
            put(76 + 4 * i, &p.to_le_bytes());
        }
        put(108, &self.vox_offset.to_le_bytes());
        put(112, &self.scl_slope.to_le_bytes());
        put(116, &self.scl_inter.to_le_bytes());
        put(120, &self.slice_end.to_le_bytes());
        put(122, &[self.slice_code, self.xyzt_units]);
        put(124, &self.cal_max.to_le_bytes());
        put(128, &self.cal_min.to_le_bytes());
        put(132, &self.slice_duration.to_le_bytes());
        put(136, &self.toffset.to_le_bytes());
        put(148, &self.descrip);
        put(228, &self.aux_file);
        put(252, &self.qform_code.to_le_bytes());
        put(254, &self.sform_code.to_le_bytes());
        put(256, &self.quatern_b.to_le_bytes());
        put(260, &self.quatern_c.to_le_bytes());
        put(264, &self.quatern_d.to_le_bytes());
        put(268, &self.qoffset_x.to_le_bytes());
        put(272, &self.qoffset_y.to_le_bytes());
        put(276, &self.qoffset_z.to_le_bytes());
        for (row, offset) in [(&self.srow_x, 280), (&self.srow_y, 296), (&self.srow_z, 312)] {
            for (i, v) in row.iter().enumerate() {
                put(offset + 4 * i, &v.to_le_bytes());
            }
        }
        put(328, &self.intent_name);
        put(344, &self.magic);
        b
    }
}

/// A NIfTI-1 image: header plus voxel data.
#[derive(Clone, Debug)]
pub struct Nifti {
    /// The header as nibabel sees it after loading.
    pub header: NiftiHeader,
    /// The stored (unscaled) voxel values.
    pub data: VoxelData,
}

impl Nifti {
    /// Reads a `.nii` or `.nii.gz` file (gzip is detected from the content, not the name).
    ///
    /// Mirrors `nibabel.load(path)`.
    pub fn read(path: impl AsRef<Path>) -> Result<Nifti, NiftiError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .map_err(|source| NiftiError::Io { path: path.display().to_string(), source })?;
        Nifti::from_bytes(&bytes)
    }

    /// Parses an image from the bytes of a `.nii` or `.nii.gz` file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Nifti, NiftiError> {
        let raw: Cow<[u8]> =
            if bytes.starts_with(&[0x1f, 0x8b]) { Cow::Owned(gunzip(bytes)?) } else { Cow::Borrowed(bytes) };
        let (header, endian) = NiftiHeader::parse(&raw)?;
        let shape = header.shape();
        let dtype = header.data_type();
        let offset = header.vox_offset as usize;
        let len = shape.iter().product::<usize>() * dtype.size();
        let Some(voxels) = raw.get(offset..offset + len) else {
            return Err(NiftiError::Format(format!(
                "file holds {} data bytes, shape {:?} of {} needs {len}",
                raw.len().saturating_sub(offset),
                shape,
                dtype.name()
            )));
        };
        let data = VoxelData::from_bytes(dtype, voxels, &shape, endian);
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
    /// codes, zooms, units).
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
        header.vox_offset = DATA_OFFSET as f32;
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

    /// Header voxel sizes (`img.header.get_zooms()`).
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

    /// The file bytes of this image as a `.nii`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut header = self.header.clone();
        header.vox_offset = DATA_OFFSET as f32;
        header.datatype = self.data.dtype().code();
        header.bitpix = 8 * self.data.dtype().size() as i16;
        let mut out = header.to_bytes().to_vec();
        out.extend_from_slice(&[0; DATA_OFFSET - HEADER_SIZE]);
        out.extend_from_slice(&self.data.to_le_bytes());
        out
    }

    /// Writes the image, gzip-compressed if `path` ends in `.gz`.
    ///
    /// Like `nib.save(img, path)`.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<(), NiftiError> {
        let path = path.as_ref();
        let io_err = |source| NiftiError::Io { path: path.display().to_string(), source };
        let bytes = self.to_bytes();
        let gzip = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gz"));
        if gzip {
            let file = std::fs::File::create(path).map_err(io_err)?;
            let mut encoder = GzEncoder::new(std::io::BufWriter::new(file), Compression::new(6));
            encoder.write_all(&bytes).map_err(io_err)?;
            encoder.finish().and_then(|mut w| w.flush()).map_err(io_err)?;
        } else {
            std::fs::write(path, bytes).map_err(io_err)?;
        }
        Ok(())
    }
}

/// Decompresses a (possibly multi-member) gzip stream.
fn gunzip(bytes: &[u8]) -> Result<Vec<u8>, NiftiError> {
    // the trailer's ISIZE (size mod 2^32 of the last member) is a good capacity hint
    let hint =
        bytes.len().checked_sub(4).map_or(0, |i| u32::from_le_bytes(bytes[i..].try_into().unwrap()) as usize);
    let mut out = Vec::with_capacity(hint.max(bytes.len()));
    MultiGzDecoder::new(bytes)
        .read_to_end(&mut out)
        .map_err(|e| NiftiError::Format(format!("gzip stream: {e}")))?;
    Ok(out)
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

    #[test]
    fn quaternion_matrix_round_trip() {
        let q = [0.9f64.sqrt(), 0.1f64.sqrt() * 0.6, 0.1f64.sqrt() * 0.8, 0.0];
        let back = mat2quat(quat2mat(q));
        for (a, b) in q.iter().zip(back) {
            assert!((a - b).abs() < 1e-15);
        }
    }
}
