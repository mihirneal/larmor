//! A file's stored samples, decoded from any transfer syntax Larmor's dicom-rs build handles.

use dicom_encoding::TransferSyntaxIndex;
use dicom_object::DefaultDicomObject;
use dicom_pixeldata::PixelDecoder;
use dicom_transfer_syntax_registry::TransferSyntaxRegistry;

use crate::error::DicomError;
use crate::header::SliceHeader;
use crate::series::{Storage, describe};

/// One slice's samples in the storage type of its NIfTI image, row by row (column index fastest,
/// the NIfTI x axis).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Samples {
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    I32(Vec<i32>),
}

/// Why files in `transfer_syntax` cannot be decoded, or `None` when they can.
pub(crate) fn transfer_syntax_problem(transfer_syntax: &str) -> Option<String> {
    let uid = transfer_syntax.trim_end_matches('\0');
    match TransferSyntaxRegistry.get(uid) {
        Some(ts) if ts.can_decode_all() => None,
        _ if uid.starts_with("1.2.840.10008.1.2.4.8") => Some(format!(
            "JPEG-LS compressed ({uid}), which Larmor does not decode; decompress the files first (`gdcmconv \
             --raw` or `dcmdjpls`) or convert the series with dcm2niix"
        )),
        Some(ts) => Some(format!("transfer syntax {} ({uid}) is not decoded by Larmor", ts.name())),
        None => Some(format!("unknown transfer syntax {uid}")),
    }
}

/// The stored sample bytes of a single-frame image, decompressed and in this machine's byte order
/// (which Larmor requires to be little-endian, like every Mac and Linux host it builds for).
pub(crate) fn sample_bytes(object: &DefaultDicomObject, h: &SliceHeader) -> Result<Vec<u8>, DicomError> {
    let decoded = object.decode_pixel_data().map_err(|e| {
        DicomError::Unsupported(format!(
            "{}: cannot decode the pixel data of {} (transfer syntax {}): {e}",
            describe(h),
            h.path.display(),
            h.transfer_syntax
        ))
    })?;
    let expected = usize::from(h.rows) * usize::from(h.columns) * usize::from(h.bits_allocated).div_ceil(8);
    let data = decoded.data();
    if data.len() < expected {
        return Err(DicomError::read(
            &h.path,
            format!(
                "the pixel data hold {} bytes, a {}x{} image needs {expected}",
                data.len(),
                h.columns,
                h.rows
            ),
        ));
    }
    // a trailing pad byte (or a second frame) is ignored, as dcm2niix reads only the first image
    Ok(data[..expected].to_vec())
}

/// The samples of one slice as `storage` keeps them (`nii_mask12bit` included).
pub(crate) fn samples(bytes: &[u8], storage: Storage) -> Samples {
    match storage {
        Storage::U8 => Samples::U8(bytes.to_vec()),
        Storage::U16 => {
            Samples::U16(bytes.as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)).collect())
        }
        Storage::I16 { mask12 } => {
            let values = bytes.as_chunks::<2>().0.iter().map(|&b| i16::from_le_bytes(b));
            Samples::I16(match mask12 {
                None => values.collect(),
                // int12toint16: the low 12 bits as a two's complement number
                Some(true) => {
                    values.map(|v| ((v & 0xFFF) as i32 - ((v as i32 & 0x800) << 1)) as i16).collect()
                }
                Some(false) => values.map(|v| v & 4095).collect(),
            })
        }
        Storage::I32 => {
            Samples::I32(bytes.as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect())
        }
    }
}

impl Samples {
    /// The samples scaled into float32 with this slice's own slope and intercept (`nii_iVaries`:
    /// `value * slope + intercept`, fused by clang).
    pub(crate) fn scaled(&self, slope: f32, intercept: f32) -> Vec<f32> {
        let scale = |v: f32| v.mul_add(slope, intercept);
        match self {
            Samples::U8(v) => v.iter().map(|&x| scale(f32::from(x))).collect(),
            Samples::I16(v) => v.iter().map(|&x| scale(f32::from(x))).collect(),
            Samples::U16(v) => v.iter().map(|&x| scale(f32::from(x))).collect(),
            Samples::I32(v) => v.iter().map(|&x| scale(x as f32)).collect(),
        }
    }
}
