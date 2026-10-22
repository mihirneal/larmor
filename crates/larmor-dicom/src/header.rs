//! The attributes of one DICOM file that the conversion needs, read the way dcm2niix reads them.

use std::path::{Path, PathBuf};

use dicom_core::{Tag, VR};
use dicom_dictionary_std::tags;
use dicom_object::{DefaultDicomObject, InMemDicomObject, OpenFileOptions};

use crate::error::DicomError;

/// Philips' private ScaleSlope (2005,100E), FL.
const PHILIPS_SCALE_SLOPE: Tag = Tag(0x2005, 0x100E);
/// Philips' private MRAcquisitionType copy (2005,106F), CS.
const PHILIPS_ACQUISITION_TYPE: Tag = Tag(0x2005, 0x106F);
/// SharedFunctionalGroupsSequence, present in enhanced multi-frame images.
const SHARED_FUNCTIONAL_GROUPS: Tag = Tag(0x5200, 0x9229);

/// Scanner manufacturers whose DICOM the conversion treats differently (dcm2niix's
/// `dcmStrManufacturer`, by the first two letters of (0008,0070)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Manufacturer {
    /// `SI…`, or `CP…` (CPS, a Siemens venture).
    Siemens,
    /// `GE…`.
    Ge,
    /// `PH…`: Philips' precise-value scaling applies.
    Philips,
    /// `TO…`.
    Toshiba,
    /// `CA…`.
    Canon,
    /// `UI…` (United Imaging).
    Uih,
    /// Anything else.
    Other,
}

impl Manufacturer {
    fn parse(text: &str) -> Manufacturer {
        let start: Vec<u8> = text.bytes().take(2).map(|b| b.to_ascii_uppercase()).collect();
        match start.as_slice() {
            b"SI" | b"CP" => Manufacturer::Siemens,
            b"GE" => Manufacturer::Ge,
            b"PH" => Manufacturer::Philips,
            b"TO" => Manufacturer::Toshiba,
            b"CA" => Manufacturer::Canon,
            b"UI" => Manufacturer::Uih,
            _ => Manufacturer::Other,
        }
    }
}

/// What the conversion reads from one DICOM file, as dcm2niix parses it: decimal strings with
/// `atof` rounded to float, integer strings with `atoi`.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceHeader {
    /// The file.
    pub path: PathBuf,
    /// SOPInstanceUID (0008,0018): two files with the same one are copies.
    pub sop_instance_uid: String,
    /// SeriesInstanceUID (0020,000E): the files of one series share it.
    pub series_instance_uid: String,
    /// SOPClassUID (0008,0016).
    pub sop_class_uid: String,
    /// Modality (0008,0060), such as `MR`.
    pub modality: String,
    /// Manufacturer (0008,0070) as dcm2niix classifies it.
    pub manufacturer: Manufacturer,
    /// Manufacturer (0008,0070) as written.
    pub manufacturer_name: String,
    /// SeriesNumber (0020,0011).
    pub series_number: Option<i32>,
    /// SeriesDescription (0008,103E).
    pub series_description: String,
    /// ProtocolName (0018,1030).
    pub protocol_name: String,
    /// InstanceNumber (0020,0013), 1 when absent; dcm2niix stacks slices in this order.
    pub instance_number: i32,
    /// AcquisitionTime (0008,0032) as `HHMMSS.FFFFFF`, the tie-break for equal instance numbers.
    pub acquisition_time: f64,
    /// EchoNumbers (0018,0086).
    pub echo_number: Option<i32>,
    /// ImageType (0008,0008) values.
    pub image_type: Vec<String>,
    /// Rows (0028,0010).
    pub rows: u16,
    /// Columns (0028,0011).
    pub columns: u16,
    /// BitsAllocated (0028,0100).
    pub bits_allocated: u16,
    /// BitsStored (0028,0101).
    pub bits_stored: u16,
    /// PixelRepresentation (0028,0103) is 1: signed samples.
    pub signed: bool,
    /// SamplesPerPixel (0028,0002).
    pub samples_per_pixel: u16,
    /// NumberOfFrames (0028,0008), 1 when absent.
    pub frames: u32,
    /// The file has the functional groups of an enhanced multi-frame image.
    pub enhanced: bool,
    /// PixelSpacing (0028,0030) as `[column spacing, row spacing]`, the NIfTI x and y voxel sizes.
    pub pixel_spacing: Option<[f32; 2]>,
    /// SliceThickness (0018,0050).
    pub slice_thickness: Option<f32>,
    /// SpacingBetweenSlices (0018,0088).
    pub spacing_between_slices: Option<f32>,
    /// ImagePositionPatient (0020,0032), LPS mm.
    pub position: Option<[f32; 3]>,
    /// ImageOrientationPatient (0020,0037): the row and column direction cosines, LPS.
    pub orientation: Option<[f32; 6]>,
    /// RescaleSlope (0028,1053), 1 when absent.
    pub rescale_slope: f32,
    /// RescaleIntercept (0028,1052), 0 when absent.
    pub rescale_intercept: f32,
    /// Philips ScaleSlope (2005,100E), for Philips' precise values.
    pub philips_scale_slope: Option<f32>,
    /// MRAcquisitionType (0018,0023, or Philips' 2005,106F) is `3D`.
    pub acquisition_3d: bool,
    /// An echo-planar sequence (0018,9018 `YES`, or `EP` in ScanningSequence 0018,0020).
    pub epi: bool,
    /// GantryDetectorTilt (0018,1120), degrees.
    pub gantry_tilt: f32,
    /// TransferSyntaxUID (0002,0010).
    pub transfer_syntax: String,
}

/// True when `bytes` start like a DICOM Part 10 file: `DICM` after the 128-byte preamble (or at
/// the start, for files written without one).
pub fn is_dicom_bytes(bytes: &[u8]) -> bool {
    bytes.get(128..132) == Some(b"DICM") || bytes.starts_with(b"DICM")
}

/// True when the file at `path` starts like a DICOM Part 10 file (reads 132 bytes).
pub fn is_dicom_file(path: &Path) -> bool {
    use std::io::Read;
    let mut head = Vec::with_capacity(132);
    match std::fs::File::open(path) {
        Ok(file) => file.take(132).read_to_end(&mut head).is_ok() && is_dicom_bytes(&head),
        Err(_) => false,
    }
}

impl SliceHeader {
    /// Reads a file's header, stopping before its pixel data.
    pub fn read(path: &Path) -> Result<SliceHeader, DicomError> {
        let object = OpenFileOptions::new()
            .read_until(tags::PIXEL_DATA)
            .open_file(path)
            .map_err(|e| DicomError::read(path, e))?;
        SliceHeader::from_object(path, &object)
    }

    /// The header of a file's bytes, parsed up to its pixel data.
    pub fn from_bytes(path: &Path, bytes: &[u8]) -> Result<SliceHeader, DicomError> {
        let object = OpenFileOptions::new()
            .read_until(tags::PIXEL_DATA)
            .from_reader(bytes)
            .map_err(|e| DicomError::read(path, e))?;
        SliceHeader::from_object(path, &object)
    }

    /// The header of a parsed file.
    pub fn from_object(path: &Path, object: &DefaultDicomObject) -> Result<SliceHeader, DicomError> {
        let o: &InMemDicomObject = object;
        let text = |tag: Tag| -> String {
            o.get(tag)
                .and_then(|e| e.to_str().ok())
                .map(|s| s.trim_matches(['\0', ' ']).to_string())
                .unwrap_or_default()
        };
        let raw =
            |tag: Tag| -> Option<String> { o.get(tag).and_then(|e| e.to_str().ok()).map(|s| s.into_owned()) };
        let int = |tag: Tag| -> Option<u16> { o.get(tag).and_then(|e| e.to_int::<u16>().ok()) };
        let is = |tag: Tag| raw(tag).filter(|s| !s.trim().is_empty()).map(|s| atoi(&s));
        let ds = |tag: Tag| raw(tag).filter(|s| !s.trim().is_empty()).map(|s| atof(&s) as f32);
        let floats = |tag: Tag, n: usize| raw(tag).and_then(|s| multi_float(&s, n));

        let manufacturer_name = text(tags::MANUFACTURER);
        let manufacturer = Manufacturer::parse(&manufacturer_name);
        let required = |tag: Tag, what: &'static str| {
            let value = text(tag);
            if value.is_empty() { Err(DicomError::missing(path, what)) } else { Ok(value) }
        };
        let philips_scale_slope = if manufacturer == Manufacturer::Philips {
            o.get(PHILIPS_SCALE_SLOPE).and_then(|e| {
                // dcm2niix reads it only when it holds exactly one float32; in an implicit VR file
                // the private tag's VR is unknown and its 4 bytes are little-endian
                if e.vr() == VR::FL {
                    return e.to_multi_float32().ok().filter(|v| v.len() == 1).map(|v| v[0]);
                }
                let bytes = e.to_bytes().ok()?;
                let bytes: [u8; 4] = bytes.as_ref().try_into().ok()?;
                Some(f32::from_le_bytes(bytes))
            })
        } else {
            None
        };
        let is_3d = |tag: Tag| text(tag).to_ascii_uppercase().starts_with("3D");
        let image_type: Vec<String> = raw(tags::IMAGE_TYPE)
            .map(|s| s.split('\\').map(|v| v.trim_matches(['\0', ' ']).to_ascii_uppercase()).collect())
            .unwrap_or_default();
        let pixel_spacing = floats(tags::PIXEL_SPACING, 2).map(|yx| [yx[1], yx[0]]);
        Ok(SliceHeader {
            path: path.to_path_buf(),
            sop_instance_uid: text(tags::SOP_INSTANCE_UID),
            series_instance_uid: required(tags::SERIES_INSTANCE_UID, "SeriesInstanceUID (0020,000E)")?,
            sop_class_uid: text(tags::SOP_CLASS_UID),
            modality: text(tags::MODALITY),
            manufacturer,
            manufacturer_name,
            series_number: is(tags::SERIES_NUMBER),
            series_description: text(tags::SERIES_DESCRIPTION),
            protocol_name: text(tags::PROTOCOL_NAME),
            instance_number: is(tags::INSTANCE_NUMBER).unwrap_or(1),
            acquisition_time: raw(tags::ACQUISITION_TIME).map_or(0.0, |s| atof(&s)),
            echo_number: is(tags::ECHO_NUMBERS),
            image_type,
            rows: int(tags::ROWS).unwrap_or(0),
            columns: int(tags::COLUMNS).unwrap_or(0),
            bits_allocated: int(tags::BITS_ALLOCATED).unwrap_or(0),
            bits_stored: int(tags::BITS_STORED).unwrap_or(0),
            signed: int(tags::PIXEL_REPRESENTATION) == Some(1),
            samples_per_pixel: int(tags::SAMPLES_PER_PIXEL).unwrap_or(1),
            frames: is(tags::NUMBER_OF_FRAMES).map_or(1, |n| n.max(1) as u32),
            enhanced: o.get(SHARED_FUNCTIONAL_GROUPS).is_some(),
            pixel_spacing,
            slice_thickness: ds(tags::SLICE_THICKNESS),
            spacing_between_slices: ds(tags::SPACING_BETWEEN_SLICES),
            position: floats(tags::IMAGE_POSITION_PATIENT, 3).map(|p| [p[0], p[1], p[2]]),
            orientation: floats(tags::IMAGE_ORIENTATION_PATIENT, 6)
                .map(|v| [v[0], v[1], v[2], v[3], v[4], v[5]]),
            rescale_slope: ds(tags::RESCALE_SLOPE).unwrap_or(1.0),
            rescale_intercept: ds(tags::RESCALE_INTERCEPT).unwrap_or(0.0),
            philips_scale_slope,
            acquisition_3d: is_3d(tags::MR_ACQUISITION_TYPE)
                || (manufacturer == Manufacturer::Philips && is_3d(PHILIPS_ACQUISITION_TYPE)),
            epi: text(tags::ECHO_PLANAR_PULSE_SEQUENCE).to_ascii_uppercase().starts_with('Y')
                || text(tags::SCANNING_SEQUENCE).contains("EP"),
            gantry_tilt: ds(tags::GANTRY_DETECTOR_TILT).unwrap_or(0.0),
            transfer_syntax: object.meta().transfer_syntax().trim_end_matches('\0').to_string(),
        })
    }

    /// True for Siemens mosaics (several slices tiled in one image: fMRI and DWI).
    pub fn is_mosaic(&self) -> bool {
        self.image_type.iter().any(|v| v == "MOSAIC")
    }
}

/// C's `atoi`: optional leading whitespace and sign, then digits; 0 when there are none.
pub(crate) fn atoi(text: &str) -> i32 {
    let s = text.trim_start();
    let (sign, digits) = match s.as_bytes().first() {
        Some(b'-') => (-1i64, &s[1..]),
        Some(b'+') => (1, &s[1..]),
        _ => (1, s),
    };
    let end = digits.bytes().position(|b| !b.is_ascii_digit()).unwrap_or(digits.len());
    let value = digits[..end].bytes().fold(0i64, |v, d| (v * 10 + i64::from(d - b'0')).min(1 << 32));
    (sign * value).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// C's `atof` on a decimal number: the longest prefix (after whitespace) of the form
/// `[+-]digits[.digits][e[+-]digits]`, correctly rounded to double; 0 when there is none.
pub(crate) fn atof(text: &str) -> f64 {
    let s = text.trim_start();
    let b = s.as_bytes();
    let mut i = 0;
    if matches!(b.first(), Some(b'+' | b'-')) {
        i += 1;
    }
    let digits_start = i;
    while b.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    let mut mantissa_digits = i - digits_start;
    if b.get(i) == Some(&b'.') {
        let mut j = i + 1;
        while b.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        mantissa_digits += j - i - 1;
        if mantissa_digits > 0 {
            i = j;
        }
    }
    if mantissa_digits == 0 {
        return 0.0;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        let mut j = i + 1;
        if matches!(b.get(j), Some(b'+' | b'-')) {
            j += 1;
        }
        let exponent_start = j;
        while b.get(j).is_some_and(u8::is_ascii_digit) {
            j += 1;
        }
        if j > exponent_start {
            i = j;
        }
    }
    s[..i].parse().unwrap_or(0.0)
}

/// dcm2niix's `dcmMultiFloat`: up to `n` numbers from a multi-valued decimal string. A token
/// ends at `\`, `/`, a space or the end, once it has a digit; each is read with `atof` and
/// rounded to float. `None` when fewer than `n` are found.
pub(crate) fn multi_float(text: &str, n: usize) -> Option<Vec<f32>> {
    let bytes = text.as_bytes();
    let mut values = Vec::with_capacity(n);
    let mut start = 0;
    let mut has_digit = false;
    for i in 0..=bytes.len() {
        let byte = bytes.get(i).copied();
        if byte.is_some_and(|b| b.is_ascii_digit()) {
            has_digit = true;
        }
        let separator = matches!(byte, None | Some(b'/' | b' ' | b'\\'));
        if has_digit && separator {
            if values.len() < n {
                values.push(atof(&text[start..i]) as f32);
                has_digit = false;
            }
            start = i + 1;
        }
    }
    (values.len() == n).then_some(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_strings_parse_like_atof() {
        assert_eq!(atof(" -12.5e1junk"), -125.0);
        assert_eq!(atof("0.0452015"), 0.0452015);
        assert_eq!(atof(".5"), 0.5);
        assert_eq!(atof("5."), 5.0);
        assert_eq!(atof("1e"), 1.0);
        assert_eq!(atof("abc"), 0.0);
        assert_eq!(atof("-"), 0.0);
        assert_eq!(atoi(" 42 "), 42);
        assert_eq!(atoi("-7x"), -7);
        assert_eq!(atoi(""), 0);
        assert_eq!(
            multi_float("0.99801\\0.0452015\\0.0439564\\-0.0342042\\0.973805\\-0.224796", 6).unwrap(),
            [0.99801f32, 0.0452015, 0.0439564, -0.0342042, 0.973805, -0.224796]
        );
        assert_eq!(multi_float("1.0\\2.0", 3), None);
        assert_eq!(multi_float("-96.9 \\-101.6", 2).unwrap()[0], -96.9f32);
        // float values are the double rounded to float, as `(float)atof(...)`
        assert_eq!(multi_float("0.1", 1).unwrap()[0], 0.1f64 as f32);
    }

    #[test]
    fn manufacturers_follow_dcm2niix() {
        assert_eq!(Manufacturer::parse("SIEMENS"), Manufacturer::Siemens);
        assert_eq!(Manufacturer::parse("GE MEDICAL SYSTEMS"), Manufacturer::Ge);
        assert_eq!(Manufacturer::parse("Philips Medical Systems"), Manufacturer::Philips);
        assert_eq!(Manufacturer::parse("UIH"), Manufacturer::Uih);
        assert_eq!(Manufacturer::parse("x"), Manufacturer::Other);
    }
}
