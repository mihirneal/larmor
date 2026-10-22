//! Synthetic classic DICOM series, written from a volume, for tests and examples.

use std::path::{Path, PathBuf};

use dicom_core::value::PrimitiveValue;
use dicom_core::{DataElement, VR};
use dicom_dictionary_std::{tags, uids};
use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
use larmor_volume::ndarray::Array3;

/// How [`write_series`] writes a volume `[columns, rows, slices]` as one file per slice.
#[derive(Clone, Debug)]
pub struct SeriesSpec {
    /// The row direction cosines (ImageOrientationPatient 1–3, LPS): moving along a row, across
    /// columns.
    pub row: [f64; 3],
    /// The column direction cosines (ImageOrientationPatient 4–6, LPS).
    pub column: [f64; 3],
    /// ImagePositionPatient of slice 0 (LPS mm).
    pub origin: [f64; 3],
    /// Column spacing, row spacing and the step from one slice to the next along the slice normal
    /// (row × column), mm; a negative step stacks the slices against the normal.
    pub spacing: [f64; 3],
    /// SeriesInstanceUID.
    pub series_uid: String,
    /// SeriesNumber.
    pub series_number: i32,
    /// SeriesDescription.
    pub description: String,
    /// Manufacturer (0008,0070).
    pub manufacturer: String,
    /// MRAcquisitionType: `2D` or `3D` (3-D acquisitions are reoriented like dcm2niix's).
    pub acquisition: String,
    /// RescaleSlope and RescaleIntercept per slice (one value for all when it has one entry).
    pub rescale: Vec<(f64, f64)>,
    /// InstanceNumber of each slice, in slice order (default `1..=n`).
    pub instance_numbers: Option<Vec<i32>>,
    /// Override the position of slice `k` (to make gaps or repeats).
    pub positions: Option<Vec<[f64; 3]>>,
}

impl SeriesSpec {
    /// An axial series (rows along +x L, columns along +y P, slices along +z S) starting at
    /// `origin`, with `spacing`.
    pub fn axial(origin: [f64; 3], spacing: [f64; 3]) -> SeriesSpec {
        SeriesSpec {
            row: [1.0, 0.0, 0.0],
            column: [0.0, 1.0, 0.0],
            origin,
            spacing,
            series_uid: "2.25.1234567890".into(),
            series_number: 3,
            description: "T1w synthetic".into(),
            manufacturer: "SIEMENS".into(),
            acquisition: "2D".into(),
            rescale: vec![(1.0, 0.0)],
            instance_numbers: None,
            positions: None,
        }
    }

    /// The position of slice `k`.
    pub fn position(&self, k: usize) -> [f64; 3] {
        if let Some(positions) = &self.positions {
            return positions[k];
        }
        let n = cross(self.row, self.column);
        std::array::from_fn(|i| self.origin[i] + k as f64 * self.spacing[2] * n[i])
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A decimal string of at most 16 characters (the DS limit).
fn ds(value: f64) -> String {
    for precision in (0..=12).rev() {
        let text = format!("{value:.precision$}");
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            text
        };
        let text = if text == "-0" { "0".to_string() } else { text };
        if text.len() <= 16 {
            return text;
        }
    }
    format!("{value:e}")
}

fn multi_ds(values: &[f64]) -> String {
    values.iter().map(|&v| ds(v)).collect::<Vec<_>>().join("\\")
}

/// Writes `volume` (`[columns, rows, slices]`, int16) as classic MR DICOM, one explicit VR little
/// endian file per slice, `<dir>/slice_<k>.dcm`. Returns the files.
pub fn write_series(dir: &Path, volume: &Array3<i16>, spec: &SeriesSpec) -> std::io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let [columns, rows, slices] = [volume.shape()[0], volume.shape()[1], volume.shape()[2]];
    let mut files = Vec::with_capacity(slices);
    for k in 0..slices {
        let instance = spec.instance_numbers.as_ref().map_or(k as i32 + 1, |n| n[k]);
        let sop_uid = format!("{}.{}", spec.series_uid, k + 1);
        let (slope, intercept) = spec.rescale[k.min(spec.rescale.len() - 1)];
        let mut pixels = Vec::with_capacity(rows * columns * 2);
        for r in 0..rows {
            for c in 0..columns {
                pixels.extend_from_slice(&volume[[c, r, k]].to_le_bytes());
            }
        }
        let text = |tag, vr, value: String| DataElement::new(tag, vr, PrimitiveValue::from(value));
        let us = |tag, value: u16| DataElement::new(tag, VR::US, PrimitiveValue::from(value));
        let elements = vec![
            text(tags::SOP_CLASS_UID, VR::UI, uids::MR_IMAGE_STORAGE.into()),
            text(tags::SOP_INSTANCE_UID, VR::UI, sop_uid.clone()),
            text(tags::IMAGE_TYPE, VR::CS, "ORIGINAL\\PRIMARY\\M\\ND".into()),
            text(tags::MODALITY, VR::CS, "MR".into()),
            text(tags::MANUFACTURER, VR::LO, spec.manufacturer.clone()),
            text(tags::SERIES_DESCRIPTION, VR::LO, spec.description.clone()),
            text(tags::ACQUISITION_TIME, VR::TM, "101500.000000".into()),
            text(tags::MR_ACQUISITION_TYPE, VR::CS, spec.acquisition.clone()),
            text(tags::SLICE_THICKNESS, VR::DS, ds(spec.spacing[2].abs())),
            text(tags::SERIES_INSTANCE_UID, VR::UI, spec.series_uid.clone()),
            text(tags::SERIES_NUMBER, VR::IS, spec.series_number.to_string()),
            text(tags::INSTANCE_NUMBER, VR::IS, instance.to_string()),
            text(tags::IMAGE_POSITION_PATIENT, VR::DS, multi_ds(&spec.position(k))),
            text(
                tags::IMAGE_ORIENTATION_PATIENT,
                VR::DS,
                multi_ds(&[
                    spec.row[0],
                    spec.row[1],
                    spec.row[2],
                    spec.column[0],
                    spec.column[1],
                    spec.column[2],
                ]),
            ),
            us(tags::SAMPLES_PER_PIXEL, 1),
            text(tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2".into()),
            us(tags::ROWS, rows as u16),
            us(tags::COLUMNS, columns as u16),
            text(tags::PIXEL_SPACING, VR::DS, multi_ds(&[spec.spacing[1], spec.spacing[0]])),
            us(tags::BITS_ALLOCATED, 16),
            us(tags::BITS_STORED, 16),
            us(tags::HIGH_BIT, 15),
            us(tags::PIXEL_REPRESENTATION, 1),
            text(tags::RESCALE_INTERCEPT, VR::DS, ds(intercept)),
            text(tags::RESCALE_SLOPE, VR::DS, ds(slope)),
            DataElement::new(tags::PIXEL_DATA, VR::OW, PrimitiveValue::from(pixels)),
        ];
        let meta = FileMetaTableBuilder::new()
            .media_storage_sop_class_uid(uids::MR_IMAGE_STORAGE)
            .media_storage_sop_instance_uid(sop_uid)
            .transfer_syntax(uids::EXPLICIT_VR_LITTLE_ENDIAN);
        let object =
            InMemDicomObject::from_element_iter(elements).with_meta(meta).map_err(std::io::Error::other)?;
        let path = dir.join(format!("slice_{k:03}.dcm"));
        object.write_to_file(&path).map_err(std::io::Error::other)?;
        files.push(path);
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_strings_fit_sixteen_characters() {
        assert_eq!(ds(0.5), "0.5");
        assert_eq!(ds(-120.0), "-120");
        assert_eq!(ds(-0.0), "0");
        assert!(ds(std::f64::consts::PI * 1000.0).len() <= 16);
        assert_eq!(ds(0.9659258262890683), "0.965925826289");
    }
}
