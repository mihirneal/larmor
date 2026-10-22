//! One series' files in slice order, and the NIfTI geometry dcm2niix gives them.
//!
//! This is `saveDcm2NiiCore` of `nii_dicom_batch.cpp` for one 3-D volume of classic (one slice per
//! file) DICOM: slices sorted by InstanceNumber, re-sorted by position if that order is not
//! spatial (`ensureSequentialSlicePositions`), the slice spacing measured between the first two
//! positions, the sform from the first slice (`headerDcm2Nii2`), the slices reversed when they
//! run against the slice normal (`nii_flipZ`), then either a 3-D acquisition turned to its nearest
//! orthogonal orientation (`nii_setOrtho`) or the rows flipped (`nii_flipY`). What dcm2niix
//! converts into something other than one volume (localizers, 4-D series, slice spacings NIfTI
//! cannot hold, gantry tilt) is refused with the reason.

use crate::error::DicomError;
use crate::geometry::{
    self, Mat44, interslice_distance, interslice_distance_signed, is_same_float, is_same_float_ge,
};
use crate::header::{Manufacturer, SliceHeader};

/// dcm2niix's `kSliceTolerance`: slice steps that differ by more than this (mm) vary.
const SLICE_TOLERANCE: f32 = 0.2;

/// How the stored samples become NIfTI voxels (`headerDcm2Nii`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Storage {
    /// 8-bit samples.
    U8,
    /// Signed 16-bit samples, or unsigned ones with fewer than 16 bits stored (dcm2niix stores those
    /// as int16, the same bytes); `mask12` clears the bits above 12 (`nii_mask12bit`), keeping
    /// the sign for signed samples.
    I16 { mask12: Option<bool> },
    /// Unsigned 16-bit samples with all 16 bits stored.
    U16,
    /// 32-bit integer samples (stored as int32 whatever their sign, like dcm2niix).
    I32,
}

/// How voxel values are scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Scaling {
    /// One slope and intercept for every slice, in the NIfTI header.
    Header { slope: f32, intercept: f32 },
    /// Slopes or intercepts differ between slices: every slice is scaled into float32 voxels
    /// (`nii_iVaries`).
    PerSlice,
}

/// What follows the slice order: dcm2niix's last reorientation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reorient {
    /// Rows reversed (`nii_flipY`): 2-D and echo-planar acquisitions.
    FlipY,
    /// Axes permuted and reversed (`nii_setOrtho`): 3-D acquisitions; see [`geometry::Ortho`].
    Ortho([i32; 3]),
    /// A 3-D acquisition already in its best orthogonal orientation.
    None,
}

/// A series' slices in stacking order and the NIfTI image dcm2niix makes of them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Plan {
    /// Indices into the headers, in the order dcm2niix stacks the slices.
    pub(crate) order: Vec<usize>,
    /// Columns and rows of every slice.
    pub(crate) columns: usize,
    pub(crate) rows: usize,
    pub(crate) storage: Storage,
    pub(crate) scaling: Scaling,
    /// Each slice's slope and intercept, in stacking order (used with [`Scaling::PerSlice`]).
    pub(crate) slice_scaling: Vec<(f32, f32)>,
    /// The stack is reversed (`nii_flipZ`).
    pub(crate) flip_z: bool,
    pub(crate) reorient: Reorient,
    /// The final voxel-to-RAS sform, float32 like the NIfTI header.
    pub(crate) sform: Mat44,
    /// The final header voxel sizes (`pixdim[1..3]`).
    pub(crate) pixdim: [f32; 3],
    /// The final array shape.
    pub(crate) shape: [usize; 3],
    /// Things worth telling the user (duplicate files dropped, instance order not spatial).
    pub(crate) notes: Vec<String>,
}

/// `series 7 "T1_MPRAGE"`, for messages.
pub(crate) fn describe(h: &SliceHeader) -> String {
    let name = if h.series_description.is_empty() { &h.protocol_name } else { &h.series_description };
    match (h.series_number, name.is_empty()) {
        (Some(n), false) => format!("series {n} \"{name}\""),
        (Some(n), true) => format!("series {n}"),
        (None, false) => format!("series \"{name}\""),
        (None, true) => format!("series {}", h.series_instance_uid),
    }
}

fn unsupported(h: &SliceHeader, reason: impl std::fmt::Display) -> DicomError {
    DicomError::Unsupported(format!("{}: {reason}", describe(h)))
}

/// The slice order and geometry of one series, from its files' headers (all with one
/// SeriesInstanceUID). Fails with the reason when the series is not one 3-D volume of classic
/// DICOM that dcm2niix would write as one NIfTI image.
pub(crate) fn plan(headers: &[SliceHeader]) -> Result<Plan, DicomError> {
    let first = headers.first().ok_or_else(|| DicomError::Unsupported("no DICOM files".into()))?;
    let mut notes = Vec::new();

    // copies of one file (same SOPInstanceUID) count once
    let mut seen = std::collections::HashSet::new();
    let mut kept: Vec<usize> = Vec::with_capacity(headers.len());
    for (i, h) in headers.iter().enumerate() {
        if h.sop_instance_uid.is_empty() || seen.insert(h.sop_instance_uid.as_str()) {
            kept.push(i);
        }
    }
    if kept.len() < headers.len() {
        let n = headers.len() - kept.len();
        notes.push(format!(
            "{n} duplicate file{} (same SOPInstanceUID) ignored",
            if n == 1 { "" } else { "s" }
        ));
    }

    for &i in &kept {
        check_slice(&headers[i])?;
    }
    check_consistent(headers, &kept)?;
    if kept.len() < 2 {
        return Err(unsupported(first, "a single 2-D slice, not a 3-D volume"));
    }

    // dcm2niix's sort key: (SeriesNumber << 32) + InstanceNumber, then the acquisition time in ms
    let key = |h: &SliceHeader| -> (u64, u32) {
        let series = h.series_number.unwrap_or(1) as i64 as u64;
        let img = (series << 32).wrapping_add(h.instance_number as i64 as u64);
        (img, acquisition_ms(h.acquisition_time))
    };
    let mut order = kept;
    order.sort_by(|&a, &b| {
        key(&headers[a]).cmp(&key(&headers[b])).then(headers[a].path.cmp(&headers[b].path))
    });
    // hdr0 (datatype, header slope and intercept, first pixdim[3]) comes from the first file in
    // instance order, before any re-sort by position
    let hdr0 = &headers[order[0]];

    let position = |i: usize| headers[i].position.expect("checked");
    let at_first = order.iter().filter(|&&i| same_position(position(i), position(order[0]))).count();
    if at_first > 1 {
        return Err(unsupported(
            hdr0,
            format!(
                "{} files at one slice position: a 4-D series (fMRI, diffusion, several echoes or repeats) \
                 rather than one 3-D volume",
                at_first
            ),
        ));
    }

    let orientation = headers[order[0]].orientation.expect("checked");
    if !is_sequential(&order, &position, orientation) {
        // ensureSequentialSlicePositions: by signed distance from the first slice, ties by index
        let origin = position(order[0]);
        let mut by_position: Vec<(f32, usize, usize)> = order
            .iter()
            .enumerate()
            .map(|(index, &i)| (interslice_distance_signed(origin, position(i), orientation), index, i))
            .collect();
        by_position.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        order = by_position.into_iter().map(|(_, _, i)| i).collect();
        notes.push("InstanceNumber order is not spatial; slices sorted by position".into());
    }

    let dx = interslice_distance(position(order[0]), position(order[1]));
    let steps: Vec<f32> =
        order.windows(2).map(|w| interslice_distance(position(w[0]), position(w[1]))).collect();
    if steps.iter().any(|&step| is_same_float_ge(step, 0.0)) {
        return Err(unsupported(
            hdr0,
            "two slices share a position: a 4-D series rather than one 3-D volume",
        ));
    }
    if steps.iter().any(|&step| (dx - step).abs() > SLICE_TOLERANCE) {
        let (lo, hi) = steps.iter().fold((f32::INFINITY, 0f32), |(lo, hi), &s| (lo.min(s), hi.max(s)));
        return Err(unsupported(
            hdr0,
            format!(
                "the slice spacing varies from {lo:.3} to {hi:.3} mm (missing slices?), which a NIfTI image \
                 cannot represent; dcm2niix would write it with a wrong geometry"
            ),
        ));
    }

    // pixdim[3]: SpacingBetweenSlices, else SliceThickness, else 1; the measured spacing when they
    // differ by more than 0.0001 mm. The sform always uses the measured spacing.
    let header_spacing = match hdr0.spacing_between_slices {
        Some(z) if z > 0.0 => z,
        _ => hdr0.slice_thickness.unwrap_or(1.0),
    };
    let pixdim_z = if dx > 0.0 && !is_same_float_ge(dx, header_spacing) { dx } else { header_spacing };

    let s0 = &headers[order[0]];
    let s1 = &headers[order[1]];
    let [column_mm, row_mm] = s0.pixel_spacing.expect("checked");
    let mut q =
        geometry::dicom2mat(s0.orientation.expect("checked"), position(order[0]), [column_mm, row_mm, dx]);
    let slice_dir =
        geometry::verify_slice_dir(position(order[0]), position(order[1]), order.len(), &mut q)
            .ok_or_else(|| unsupported(s1, "the first two slices do not move along the slice axis"))?;
    geometry::lps_to_ras(&mut q);
    let mut sform = q;
    let flip_z = slice_dir < 0;
    if flip_z {
        sform = geometry::flip_z(&sform, order.len());
    }

    let columns = usize::from(s0.columns);
    let rows = usize::from(s0.rows);
    let mut shape = [columns, rows, order.len()];
    let mut pixdim = [column_mm, row_mm, pixdim_z];
    let reorient = if s0.acquisition_3d && !s0.epi {
        match geometry::set_ortho(&sform, shape) {
            Some(ortho) => {
                let source = |i: usize| ortho.axes[i].unsigned_abs() as usize - 1;
                shape = std::array::from_fn(|i| shape[source(i)]);
                pixdim = std::array::from_fn(|i| pixdim[source(i)]);
                sform = ortho.sform;
                Reorient::Ortho(ortho.axes)
            }
            None => Reorient::None,
        }
    } else {
        sform = geometry::flip_y(&sform, rows);
        Reorient::FlipY
    };

    // headerDcm2Nii's datatype, from the first file in instance order
    let storage = match (hdr0.bits_allocated, hdr0.signed) {
        (8, _) => Storage::U8,
        (16, signed) if signed || (hdr0.bits_stored > 0 && hdr0.bits_stored < 16) => {
            Storage::I16 { mask12: (hdr0.bits_stored == 12).then_some(signed) }
        }
        (16, _) => Storage::U16,
        (32, _) => Storage::I32,
        (bits, _) => return Err(unsupported(hdr0, format!("{bits}-bit samples are not supported"))),
    };

    let slice_scaling: Vec<(f32, f32)> =
        order.iter().map(|&i| (headers[i].rescale_slope, headers[i].rescale_intercept)).collect();
    // intensityScaleVaries, over the slices in instance order
    let varies = order.iter().map(|&i| &headers[i]).any(|h| {
        (h.rescale_slope - hdr0.rescale_slope).abs() > f32::EPSILON
            || (h.rescale_intercept - hdr0.rescale_intercept).abs() > f32::EPSILON
    });
    let scaling = if varies {
        if hdr0.manufacturer == Manufacturer::Philips {
            return Err(unsupported(
                hdr0,
                "RescaleSlope or RescaleIntercept varies between slices of a Philips series, which Larmor does \
                 not convert",
            ));
        }
        Scaling::PerSlice
    } else {
        let (slope, intercept) = philips_precise(s0, hdr0.rescale_slope, hdr0.rescale_intercept);
        Scaling::Header { slope, intercept }
    };

    Ok(Plan {
        order,
        columns,
        rows,
        storage,
        scaling,
        slice_scaling,
        flip_z,
        reorient,
        sform,
        pixdim,
        shape,
        notes,
    })
}

/// A slice Larmor can stack: one frame of grey levels with its position, orientation and spacing.
fn check_slice(h: &SliceHeader) -> Result<(), DicomError> {
    if h.is_mosaic() {
        return Err(unsupported(h, "a Siemens mosaic (fMRI or diffusion), not an anatomical volume"));
    }
    if h.enhanced || h.frames > 1 {
        return Err(unsupported(
            h,
            "an enhanced (multi-frame) DICOM file, which Larmor does not read yet; export the series as \
             classic DICOM (one slice per file) or convert it with dcm2niix",
        ));
    }
    if h.rows == 0 || h.columns == 0 {
        return Err(unsupported(h, format!("{}: no image (Rows and Columns are missing)", h.path.display())));
    }
    if h.samples_per_pixel != 1 {
        return Err(unsupported(h, "a colour image (SamplesPerPixel is not 1)"));
    }
    let missing = [
        (h.position.is_none(), "ImagePositionPatient (0020,0032)"),
        (h.orientation.is_none(), "ImageOrientationPatient (0020,0037)"),
        (h.pixel_spacing.is_none(), "PixelSpacing (0028,0030)"),
    ];
    if let Some((_, what)) = missing.iter().find(|(m, _)| *m) {
        return Err(unsupported(
            h,
            format!("{} has no {what}, so it is not a spatial image", h.path.display()),
        ));
    }
    if let Some(problem) = crate::pixels::transfer_syntax_problem(&h.transfer_syntax) {
        return Err(unsupported(h, problem));
    }
    if !is_same_float_ge(h.gantry_tilt, 0.0) {
        return Err(unsupported(
            h,
            format!("gantry tilt of {} degrees, which Larmor does not correct", h.gantry_tilt),
        ));
    }
    Ok(())
}

/// Every slice of a volume has the same size, sample format, spacing, orientation and echo.
fn check_consistent(headers: &[SliceHeader], kept: &[usize]) -> Result<(), DicomError> {
    let first = &headers[kept[0]];
    for &i in &kept[1..] {
        let h = &headers[i];
        let differs = |what: &str| {
            unsupported(
                first,
                format!(
                    "{what} differs between {} and {}; Larmor needs one 3-D volume",
                    first.path.display(),
                    h.path.display()
                ),
            )
        };
        if (h.rows, h.columns) != (first.rows, first.columns) {
            return Err(differs("the image size"));
        }
        if (h.bits_allocated, h.bits_stored, h.signed)
            != (first.bits_allocated, first.bits_stored, first.signed)
        {
            return Err(differs("the sample format"));
        }
        let (a, b) = (h.pixel_spacing.expect("checked"), first.pixel_spacing.expect("checked"));
        if !(is_same_float(a[0], b[0]) && is_same_float(a[1], b[1])) {
            return Err(differs("PixelSpacing"));
        }
        let (a, b) = (h.orientation.expect("checked"), first.orientation.expect("checked"));
        if !(0..6).all(|k| is_same_float_ge(a[k], b[k])) {
            return Err(unsupported(
                first,
                "its slices have different orientations (a localizer?); Larmor needs one 3-D volume",
            ));
        }
        if h.echo_number != first.echo_number {
            return Err(unsupported(
                first,
                "it holds more than one echo (EchoNumbers differ); encode a combined or single-echo series",
            ));
        }
        if h.acquisition_3d != first.acquisition_3d {
            return Err(differs("MRAcquisitionType"));
        }
    }
    Ok(())
}

/// Within 0.0001 mm on every axis.
fn same_position(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|k| is_same_float_ge(a[k], b[k]))
}

/// The first pass of `ensureSequentialSlicePositions`: every step along the slice normal equals
/// the first one within 0.2 mm.
fn is_sequential(order: &[usize], position: &impl Fn(usize) -> [f32; 3], orientation: [f32; 6]) -> bool {
    if order.len() < 3 {
        return true;
    }
    let step = |a: usize, b: usize| interslice_distance_signed(position(a), position(b), orientation);
    let first = step(order[0], order[1]);
    order.windows(2).all(|w| (first - step(w[0], w[1])).abs() <= SLICE_TOLERANCE)
}

/// `fillTDCMsort`: AcquisitionTime `HHMMSS.FFFF` as milliseconds since midnight.
fn acquisition_ms(time: f64) -> u32 {
    let hours = (time / 10000.0).trunc();
    let rest = time - hours * 10000.0;
    let minutes = (rest / 100.0).trunc();
    let seconds = rest - minutes * 100.0;
    let ms = (seconds * 1000.0).round() + hours * 3_600_000.0 + minutes * 60_000.0;
    ms.clamp(0.0, f64::from(u32::MAX)) as u32
}

/// `PhilipsPrecise`: Philips' precise values `P = (R * RS + RI) / (RS * SS)` replace the displayed
/// ones when the series has a ScaleSlope (dcm2niix's default `-p y`).
fn philips_precise(first: &SliceHeader, slope: f32, intercept: f32) -> (f32, f32) {
    let (Manufacturer::Philips, Some(scale)) = (first.manufacturer, first.philips_scale_slope) else {
        return (slope, intercept);
    };
    if scale == 0.0 {
        return (slope, intercept);
    }
    let (rs, ri) = (first.rescale_slope, first.rescale_intercept);
    let precise = |value: f32| if rs * scale == 0.0 { 0.0 } else { value.mul_add(rs, ri) / (rs * scale) };
    let (l0, l1) = (precise(0.0), precise(1.0));
    let (precise_slope, precise_intercept) = if l0 != l1 { (l1 - l0, l0) } else { (rs, ri) };
    if is_same_float(ri, precise_intercept) && is_same_float(rs, precise_slope) {
        return (slope, intercept);
    }
    (precise_slope, precise_intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquisition_times_become_milliseconds() {
        assert_eq!(acquisition_ms(0.0), 0);
        assert_eq!(acquisition_ms(102030.5), (10 * 3600 + 20 * 60 + 30) * 1000 + 500);
    }
}
