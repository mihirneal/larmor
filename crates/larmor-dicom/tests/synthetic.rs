//! Conversion of synthetic series: every voxel lands where its DICOM pixel is in space, for
//! every orientation and slice order, and what is not one 3-D volume is refused with the reason.
//! (Bit-exactness against dcm2niix itself is checked on real series by `larmor check-parity`.)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use larmor_dicom::synthetic::{SeriesSpec, write_series};
use larmor_dicom::{Converted, DicomError, SliceHeader, convert_paths, group_series, series_sha256};
use larmor_volume::ndarray::{Array3, Ix3};
use larmor_volume::{VoxelData, aff2axcodes};

/// A volume whose every voxel has its own value, so a converted voxel identifies its source.
fn unique_volume(shape: [usize; 3]) -> Array3<i16> {
    Array3::from_shape_fn(shape, |(c, r, k)| (1 + c + shape[0] * (r + shape[1] * k)) as i16)
}

fn convert(files: &[PathBuf]) -> Converted {
    convert_paths(files).unwrap_or_else(|e| panic!("{e}"))
}

fn refused(files: &[PathBuf]) -> String {
    match convert_paths(files) {
        Ok(_) => panic!("conversion should have failed"),
        Err(DicomError::Unsupported(message)) => message,
        Err(other) => panic!("unexpected error {other}"),
    }
}

/// Every converted voxel's RAS position (its affine applied to its index) is the RAS position of
/// the DICOM pixel holding its value.
fn assert_voxels_in_place(converted: &Converted, volume: &Array3<i16>, spec: &SeriesSpec) {
    let VoxelData::I16(data) = &converted.nifti.data else { panic!("expected int16") };
    let data = data.view().into_dimensionality::<Ix3>().unwrap();
    let mut source = HashMap::new();
    for ((c, r, k), &v) in volume.indexed_iter() {
        source.insert(v, (c, r, k));
    }
    let affine = converted.nifti.affine().unwrap();
    assert_eq!(data.len(), volume.len());
    for ((i, j, k), v) in data.indexed_iter() {
        let (c, r, slice) = source[v];
        let ipp = spec.position(slice);
        let lps: [f64; 3] = std::array::from_fn(|a| {
            ipp[a] + c as f64 * spec.spacing[0] * spec.row[a] + r as f64 * spec.spacing[1] * spec.column[a]
        });
        let expected = [-lps[0], -lps[1], lps[2]];
        let got = affine.apply([i as f64, j as f64, k as f64]);
        for a in 0..3 {
            assert!((got[a] - expected[a]).abs() < 1e-3, "voxel {:?}: {got:?} vs {expected:?}", (i, j, k));
        }
    }
}

fn rotation(axis: usize, degrees: f64) -> [[f64; 3]; 3] {
    let (s, c) = degrees.to_radians().sin_cos();
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut m = [[0.0; 3]; 3];
    m[axis][axis] = 1.0;
    m[a][a] = c;
    m[b][b] = c;
    m[a][b] = -s;
    m[b][a] = s;
    m
}

fn rotate(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| m[i][j] * v[j]).sum())
}

#[test]
fn an_axial_2d_series_flips_its_rows_like_dcm2niix() {
    let dir = tempfile::tempdir().unwrap();
    let volume = unique_volume([6, 5, 4]);
    let spec = SeriesSpec::axial([-100.0, -120.0, -50.0], [0.5, 0.75, 2.0]);
    let files = write_series(dir.path(), &volume, &spec).unwrap();
    let converted = convert(&files);
    let nifti = &converted.nifti;
    assert_eq!(nifti.shape(), [6, 5, 4]);
    assert_eq!(nifti.zooms(), [0.5, 0.75, 2.0]);
    assert_eq!((nifti.header.sform_code, nifti.header.qform_code), (1, 1));
    // LPS to RAS negates x and y; the row flip moves the origin to the last row
    assert_eq!(
        nifti.affine().unwrap().0,
        [[-0.5, 0.0, 0.0, 100.0], [0.0, 0.75, 0.0, 117.0], [0.0, 0.0, 2.0, -50.0], [0.0, 0.0, 0.0, 1.0]]
    );
    assert_eq!(aff2axcodes(&nifti.affine().unwrap()).unwrap(), "LAS");
    assert_voxels_in_place(&converted, &volume, &spec);
    assert_eq!(converted.series.slices, 4);
    assert_eq!(converted.series.series_number, Some(3));
    assert!(converted.notes.is_empty(), "{:?}", converted.notes);
}

#[test]
fn instance_order_and_file_names_do_not_change_the_image() {
    let volume = unique_volume([6, 5, 7]);
    let spec = SeriesSpec::axial([-10.0, -20.0, -30.0], [1.0, 1.0, 3.0]);
    let base = tempfile::tempdir().unwrap();
    let reference = convert(&write_series(base.path(), &volume, &spec).unwrap());

    // descending positions in instance order: dcm2niix reverses the stack (nii_flipZ)
    let dir = tempfile::tempdir().unwrap();
    let descending = SeriesSpec { instance_numbers: Some((1..=7).rev().collect()), ..spec.clone() };
    let flipped = convert(&write_series(dir.path(), &volume, &descending).unwrap());
    assert_eq!(flipped.nifti.data, reference.nifti.data);
    assert_eq!(flipped.nifti.affine().unwrap(), reference.nifti.affine().unwrap());

    // a non-spatial instance order is re-sorted by position, with a note
    let dir = tempfile::tempdir().unwrap();
    let shuffled = SeriesSpec { instance_numbers: Some(vec![3, 1, 7, 2, 6, 4, 5]), ..spec.clone() };
    let sorted = convert(&write_series(dir.path(), &volume, &shuffled).unwrap());
    assert_eq!(sorted.nifti.data, reference.nifti.data);
    assert_eq!(sorted.nifti.affine().unwrap(), reference.nifti.affine().unwrap());
    assert!(sorted.notes.iter().any(|n| n.contains("not spatial")), "{:?}", sorted.notes);

    // a copy of a file under another name is ignored
    let dir = tempfile::tempdir().unwrap();
    let mut files = write_series(dir.path(), &volume, &spec).unwrap();
    let copy = dir.path().join("copy_of_slice_2.dcm");
    std::fs::copy(&files[2], &copy).unwrap();
    files.push(copy);
    let deduplicated = convert(&files);
    assert_eq!(deduplicated.nifti.data, reference.nifti.data);
    assert!(deduplicated.notes.iter().any(|n| n.contains("duplicate")), "{:?}", deduplicated.notes);
}

#[test]
fn every_orientation_puts_voxels_in_place() {
    let volume = unique_volume([5, 4, 6]);
    // axial, sagittal, coronal and oblique slices, as 2-D (row flip) and 3-D (reoriented)
    // acquisitions, stacked with and against the slice normal
    let orientations: [([f64; 3], [f64; 3]); 4] = [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [0.0, 0.0, -1.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        (
            rotate(rotation(2, 12.0), rotate(rotation(0, 20.0), [1.0, 0.0, 0.0])),
            rotate(rotation(2, 12.0), rotate(rotation(0, 20.0), [0.0, 1.0, 0.0])),
        ),
    ];
    for (row, column) in orientations {
        for acquisition in ["2D", "3D"] {
            for step in [2.5, -2.5] {
                let dir = tempfile::tempdir().unwrap();
                let spec = SeriesSpec {
                    row,
                    column,
                    acquisition: acquisition.into(),
                    ..SeriesSpec::axial([12.0, -30.0, 41.0], [1.25, 0.8, step])
                };
                let converted = convert(&write_series(dir.path(), &volume, &spec).unwrap());
                assert_voxels_in_place(&converted, &volume, &spec);
                if acquisition == "3D" {
                    // dcm2niix turns 3-D acquisitions to the nearest orthogonal RAS orientation
                    let affine = converted.nifti.affine().unwrap();
                    assert_eq!(aff2axcodes(&affine).unwrap(), "RAS", "{row:?} {column:?} {step}");
                }
            }
        }
    }
}

#[test]
fn varying_scaling_becomes_float_voxels() {
    let dir = tempfile::tempdir().unwrap();
    let volume = unique_volume([4, 3, 3]);
    let spec = SeriesSpec {
        rescale: vec![(2.0, 1.0), (0.5, -3.0), (1.0, 0.0)],
        ..SeriesSpec::axial([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    };
    let converted = convert(&write_series(dir.path(), &volume, &spec).unwrap());
    let VoxelData::F32(data) = &converted.nifti.data else { panic!("expected float32 voxels") };
    let data = data.view().into_dimensionality::<Ix3>().unwrap();
    assert_eq!((converted.nifti.header.scl_slope, converted.nifti.header.scl_inter), (1.0, 0.0));
    // slice 1 (rows flipped): value * 0.5 - 3
    let raw = f32::from(volume[[2, 0, 1]]);
    assert_eq!(data[[2, 2, 1]], raw * 0.5 - 3.0);

    let dir = tempfile::tempdir().unwrap();
    let scaled = SeriesSpec { rescale: vec![(0.25, 10.0)], ..spec.clone() };
    let converted = convert(&write_series(dir.path(), &volume, &scaled).unwrap());
    assert_eq!(converted.nifti.data.dtype().name(), "int16");
    assert_eq!((converted.nifti.header.scl_slope, converted.nifti.header.scl_inter), (0.25, 10.0));
}

#[test]
fn what_is_not_one_volume_is_refused_with_the_reason() {
    let volume = unique_volume([4, 4, 6]);
    let spec = SeriesSpec::axial([0.0, 0.0, 0.0], [1.0, 1.0, 2.0]);

    // a missing slice: the spacing varies
    let dir = tempfile::tempdir().unwrap();
    let mut files = write_series(dir.path(), &volume, &spec).unwrap();
    files.remove(3);
    let message = refused(&files);
    assert!(message.contains("slice spacing varies from 2.000 to 4.000 mm"), "{message}");

    // two volumes at the same positions: 4-D
    let dir = tempfile::tempdir().unwrap();
    let mut positions: Vec<[f64; 3]> = (0..3).map(|k| spec.position(k)).collect();
    positions.extend((0..3).map(|k| spec.position(k)));
    let repeated = SeriesSpec { positions: Some(positions), ..spec.clone() };
    let message = refused(&write_series(dir.path(), &volume, &repeated).unwrap());
    assert!(message.contains("4-D series"), "{message}");

    // one slice
    let dir = tempfile::tempdir().unwrap();
    let files = write_series(dir.path(), &volume, &spec).unwrap();
    let message = refused(&files[..1]);
    assert!(message.contains("single 2-D slice"), "{message}");

    // two series
    let dir = tempfile::tempdir().unwrap();
    let mut files = write_series(&dir.path().join("a"), &volume, &spec).unwrap();
    let other = SeriesSpec { series_uid: "2.25.99".into(), series_number: 4, ..spec.clone() };
    files.extend(write_series(&dir.path().join("b"), &volume, &other).unwrap());
    let message = refused(&files);
    assert!(message.contains("different series"), "{message}");
}

#[test]
fn a_localizer_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let volume = unique_volume([4, 4, 3]);
    let spec = SeriesSpec::axial([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let mut files = write_series(&dir.path().join("ax"), &volume, &spec).unwrap();
    let sagittal = SeriesSpec { row: [0.0, 1.0, 0.0], column: [0.0, 0.0, -1.0], ..spec.clone() };
    let mut extra = write_series(&dir.path().join("sag"), &volume, &sagittal).unwrap();
    // same series, other files: give them their own SOP instance UIDs by renumbering the instances
    for (i, path) in extra.iter_mut().enumerate() {
        let target = dir.path().join(format!("sag_{i}.dcm"));
        let mut object = dicom_object::open_file(&path).unwrap();
        let uid = format!("2.25.777.{i}");
        object.put_str(dicom_dictionary_std::tags::SOP_INSTANCE_UID, dicom_core::VR::UI, uid.clone());
        object.write_to_file(&target).unwrap();
        *path = target;
    }
    files.extend(extra);
    let message = refused(&files);
    assert!(message.contains("different orientations"), "{message}");
}

#[test]
fn series_are_hashed_by_content() {
    let dir = tempfile::tempdir().unwrap();
    let volume = unique_volume([4, 4, 3]);
    let spec = SeriesSpec::axial([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let files = write_series(dir.path(), &volume, &spec).unwrap();
    let series = group_series(files.iter().map(|p| SliceHeader::read(p).unwrap()).collect());
    assert_eq!(series.len(), 1);
    assert_eq!(series[0].label(), "series 3 \"T1w synthetic\"");
    assert_eq!(series[0].folder(), dir.path());
    series[0].check().unwrap();

    // file order and names do not matter, content does
    let bytes: Vec<Vec<u8>> = files.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let forward = series_sha256(bytes.iter().map(Vec::as_slice));
    let backward = series_sha256(bytes.iter().rev().map(Vec::as_slice));
    assert_eq!(forward, backward);
    assert_ne!(forward, series_sha256(bytes[..2].iter().map(Vec::as_slice)));
}

#[test]
fn a_folder_of_two_series_groups_into_two() {
    let dir = tempfile::tempdir().unwrap();
    let volume = unique_volume([4, 4, 3]);
    let spec = SeriesSpec::axial([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    write_series(&dir.path().join("t1"), &volume, &spec).unwrap();
    let flair = SeriesSpec { series_uid: "2.25.42".into(), series_number: 9, ..spec };
    write_series(&dir.path().join("flair"), &volume, &flair).unwrap();
    let headers = read_tree(dir.path());
    let series = group_series(headers);
    assert_eq!(series.len(), 2);
    assert_eq!(series[0].folder(), dir.path().join("flair"));
    assert_eq!(series[1].folder(), dir.path().join("t1"));
}

fn read_tree(dir: &Path) -> Vec<SliceHeader> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            out.extend(read_tree(&path));
        } else if larmor_dicom::is_dicom_file(&path) {
            out.push(SliceHeader::read(&path).unwrap());
        }
    }
    out
}
