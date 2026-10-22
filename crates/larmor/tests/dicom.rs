//! DICOM series through the CLI: `encode` reads a series folder like the NIfTI it converts to,
//! folder searches group and skip series, `labels.tsv` and `--reference` name series folders, and
//! `watch` encodes settled scans once, logging what it does.

mod common;

use std::path::{Path, PathBuf};

use common::{SCAN_SHAPE, Subject, VOXEL_MM, Workspace, head_image, read_json, read_npy_f32, write_scan};
use larmor_dicom::synthetic::{SeriesSpec, write_series};
use larmor_volume::ndarray::Array3;

/// The synthetic head as an axial DICOM series (rows along x, columns along y, slices along z),
/// centred like the NIfTI scans of the other tests.
fn spec(series_uid: &str, series_number: i32, description: &str) -> SeriesSpec {
    let origin = SCAN_SHAPE.map(|n| -(n as f64) * VOXEL_MM / 2.0);
    // LPS: the scans' RAS x and y are negated
    let row = [-1.0, 0.0, 0.0];
    let column = [0.0, -1.0, 0.0];
    SeriesSpec {
        row,
        column,
        series_uid: series_uid.into(),
        series_number,
        description: description.into(),
        ..SeriesSpec::axial([-origin[0], -origin[1], origin[2]], [VOXEL_MM, VOXEL_MM, VOXEL_MM])
    }
}

fn write_head_series(dir: &Path, subject: &Subject, spec: &SeriesSpec) -> Vec<PathBuf> {
    write_series(dir, &head_image(subject), spec).unwrap()
}

#[test]
fn encode_reads_a_series_folder_as_dcm2niix_would_convert_it() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let subject = Subject::nth(0);
    let files = write_head_series(&ws.path("dicom/sub-01/T1w"), &subject, &spec("2.25.101", 5, "T1w MPRAGE"));
    // the NIfTI dcm2niix would write: Larmor's own conversion, saved
    let converted = larmor_dicom::convert_paths(&files).unwrap();
    converted.nifti.write(ws.path("nifti/sub-01/anat/sub-01_T1w.nii.gz")).unwrap();

    let out = ws.ok(&["encode", "dicom", "--model", "tiny", "--out", "out"]);
    assert!(out.contains("1 encoded, 0 cached, 0 failed"), "{out}");
    let dicom_dir = ws.path("out/sub-01");
    let (shape, from_dicom) = read_npy_f32(&dicom_dir.join("T1w_model-tiny_embedding.npy"));
    assert_eq!(shape, [32]);
    let run = read_json(&dicom_dir.join("T1w_model-tiny_run.json"));
    let canonical = |p: &str| std::fs::canonicalize(ws.path(p)).unwrap().display().to_string();
    assert_eq!(run["scan"]["path"], canonical("dicom/sub-01/T1w"));
    assert_eq!(run["scan"]["dicom"]["series_number"], 5);
    assert_eq!(run["scan"]["dicom"]["series_description"], "T1w MPRAGE");
    assert_eq!(run["scan"]["dicom"]["slices"], SCAN_SHAPE[2]);
    assert!(run["scan"]["dicom"]["conversion"].as_str().unwrap().contains("dcm2niix v1.0.20260724"));
    // dcm2niix flips the rows of a 2-D acquisition
    assert_eq!(run["scan"]["orientation"], "RPS");

    ws.ok(&["encode", "nifti", "--model", "tiny", "--out", "out_nifti"]);
    let (_, from_nifti) = read_npy_f32(&ws.path("out_nifti/sub-01/anat/sub-01_T1w_model-tiny_embedding.npy"));
    assert_eq!(from_dicom, from_nifti, "the series and its converted NIfTI embed differently");

    // the series hash ignores file names: renamed copies are a cache hit
    for (i, file) in files.iter().enumerate() {
        std::fs::create_dir_all(ws.path("renamed")).unwrap();
        std::fs::copy(file, ws.path(&format!("renamed/IM{:04}", 100 - i))).unwrap();
    }
    let out = ws.ok(&["encode", "renamed", "--model", "tiny", "--out", "out"]);
    assert!(out.contains("0 encoded, 1 cached"), "{out}");

    // a single file of a series is not a scan
    let err = ws.fails(&["encode", &files[0].display().to_string(), "--model", "tiny"]);
    assert!(err.contains("name the folder of its series"), "{err}");
}

#[test]
fn folder_searches_group_series_and_skip_what_is_not_one_volume() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let (a, b) = (Subject::nth(0), Subject::nth(1));
    // two series in one folder (a flat export), a localizer and a NIfTI next to them
    write_head_series(&ws.path("exam/flat"), &a, &spec("2.25.201", 3, "T1"));
    let flair = SeriesSpec { ..spec("2.25.202", 4, "FLAIR") };
    let flair_dir = ws.path("exam/flair_tmp");
    for (i, file) in write_head_series(&flair_dir, &b, &flair).iter().enumerate() {
        std::fs::rename(file, ws.path(&format!("exam/flat/flair_{i:03}.dcm"))).unwrap();
    }
    std::fs::remove_dir(&flair_dir).unwrap();
    let small = Array3::from_shape_fn((8, 8, 3), |(x, y, z)| (x + y + z) as i16);
    let axial = SeriesSpec {
        series_uid: "2.25.301".into(),
        series_number: 1,
        ..SeriesSpec::axial([0.0; 3], [2.0; 3])
    };
    write_series(&ws.path("exam/scout/ax"), &small, &axial).unwrap();
    let sagittal = SeriesSpec { row: [0.0, 1.0, 0.0], column: [0.0, 0.0, -1.0], ..axial.clone() };
    let sag = write_series(&ws.path("exam/scout/sag_tmp"), &small, &sagittal).unwrap();
    for (i, file) in sag.iter().enumerate() {
        // same series, its own SOP instance UIDs
        let mut object = dicom_object::open_file(file).unwrap();
        object.put_str(
            dicom_dictionary_std::tags::SOP_INSTANCE_UID,
            dicom_core::VR::UI,
            format!("2.25.399.{i}"),
        );
        object.write_to_file(ws.path(&format!("exam/scout/ax/sag_{i}.dcm"))).unwrap();
    }
    std::fs::remove_dir_all(ws.path("exam/scout/sag_tmp")).unwrap();
    write_scan(&ws.path("exam/brain.nii.gz"), &a);
    std::fs::write(ws.path("exam/flat/notes.txt"), "not DICOM").unwrap();

    let output = ws.command(&["encode", "exam", "--model", "tiny", "--out", "out"]).output().unwrap();
    let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "{stdout}\n{stderr}");
    assert!(stdout.contains("3 encoded, 0 cached, 0 failed"), "{stdout}");
    assert!(
        stderr.contains("skipped the DICOM series in") && stderr.contains("different orientations"),
        "{stderr}"
    );
    // series sharing a folder get their series number in their names
    for file in [
        "out/flat/flat_series-3_model-tiny_embedding.npy",
        "out/flat/flat_series-4_model-tiny_embedding.npy",
        "out/brain/brain_model-tiny_embedding.npy",
    ] {
        assert!(ws.path(file).is_file(), "{file} is missing: {:?}", common::files_below(&ws.path("out")));
    }
}

#[test]
fn labels_and_references_name_series_folders() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let mut tsv = String::from("scan\tage\n");
    for i in 0..6 {
        let subject = Subject::nth(i);
        write_head_series(
            &ws.path(&format!("data/s{i}/T1w")),
            &subject,
            &spec(&format!("2.25.5{i}"), 2, "T1w"),
        );
        tsv.push_str(&format!("data/s{i}/T1w\t{}\n", 30 + 5 * i));
    }
    std::fs::write(ws.path("labels.tsv"), tsv).unwrap();
    let out = ws.ok(&[
        "probe",
        "fit",
        "labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "age",
        "--encode",
    ]);
    assert!(out.contains("deployable probe refit on all 6 scans"), "{out}");

    // a row naming a folder with two series is refused with the line
    write_head_series(&ws.path("data/s0/T1w_b"), &Subject::nth(7), &spec("2.25.599", 9, "T1w"));
    for entry in std::fs::read_dir(ws.path("data/s0/T1w_b")).unwrap() {
        let path = entry.unwrap().path();
        std::fs::rename(
            &path,
            ws.path("data/s0/T1w").join(format!("b_{}", path.file_name().unwrap().to_string_lossy())),
        )
        .unwrap();
    }
    let err = ws.fails(&[
        "probe",
        "fit",
        "labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "age",
    ]);
    assert!(err.contains("labels.tsv line 2") && err.contains("holds 2 scans"), "{err}");

    // a second modality on a series' head mask and grid
    let flair = spec("2.25.777", 6, "FLAIR");
    write_series(&ws.path("data/s1/FLAIR"), &head_image(&Subject::nth(1)).mapv(|v| v / 2 + 30), &flair)
        .unwrap();
    let out =
        ws.ok(&["encode", "data/s1/FLAIR", "--reference", "data/s1/T1w", "--model", "tiny", "--out", "out"]);
    assert!(out.contains("on the head mask of") && out.contains("1 encoded"), "{out}");
    let run = read_json(&ws.path("out/FLAIR/FLAIR_model-tiny_run.json"));
    let canonical = std::fs::canonicalize(ws.path("data/s1/T1w")).unwrap();
    assert_eq!(run["reference"]["path"], canonical.display().to_string());
}

fn write_site(ws: &Workspace, settle: f64) -> PathBuf {
    let site = ws.path("site.toml");
    std::fs::write(
        &site,
        format!(
            "[[source]]\nname = \"box\"\npath = \"incoming\"\n\n[output]\ndir = \"derivatives\"\n\n[encode]\n\
             models = [\"tiny\"]\ndevice = \"cpu\"\n\n[watch]\npoll_seconds = 0.2\nsettle_seconds = {settle}\n"
        ),
    )
    .unwrap();
    site
}

#[test]
fn watch_encodes_settled_scans_once_and_logs_every_action() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let site = write_site(&ws, 0.0).display().to_string();
    write_scan(&ws.path("incoming/sub-01/anat/sub-01_T1w.nii.gz"), &Subject::nth(0));
    write_head_series(&ws.path("incoming/exam2/T1_MPRAGE"), &Subject::nth(1), &spec("2.25.901", 3, "T1"));

    let out = ws.ok(&["watch", "--site", &site, "--once"]);
    assert!(out.contains("encoded sub-01/anat/sub-01_T1w.nii.gz with tiny"), "{out}");
    assert!(out.contains("encoded exam2/T1_MPRAGE with tiny") && out.contains("done (--once)"), "{out}");
    for file in [
        "derivatives/box/sub-01/anat/sub-01_T1w_model-tiny_embedding.npy",
        "derivatives/box/exam2/T1_MPRAGE_model-tiny_embedding.npy",
        "derivatives/box/dataset_description.json",
        "derivatives/.larmor-watch.json",
    ] {
        assert!(
            ws.path(file).is_file(),
            "{file} missing: {:?}",
            common::files_below(&ws.path("derivatives"))
        );
    }
    let log = std::fs::read_to_string(ws.path("derivatives/larmor-watch.log")).unwrap();
    assert!(log.contains("settled; encoding") && log.contains("encoded exam2/T1_MPRAGE"), "{log}");

    // a restart finds nothing new
    let out = ws.ok(&["watch", "--site", &site, "--once"]);
    assert!(!out.contains("encoding"), "{out}");

    // a changed file is encoded again; a deleted output is written again from the cache
    write_scan(&ws.path("incoming/sub-01/anat/sub-01_T1w.nii.gz"), &Subject::nth(2));
    std::fs::remove_file(ws.path("derivatives/box/exam2/T1_MPRAGE_model-tiny_run.json")).unwrap();
    let out = ws.ok(&["watch", "--site", &site, "--once"]);
    assert!(out.contains("encoded sub-01/anat/sub-01_T1w.nii.gz"), "{out}");
    assert!(out.contains("cached exam2/T1_MPRAGE"), "{out}");
}

#[test]
fn watch_waits_for_a_series_to_settle() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let site = write_site(&ws, 1.5).display().to_string();
    let files = write_head_series(&ws.path("staging/T1"), &Subject::nth(0), &spec("2.25.902", 3, "T1"));
    let target = ws.path("incoming/exam/T1");
    std::fs::create_dir_all(&target).unwrap();
    // half the series is there when the watcher starts; the rest arrives while it settles
    let half = files.len() / 2;
    for file in &files[..half] {
        std::fs::copy(file, target.join(file.file_name().unwrap())).unwrap();
    }
    let mut child =
        ws.command(&["watch", "--site", &site]).stdout(std::process::Stdio::piped()).spawn().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(800));
    for file in &files[half..] {
        std::fs::copy(file, target.join(file.file_name().unwrap())).unwrap();
    }
    let embedding = ws.path("derivatives/box/exam/T1_model-tiny_embedding.npy");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !embedding.is_file() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    child.kill().unwrap();
    let _ = child.wait();
    assert!(embedding.is_file(), "nothing encoded");
    let run = read_json(&ws.path("derivatives/box/exam/T1_model-tiny_run.json"));
    // the whole series, not the half that was there first
    assert_eq!(run["scan"]["dicom"]["slices"], files.len());
    let log = std::fs::read_to_string(ws.path("derivatives/larmor-watch.log")).unwrap();
    assert_eq!(log.matches("settled; encoding").count(), 1, "{log}");
}
