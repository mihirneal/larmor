//! The `larmor` binary on the real reference fixtures (`$LARMOR_FIXTURES`, default
//! `/Users/mihir/Developer/larmor/fixtures`): parity on every scan, then encode and the three probe
//! kinds on `fixtures/scans`. They read full-size scans and run the ViT-S reference model, so they
//! are ignored by default:
//!
//! ```text
//! cargo test --release -p larmor --test fixtures -- --ignored --nocapture --test-threads 1
//! ```

mod common;

use std::path::PathBuf;

use common::{Workspace, read_json, read_nifti};
use larmor_volume::ndarray::Array3;
use larmor_volume::{Affine, Nifti, VoxelData};

const MODEL: &str = "nanobrain-vits-p8-ref";

fn fixtures_root() -> PathBuf {
    std::env::var_os("LARMOR_FIXTURES")
        .map_or_else(|| PathBuf::from("/Users/mihir/Developer/larmor/fixtures"), PathBuf::from)
}

/// `(name, file, modality)` of every scan in the fixture manifest.
fn fixture_scans() -> Vec<(String, PathBuf, String)> {
    let root = fixtures_root();
    let manifest: toml::Table = std::fs::read_to_string(root.join("manifest.toml")).unwrap().parse().unwrap();
    manifest["scan"]
        .as_array()
        .unwrap()
        .iter()
        .map(|scan| {
            let field = |key: &str| scan[key].as_str().unwrap().to_string();
            (field("name"), root.join(field("file")), field("modality"))
        })
        .collect()
}

#[test]
#[ignore = "needs the reference fixtures; run with --ignored --release"]
fn check_parity_passes_on_every_fixture_scan() {
    let ws = Workspace::new();
    let root = fixtures_root().display().to_string();
    let output = ws.command(&["check-parity", &root]).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(output.status.success(), "check-parity failed");
    let n = fixture_scans().len();
    assert!(stdout.contains(&format!("{n} scans,")) && stdout.contains("every gate passes"));
}

#[test]
#[ignore = "needs the reference fixtures; run with --ignored --release"]
fn encode_and_probes_on_fixture_scans() {
    let ws = Workspace::new();
    let model = fixtures_root().join(format!("models/{MODEL}.toml")).display().to_string();
    let scans_dir = fixtures_root().join("scans").display().to_string();
    let scans = fixture_scans();

    let out = ws.ok(&["encode", &scans_dir, "--model", &model]);
    println!("{out}");
    assert!(out.contains(&format!("{} encoded, 0 cached, 0 failed", scans.len())), "{out}");
    let out = ws.ok(&["encode", &scans_dir, "--model", &model]);
    assert!(out.contains(&format!("0 encoded, {} cached, 0 failed", scans.len())), "{out}");

    // scan-level labels for every scan: head volume (regression) and MNI template or OpenNeuro scan
    // (classification: 15 vs 21, so every outer training fold keeps at least 5 of each class, which
    // LogisticRegressionCV's inner StratifiedKFold(5) needs);
    // for ten scans a 25 mm sphere above the grid centre drawn in each scan's own voxels
    // (segmentation, two folds fixed by the fold column)
    let mut scan_tsv = String::from("scan\thead_ml\ttemplate\n");
    let mut seg_tsv = String::from("scan\tsphere\tfold\n");
    for (i, (name, file, _)) in scans.iter().enumerate() {
        let run = read_json(&ws.path(&format!("derivatives/larmor/{name}/{name}_model-{MODEL}_run.json")));
        let head_ml = run["head_volume_ml"].as_f64().unwrap();
        let template = u8::from(name.starts_with("mni152"));
        scan_tsv.push_str(&format!("{}\t{head_ml}\t{template}\n", file.display()));
        if i < 10 {
            let fit: [[f64; 4]; 4] = serde_json::from_value(run["grid"]["fit_affine"].clone()).unwrap();
            let path = ws.path(&format!("labels/{name}_label-sphere_roi.nii.gz"));
            write_sphere(file, Affine(fit).apply([96.0, 120.0, 110.0]), 25.0, &path);
            seg_tsv.push_str(&format!("{}\t{}\t{}\n", file.display(), path.display(), i % 2));
        }
    }
    std::fs::write(ws.path("labels.tsv"), scan_tsv).unwrap();
    std::fs::write(ws.path("seg_labels.tsv"), seg_tsv).unwrap();

    for (table, kind, label) in [
        ("labels.tsv", "regression", "head_ml"),
        ("labels.tsv", "classification", "template"),
        ("seg_labels.tsv", "segmentation", "sphere"),
    ] {
        let probe = format!("probes/{label}.toml");
        let mut args =
            vec!["probe", "fit", table, "--model", &model, "--kind", kind, "--label", label, "--out", &probe];
        if kind == "segmentation" {
            args.extend(["--max-negative-ratio", "2"]);
        }
        let out = ws.ok(&args);
        println!("{out}");
    }

    let three: Vec<String> = scans.iter().take(3).map(|(_, file, _)| file.display().to_string()).collect();
    for label in ["head_ml", "template"] {
        let probe = format!("probes/{label}.toml");
        let mut args = vec!["probe", "apply", probe.as_str(), "--model", &model, "--out", "applied"];
        args.extend(three.iter().map(String::as_str));
        println!("{}", ws.ok(&args));
        let tsv =
            std::fs::read_to_string(ws.path(&format!("applied/{}_predictions.tsv", label.replace('_', ""))))
                .unwrap();
        assert_eq!(tsv.lines().count(), 4, "{tsv}");
    }

    let mut args = vec!["probe", "apply", "probes/sphere.toml", "--model", &model, "--out", "applied"];
    args.extend(three.iter().take(2).map(String::as_str));
    println!("{}", ws.ok(&args));
    for (name, file, _) in scans.iter().take(2) {
        let scan = read_nifti(file);
        let mask = read_nifti(&ws.path(&format!("applied/{name}/{name}_desc-sphere_mask.nii.gz")));
        assert_eq!(mask.shape(), scan.shape());
        assert_eq!(mask.affine().unwrap(), scan.affine().unwrap());
    }
}

/// A 0/1 sphere around a world point, on the voxel grid of `scan`.
fn write_sphere(scan: &std::path::Path, center: [f64; 3], radius: f64, path: &std::path::Path) {
    let nifti = Nifti::read(scan).unwrap();
    let affine = nifti.affine().unwrap();
    let shape: [usize; 3] = nifti.shape().try_into().unwrap();
    let sphere = Array3::from_shape_fn(shape, |(x, y, z)| {
        let p = affine.apply([x as f64, y as f64, z as f64]);
        u8::from((0..3).map(|i| (p[i] - center[i]).powi(2)).sum::<f64>() < radius * radius)
    });
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    nifti.with_data(VoxelData::U8(sphere.into_dyn())).write(path).unwrap();
}
