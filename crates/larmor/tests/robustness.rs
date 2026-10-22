//! The `larmor` binary on inputs the happy path does not cover: corrected manifests, corrupt files,
//! BIDS folders with functional series, scans with the same file name, duplicated label rows,
//! untrusted model manifests, and a second modality prepared on a reference scan's head mask.

mod common;

use std::path::Path;

use common::{Subject, Workspace, read_json, read_npy_f32, write_lesion, write_scan};
use larmor_volume::ndarray::{Array3, ArrayD, IxDyn};
use larmor_volume::{Affine, Nifti, VoxelData};

const CPU: [&str; 2] = ["--device", "cpu"];

fn args<'a>(base: &[&'a str]) -> Vec<&'a str> {
    let mut all = base.to_vec();
    all.extend(CPU);
    all
}

/// A copy of the tiny model's manifest and weights in `dir`, with `from` replaced by `to`.
fn edited_model(ws: &Workspace, dir: &str, from: &str, to: &str) -> String {
    let text = std::fs::read_to_string(ws.path("models/tiny.toml")).unwrap();
    assert!(text.contains(from), "{from}");
    std::fs::create_dir_all(ws.path(dir)).unwrap();
    std::fs::copy(ws.path("models/tiny.safetensors"), ws.path(&format!("{dir}/tiny.safetensors"))).unwrap();
    let path = format!("{dir}/tiny.toml");
    std::fs::write(ws.path(&path), text.replace(from, to)).unwrap();
    path
}

#[test]
fn a_corrected_manifest_is_not_served_the_old_embeddings() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("scan.nii.gz"), &Subject::nth(0));
    let out = ws.ok(&args(&["encode", "scan.nii.gz", "--model", "tiny", "--out", "a"]));
    assert!(out.contains("1 encoded, 0 cached"), "{out}");

    // same id and weights, but heads changed: another encoder, so another cache entry
    let edited = edited_model(&ws, "edited", "heads = 2", "heads = 4");
    let out = ws.ok(&args(&["encode", "scan.nii.gz", "--model", &edited, "--out", "b"]));
    assert!(out.contains("1 encoded, 0 cached"), "{out}");
    let (_, a) = read_npy_f32(&ws.path("a/scan/scan_model-tiny_embedding.npy"));
    let (_, b) = read_npy_f32(&ws.path("b/scan/scan_model-tiny_embedding.npy"));
    assert_ne!(a, b, "heads changes the embedding");
    // and the original manifest still finds its own entry
    let out = ws.ok(&args(&["encode", "scan.nii.gz", "--model", "tiny", "--out", "a"]));
    assert!(out.contains("0 encoded, 1 cached"), "{out}");
    let run = read_json(&ws.path("a/scan/scan_model-tiny_run.json"));
    assert_eq!(run["precision"], "f32");
    assert!(run["model"]["encoder_key"].as_str().unwrap().len() == 64);
    let entry = run["cache"]["path"].as_str().unwrap().to_string();
    assert!(entry.contains("/f32/"), "{entry}");

    // a probe fit on the original encoder refuses the edited one
    write_cohort(&ws, 8);
    ws.ok(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "age",
        "--encode",
        "--out",
        "age.toml",
    ]));
    let probe: toml::Table = std::fs::read_to_string(ws.path("age.toml")).unwrap().parse().unwrap();
    assert_eq!(probe["precision"].as_str(), Some("f32"));
    assert_eq!(probe["encoder_key"].as_str().map(str::len), Some(64));
    let err = ws.fails(&args(&["probe", "apply", "age.toml", "scan.nii.gz", "--model", &edited]));
    assert!(err.contains("refusing to apply") && err.contains("encoder"), "{err}");
}

/// `n` subjects in a BIDS dataset `cohort/` with age and lesion labels in `cohort/labels.tsv`, and
/// the same with a two-fold `fold` column in `cohort/folds.tsv`.
fn write_cohort(ws: &Workspace, n: usize) {
    std::fs::create_dir_all(ws.path("cohort")).unwrap();
    std::fs::write(ws.path("cohort/dataset_description.json"), r#"{"Name": "c", "BIDSVersion": "1.10.0"}"#)
        .unwrap();
    let mut tsv = String::from("scan\tage\tlesion\n");
    let mut folds = String::from("scan\tage\tlesion\tfold\n");
    for i in 0..n {
        let scan = format!("sub-{i:02}/anat/sub-{i:02}_T1w.nii.gz");
        let lesion = format!("sub-{i:02}/anat/sub-{i:02}_desc-lesion_mask.nii.gz");
        write_scan(&ws.path(&format!("cohort/{scan}")), &Subject::nth(i));
        write_lesion(&ws.path(&format!("cohort/{lesion}")), &Subject::nth(i));
        tsv.push_str(&format!("{scan}\t{}\t{lesion}\n", 20 + 7 * i));
        folds.push_str(&format!("{scan}\t{}\t{lesion}\t{}\n", 20 + 7 * i, i % 2));
    }
    std::fs::write(ws.path("cohort/labels.tsv"), tsv).unwrap();
    std::fs::write(ws.path("cohort/folds.tsv"), folds).unwrap();
}

/// The bytes of a synthetic scan as an uncompressed `.nii`, on `affine`.
fn nii_bytes(subject: &Subject, affine: &Affine) -> Vec<u8> {
    Nifti::new(VoxelData::I16(common::head_image(subject).into_dyn()), affine).to_bytes()
}

#[test]
fn malformed_headers_fail_their_scan_and_the_batch_goes_on() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    // dims whose product (16384^5 = 2^70) wraps to zero bytes without checked arithmetic
    let mut wrapped = nii_bytes(&Subject::nth(0), &common::scan_affine());
    let dims: [i16; 8] = [5, 16384, 16384, 16384, 16384, 16384, 1, 1];
    for (i, d) in dims.iter().enumerate() {
        wrapped[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
    }
    wrapped.truncate(352 + 16);
    std::fs::create_dir_all(ws.path("in")).unwrap();
    std::fs::write(ws.path("in/a_wrapped.nii"), &wrapped).unwrap();
    // a 2 mm scan whose header voxel size is 1e-30 mm: the mask's smoothing sigma is 4e30 voxels
    let two_mm =
        Affine::from_matvec([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], [-40.0, -48.0, -40.0]);
    let mut tiny_voxels = nii_bytes(&Subject::nth(1), &two_mm);
    tiny_voxels[80..84].copy_from_slice(&1e-30f32.to_le_bytes());
    std::fs::write(ws.path("in/b_tiny_voxels.nii"), &tiny_voxels).unwrap();
    write_scan(&ws.path("in/c_good.nii.gz"), &Subject::nth(2));

    let files = ["in/a_wrapped.nii", "in/b_tiny_voxels.nii", "in/c_good.nii.gz"];
    let output = ws.run(&args(&["encode", files[0], files[1], files[2], "--model", "tiny"]));
    let text =
        format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(!text.contains("panicked"), "{text}");
    assert!(text.contains("1 encoded, 0 cached, 2 failed"), "{text}");
    // the 5-D header is refused from its header alone; the 1e-30 mm voxel by its kernel size
    assert!(text.contains("expected a 3-D image") && text.contains("kernel radius"), "{text}");
    // a folder search leaves the 5-D image out with a note
    let output = ws.run(&args(&["encode", "in", "--model", "tiny"]));
    let text =
        format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert!(text.contains("skipped 1 image that is not 3-D") && text.contains("1 failed"), "{text}");
}

#[test]
fn a_bids_dataset_with_functional_series_encodes_its_anatomy() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    std::fs::create_dir_all(ws.path("bids")).unwrap();
    std::fs::write(ws.path("bids/dataset_description.json"), r#"{"Name": "b", "BIDSVersion": "1.10.0"}"#)
        .unwrap();
    write_scan(&ws.path("bids/sub-01/anat/sub-01_T1w.nii.gz"), &Subject::nth(0));
    let bold = ArrayD::<i16>::zeros(IxDyn(&[8, 8, 6, 5]));
    std::fs::create_dir_all(ws.path("bids/sub-01/func")).unwrap();
    Nifti::new(VoxelData::I16(bold), &Affine::IDENTITY)
        .write(ws.path("bids/sub-01/func/sub-01_task-rest_bold.nii.gz"))
        .unwrap();
    let out = ws.ok(&args(&["encode", "bids", "--model", "tiny"]));
    assert!(out.contains("1 scan ") && out.contains("1 encoded, 0 cached, 0 failed"), "{out}");
    let err =
        ws.fails(&args(&["encode", "bids/sub-01/func/sub-01_task-rest_bold.nii.gz", "--model", "tiny"]));
    assert!(err.contains("expected a 3-D image") && err.contains("fMRI"), "{err}");
}

#[test]
fn probe_apply_reports_a_bad_scan_and_predicts_the_others() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_cohort(&ws, 8);
    ws.ok(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "age",
        "--encode",
        "--out",
        "age.toml",
    ]));
    std::fs::create_dir_all(ws.path("new")).unwrap();
    write_scan(&ws.path("new/a.nii.gz"), &Subject::nth(9));
    std::fs::write(ws.path("new/b.nii.gz"), b"not a nifti").unwrap();
    write_scan(&ws.path("new/c.nii.gz"), &Subject::nth(10));
    let output = ws.run(&args(&["probe", "apply", "age.toml", "new", "--out", "pred"]));
    let text =
        format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
    assert_eq!(output.status.code(), Some(1), "{text}");
    assert!(text.contains("failed") && text.contains("1 of 3 scans failed"), "{text}");
    let tsv = std::fs::read_to_string(ws.path("pred/age_predictions.tsv")).unwrap();
    assert_eq!(tsv.lines().count(), 3, "{tsv}");
    assert!(tsv.contains("a.nii.gz") && tsv.contains("c.nii.gz") && !tsv.contains("b.nii.gz"), "{tsv}");
}

#[test]
fn scans_with_the_same_file_name_get_their_own_outputs() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("data/subj01/T1.nii.gz"), &Subject::nth(0));
    write_scan(&ws.path("data/subj02/T1.nii.gz"), &Subject::nth(1));
    let out = ws.ok(&args(&["encode", "data", "--model", "tiny", "--out", "enc"]));
    assert!(out.contains("2 encoded"), "{out}");
    assert!(ws.path("enc/subj01/T1_model-tiny_embedding.npy").is_file());
    assert!(ws.path("enc/subj02/T1_model-tiny_embedding.npy").is_file());

    // one folder at a time into one --out: the second run would replace the first's outputs
    ws.ok(&args(&["encode", "data/subj01", "--model", "tiny", "--out", "one"]));
    let err = ws.fails(&args(&["encode", "data/subj02", "--model", "tiny", "--out", "one"]));
    assert!(err.contains("holds the outputs of another scan") && err.contains("--force"), "{err}");
    ws.ok(&args(&["encode", "data/subj02", "--model", "tiny", "--out", "one", "--force"]));

    // file names that differ only in case are one file on macOS's default file system
    write_scan(&ws.path("ds1/sub-01/anat/sub-01_T1w.nii.gz"), &Subject::nth(2));
    write_scan(&ws.path("ds2/sub-01/anat/sub-01_t1w.nii.gz"), &Subject::nth(3));
    let err = ws.fails(&args(&["encode", "ds1", "ds2", "--model", "tiny", "--out", "x"]));
    assert!(err.contains("would both write") && err.contains("only in case"), "{err}");
}

#[test]
fn a_cache_hit_names_this_scan_in_run_json() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("x/scan.nii.gz"), &Subject::nth(0));
    ws.ok(&args(&["encode", "x/scan.nii.gz", "--model", "tiny", "--out", "out_x"]));
    std::fs::create_dir_all(ws.path("y")).unwrap();
    std::fs::copy(ws.path("x/scan.nii.gz"), ws.path("y/copy.nii.gz")).unwrap();
    let out = ws.ok(&args(&["encode", "y/copy.nii.gz", "--model", "tiny", "--out", "out_y"]));
    assert!(out.contains("1 cached"), "{out}");
    let run = read_json(&ws.path("out_y/copy/copy_model-tiny_run.json"));
    assert!(run["scan"]["path"].as_str().unwrap().ends_with("y/copy.nii.gz"), "{run}");
    assert!(run["encoded_from"]["scan_path"].as_str().unwrap().ends_with("x/scan.nii.gz"), "{run}");
    assert_eq!(run["cache"]["hit"], true);
}

#[test]
fn saved_grid_images_are_not_encoded_as_scans_on_the_next_run() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("data/T1.nii.gz"), &Subject::nth(0));
    for _ in 0..2 {
        let out = ws.ok(&args(&[
            "encode",
            "data",
            "--out",
            "data/larmor",
            "--model",
            "tiny",
            "--save-preprocessed",
        ]));
        assert!(out.contains("larmor encode  1 scan "), "{out}");
    }
    assert!(ws.path("data/larmor/T1/T1_model-tiny_desc-quantized.nii.gz").is_file());
}

#[test]
fn an_encoder_that_cannot_load_is_reported_once() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let text = std::fs::read_to_string(ws.path("models/tiny.toml")).unwrap();
    let sha = text.split("sha256 = \"").nth(1).unwrap()[..64].to_string();
    let wrong = format!("{}{}", if sha.starts_with('0') { "1" } else { "0" }, &sha[1..]);
    std::fs::write(ws.path("models/tiny.toml"), text.replace(&sha, &wrong)).unwrap();
    for i in 0..3 {
        write_scan(&ws.path(&format!("in/s{i}.nii.gz")), &Subject::nth(i));
    }
    let err = ws.fails(&args(&["encode", "in", "--model", "tiny"]));
    assert_eq!(err.matches("does not match").count(), 1, "{err}");
    assert!(err.contains("larmor models verify tiny"), "{err}");
    assert_eq!(err.matches("could not be loaded").count(), 2, "{err}");
    assert!(err.contains("3 of 3 scans failed"), "{err}");
}

#[test]
fn probe_fit_refuses_duplicate_scans_and_re_encodes_corrupt_entries() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_cohort(&ws, 8);
    let fit = |table: &str, extra: &[&'static str]| {
        let mut a = vec!["probe", "fit", table, "--model", "tiny", "--kind", "regression", "--label", "age"];
        a.extend(extra);
        args(&a).into_iter().map(String::from).collect::<Vec<_>>()
    };
    let run = |a: Vec<String>| ws.run(&a.iter().map(String::as_str).collect::<Vec<_>>());
    let text = |o: std::process::Output| {
        format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
    };

    // a row repeated, and a byte-identical copy under another name
    let tsv = std::fs::read_to_string(ws.path("cohort/labels.tsv")).unwrap();
    let rows: Vec<&str> = tsv.lines().collect();
    std::fs::copy(ws.path("cohort/sub-01/anat/sub-01_T1w.nii.gz"), ws.path("cohort/copy.nii.gz")).unwrap();
    let dup = format!("{}\n{}\ncopy.nii.gz\t99\tn/a\n", tsv.trim_end(), rows[3]);
    std::fs::write(ws.path("cohort/dup.tsv"), dup).unwrap();
    let out = run(fit("cohort/dup.tsv", &["--encode"]));
    assert_eq!(out.status.code(), Some(1));
    let out = text(out);
    assert!(out.contains("lines 4, 10 name the same scan"), "{out}");
    assert!(out.contains("lines 3, 11 name files with identical bytes"), "{out}");

    // a fold column that keeps the copies together is fine
    let mut folded = String::from("scan\tage\tlesion\tfold\n");
    for (i, row) in rows[1..].iter().enumerate() {
        folded.push_str(&format!("{row}\t{}\n", i % 2));
    }
    folded.push_str(&format!("{}\t{}\n", rows[3], 0));
    std::fs::write(ws.path("cohort/folded.tsv"), folded).unwrap();
    let out = text(run(fit("cohort/folded.tsv", &["--encode", "--out", "f.toml"])));
    assert!(out.contains("repeated scans are kept"), "{out}");

    // a truncated cache entry: refused with a hint without --encode, re-encoded with it
    let run_json = ws.path("cohort/derivatives/larmor/sub-02/anat/sub-02_T1w_model-tiny_run.json");
    ws.ok(&args(&["encode", "cohort/sub-02", "--model", "tiny"]));
    let entry = read_json(&run_json)["cache"]["path"].as_str().unwrap().to_string();
    let bytes = std::fs::read(&entry).unwrap();
    std::fs::write(&entry, &bytes[..bytes.len() / 2]).unwrap();
    let out = run(fit("cohort/labels.tsv", &[]));
    assert_eq!(out.status.code(), Some(1));
    let out = text(out);
    assert!(out.contains("unreadable cache entry") && out.contains("--force"), "{out}");
    let out = text(run(fit("cohort/labels.tsv", &["--encode", "--out", "g.toml"])));
    assert!(out.contains("re-encoding") && out.contains("wrote g.toml"), "{out}");
    // the entry is whole again
    let out = text(run(fit("cohort/labels.tsv", &["--out", "h.toml"])));
    assert!(out.contains("wrote h.toml"), "{out}");
}

#[test]
fn segmentation_masks_of_same_named_scans_and_fresh_versus_cached_tokens() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_cohort(&ws, 10);
    ws.ok(&args(&[
        "probe",
        "fit",
        "cohort/folds.tsv",
        "--model",
        "tiny",
        "--kind",
        "segmentation",
        "--label",
        "lesion",
        "--encode",
        "--out",
        "les.toml",
        "--max-negative-ratio",
        "4",
    ]));
    write_scan(&ws.path("a/T1.nii.gz"), &Subject::nth(11));
    write_scan(&ws.path("b/T1.nii.gz"), &Subject::nth(12));
    // the first apply encodes a/T1 (fresh tokens), the second reads it from the cache
    ws.ok(&args(&["probe", "apply", "les.toml", "a", "b", "--out", "first"]));
    ws.ok(&args(&["probe", "apply", "les.toml", "a", "b", "--out", "second"]));
    for subject in ["a", "b"] {
        let mask = ws.path(&format!("first/{subject}/T1_desc-les_mask.nii.gz"));
        assert!(mask.is_file(), "{}", mask.display());
        let first = common::read_nifti(&ws.path(&format!("first/{subject}/T1_desc-les_probseg.nii.gz")));
        let second = common::read_nifti(&ws.path(&format!("second/{subject}/T1_desc-les_probseg.nii.gz")));
        assert!(
            first.data == second.data,
            "{subject}: probabilities differ between a fresh encode and a cache hit"
        );
    }
}

#[test]
fn model_manifests_cannot_write_outside_their_folder() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let text = std::fs::read_to_string(ws.path("models/tiny.toml")).unwrap();
    std::fs::create_dir_all(ws.path("evil")).unwrap();
    let evil = text
        .replace("file = \"tiny.safetensors\"", "file = \"../victim.txt\"")
        .replace("url = \"\"", "url = \"https://example.org/payload\"");
    std::fs::write(ws.path("evil/evil.toml"), evil).unwrap();
    std::fs::write(ws.path("victim.txt"), "original").unwrap();
    let err = ws.fails(&["models", "download", "evil/evil.toml"]);
    assert!(err.contains("weights.file") && err.contains("plain file name"), "{err}");
    assert_eq!(std::fs::read_to_string(ws.path("victim.txt")).unwrap(), "original");

    let id = text.replace("id = \"tiny\"", "id = \"../../escaped\"");
    std::fs::write(ws.path("evil/id.toml"), id).unwrap();
    let err = ws.fails(&["models", "verify", "evil/id.toml"]);
    assert!(err.contains("id") && err.contains("org--name"), "{err}");

    // weights that do not match are not replaced without --force
    std::fs::write(ws.path("models/tiny.safetensors"), b"something else").unwrap();
    let url = text.replace("url = \"\"", "url = \"https://example.org/w.safetensors\"");
    std::fs::write(ws.path("models/tiny.toml"), url).unwrap();
    let err = ws.fails(&["models", "download", "tiny"]);
    assert!(err.contains("refusing to replace") && err.contains("--force"), "{err}");
    assert_eq!(std::fs::read(ws.path("models/tiny.safetensors")).unwrap(), b"something else");
}

#[test]
fn a_second_modality_is_encoded_on_the_reference_head_mask() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let subject = Subject::nth(0);
    write_scan(&ws.path("sub/t1.nii.gz"), &subject);
    // another contrast on the same grid: the brain inverted, scalp and fluid dark, so its own head
    // mask would be the brain alone
    let t1 = common::head_image(&subject);
    let radii = [56.0, 68.0, 60.0].map(|r| r * subject.scale);
    let t2 = Array3::from_shape_fn(t1.raw_dim(), |(x, y, z)| {
        let p = common::scan_affine().apply([x as f64, y as f64, z as f64]);
        let r = (0..3).map(|i| (p[i] / radii[i]).powi(2)).sum::<f64>().sqrt();
        if r < 0.84 { (1500 - t1[[x, y, z]].min(1500)).max(50) } else { t1[[x, y, z]] / 20 }
    });
    Nifti::new(VoxelData::I16(t2.into_dyn()), &common::scan_affine())
        .write(ws.path("sub/t2.nii.gz"))
        .unwrap();

    ws.ok(&args(&["encode", "sub/t2.nii.gz", "--model", "tiny", "--out", "alone"]));
    let out = ws.ok(&args(&[
        "encode",
        "sub/t2.nii.gz",
        "--reference",
        "sub/t1.nii.gz",
        "--model",
        "tiny",
        "--out",
        "ref",
    ]));
    assert!(out.contains("1 encoded") && out.contains("head mask of sub/t1.nii.gz"), "{out}");
    let alone = read_json(&ws.path("alone/t2/t2_model-tiny_run.json"));
    let referenced = read_json(&ws.path("ref/t2/t2_model-tiny_run.json"));
    let t1_sha = common::sha256_hex(&std::fs::read(ws.path("sub/t1.nii.gz")).unwrap());
    assert_eq!(referenced["reference"]["sha256"], t1_sha.as_str());
    assert!(alone.get("reference").is_none());
    assert!(referenced["cache"]["path"].as_str().unwrap().ends_with(&format!("_ref-{t1_sha}.safetensors")));
    assert_ne!(alone["head_volume_ml"], referenced["head_volume_ml"], "the mask is the T1's");
    let t1 = ws.ok(&args(&["encode", "sub/t1.nii.gz", "--model", "tiny", "--out", "t1"]));
    assert!(t1.contains("1 encoded"), "{t1}");
    let t1_run = read_json(&ws.path("t1/t1/t1_model-tiny_run.json"));
    assert_eq!(t1_run["head_volume_ml"], referenced["head_volume_ml"]);
    // the same request again is a cache hit
    let out = ws.ok(&args(&[
        "encode",
        "sub/t2.nii.gz",
        "--reference",
        "sub/t1.nii.gz",
        "--model",
        "tiny",
        "--out",
        "ref",
    ]));
    assert!(out.contains("1 cached"), "{out}");
    assert!(Path::new(&ws.path("ref/t2/t2_model-tiny_embedding.npy")).is_file());

    // a reference on another grid is refused for that scan
    std::fs::create_dir_all(ws.path("other")).unwrap();
    let shifted =
        Affine::from_matvec([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]], [-75.0, -96.0, -80.0]);
    Nifti::new(VoxelData::I16(common::head_image(&subject).into_dyn()), &shifted)
        .write(ws.path("other/t1.nii.gz"))
        .unwrap();
    let err =
        ws.fails(&args(&["encode", "sub/t2.nii.gz", "--reference", "other/t1.nii.gz", "--model", "tiny"]));
    assert!(err.contains("same voxel grid"), "{err}");
}
