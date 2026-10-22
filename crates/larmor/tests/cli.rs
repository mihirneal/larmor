//! The `larmor` binary end to end on tiny synthetic inputs: head-like scans of 40 x 48 x 40 voxels
//! and a one-block ViT of width 32 on the CPU, all in a temporary folder.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::Path;

use common::{
    Subject, Workspace, as3, files_below, read_json, read_nifti, read_npy_f32, write_lesion, write_scan,
};
use larmor_model::{DeviceChoice, Encoder, Manifest, Precision};
use larmor_prep::Recipe;
use larmor_volume::{Nifti, VoxelData};
use safetensors::tensor::{Dtype, TensorView};

const CPU: [&str; 2] = ["--device", "cpu"];

fn args<'a>(base: &[&'a str]) -> Vec<&'a str> {
    let mut all = base.to_vec();
    all.extend(CPU);
    all
}

#[test]
fn encode_writes_bids_outputs_and_reuses_the_cache() {
    let ws = Workspace::new();
    let sha = ws.tiny_model("tiny", 1);
    std::fs::create_dir_all(ws.path("ds")).unwrap();
    std::fs::write(ws.path("ds/dataset_description.json"), r#"{"Name": "test", "BIDSVersion": "1.10.0"}"#)
        .unwrap();
    write_scan(&ws.path("ds/sub-01/anat/sub-01_T1w.nii.gz"), &Subject::nth(0));
    write_scan(&ws.path("ds/sub-02/ses-1/anat/sub-02_ses-1_T1w.nii.gz"), &Subject::nth(1));
    // outputs from an earlier run are not inputs
    write_scan(&ws.path("ds/derivatives/other/sub-01_T1w.nii.gz"), &Subject::nth(2));
    write_scan(&ws.path("loose/brain.nii"), &Subject::nth(3));

    let out = ws.ok(&args(&["encode", "ds", "loose/brain.nii", "--model", "tiny", "--save-preprocessed"]));
    assert_eq!(out.matches("encoded ").count(), 3, "{out}");
    assert!(out.contains("3 encoded, 0 cached, 0 failed"), "{out}");

    let derivatives = ws.path("ds/derivatives/larmor");
    let description = read_json(&derivatives.join("dataset_description.json"));
    assert_eq!(description["DatasetType"], "derivative");
    assert_eq!(description["GeneratedBy"][0]["Name"], "larmor");
    assert_eq!(
        files_below(&derivatives),
        [
            "dataset_description.json",
            "sub-01/anat/sub-01_T1w_model-tiny_desc-head_mask.nii.gz",
            "sub-01/anat/sub-01_T1w_model-tiny_desc-quantized.nii.gz",
            "sub-01/anat/sub-01_T1w_model-tiny_embedding.npy",
            "sub-01/anat/sub-01_T1w_model-tiny_run.json",
            "sub-02/ses-1/anat/sub-02_ses-1_T1w_model-tiny_desc-head_mask.nii.gz",
            "sub-02/ses-1/anat/sub-02_ses-1_T1w_model-tiny_desc-quantized.nii.gz",
            "sub-02/ses-1/anat/sub-02_ses-1_T1w_model-tiny_embedding.npy",
            "sub-02/ses-1/anat/sub-02_ses-1_T1w_model-tiny_run.json",
        ]
    );
    // a scan outside any BIDS dataset goes to ./derivatives/larmor/, in its folder below the
    // inputs' common folder (here the workspace: ds and loose/brain.nii)
    let loose = ws.path("derivatives/larmor/loose");
    assert!(loose.join("brain_model-tiny_embedding.npy").is_file());
    assert!(ws.path("derivatives/larmor/dataset_description.json").is_file());

    let (shape, pooled) = read_npy_f32(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_embedding.npy"));
    assert_eq!(shape, [32]);
    assert!(pooled.iter().all(|v| v.is_finite()) && pooled.iter().any(|&v| v != 0.0));

    let run = read_json(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_run.json"));
    assert_eq!(run["larmor_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(run["model"]["id"], "tiny");
    assert_eq!(run["model"]["sha256"], sha.as_str());
    let prep_version = run["preprocessing"]["version"].as_str().unwrap();
    assert_eq!(prep_version, Recipe::nanobrain_eval().version_hash());
    assert_eq!(run["preprocessing"]["steps"].as_array().unwrap().len(), 4);
    let scan_bytes = std::fs::read(ws.path("ds/sub-01/anat/sub-01_T1w.nii.gz")).unwrap();
    assert_eq!(run["scan"]["sha256"], common::sha256_hex(&scan_bytes).as_str());
    assert_eq!(run["scan"]["shape"], serde_json::json!([40, 48, 40]));
    assert_eq!(run["scan"]["zooms"], serde_json::json!([4.0, 4.0, 4.0]));
    assert_eq!(run["scan"]["orientation"], "RAS");
    assert_eq!(run["scan"]["dtype"], "int16");
    assert_eq!(run["grid"]["shape"], serde_json::json!([192, 240, 192]));
    assert_eq!(run["grid"]["fit_affine"].as_array().unwrap().len(), 4);
    assert!(run["vmin"].as_f64().unwrap() < run["vmax"].as_f64().unwrap());
    let head_ml = run["head_volume_ml"].as_f64().unwrap();
    assert!((300.0..1500.0).contains(&head_ml), "head volume {head_ml} ml");
    let tokens = run["tokens"].as_u64().unwrap();
    assert!(tokens > 500, "{tokens} tokens");
    assert_eq!(run["width"], 32);
    assert_eq!((run["device"].as_str(), run["precision"].as_str()), (Some("cpu"), Some("f32")));
    for stage in [
        "read",
        "hash",
        "mask",
        "conform",
        "quantize",
        "normalize",
        "tokens",
        "encode",
        "cache_write",
        "total",
    ] {
        assert!(run["seconds"][stage].as_f64().unwrap() >= 0.0, "seconds.{stage}");
    }
    assert!(run["started"].as_str().unwrap().len() >= 19 && run["finished"].is_string());
    assert_eq!(run["cache"]["hit"], false);
    let cache_path = run["cache"]["path"].as_str().unwrap();
    // the cache folder is named after the encoder key (weights, architecture, readout), then the
    // preprocessing version and the precision
    let encoder_key = run["model"]["encoder_key"].as_str().unwrap();
    let manifest = larmor_model::Manifest::load(ws.path("models/tiny.toml")).unwrap();
    assert_eq!(encoder_key, manifest.encoder_key());
    let expected =
        ws.cache().join(format!("tiny-{}", &encoder_key[..8])).join(&prep_version[..8]).join("f32");
    assert!(Path::new(cache_path).starts_with(&expected), "{cache_path}");
    assert!(Path::new(cache_path).is_file());

    let quantized = read_nifti(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_desc-quantized.nii.gz"));
    let VoxelData::U16(q) = &quantized.data else { panic!("quantized is {:?}", quantized.data.dtype()) };
    assert_eq!(q.shape(), [192, 240, 192]);
    assert_eq!(*q.iter().max().unwrap(), 65535);
    let mask = read_nifti(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_desc-head_mask.nii.gz"));
    let VoxelData::U8(m) = &mask.data else { panic!("mask is {:?}", mask.data.dtype()) };
    assert_eq!(m.shape(), [192, 240, 192]);
    let fit_affine: [[f64; 4]; 4] = serde_json::from_value(run["grid"]["fit_affine"].clone()).unwrap();
    assert_eq!(mask.affine().unwrap().0, fit_affine.map(|row| row.map(|v| v as f32 as f64)));

    // the second run reads the cache, and its run.json says so
    let out = ws.ok(&args(&["encode", "ds", "loose/brain.nii", "--model", "tiny"]));
    assert!(out.contains("0 encoded, 3 cached, 0 failed"), "{out}");
    let again = read_json(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_run.json"));
    assert_eq!(again["cache"]["hit"], true);
    assert_eq!(again["tokens"], run["tokens"]);
    let (_, pooled_again) =
        read_npy_f32(&derivatives.join("sub-01/anat/sub-01_T1w_model-tiny_embedding.npy"));
    assert_eq!(pooled_again, pooled);

    // --force re-encodes; --out moves the outputs
    let out = ws.ok(&args(&["encode", "ds/sub-01", "--model", "tiny", "--force", "--out", "elsewhere"]));
    assert!(out.contains("1 encoded, 0 cached"), "{out}");
    assert!(ws.path("elsewhere/sub-01/anat/sub-01_T1w_model-tiny_embedding.npy").is_file());

    let err = ws.fails(&["encode", "nothing-here.nii.gz", "--model", "tiny"]);
    assert!(err.contains("does not exist"), "{err}");
    let err = ws.fails(&["encode", "ds", "--model", "no-such-model"]);
    assert!(err.contains("no model \"no-such-model\"") && err.contains("tiny"), "{err}");
}

#[test]
fn half_precision_on_the_cpu_is_refused() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("scan.nii.gz"), &Subject::nth(0));
    let err =
        ws.fails(&["encode", "scan.nii.gz", "--model", "tiny", "--device", "cpu", "--precision", "f16"]);
    assert!(err.contains("f16 needs a GPU") && err.contains("1 of 1 scans failed"), "{err}");
}

#[test]
fn models_list_verify_and_download() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    let out = ws.ok(&["models", "list"]);
    let line = out.lines().find(|l| l.starts_with("tiny ")).unwrap_or_else(|| panic!("{out}"));
    assert!(line.contains("vit3d") && line.contains("192x240x192") && line.contains(" ok "), "{line}");
    assert!(!out.contains("shadowed"), "{out}");
    assert!(ws.ok(&["models", "verify", "tiny"]).contains("ok"));

    // weights that do not match their manifest fail verification
    let models = ws.path("models");
    common::write_tiny_model(&models, "corrupt", 2);
    std::fs::write(models.join("corrupt.safetensors"), b"not the weights").unwrap();
    let err = ws.fails(&["models", "verify"]);
    assert!(err.contains("MISMATCH") && err.contains("1 of 2 models failed"), "{err}");
    std::fs::remove_file(models.join("corrupt.toml")).unwrap();

    // a model whose weights are missing downloads them from weights.url and verifies them
    let sha = common::write_tiny_model(&models, "remote", 3);
    let weights = std::fs::read(models.join("remote.safetensors")).unwrap();
    std::fs::remove_file(models.join("remote.safetensors")).unwrap();
    let url = serve_once(weights.clone());
    let manifest = std::fs::read_to_string(models.join("remote.toml")).unwrap();
    std::fs::write(models.join("remote.toml"), manifest.replace("url = \"\"", &format!("url = \"{url}\"")))
        .unwrap();
    let out = ws.ok(&["models", "list"]);
    assert!(
        out.lines().any(|l| l.starts_with("remote ") && l.contains("missing") && l.contains("download")),
        "{out}"
    );
    // plain http needs --allow-http
    let err = ws.fails(&["models", "download", "remote"]);
    assert!(err.contains("--allow-http"), "{err}");
    let out = ws.ok(&["models", "download", "remote", "--allow-http"]);
    assert!(out.contains("verified"), "{out}");
    assert_eq!(common::sha256_hex(&std::fs::read(models.join("remote.safetensors")).unwrap()), sha);
    assert!(ws.ok(&["models", "verify"]).lines().filter(|l| l.contains(" ok ")).count() == 2);

    let err = ws.fails(&["models", "download", "tiny-missing"]);
    assert!(err.contains("no model"), "{err}");
}

/// Serves `body` to one HTTP GET on localhost; returns the URL.
fn serve_once(body: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/weights.safetensors", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        let mut line = String::new();
        // read the request head
        while reader.read_line(&mut line).unwrap() > 0 && line != "\r\n" {
            line.clear();
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(head.as_bytes()).unwrap();
        stream.write_all(&body).unwrap();
    });
    url
}

/// `n` subjects in a BIDS dataset with age, group and lesion labels. (Classification needs at
/// least 7 per class: sklearn's inner StratifiedKFold(5) runs on 4/5 of each class.)
fn write_cohort(ws: &Workspace, n: usize, with_folds: bool) {
    std::fs::create_dir_all(ws.path("cohort")).unwrap();
    std::fs::write(
        ws.path("cohort/dataset_description.json"),
        r#"{"Name": "cohort", "BIDSVersion": "1.10.0"}"#,
    )
    .unwrap();
    let mut tsv = String::from(if with_folds {
        "scan\tage\tgroup\tlesion\tfold\n"
    } else {
        "scan\tage\tgroup\tlesion\n"
    });
    for i in 0..n {
        let subject = Subject::nth(i);
        let scan = format!("sub-{i:02}/anat/sub-{i:02}_T1w.nii.gz");
        let lesion = format!("sub-{i:02}/anat/sub-{i:02}_desc-lesion_mask.nii.gz");
        write_scan(&ws.path(&format!("cohort/{scan}")), &subject);
        write_lesion(&ws.path(&format!("cohort/{lesion}")), &subject);
        let age = 20.0 + 40.0 * (subject.scale - 0.9) / 0.25 + (i % 3) as f64;
        let group = if subject.is_patient() { "patient" } else { "control" };
        tsv.push_str(&format!("{scan}\t{age}\t{group}\t{lesion}"));
        if with_folds {
            tsv.push_str(&format!("\t{}", i % 2));
        }
        tsv.push('\n');
    }
    std::fs::write(ws.path("cohort/labels.tsv"), tsv).unwrap();
}

#[test]
fn scan_level_probes_fit_apply_and_refuse_other_models() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_cohort(&ws, 14, false);

    // nothing is cached yet
    let err = ws.fails(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "age",
    ]));
    assert!(
        err.contains("14 of 14 scans are not encoded with tiny") && err.contains("larmor encode"),
        "{err}"
    );

    let out = ws.ok(&args(&[
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
        "probes/age.toml",
    ]));
    assert!(out.contains("MAE") && out.contains("R²"), "{out}");
    let probe: toml::Table = std::fs::read_to_string(ws.path("probes/age.toml")).unwrap().parse().unwrap();
    assert_eq!(probe["kind"].as_str(), Some("regression"));
    assert_eq!(probe["model_id"].as_str(), Some("tiny"));
    assert_eq!(probe["n_train"].as_integer(), Some(14));
    assert_eq!(probe["embed_dim"].as_integer(), Some(32));
    assert_eq!(probe["cv"]["n_folds"].as_integer(), Some(5));
    assert!(probe["cv"]["mae"].as_float().unwrap().is_finite());
    assert!(ws.path("probes/age.safetensors").is_file());

    // the embeddings are cached now, so classification needs no --encode
    let out = ws.ok(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "classification",
        "--label",
        "group",
        "--out",
        "probes/group.toml",
    ]));
    assert!(out.contains("AUROC") && out.contains("patient vs control"), "{out}");
    let probe: toml::Table = std::fs::read_to_string(ws.path("probes/group.toml")).unwrap().parse().unwrap();
    let names: Vec<&str> =
        probe["label_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(names, ["group", "control", "patient"]);

    let out = ws.ok(&args(&[
        "probe",
        "apply",
        "probes/age.toml",
        "cohort/sub-00",
        "cohort/sub-01",
        "--out",
        "predicted",
    ]));
    assert!(out.contains("sub-00_T1w") && out.contains("sub-01_T1w"), "{out}");
    let tsv = std::fs::read_to_string(ws.path("predicted/age_predictions.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = tsv.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows[0], ["scan", "prediction"]);
    assert_eq!(rows.len(), 3);
    assert!(rows[1..].iter().all(|r| r[1].parse::<f64>().unwrap().is_finite()));

    ws.ok(&args(&["probe", "apply", "probes/group.toml", "cohort", "--out", "predicted"]));
    let tsv = std::fs::read_to_string(ws.path("predicted/group_predictions.tsv")).unwrap();
    let rows: Vec<Vec<&str>> = tsv.lines().map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows[0], ["scan", "prediction", "probability"]);
    assert_eq!(rows.len(), 15);
    for row in &rows[1..] {
        assert!(row[1] == "patient" || row[1] == "control", "{row:?}");
        let p: f64 = row[2].parse().unwrap();
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(row[1] == "patient", p > 0.5);
    }

    // a model with other weights is refused, even under the same id
    common::write_tiny_model(&ws.path("other"), "tiny", 99);
    let err = ws.fails(&args(&[
        "probe",
        "apply",
        "probes/age.toml",
        "cohort/sub-00",
        "--model",
        "other/tiny.toml",
    ]));
    assert!(err.contains("refusing to apply") && err.contains("trained on model weights"), "{err}");

    // and so is a probe trained with another preprocessing version
    let text = std::fs::read_to_string(ws.path("probes/age.toml")).unwrap();
    let version = Recipe::nanobrain_eval().version_hash();
    std::fs::write(ws.path("probes/age.toml"), text.replace(&version, &"ab".repeat(32))).unwrap();
    let err = ws.fails(&args(&["probe", "apply", "probes/age.toml", "cohort/sub-00"]));
    assert!(err.contains("preprocessing"), "{err}");

    let err = ws.fails(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "regression",
        "--label",
        "height",
    ]));
    assert!(err.contains("no column \"height\""), "{err}");
}

#[test]
fn segmentation_probe_fits_and_writes_native_masks() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_cohort(&ws, 10, true);
    ws.ok(&args(&["encode", "cohort", "--model", "tiny"]));

    let out = ws.ok(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "segmentation",
        "--label",
        "lesion",
        "--out",
        "lesion.toml",
        "--max-negative-ratio",
        "4",
    ]));
    println!("{out}");
    assert!(out.contains("Dice") && out.contains("voxel AUROC"), "{out}");
    let probe: toml::Table = std::fs::read_to_string(ws.path("lesion.toml")).unwrap().parse().unwrap();
    assert_eq!(probe["kind"].as_str(), Some("segmentation"));
    assert_eq!(probe["pooling"].as_str(), Some("none"));
    assert_eq!(probe["cv"]["n_folds"].as_integer(), Some(2), "the fold column sets the outer folds");
    assert!(probe["hyperparameters"]["threshold"].as_float().is_some());

    let out = ws.ok(&args(&["probe", "apply", "lesion.toml", "cohort/sub-01", "cohort/sub-02"]));
    assert!(out.contains("ml at threshold"), "{out}");
    let dir = ws.path("cohort/derivatives/larmor/sub-01/anat");
    let scan = read_nifti(&ws.path("cohort/sub-01/anat/sub-01_T1w.nii.gz"));
    let mask = read_nifti(&dir.join("sub-01_T1w_desc-lesion_mask.nii.gz"));
    let VoxelData::U8(values) = &mask.data else { panic!("mask is {:?}", mask.data.dtype()) };
    assert_eq!(values.shape(), scan.shape());
    assert!(values.iter().all(|&v| v <= 1));
    assert_eq!(mask.affine().unwrap(), scan.affine().unwrap());
    let probseg = read_nifti(&dir.join("sub-01_T1w_desc-lesion_probseg.nii.gz"));
    let VoxelData::F32(p) = &probseg.data else { panic!("probseg is {:?}", probseg.data.dtype()) };
    assert_eq!(p.shape(), [192, 240, 192]);
    assert!(p.iter().all(|v| (0.0..=1.0).contains(v)));
    assert!(p.iter().any(|&v| v > 0.0));

    // segmentation labels must be binary
    let subject = Subject::nth(0);
    let twos = common::lesion_image(&subject).mapv(|v| v * 2);
    Nifti::new(VoxelData::U8(twos.into_dyn()), &common::scan_affine())
        .write(ws.path("cohort/sub-00/anat/sub-00_desc-lesion_mask.nii.gz"))
        .unwrap();
    let err = ws.fails(&args(&[
        "probe",
        "fit",
        "cohort/labels.tsv",
        "--model",
        "tiny",
        "--kind",
        "segmentation",
        "--label",
        "lesion",
    ]));
    assert!(err.contains("must be 0 or 1"), "{err}");
}

/// A fixture set in the layout of `fixtures/`, its references made by larmor's own crates.
fn write_fixture_set(ws: &Workspace) -> std::path::PathBuf {
    let root = ws.path("fixtures");
    let sha = common::write_tiny_model(&root.join("models"), "tiny-ref", 5);
    let manifest = Manifest::load(root.join("models/tiny-ref.toml")).unwrap();
    assert_eq!(manifest.weights.sha256, sha);
    let encoder = Encoder::load(&manifest, DeviceChoice::Cpu, Precision::F32).unwrap();
    let recipe = Recipe::nanobrain_eval();
    let mut list = String::new();
    for (i, name) in ["synth-a", "synth-b"].iter().enumerate() {
        let file = format!("scans/{name}.nii.gz");
        write_scan(&root.join(&file), &Subject::nth(i));
        let bytes = std::fs::read(root.join(&file)).unwrap();
        let scan_sha = common::sha256_hex(&bytes);
        list.push_str(&format!("[[scan]]\nname = \"{name}\"\nfile = \"{file}\"\nsha256 = \"{scan_sha}\"\nencode = [\"tiny-ref\"]\n\n"));

        let nifti = Nifti::from_bytes(&bytes).unwrap();
        let prepared = recipe.run(&nifti).unwrap();
        let dir = root.join("reference/prep").join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let grid = |data| Nifti::new(data, &prepared.fit_affine);
        Nifti::new(VoxelData::U8(prepared.native_mask.mapv(u8::from).into_dyn()), &prepared.native_affine)
            .write(dir.join("head_mask.nii.gz"))
            .unwrap();
        grid(VoxelData::F32(prepared.conformed.clone().into_dyn()))
            .write(dir.join("conform.nii.gz"))
            .unwrap();
        grid(VoxelData::U8(prepared.mask.mapv(u8::from).into_dyn()))
            .write(dir.join("fit_mask.nii.gz"))
            .unwrap();
        grid(VoxelData::U16(prepared.quantized.clone().into_dyn()))
            .write(dir.join("quantized.nii.gz"))
            .unwrap();
        let info = serde_json::json!({
            "scan": file,
            "scan_sha256": scan_sha,
            "head_mask_voxels": prepared.native_mask.iter().filter(|&&v| v).count(),
            "fit_mask_voxels": prepared.mask.iter().filter(|&&v| v).count(),
            "centroid": prepared.centroid,
            "fit_affine": prepared.fit_affine.0,
            "vmin": prepared.vmin,
            "vmax": prepared.vmax,
            "fwhm": null,
        });
        std::fs::write(dir.join("info.json"), serde_json::to_string_pretty(&info).unwrap()).unwrap();

        let tokens = prepared.tokens(8).unwrap();
        let encoded = encoder.encode(&tokens.patches, &tokens.coords).unwrap();
        let f32_bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let coords: Vec<u8> =
            tokens.coords.iter().flatten().flat_map(|&c| i32::from(c).to_le_bytes()).collect();
        let (n, d) = (tokens.len(), encoded.width);
        let (patches, toks, cls, pooled) = (
            f32_bytes(&tokens.patches),
            f32_bytes(&encoded.tokens),
            f32_bytes(&encoded.cls),
            f32_bytes(&encoded.pooled),
        );
        let views = vec![
            ("coords", TensorView::new(Dtype::I32, vec![n, 3], &coords).unwrap()),
            ("patches", TensorView::new(Dtype::F32, vec![n, 512], &patches).unwrap()),
            ("tokens", TensorView::new(Dtype::F32, vec![n, d], &toks).unwrap()),
            ("cls", TensorView::new(Dtype::F32, vec![1, d], &cls).unwrap()),
            ("pooled", TensorView::new(Dtype::F32, vec![d], &pooled).unwrap()),
        ];
        let metadata = std::collections::HashMap::from([
            ("model_id".to_string(), "tiny-ref".to_string()),
            ("bf16_min_token_cosine".to_string(), "0.999900".to_string()),
            ("bf16_pooled_cosine".to_string(), "0.999990".to_string()),
        ]);
        let bytes = safetensors::serialize(views, Some(metadata)).unwrap();
        let encode_dir = root.join("reference/encode/tiny-ref");
        std::fs::create_dir_all(&encode_dir).unwrap();
        std::fs::write(encode_dir.join(format!("{name}.safetensors")), bytes).unwrap();
    }
    std::fs::write(root.join("manifest.toml"), list).unwrap();
    root
}

#[test]
fn check_parity_passes_on_matching_references_and_fails_on_perturbed_ones() {
    let ws = Workspace::new();
    let root = write_fixture_set(&ws);
    let fixtures = root.display().to_string();
    let out = ws.ok(&args(&["check-parity", &fixtures]));
    assert!(out.contains("2 scans, 2 encodes: every gate passes"), "{out}");
    assert!(out.contains("tiny-ref: min token cosine") && out.contains("noise floor"), "{out}");
    assert_eq!(out.matches(" pass\n").count(), 2 + 8, "{out}");

    // one quantized voxel in a thousand off by one fails the 99.99% gate, and only for that scan
    let path = root.join("reference/prep/synth-b/quantized.nii.gz");
    let image = read_nifti(&path);
    let VoxelData::U16(values) = &image.data else { panic!() };
    let mut values = as3(values);
    let mut changed = 0;
    for v in values.iter_mut().filter(|v| **v > 0).step_by(1000) {
        *v = if *v == u16::MAX { *v - 1 } else { *v + 1 };
        changed += 1;
    }
    assert!(changed > 10);
    image.with_data(VoxelData::U16(values.into_dyn())).write(&path).unwrap();
    let out = ws.fails(&args(&["check-parity", &fixtures]));
    assert!(out.contains("1 of 2 scans failed") && out.contains("synth-b: quantized exact"), "{out}");

    let err = ws.fails(&args(&["check-parity", &fixtures, "--models", "nanobrain-vits-p8-ref"]));
    assert!(err.contains("no encode references for nanobrain-vits-p8-ref"), "{err}");
}

#[test]
fn bench_reports_every_stage_as_json() {
    let ws = Workspace::new();
    ws.tiny_model("tiny", 1);
    write_scan(&ws.path("scan.nii.gz"), &Subject::nth(0));
    let out = ws.ok(&args(&["bench", "--model", "tiny", "--scan", "scan.nii.gz", "--runs", "2", "--json"]));
    let report: serde_json::Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    let stages: Vec<&str> =
        report["stages"].as_array().unwrap().iter().map(|s| s["stage"].as_str().unwrap()).collect();
    assert_eq!(
        stages,
        [
            "read",
            "hash",
            "mask",
            "conform",
            "quantize",
            "normalize",
            "tokens",
            "encode (warm)",
            "cache write",
            "total (warm)"
        ]
    );
    assert!(report["machine"]["cpu_cores"].as_u64().unwrap() >= 1);
    assert!(report["tokens"].as_u64().unwrap() > 0);
    assert_eq!(report["device"], "cpu");
}
