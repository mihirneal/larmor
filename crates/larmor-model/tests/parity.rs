//! Parity with the PyTorch encode references (Phase 0 gate) on every file in
//! `fixtures/reference/encode/<model_id>/*.safetensors`. Needs the fixtures, so ignored by default:
//!
//! ```sh
//! cargo test --release -p larmor-model --test parity -- --ignored --nocapture --test-threads 1
//! ```
//!
//! `LARMOR_FIXTURES` points at the fixtures directory (default: the checkout's `fixtures/`). Each
//! file must reach a minimum token cosine and a pooled cosine of at least 0.999 against PyTorch
//! fp32. The tables also show the centered cosine, the largest absolute difference, and how far
//! PyTorch's own bf16 autocast is from its fp32 (the reference's stored noise floor). Cosines are
//! printed as `1 - cos`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use larmor_model::parity::{EncodeReference, Parity, compare};
use larmor_model::{DeviceChoice, Encoder, Manifest, Precision};

const GATE: f64 = 0.999;

fn fixtures() -> PathBuf {
    std::env::var_os("LARMOR_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

/// Every encode reference, grouped by model id (the directory name), in name order.
fn references() -> BTreeMap<String, Vec<PathBuf>> {
    let root = fixtures().join("reference/encode");
    let mut by_model = BTreeMap::new();
    let entries = std::fs::read_dir(&root).unwrap_or_else(|err| panic!("{}: {err}", root.display()));
    for model_dir in entries.map(|entry| entry.unwrap().path()).filter(|path| path.is_dir()) {
        let model_id = model_dir.file_name().unwrap().to_string_lossy().into_owned();
        let mut files: Vec<PathBuf> = std::fs::read_dir(&model_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        files.sort();
        by_model.insert(model_id, files);
    }
    by_model
}

fn scan_name(path: &Path) -> String {
    path.file_stem().unwrap().to_string_lossy().into_owned()
}

/// Encode each reference's own patches with `model_id` and compare with its fp32 outputs.
/// Returns the scans that failed the gate.
fn check_model(model_id: &str, files: &[PathBuf], device: DeviceChoice, precision: Precision) -> Vec<String> {
    let manifest = Manifest::load(fixtures().join(format!("models/{model_id}.toml"))).expect("manifest");
    let encoder = Encoder::load(&manifest, device, precision).expect("encoder");
    let options = encoder.options();
    println!("\n{model_id} on {} in {precision}, {} attention", options.device, options.attention.kernel);
    println!(
        "{:<26} {:>6} {:>9} {:>9} {:>9} {:>9} {:>11} {:>11} {:>7}  gate",
        "scan", "tokens", "1-mincos", "1-pooled", "1-minctr", "max|d|", "bf16 1-min", "bf16 1-pool", "sec"
    );
    let mut failures = Vec::new();
    for path in files {
        let scan = scan_name(path);
        let reference = EncodeReference::load(path).expect("reference");
        let encoded = encoder.encode(&reference.patches, &reference.coords).expect("encode");
        assert_eq!(encoded.n_tokens, reference.n_tokens());
        let parity = compare(&encoded, &reference);
        let passes = parity.passes(GATE);
        print_row(&scan, &reference, &parity, encoded.seconds, passes);
        if !passes {
            failures.push(format!("{model_id}/{scan}: {parity}"));
        }
    }
    failures
}

fn print_row(scan: &str, reference: &EncodeReference, parity: &Parity, seconds: f64, passes: bool) {
    let floor = reference.bf16_noise_floor();
    let gap = |cosine: f64| format!("{:.2e}", 1.0 - cosine);
    println!(
        "{scan:<26} {:>6} {:>9} {:>9} {:>9} {:>9.2e} {:>11} {:>11} {seconds:>7.3}  {}",
        reference.n_tokens(),
        gap(parity.min_token_cosine),
        gap(parity.pooled_cosine),
        gap(parity.min_centered_cosine),
        parity.max_abs_diff,
        floor.map_or("-".into(), |f| gap(f.min_token_cosine)),
        floor.map_or("-".into(), |f| gap(f.pooled_cosine)),
        if passes { "PASS" } else { "FAIL" },
    );
}

/// Run every reference on `device` in `precision` and fail if any file misses the gate.
fn check_all(device: DeviceChoice, precision: Precision) {
    let mut failures = Vec::new();
    let mut count = 0;
    for (model_id, files) in references() {
        count += files.len();
        failures.extend(check_model(&model_id, &files, device, precision));
    }
    assert!(count > 0, "no encode references under {}", fixtures().display());
    println!("\n{} of {count} references pass the {GATE} gate", count - failures.len());
    assert!(failures.is_empty(), "parity gate {GATE} failed:\n{}", failures.join("\n"));
}

#[test]
#[ignore = "needs fixtures and a GPU"]
fn every_reference_f32() {
    check_all(DeviceChoice::Auto, Precision::F32);
}

#[test]
#[ignore = "needs fixtures and a GPU"]
fn every_reference_f16() {
    check_all(DeviceChoice::Auto, Precision::F16);
}

#[test]
#[ignore = "needs fixtures and a GPU"]
fn every_reference_bf16() {
    check_all(DeviceChoice::Auto, Precision::Bf16);
}

#[test]
#[ignore = "needs fixtures"]
fn vits_p8_cpu_f32() {
    let file = fixtures().join("reference/encode/nanobrain-vits-p8-ref/mni152-t1w.safetensors");
    let failures = check_model("nanobrain-vits-p8-ref", &[file], DeviceChoice::Cpu, Precision::F32);
    assert!(failures.is_empty(), "{failures:?}");
}
