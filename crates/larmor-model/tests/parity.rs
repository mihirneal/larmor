//! Parity with the PyTorch encode references (Phase 0 gate). Needs the fixtures, so ignored by
//! default:
//!
//! ```sh
//! cargo test --release -p larmor-model --test parity -- --ignored --nocapture
//! ```
//!
//! `LARMOR_FIXTURES` points at the fixtures directory (default: the checkout's `fixtures/`).

use std::path::PathBuf;

use larmor_model::parity::{EncodeReference, compare};
use larmor_model::{DeviceChoice, Encoder, Manifest, Precision};

const GATE: f64 = 0.999;

fn fixtures() -> PathBuf {
    std::env::var_os("LARMOR_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures"))
}

/// Encode the reference's own patches and check the fp32 gate.
fn check(model_id: &str, scan: &str, device: DeviceChoice, precision: Precision) {
    let root = fixtures();
    let reference_path = root.join(format!("reference/encode/{model_id}/{scan}.safetensors"));
    if !reference_path.exists() {
        panic!("missing reference {}", reference_path.display());
    }
    let manifest = Manifest::load(root.join(format!("models/{model_id}.toml"))).expect("manifest");
    let encoder = Encoder::load(&manifest, device, precision).expect("encoder");
    let reference = EncodeReference::load(&reference_path).expect("reference");
    let encoded = encoder.encode(&reference.patches, &reference.coords).expect("encode");
    let parity = compare(&encoded, &reference);
    println!(
        "{model_id} {scan} on {} {precision}: {} tokens in {:.2}s\n  {parity}",
        encoder.options().device,
        encoded.n_tokens,
        encoded.seconds
    );
    assert_eq!(encoded.n_tokens, reference.n_tokens());
    assert!(parity.passes(GATE), "parity gate {GATE} failed: {parity}");
    assert!(parity.min_centered_cosine > 0.99, "centered cosine too low: {parity}");
}

#[test]
#[ignore = "needs fixtures"]
fn vits_p8_cpu_f32() {
    check("nanobrain-vits-p8-ref", "mni152-t1w", DeviceChoice::Cpu, Precision::F32);
}

#[test]
#[ignore = "needs fixtures and a GPU"]
fn vits_p8_gpu_f32() {
    check("nanobrain-vits-p8-ref", "mni152-t1w", DeviceChoice::Auto, Precision::F32);
}

#[test]
#[ignore = "needs fixtures and a GPU"]
fn vits_p4_gpu_f32() {
    check("nanobrain-vits-p4-ref", "mni152-t1w", DeviceChoice::Auto, Precision::F32);
}
