//! Checks against the PyTorch reference fixtures. Each test returns early (passes) when the
//! fixtures or a Metal device are missing, so `cargo test` still works on a fresh checkout.
//! Run with `--release`: debug builds of the encoder are slow.

use std::path::{Path, PathBuf};

use candle_core::Device;
use candle_spike::config::Manifest;
use candle_spike::metrics::compare;
use candle_spike::model::{BiasAdd, Encoder, EncoderInput, Options};
use candle_spike::reference::EncodeReference;
use candle_spike::tokens::in_mask_patches;
use candle_spike::weights::Weights;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn reference_path(model_id: &str) -> PathBuf {
    fixtures().join(format!("reference/encode/{model_id}/mni152-t1w.safetensors"))
}

/// Phase 0 gate (`docs/DESIGN.md`): token and pooled cosine >= 0.999 against PyTorch fp32.
#[test]
fn vits_p8_matches_pytorch_fp32() {
    let model = fixtures().join("models/nanobrain-vits-p8-ref.toml");
    let reference = reference_path("nanobrain-vits-p8-ref");
    if !model.exists() || !reference.exists() {
        return;
    }
    let Ok(device) = Device::new_metal(0) else {
        return;
    };
    let manifest = Manifest::load(&model).unwrap();
    let reference = EncodeReference::load(&reference).unwrap();
    let arch = &manifest.architecture;
    let weights = Weights::load(&manifest.weights_path, arch, &device).unwrap();
    let input =
        EncoderInput::new(&reference.patches.values, &reference.patches.coords, arch, &device).unwrap();
    for bias in [BiasAdd::Broadcast, BiasAdd::Folded] {
        let encoder = Encoder::new(arch, &weights, Options { bias, ..Options::DEFAULT }).unwrap();
        let output = encoder.forward(&input).unwrap();
        let tokens = output.tokens.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let cls = output.cls.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let parity = compare(
            &tokens,
            &reference.tokens,
            &reference.pooled,
            Some((&cls, &reference.cls)),
            reference.dim,
        );
        assert!(parity.min_token_cosine >= 0.999, "{bias:?}: {parity:?}");
        assert!(parity.pooled_cosine >= 0.999, "{bias:?}: {parity:?}");
        // fp32 on the GPU should agree far better than the gate
        assert!(parity.max_abs_diff < 1e-3, "{bias:?}: {parity:?}");
        assert!(parity.min_centered_cosine > 0.99999, "{bias:?}: {parity:?}");
    }
}

/// `tokens::in_mask_patches` reproduces einops' `patchify3d` and dense_embed's mask selection:
/// re-patchifying the p8 reference's image at patch 4 gives the p4 reference's patches exactly.
#[test]
fn patchify_matches_the_p4_reference() {
    let p8 = reference_path("nanobrain-vits-p8-ref");
    let p4 = reference_path("nanobrain-vits-p4-ref");
    if !p8.exists() || !p4.exists() {
        return;
    }
    let p8 = EncodeReference::load(&p8).unwrap();
    let p4 = EncodeReference::load(&p4).unwrap();
    let ours = in_mask_patches(&p8.image, &p8.mask, p8.shape, 4).unwrap();
    assert_eq!(ours.coords, p4.patches.coords);
    assert_eq!(ours.values, p4.patches.values);
    // and at patch 8 it reproduces the p8 reference's own patches
    let ours = in_mask_patches(&p8.image, &p8.mask, p8.shape, 8).unwrap();
    assert_eq!(ours, p8.patches);
}
