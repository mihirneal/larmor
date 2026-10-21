//! Unit tests that need no fixtures: the committed tiny model (`tests/data/make_tiny.py`), random
//! weights and small attention problems. All run on the CPU unless marked as needing a GPU.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use larmor_model::attention::{AttentionKernel, AttentionOptions, chunked_attention, fused_attention};
use larmor_model::parity::{EncodeReference, compare, cosine};
use larmor_model::synthetic::random_safetensors;
use larmor_model::{
    Architecture, DeviceChoice, Encoder, EncoderError, EncoderOptions, Manifest, ManifestError, Precision,
};

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn tiny_manifest() -> Manifest {
    Manifest::load(data("tiny-vit.toml")).expect("tiny manifest")
}

fn tiny_encoder(options: EncoderOptions) -> Encoder {
    Encoder::load_with(&tiny_manifest(), options).expect("tiny encoder")
}

fn cpu() -> EncoderOptions {
    EncoderOptions { device: DeviceChoice::Cpu, ..EncoderOptions::default() }
}

fn tiny_reference() -> EncodeReference {
    EncodeReference::load(data("tiny-vit-encode.safetensors")).expect("tiny reference")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
}

// ---- manifest ----

#[test]
fn manifest_parses_and_resolves_weights() {
    let manifest = tiny_manifest();
    assert_eq!(manifest.id, "tiny-vit");
    assert_eq!(manifest.architecture.grid, [16, 24, 12]);
    assert_eq!(manifest.architecture.patch_grid(), [4, 6, 3]);
    assert_eq!(manifest.architecture.head_dim(), 8);
    assert_eq!(manifest.architecture.mlp_hidden(), 128);
    assert_eq!(manifest.weights_path(), data("tiny-vit.safetensors"));
    assert_eq!(manifest.preprocessing.steps.len(), 4);
    assert_eq!(manifest.preprocessing.steps[0]["op"].as_str(), Some("threshold_mask"));
    manifest.verify_weights().expect("sha256 matches");
}

fn tiny_text() -> String {
    std::fs::read_to_string(data("tiny-vit.toml")).unwrap()
}

fn parse_edited(from: &str, to: &str) -> Result<Manifest, ManifestError> {
    let text = tiny_text();
    assert!(text.contains(from), "test edit {from:?} not found");
    Manifest::parse(&text.replacen(from, to, 1), data(""))
}

#[test]
fn manifest_rejects_unsupported_values() {
    let cases = [
        ("qkv = \"separate\"", "qkv = \"fused\"", "architecture.qkv"),
        ("activation = \"gelu\"", "activation = \"gelu_tanh\"", "architecture.activation"),
        ("pos_embed = \"separable\"", "pos_embed = \"learned\"", "architecture.pos_embed"),
        ("family = \"vit3d\"", "family = \"vit2d\"", "family"),
        ("pool = \"mean\"", "pool = \"cls\"", "readout.pool"),
        ("naming = \"nanobrain\"", "naming = \"timm\"", "weights.naming"),
    ];
    for (from, to, field) in cases {
        match parse_edited(from, to) {
            Err(ManifestError::Unsupported { field: f, .. }) => assert_eq!(f, field),
            other => panic!("{to}: expected Unsupported({field}), got {other:?}"),
        }
    }
}

#[test]
fn manifest_rejects_inconsistent_shapes() {
    for (from, to, field) in [
        ("heads = 4", "heads = 5", "architecture.heads"),
        ("grid = [16, 24, 12]", "grid = [16, 24, 10]", "architecture.grid"),
        ("norm_eps = 1e-06", "norm_eps = 0.0", "architecture.norm_eps"),
        ("depth = 2", "depth = 0", "architecture.depth"),
    ] {
        match parse_edited(from, to) {
            Err(ManifestError::Invalid { field: f, .. }) => assert_eq!(f, field),
            other => panic!("{to}: expected Invalid({field}), got {other:?}"),
        }
    }
}

#[test]
fn manifest_rejects_unknown_keys() {
    let err = parse_edited("qkv_bias = true", "qkv_bias = true\nqkv_bais = true").unwrap_err();
    assert!(matches!(err, ManifestError::Parse(_)), "{err:?}");
    assert!(err.to_string().contains("qkv_bais"), "{err}");
}

#[test]
fn manifest_checksum_mismatch_is_an_error() {
    let manifest = parse_edited("sha256 = \"29c7", "sha256 = \"39c7").unwrap();
    assert!(matches!(manifest.verify_weights(), Err(ManifestError::Checksum { .. })));
    assert!(matches!(Encoder::load_with(&manifest, cpu()), Err(EncoderError::Manifest(_))));
}

// ---- the encoder against PyTorch ----

#[test]
fn tiny_model_matches_pytorch_on_cpu() {
    let reference = tiny_reference();
    let encoder = tiny_encoder(cpu());
    assert_eq!(encoder.options().device, DeviceChoice::Cpu);
    assert_eq!(encoder.options().attention.kernel, AttentionKernel::Chunked);
    let encoded = encoder.encode(&reference.patches, &reference.coords).unwrap();
    assert_eq!((encoded.n_tokens, encoded.width), (50, 32));
    assert_eq!(encoded.cls.len(), 2 * 32);
    assert!(max_abs_diff(&encoded.tokens, &reference.tokens) < 1e-5);
    assert!(max_abs_diff(&encoded.cls, &reference.cls) < 1e-5);
    assert!(max_abs_diff(&encoded.pooled, &reference.pooled) < 1e-5);
    let parity = compare(&encoded, &reference);
    assert!(parity.min_centered_cosine > 0.99999, "{parity}");
}

#[test]
fn chunk_size_does_not_change_the_encoding() {
    let reference = tiny_reference();
    let encode = |chunk_rows| {
        let attention = AttentionOptions { chunk_rows, ..AttentionOptions::default() };
        tiny_encoder(EncoderOptions { attention, ..cpu() })
            .encode(&reference.patches, &reference.coords)
            .unwrap()
    };
    let whole = encode(None);
    // 52 tokens: chunks of 1, of 7 (the last one short) and one chunk larger than the sequence
    for rows in [1, 7, 1000] {
        let other = encode(Some(rows));
        assert!(max_abs_diff(&whole.tokens, &other.tokens) < 1e-5, "{rows} rows");
    }
}

#[test]
fn tokens_follow_their_coordinates_not_their_order() {
    // attention without a mask is permutation-equivariant, so reversing the input reverses the output
    let reference = tiny_reference();
    let encoder = tiny_encoder(cpu());
    let forward = encoder.encode(&reference.patches, &reference.coords).unwrap();
    let p = encoder.architecture().patch_dim();
    let patches: Vec<f32> = reference.patches.chunks_exact(p).rev().flatten().copied().collect();
    let coords: Vec<[u16; 3]> = reference.coords.iter().rev().copied().collect();
    let backward = encoder.encode(&patches, &coords).unwrap();
    let width = encoder.architecture().width;
    let reversed: Vec<f32> = backward.tokens.chunks_exact(width).rev().flatten().copied().collect();
    assert!(max_abs_diff(&forward.tokens, &reversed) < 1e-5);

    // moving one patch along each axis in turn changes its token
    for axis in 0..3 {
        let mut moved = reference.coords.clone();
        let grid = encoder.architecture().patch_grid()[axis] as u16;
        moved[0][axis] = (moved[0][axis] + 1) % grid;
        let encoded = encoder.encode(&reference.patches, &moved).unwrap();
        assert!(cosine(&encoded.tokens[..32], &forward.tokens[..32]) < 0.9999, "axis {axis} ignored");
    }
}

#[test]
fn half_precision_is_refused_on_cpu() {
    // Candle's CPU backend has no bf16 matmul and accumulates f16 in f16 (see `Precision`);
    // `tiny_model_matches_pytorch_on_gpu` covers the half precisions
    for precision in [Precision::F16, Precision::Bf16] {
        let options = EncoderOptions { precision, ..cpu() };
        let err = Encoder::load_with(&tiny_manifest(), options).unwrap_err();
        assert!(matches!(err, EncoderError::Device(_)), "{precision}: {err}");
    }
}

// ---- inputs, weights and options that do not fit ----

#[test]
fn bad_inputs_are_errors() {
    let reference = tiny_reference();
    let encoder = tiny_encoder(cpu());
    let short = &reference.patches[..reference.patches.len() - 1];
    assert!(matches!(encoder.encode(short, &reference.coords), Err(EncoderError::Input(_))));
    let mut off_grid = reference.coords.clone();
    off_grid[3][1] = 6;
    assert!(matches!(encoder.encode(&reference.patches, &off_grid), Err(EncoderError::Input(_))));
    assert!(matches!(encoder.encode(&[], &[]), Err(EncoderError::Input(_))));
}

#[test]
fn attention_over_budget_is_refused() {
    let reference = tiny_reference();
    // 52 tokens x 4 heads x 2 buffers x 4 bytes = 1664 bytes per query row
    let attention = AttentionOptions { score_budget_bytes: 1000, ..AttentionOptions::default() };
    let encoder = tiny_encoder(EncoderOptions { attention, ..cpu() });
    let err = encoder.encode(&reference.patches, &reference.coords).unwrap_err();
    assert!(matches!(err, EncoderError::MemoryBudget { needed_bytes: 1664, .. }), "{err}");

    // one chunk of every row, a row over budget
    let attention = AttentionOptions {
        chunk_rows: Some(52),
        score_budget_bytes: 51 * 1664,
        ..AttentionOptions::default()
    };
    let encoder = tiny_encoder(EncoderOptions { attention, ..cpu() });
    let err = encoder.encode(&reference.patches, &reference.coords).unwrap_err();
    assert!(matches!(err, EncoderError::MemoryBudget { .. }), "{err}");
}

#[test]
fn fused_attention_is_refused_where_it_does_not_exist() {
    // the CPU has no fused kernel
    let attention = AttentionOptions { kernel: AttentionKernel::Fused, ..AttentionOptions::default() };
    let err = Encoder::load_with(&tiny_manifest(), EncoderOptions { attention, ..cpu() }).unwrap_err();
    assert!(matches!(err, EncoderError::Device(_)), "{err}");
}

#[test]
fn missing_and_extra_tensors_are_errors() {
    let arch = tiny_manifest().architecture;
    let without_proj_bias = Architecture { proj_bias: false, ..arch.clone() };
    let bytes = random_safetensors(&without_proj_bias, 0);
    let err = Encoder::from_safetensors(&arch, &bytes, cpu()).unwrap_err();
    assert!(matches!(err, EncoderError::MissingTensor(ref name) if name.ends_with("proj.bias")), "{err}");

    let err =
        Encoder::from_safetensors(&without_proj_bias, &random_safetensors(&arch, 0), cpu()).unwrap_err();
    assert!(
        matches!(err, EncoderError::UnexpectedTensors(ref names) if names.len() == 3 * arch.depth),
        "{err}"
    );
}

#[test]
fn random_weights_give_the_right_shapes() {
    let base = tiny_manifest().architecture;
    let arch = Architecture { class_tokens: 0, qkv_bias: false, mlp_ratio: 2.5, ..base };
    let encoder = Encoder::from_safetensors(&arch, &random_safetensors(&arch, 7), cpu()).unwrap();
    let coords = vec![[0, 0, 0], [3, 5, 2], [1, 2, 1]];
    let patches = vec![0.5; coords.len() * arch.patch_dim()];
    let encoded = encoder.encode(&patches, &coords).unwrap();
    assert_eq!(encoded.tokens.len(), 3 * 32);
    assert!(encoded.cls.is_empty());
    assert_eq!(encoded.pooled.len(), 32);
    assert!(encoded.tokens.iter().all(|v| v.is_finite()));
}

// ---- attention ----

/// Deterministic values in [-2, 2), so the test needs no random generator.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    (0..len as u64)
        .map(|i| {
            let x = (i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) as f32;
            4.0 * x / (1u64 << 24) as f32 - 2.0
        })
        .collect()
}

/// Unchunked reference attention on the host, in f64.
fn host_attention(q: &[f32], k: &[f32], v: &[f32], [b, h, n, d]: [usize; 4]) -> Vec<f32> {
    let mut out = vec![0f32; b * h * n * d];
    let scale = 1.0 / (d as f64).sqrt();
    for bh in 0..b * h {
        let (q, k, v) = (&q[bh * n * d..], &k[bh * n * d..], &v[bh * n * d..]);
        for i in 0..n {
            let scores: Vec<f64> = (0..n)
                .map(|j| {
                    (0..d).map(|c| f64::from(q[i * d + c]) * f64::from(k[j * d + c])).sum::<f64>() * scale
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for c in 0..d {
                let value: f64 = (0..n).map(|j| weights[j] * f64::from(v[j * d + c])).sum();
                out[(bh * n + i) * d + c] = (value / total) as f32;
            }
        }
    }
    out
}

fn qkv(shape: [usize; 4], device: &Device) -> ([Vec<f32>; 3], [Tensor; 3]) {
    let len = shape.iter().product();
    let host = [1, 2, 3].map(|seed| pseudo_random(len, seed));
    let tensors = host.clone().map(|values| Tensor::from_vec(values, shape.as_slice(), device).unwrap());
    (host, tensors)
}

fn to_host(tensor: &Tensor) -> Vec<f32> {
    tensor.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

#[test]
fn chunked_attention_equals_unchunked() {
    let shape = [1, 3, 200, 16];
    let ([q, k, v], [tq, tk, tv]) = qkv(shape, &Device::Cpu);
    let expected = host_attention(&q, &k, &v, shape);
    let scale = 1.0 / 4.0;
    for rows in [200, 64, 7, 1] {
        let out = chunked_attention(&tq, &tk, &tv, scale, rows).unwrap();
        assert_eq!(out.dims(), shape);
        let diff = max_abs_diff(&to_host(&out), &expected);
        assert!(diff < 1e-5, "{rows} rows: max diff {diff}");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn fused_attention_equals_unchunked_on_metal() {
    let device = Device::new_metal(0).expect("Metal");
    let shape = [1, 2, 300, 64];
    let ([q, k, v], [tq, tk, tv]) = qkv(shape, &device);
    let expected = host_attention(&q, &k, &v, shape);
    for (dtype, tolerance) in [(DType::F32, 1e-4), (DType::F16, 2e-2), (DType::BF16, 6e-2)] {
        let [tq, tk, tv] = [&tq, &tk, &tv].map(|t| t.to_dtype(dtype).unwrap());
        let fused = to_host(&fused_attention(&tq, &tk, &tv, 0.125).unwrap());
        let chunked = to_host(&chunked_attention(&tq, &tk, &tv, 0.125, 64).unwrap());
        for (name, out) in [("fused", fused), ("chunked", chunked)] {
            let diff = max_abs_diff(&out, &expected);
            assert!(diff < tolerance, "{name} {dtype:?}: max diff {diff}");
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn tiny_model_matches_pytorch_on_gpu() {
    let reference = tiny_reference();
    for (precision, tolerance) in [(Precision::F32, 1e-4), (Precision::F16, 2e-2), (Precision::Bf16, 0.1)] {
        let options = EncoderOptions { device: DeviceChoice::Auto, precision, ..EncoderOptions::default() };
        let encoder = tiny_encoder(options);
        assert_ne!(encoder.options().device, DeviceChoice::Cpu, "no GPU found");
        let encoded = encoder.encode(&reference.patches, &reference.coords).unwrap();
        let diff = max_abs_diff(&encoded.tokens, &reference.tokens);
        assert!(diff < tolerance, "{precision}: max diff {diff}");
        let parity = compare(&encoded, &reference);
        assert!(parity.min_token_cosine > 0.999, "{precision}: {parity}");
    }
}

#[test]
#[ignore = "needs a Metal GPU"]
fn fused_and_chunked_encoders_agree_on_metal() {
    // head dim 64, which the fused kernel supports (the tiny model's 8 is not)
    let arch = Architecture { width: 128, heads: 2, ..tiny_manifest().architecture };
    let bytes = random_safetensors(&arch, 3);
    let grid = arch.patch_grid();
    let coords: Vec<[u16; 3]> = (0..grid.iter().product::<usize>())
        .map(|i| [i % grid[0], i / grid[0] % grid[1], i / (grid[0] * grid[1])].map(|c| c as u16))
        .collect();
    let patches = pseudo_random(coords.len() * arch.patch_dim(), 5);
    let encode = |kernel, precision| {
        let attention = AttentionOptions { kernel, ..AttentionOptions::default() };
        let options = EncoderOptions { device: DeviceChoice::Metal, precision, attention };
        let encoder = Encoder::from_safetensors(&arch, &bytes, options).unwrap();
        assert_eq!(encoder.options().attention.kernel, kernel);
        encoder.encode(&patches, &coords).unwrap()
    };
    let cpu = Encoder::from_safetensors(&arch, &bytes, cpu()).unwrap().encode(&patches, &coords).unwrap();
    for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
        let fused = encode(AttentionKernel::Fused, precision);
        let chunked = encode(AttentionKernel::Chunked, precision);
        let tolerance = if precision == Precision::F32 { 1e-4 } else { 0.1 };
        for (name, encoded) in [("fused", &fused), ("chunked", &chunked)] {
            let diff = max_abs_diff(&encoded.tokens, &cpu.tokens);
            assert!(diff < tolerance, "{name} {precision}: max diff {diff} from the CPU");
        }
    }
}
