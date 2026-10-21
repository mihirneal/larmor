//! Random encoder weights for benchmarks and shape tests, where no checkpoint exists.

use std::collections::BTreeMap;

use safetensors::{Dtype, tensor::TensorView};

use crate::manifest::Architecture;

/// The `naming = "nanobrain"` tensor names and shapes the architecture needs, in file order.
///
/// Mirrors the encoder keys of `nanobrain.model.ViTMAE3D.state_dict()` as exported by
/// `tools/make_models.py::encoder_state`.
pub fn tensor_shapes(arch: &Architecture) -> Vec<(String, Vec<usize>)> {
    let (d, hidden) = (arch.width, arch.mlp_hidden());
    let [gx, gy, gz] = arch.patch_grid();
    let mut shapes = vec![
        ("patch_embed.weight".to_string(), vec![d, arch.patch_dim()]),
        ("patch_embed.bias".to_string(), vec![d]),
        ("pos_embed.weight_x".to_string(), vec![gx, d]),
        ("pos_embed.weight_y".to_string(), vec![gy, d]),
        ("pos_embed.weight_z".to_string(), vec![gz, d]),
    ];
    if arch.class_tokens > 0 {
        shapes.push(("cls_token".to_string(), vec![1, arch.class_tokens, d]));
    }
    for i in 0..arch.depth {
        let p = format!("blocks.{i}");
        let mut push = |name: &str, shape: Vec<usize>| shapes.push((format!("{p}.{name}"), shape));
        push("norm1.weight", vec![d]);
        push("norm1.bias", vec![d]);
        for qkv in ["q", "k", "v"] {
            push(&format!("attn.{qkv}.weight"), vec![d, d]);
            if arch.qkv_bias {
                push(&format!("attn.{qkv}.bias"), vec![d]);
            }
        }
        push("attn.proj.weight", vec![d, d]);
        push("norm2.weight", vec![d]);
        push("norm2.bias", vec![d]);
        push("mlp.fc1.weight", vec![hidden, d]);
        push("mlp.fc2.weight", vec![d, hidden]);
        if arch.proj_bias {
            push("attn.proj.bias", vec![d]);
            push("mlp.fc1.bias", vec![hidden]);
            push("mlp.fc2.bias", vec![d]);
        }
    }
    shapes.push(("norm.weight".to_string(), vec![d]));
    shapes.push(("norm.bias".to_string(), vec![d]));
    shapes
}

/// A safetensors file with random weights for `arch`: linear weights uniform within the Xavier
/// bound; everything else uniform with a standard deviation of about 0.02, around 1 for
/// LayerNorm weights and around 0 otherwise.
///
/// The values are deterministic for a given `seed` but are not PyTorch's; there is no parity to
/// check against them.
pub fn random_safetensors(arch: &Architecture, seed: u64) -> Vec<u8> {
    let mut rng = SplitMix64(seed);
    let tensors: BTreeMap<String, (Vec<usize>, Vec<u8>)> = tensor_shapes(arch)
        .into_iter()
        .map(|(name, shape)| {
            let count: usize = shape.iter().product();
            let scale = init_scale(&name, &shape);
            let offset = if name.contains("norm") && name.ends_with("weight") { 1.0 } else { 0.0 };
            let bytes = (0..count)
                .flat_map(|_| (offset + scale * rng.uniform_pm1()).to_le_bytes())
                .collect::<Vec<u8>>();
            (name, (shape, bytes))
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, (shape, bytes))| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).expect("shape matches byte count");
            (name.clone(), view)
        })
        .collect();
    safetensors::serialize(views, None).expect("in-memory serialization cannot fail")
}

/// Half-width of the uniform distribution for one tensor.
fn init_scale(name: &str, shape: &[usize]) -> f32 {
    if name.ends_with("weight") && shape.len() == 2 && !name.starts_with("pos_embed") {
        // xavier_uniform_, as `model.py::_init_weights` does for every nn.Linear
        let (fan_out, fan_in) = (shape[0] as f32, shape[1] as f32);
        (6.0 / (fan_in + fan_out)).sqrt()
    } else {
        // uniform on ±a has std a/√3; 0.035 gives about 0.02
        0.035
    }
}

/// A tiny, fast, seedable generator (Steele et al.'s SplitMix64); good enough for test weights.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`.
    fn uniform_pm1(&mut self) -> f32 {
        // the top 24 bits fill an f32 mantissa exactly
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        2.0 * unit - 1.0
    }
}
