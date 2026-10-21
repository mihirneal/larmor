//! Encoder weights in nanobrain's tensor naming (`docs/DESIGN.md`, "Model manifest").

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device, Tensor};

use crate::config::Architecture;

/// The name and PyTorch-layout shape of every encoder tensor an architecture needs.
///
/// Mirrors the encoder half of `nanobrain/model.py::ViTMAE3D.state_dict()` (what
/// `tools/make_models.py::encoder_state` exports). Linear weights are `[out, in]`.
pub fn expected_shapes(arch: &Architecture) -> Vec<(String, Vec<usize>)> {
    let d = arch.width;
    let hidden = arch.mlp_hidden();
    let [gx, gy, gz] = arch.patch_grid();
    let mut shapes = vec![
        ("patch_embed.weight".to_string(), vec![d, arch.patch_dim()]),
        ("patch_embed.bias".to_string(), vec![d]),
        ("pos_embed.weight_x".to_string(), vec![gx, d]),
        ("pos_embed.weight_y".to_string(), vec![gy, d]),
        ("pos_embed.weight_z".to_string(), vec![gz, d]),
        ("norm.weight".to_string(), vec![d]),
        ("norm.bias".to_string(), vec![d]),
    ];
    if arch.class_tokens > 0 {
        shapes.push(("cls_token".to_string(), vec![1, arch.class_tokens, d]));
    }
    for i in 0..arch.depth {
        let block = |name: &str| format!("blocks.{i}.{name}");
        for norm in ["norm1", "norm2"] {
            shapes.push((block(&format!("{norm}.weight")), vec![d]));
            shapes.push((block(&format!("{norm}.bias")), vec![d]));
        }
        for (linear, out, inp, has_bias) in [
            ("attn.q", d, d, arch.qkv_bias),
            ("attn.k", d, d, arch.qkv_bias),
            ("attn.v", d, d, arch.qkv_bias),
            ("attn.proj", d, d, arch.proj_bias),
            ("mlp.fc1", hidden, d, arch.proj_bias),
            ("mlp.fc2", d, hidden, arch.proj_bias),
        ] {
            shapes.push((block(&format!("{linear}.weight")), vec![out, inp]));
            if has_bias {
                shapes.push((block(&format!("{linear}.bias")), vec![out]));
            }
        }
    }
    shapes
}

/// Named f32 encoder tensors on one device; `get` casts each to the dtype a layer wants.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
}

impl Weights {
    /// Loads a nanobrain-naming safetensors file and checks it holds exactly the tensors the
    /// architecture needs, with the right shapes.
    pub fn load(path: &Path, arch: &Architecture, device: &Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("loading weights {}", path.display()))?;
        let weights = Self { tensors };
        weights.check(arch)?;
        Ok(weights)
    }

    /// Random weights with the architecture's shapes, for timing only. Linear and embedding
    /// weights are N(0, 0.02), LayerNorm weights 1, biases N(0, 0.02).
    pub fn random(arch: &Architecture, device: &Device) -> Result<Self> {
        let mut tensors = HashMap::new();
        for (name, shape) in expected_shapes(arch) {
            let is_norm_weight = name.ends_with("weight") && name.contains("norm");
            let tensor = if is_norm_weight {
                Tensor::ones(shape.as_slice(), DType::F32, device)?
            } else {
                Tensor::randn(0f32, 0.02, shape.as_slice(), device)?
            };
            tensors.insert(name, tensor);
        }
        Ok(Self { tensors })
    }

    fn check(&self, arch: &Architecture) -> Result<()> {
        let expected = expected_shapes(arch);
        for (name, shape) in &expected {
            let tensor = self.tensors.get(name).with_context(|| format!("weights are missing {name}"))?;
            ensure!(
                tensor.dims() == shape.as_slice(),
                "{name} has shape {:?}, expected {shape:?}",
                tensor.dims()
            );
        }
        if self.tensors.len() != expected.len() {
            let known: Vec<&str> = expected.iter().map(|(name, _)| name.as_str()).collect();
            let extra: Vec<&String> = self.tensors.keys().filter(|k| !known.contains(&k.as_str())).collect();
            bail!("weights have unexpected tensors: {extra:?}");
        }
        Ok(())
    }

    /// The tensor called `name`, cast to `dtype`.
    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let tensor = self.tensors.get(name).with_context(|| format!("weights are missing {name}"))?;
        Ok(tensor.to_dtype(dtype)?)
    }

    /// Like `get`, but `None` when the tensor is absent (a bias the architecture disables).
    pub fn get_optional(&self, name: &str, dtype: DType) -> Result<Option<Tensor>> {
        match self.tensors.get(name) {
            Some(tensor) => Ok(Some(tensor.to_dtype(dtype)?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vits_tensor_count_matches_export() {
        // 7 globals + cls_token + 16 per block (2 norms x 2, 6 linears x 2)
        let shapes = expected_shapes(&Architecture::vit(384, 12, 6, 8));
        assert_eq!(shapes.len(), 8 + 12 * 16);
    }

    #[test]
    fn random_weights_pass_their_own_check() {
        let arch = Architecture::vit(64, 2, 2, 8);
        let weights = Weights::random(&arch, &Device::Cpu).unwrap();
        weights.check(&arch).unwrap();
    }
}
