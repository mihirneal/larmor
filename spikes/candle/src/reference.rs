//! Encode reference files written by `tools/make_reference.py::run_encode`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use safetensors::{Dtype, SafeTensors};

use crate::tokens::{Patches, in_mask_patches};

/// The tensors of `fixtures/reference/encode/<model_id>/<scan>.safetensors`
/// (`docs/DESIGN.md`, "Encode reference tensors").
pub struct EncodeReference {
    pub metadata: HashMap<String, String>,
    /// Normalized model input `[X, Y, Z]`, zeros outside the mask.
    pub image: Vec<f32>,
    pub mask: Vec<u8>,
    pub shape: [usize; 3],
    /// The in-mask patches the reference encoder was fed.
    pub patches: Patches,
    pub patch_dim: usize,
    /// fp32 encoder outputs, `[N, D]`, `[C, D]` and `[D]`.
    pub tokens: Vec<f32>,
    pub cls: Vec<f32>,
    pub pooled: Vec<f32>,
    /// The same under `torch.autocast("cpu", dtype=torch.bfloat16)`.
    pub tokens_bf16: Vec<f32>,
    pub pooled_bf16: Vec<f32>,
    pub dim: usize,
}

impl EncodeReference {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let metadata = header.metadata().clone().unwrap_or_default();
        let st = SafeTensors::deserialize(&bytes)?;

        let (image, image_shape) = read_f32(&st, "image")?;
        let (mask, _) = read_u8(&st, "mask")?;
        let (values, patch_shape) = read_f32(&st, "patches")?;
        let (coords, coord_shape) = read_i32(&st, "coords")?;
        let (tokens, token_shape) = read_f32(&st, "tokens")?;
        let (cls, _) = read_f32(&st, "cls")?;
        let (pooled, _) = read_f32(&st, "pooled")?;
        let (tokens_bf16, _) = read_f32(&st, "tokens_bf16")?;
        let (pooled_bf16, _) = read_f32(&st, "pooled_bf16")?;

        ensure!(image_shape.len() == 3 && coord_shape[1] == 3, "unexpected image or coords shape");
        ensure!(
            patch_shape[0] == coord_shape[0] && token_shape[0] == coord_shape[0],
            "token counts disagree"
        );
        let coords = coords.as_chunks::<3>().0.iter().map(|c| c.map(|v| v as u32)).collect();
        Ok(Self {
            metadata,
            image,
            mask,
            shape: [image_shape[0], image_shape[1], image_shape[2]],
            patches: Patches { values, coords },
            patch_dim: patch_shape[1],
            tokens,
            cls,
            pooled,
            tokens_bf16,
            pooled_bf16,
            dim: token_shape[1],
        })
    }

    pub fn len(&self) -> usize {
        self.patches.coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The reference's own patches when they have `patch^3` voxels, otherwise its normalized
    /// image re-patchified at `patch` (for timing a model with another patch size).
    pub fn patches_for(&self, patch: usize) -> Result<Patches> {
        if self.patch_dim == patch.pow(3) {
            Ok(self.patches.clone())
        } else {
            in_mask_patches(&self.image, &self.mask, self.shape, patch)
        }
    }
}

fn view<'a>(
    st: &'a SafeTensors<'a>,
    name: &str,
    dtype: Dtype,
) -> Result<safetensors::tensor::TensorView<'a>> {
    let view = st.tensor(name).with_context(|| format!("reference has no {name}"))?;
    ensure!(view.dtype() == dtype, "{name} is {:?}, expected {dtype:?}", view.dtype());
    Ok(view)
}

fn read_f32(st: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let view = view(st, name, Dtype::F32)?;
    let data = view.data().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    Ok((data, view.shape().to_vec()))
}

fn read_i32(st: &SafeTensors, name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
    let view = view(st, name, Dtype::I32)?;
    let data = view.data().as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b)).collect();
    Ok((data, view.shape().to_vec()))
}

fn read_u8(st: &SafeTensors, name: &str) -> Result<(Vec<u8>, Vec<usize>)> {
    let view = view(st, name, Dtype::U8)?;
    Ok((view.data().to_vec(), view.shape().to_vec()))
}
