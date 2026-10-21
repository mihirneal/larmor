//! The 3D ViT encoder, layer for layer as in `nanobrain/modules.py` and `nanobrain/model.py`.
//!
//! The residual stream, LayerNorms and position embeddings are always f32. Linears and attention
//! run in `compute` (f32, f16 or bf16), which is what PyTorch autocast does: a half-precision
//! linear output added to an f32 residual promotes back to f32.

use burn::module::Param;
use burn::nn::{LayerNorm, LayerNormConfig, Linear};
use burn::tensor::activation::gelu;
use burn::tensor::{FloatDType, Int, Tensor};

use crate::attention::{AttentionOptions, chunked_attention};
use crate::error::EncoderError;
use crate::manifest::Architecture;
use crate::weights::WeightReader;

/// The encoder half of `nanobrain.model.ViTMAE3D`.
#[derive(Debug)]
pub struct VisionTransformer {
    patch_embed: Linear,
    pos_embed: SeparablePosEmbed3D,
    /// `[class_tokens, width]`, or `None` without class tokens.
    cls_token: Option<Tensor<2>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    compute: FloatDType,
}

/// `modules.py::SeparablePosEmbed3D`: one learned table per axis, looked up and summed.
#[derive(Debug)]
struct SeparablePosEmbed3D {
    weight_x: Tensor<2>,
    weight_y: Tensor<2>,
    weight_z: Tensor<2>,
}

/// `modules.py::Block`: pre-norm attention and MLP, each with a residual.
#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
    compute: FloatDType,
}

/// `modules.py::Attention` with separate q, k, v projections.
#[derive(Debug)]
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    proj: Linear,
    heads: usize,
    options: AttentionOptions,
}

/// `modules.py::Mlp`: fc1, exact GELU, fc2.
#[derive(Debug)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionTransformer {
    /// Build the encoder from `naming = "nanobrain"` tensors. `compute` is the dtype of the
    /// linears and attention.
    pub(crate) fn load(
        arch: &Architecture,
        weights: &mut WeightReader,
        compute: FloatDType,
        attention: AttentionOptions,
    ) -> Result<Self, EncoderError> {
        let width = arch.width;
        let patch_embed = load_linear(weights, "patch_embed", arch.patch_dim(), width, true, compute)?;
        let pos_embed = SeparablePosEmbed3D::load(weights, arch.patch_grid(), width)?;
        let cls_token = match arch.class_tokens {
            0 => None,
            count => {
                let cls = weights.tensor("cls_token", [1, count, width], FloatDType::F32)?;
                Some(cls.reshape([count, width]))
            }
        };
        let blocks = (0..arch.depth)
            .map(|i| Block::load(weights, &format!("blocks.{i}"), arch, compute, attention))
            .collect::<Result<_, _>>()?;
        let norm = load_layer_norm(weights, "norm", width, arch.norm_eps)?;
        Ok(Self { patch_embed, pos_embed, cls_token, blocks, norm, compute })
    }

    /// Encoder output for one scan, `[class_tokens + N, width]` in f32, class tokens first.
    ///
    /// Mirrors `model.py::ViTMAE3D.forward_encoder` with batch size 1. `patches` is `[N, p³]`;
    /// `coords` are the three patch-grid index columns (`x`, `y`, `z`), each `[N]`.
    pub fn forward(&self, patches: Tensor<2>, coords: [Tensor<1, Int>; 3]) -> Tensor<2> {
        let x = self.patch_embed.forward(patches.cast(self.compute)).cast(FloatDType::F32);
        let x = self.pos_embed.forward(x, coords);
        let x = match &self.cls_token {
            Some(cls) => Tensor::cat(vec![cls.clone(), x], 0),
            None => x,
        };
        let x = self.blocks.iter().fold(x.unsqueeze::<3>(), |x, block| block.forward(x));
        self.norm.forward(x).squeeze_dim(0)
    }
}

impl SeparablePosEmbed3D {
    fn load(weights: &mut WeightReader, grid: [usize; 3], width: usize) -> Result<Self, EncoderError> {
        let [gx, gy, gz] = grid;
        Ok(Self {
            weight_x: weights.tensor("pos_embed.weight_x", [gx, width], FloatDType::F32)?,
            weight_y: weights.tensor("pos_embed.weight_y", [gy, width], FloatDType::F32)?,
            weight_z: weights.tensor("pos_embed.weight_z", [gz, width], FloatDType::F32)?,
        })
    }

    /// `x + weight_x[ids_x] + weight_y[ids_y] + weight_z[ids_z]`, added in that order.
    ///
    /// The Python indexes by `floor(coord / unit)` with `unit = 1.0`; for the integer patch
    /// coordinates nanobrain passes, that is the coordinate itself.
    fn forward(&self, x: Tensor<2>, [ids_x, ids_y, ids_z]: [Tensor<1, Int>; 3]) -> Tensor<2> {
        let x = x + self.weight_x.clone().select(0, ids_x);
        let x = x + self.weight_y.clone().select(0, ids_y);
        x + self.weight_z.clone().select(0, ids_z)
    }
}

impl Block {
    fn load(
        weights: &mut WeightReader,
        prefix: &str,
        arch: &Architecture,
        compute: FloatDType,
        options: AttentionOptions,
    ) -> Result<Self, EncoderError> {
        let (width, hidden) = (arch.width, arch.mlp_hidden());
        let linear = |weights: &mut WeightReader, name: &str, d_in, d_out, bias| {
            load_linear(weights, &format!("{prefix}.{name}"), d_in, d_out, bias, compute)
        };
        let attn = Attention {
            q: linear(weights, "attn.q", width, width, arch.qkv_bias)?,
            k: linear(weights, "attn.k", width, width, arch.qkv_bias)?,
            v: linear(weights, "attn.v", width, width, arch.qkv_bias)?,
            proj: linear(weights, "attn.proj", width, width, arch.proj_bias)?,
            heads: arch.heads,
            options,
        };
        // nanobrain's Block passes proj_bias to the MLP as well
        let mlp = Mlp {
            fc1: linear(weights, "mlp.fc1", width, hidden, arch.proj_bias)?,
            fc2: linear(weights, "mlp.fc2", hidden, width, arch.proj_bias)?,
        };
        Ok(Self {
            norm1: load_layer_norm(weights, &format!("{prefix}.norm1"), width, arch.norm_eps)?,
            attn,
            norm2: load_layer_norm(weights, &format!("{prefix}.norm2"), width, arch.norm_eps)?,
            mlp,
            compute,
        })
    }

    /// `x + attn(norm1(x))`, then `x + mlp(norm2(x))`, with `x` in f32.
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let h = self.norm1.forward(x.clone()).cast(self.compute);
        let x = x + self.attn.forward(h).cast(FloatDType::F32);
        let h = self.norm2.forward(x.clone()).cast(self.compute);
        x + self.mlp.forward(h).cast(FloatDType::F32)
    }
}

impl Attention {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, width] = x.dims();
        let heads = self.heads;
        // [B, N, D] -> [B, heads, N, D / heads], as `reshape(B, N, h, D // h).transpose(1, 2)`
        let split_heads = |t: Tensor<3>| t.reshape([batch, tokens, heads, width / heads]).swap_dims(1, 2);
        let q = split_heads(self.q.forward(x.clone()));
        let k = split_heads(self.k.forward(x.clone()));
        let v = split_heads(self.v.forward(x));
        let out = chunked_attention(q, k, v, &self.options);
        self.proj.forward(out.swap_dims(1, 2).reshape([batch, tokens, width]))
    }
}

impl Mlp {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.fc2.forward(gelu(self.fc1.forward(x)))
    }
}

/// A PyTorch `nn.Linear(d_in, d_out, bias)` as a Burn `Linear` in the compute dtype.
fn load_linear(
    weights: &mut WeightReader,
    prefix: &str,
    d_in: usize,
    d_out: usize,
    bias: bool,
    compute: FloatDType,
) -> Result<Linear, EncoderError> {
    let weight = weights.linear_weight(&format!("{prefix}.weight"), d_in, d_out, compute)?;
    let bias = if bias {
        Some(Param::from_tensor(weights.tensor(&format!("{prefix}.bias"), [d_out], compute)?))
    } else {
        None
    };
    Ok(Linear { weight: Param::from_tensor(weight), bias })
}

/// A PyTorch `nn.LayerNorm(width, eps)` in f32.
fn load_layer_norm(
    weights: &mut WeightReader,
    prefix: &str,
    width: usize,
    eps: f64,
) -> Result<LayerNorm, EncoderError> {
    let mut norm = LayerNormConfig::new(width).with_epsilon(eps).init(weights.device());
    norm.gamma = Param::from_tensor(weights.tensor(&format!("{prefix}.weight"), [width], FloatDType::F32)?);
    norm.beta =
        Some(Param::from_tensor(weights.tensor(&format!("{prefix}.bias"), [width], FloatDType::F32)?));
    Ok(norm)
}
