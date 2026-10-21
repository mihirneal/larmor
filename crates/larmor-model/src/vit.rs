//! The 3D ViT encoder, layer for layer as in `nanobrain/modules.py` and `nanobrain/model.py`.
//!
//! Activations are `[tokens, width]` matrices: one scan at a time, so no batch axis except inside
//! attention. Linears, attention and GELU compute in [`Precision::compute_dtype`]; the residual
//! stream, LayerNorms, position embeddings and class tokens in [`Precision::residual_dtype`].

use candle_core::{DType, Tensor};
use candle_nn::{LayerNorm, Module};

use crate::attention::{AttentionPlan, attend};
use crate::device::Precision;
use crate::error::EncoderError;
use crate::manifest::Architecture;
use crate::weights::WeightReader;

/// The encoder half of `nanobrain.model.ViTMAE3D`.
pub(crate) struct VisionTransformer {
    patch_embed: Dense,
    pos_embed: SeparablePosEmbed3D,
    /// `[class_tokens, width]`, or `None` without class tokens.
    cls_token: Option<Tensor>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    compute: DType,
    residual: DType,
}

/// `modules.py::SeparablePosEmbed3D`: one learned table per axis, looked up and summed.
struct SeparablePosEmbed3D {
    /// `weight_x`, `weight_y`, `weight_z`, each `[grid size, width]`.
    tables: [Tensor; 3],
}

/// `modules.py::Block`: pre-norm attention and MLP, each with a residual.
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

/// `modules.py::Attention` with separate q, k, v projections.
struct Attention {
    /// nanobrain's q, k and v linears side by side: one matmul gives `[tokens, 3 × width]`.
    /// Each output column is the same dot product as in the separate linears.
    qkv: Dense,
    proj: Dense,
    heads: usize,
    /// `F.scaled_dot_product_attention`'s default, `1 / √head_dim`.
    scale: f64,
}

/// `modules.py::Mlp`: fc1, exact GELU, fc2.
struct Mlp {
    fc1: Dense,
    fc2: Dense,
}

/// `nn.Linear` as one matrix `[in (+ 1), out]` for `x @ W`, with the bias (if any) as its last
/// row; see `WeightReader::linear`.
struct Dense {
    weight: Tensor,
    bias: bool,
}

impl VisionTransformer {
    /// Build the encoder from `naming = "nanobrain"` tensors.
    pub(crate) fn load(
        arch: &Architecture,
        weights: &mut WeightReader,
        precision: Precision,
    ) -> Result<Self, EncoderError> {
        let (width, compute, residual) = (arch.width, precision.compute_dtype(), precision.residual_dtype());
        let patch_embed =
            Dense::load(weights, &["patch_embed".into()], arch.patch_dim(), width, true, compute)?;
        let [gx, gy, gz] = arch.patch_grid();
        let mut table =
            |axis: &str, size| weights.tensor(&format!("pos_embed.weight_{axis}"), &[size, width], residual);
        let pos_embed = SeparablePosEmbed3D { tables: [table("x", gx)?, table("y", gy)?, table("z", gz)?] };
        let cls_token = match arch.class_tokens {
            0 => None,
            count => Some(weights.tensor("cls_token", &[1, count, width], residual)?.squeeze(0)?),
        };
        let blocks = (0..arch.depth)
            .map(|i| Block::load(weights, &format!("blocks.{i}"), arch, precision))
            .collect::<Result<_, _>>()?;
        let norm = load_layer_norm(weights, "norm", width, arch.norm_eps, residual)?;
        Ok(Self { patch_embed, pos_embed, cls_token, blocks, norm, compute, residual })
    }

    /// Encoder output for one scan, `[class_tokens + N, width]` in f32, class tokens first.
    ///
    /// Mirrors `model.py::ViTMAE3D.forward_encoder` with batch size 1. `patches` is `[N, p³]`
    /// f32; `coords` are the three u32 patch-grid index columns (`x`, `y`, `z`), each `[N]`.
    pub(crate) fn forward(
        &self,
        patches: &Tensor,
        coords: &[Tensor; 3],
        plan: AttentionPlan,
    ) -> candle_core::Result<Tensor> {
        let x = self.patch_embed.forward(&patches.to_dtype(self.compute)?)?.to_dtype(self.residual)?;
        let x = self.pos_embed.forward(x, coords)?;
        let mut x = match &self.cls_token {
            Some(cls) => Tensor::cat(&[cls, &x], 0)?,
            None => x,
        };
        for block in &self.blocks {
            x = block.forward(&x, self.compute, self.residual, plan)?;
        }
        self.norm.forward(&x)?.to_dtype(DType::F32)
    }
}

impl SeparablePosEmbed3D {
    /// `x + weight_x[ids_x] + weight_y[ids_y] + weight_z[ids_z]`, added in that order.
    ///
    /// The Python indexes by `floor(coord / unit)` with `unit = 1.0`; for the integer patch
    /// coordinates nanobrain passes, that is the coordinate itself.
    fn forward(&self, mut x: Tensor, coords: &[Tensor; 3]) -> candle_core::Result<Tensor> {
        for (table, ids) in self.tables.iter().zip(coords) {
            x = (x + table.index_select(ids, 0)?)?;
        }
        Ok(x)
    }
}

impl Block {
    fn load(
        weights: &mut WeightReader,
        prefix: &str,
        arch: &Architecture,
        precision: Precision,
    ) -> Result<Self, EncoderError> {
        let (width, hidden, compute) = (arch.width, arch.mlp_hidden(), precision.compute_dtype());
        let norm = |weights: &mut WeightReader, name: &str| {
            load_layer_norm(
                weights,
                &format!("{prefix}.{name}"),
                width,
                arch.norm_eps,
                precision.residual_dtype(),
            )
        };
        let linear = |weights: &mut WeightReader, names: &[&str], d_in, d_out, bias| {
            let names: Vec<String> = names.iter().map(|name| format!("{prefix}.{name}")).collect();
            Dense::load(weights, &names, d_in, d_out, bias, compute)
        };
        let norm1 = norm(weights, "norm1")?;
        let attn = Attention {
            qkv: linear(weights, &["attn.q", "attn.k", "attn.v"], width, width, arch.qkv_bias)?,
            proj: linear(weights, &["attn.proj"], width, width, arch.proj_bias)?,
            heads: arch.heads,
            scale: 1.0 / (arch.head_dim() as f64).sqrt(),
        };
        let norm2 = norm(weights, "norm2")?;
        // nanobrain's Block passes proj_bias to the MLP as well
        let mlp = Mlp {
            fc1: linear(weights, &["mlp.fc1"], width, hidden, arch.proj_bias)?,
            fc2: linear(weights, &["mlp.fc2"], hidden, width, arch.proj_bias)?,
        };
        Ok(Self { norm1, attn, norm2, mlp })
    }

    /// `x + attn(norm1(x))`, then `x + mlp(norm2(x))`. Mirrors `Block.forward` at inference,
    /// where drop path is the identity.
    fn forward(
        &self,
        x: &Tensor,
        compute: DType,
        residual: DType,
        plan: AttentionPlan,
    ) -> candle_core::Result<Tensor> {
        let h = self.norm1.forward(x)?.to_dtype(compute)?;
        let x = (x + self.attn.forward(&h, plan)?.to_dtype(residual)?)?;
        let h = self.norm2.forward(&x)?.to_dtype(compute)?;
        &x + self.mlp.forward(&h)?.to_dtype(residual)?
    }
}

impl Attention {
    /// `Attention.forward` for self-attention: no context, no mask, not causal.
    fn forward(&self, x: &Tensor, plan: AttentionPlan) -> candle_core::Result<Tensor> {
        let (tokens, width) = x.dims2()?;
        let head_dim = width / self.heads;
        // [N, 3D] -> [3, 1, heads, N, head_dim], as `reshape(B, N, h, D // h).transpose(1, 2)`
        // on each of q, k and v
        let qkv =
            self.qkv.forward(x)?.reshape((tokens, 3, 1, self.heads, head_dim))?.permute((1, 2, 3, 0, 4))?;
        let (q, k, v) = (qkv.get(0)?, qkv.get(1)?, qkv.get(2)?);
        let out = attend(&q, &k, &v, self.scale, plan)?;
        // [1, heads, N, head_dim] -> [N, D]
        let out = out.squeeze(0)?.transpose(0, 1)?.reshape((tokens, width))?;
        self.proj.forward(&out)
    }
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // nn.GELU()'s default is the exact erf form, not the tanh approximation. Candle's Metal erf
        // is a polynomial (Abramowitz-Stegun 7.1.26, |error| <= 1.5e-7), near f32 rounding.
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

impl Dense {
    /// The linears named by `prefixes` (one, or several sharing an input), side by side.
    fn load(
        weights: &mut WeightReader,
        prefixes: &[String],
        d_in: usize,
        d_out: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Self, EncoderError> {
        Ok(Self { weight: weights.linear(prefixes, d_in, d_out, bias, dtype)?, bias })
    }

    /// `[N, in] -> [N, out]`.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if self.bias {
            let ones = Tensor::ones((x.dim(0)?, 1), x.dtype(), x.device())?;
            Tensor::cat(&[x, &ones], 1)?.matmul(&self.weight)
        } else {
            x.matmul(&self.weight)
        }
    }
}

/// A PyTorch `nn.LayerNorm(width, eps)`.
fn load_layer_norm(
    weights: &mut WeightReader,
    prefix: &str,
    width: usize,
    eps: f64,
    dtype: DType,
) -> Result<LayerNorm, EncoderError> {
    let weight = weights.tensor(&format!("{prefix}.weight"), &[width], dtype)?;
    let bias = weights.tensor(&format!("{prefix}.bias"), &[width], dtype)?;
    Ok(LayerNorm::new(weight, bias, eps))
}
