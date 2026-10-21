//! nanobrain's 3D ViT encoder on Candle.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module};

use crate::config::Architecture;
use crate::memory;
use crate::weights::Weights;

/// How the encoder computes attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKernel {
    /// `candle_nn::ops::sdpa`, Candle's fused Metal kernel (MLX "steel" attention). Metal only.
    Fused,
    /// softmax(Q K^T * scale) V from separate matmul and softmax kernels, `chunk` query rows at
    /// a time, so the score matrix is `heads x chunk x tokens` rather than
    /// `heads x tokens x tokens`.
    Chunked { chunk: usize },
}

impl AttentionKernel {
    /// Parses `fused` or `chunked:<rows>`.
    pub fn parse(text: &str) -> Result<Self> {
        match text.split_once(':') {
            None if text == "fused" => Ok(Self::Fused),
            Some(("chunked", rows)) => {
                let chunk: usize = rows.parse().with_context(|| format!("bad chunk size in {text:?}"))?;
                ensure!(chunk > 0, "chunk size must be positive");
                Ok(Self::Chunked { chunk })
            }
            _ => bail!("unknown attention kernel {text:?} (expected fused or chunked:<rows>)"),
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Fused => "fused".into(),
            Self::Chunked { chunk } => format!("chunked:{chunk}"),
        }
    }
}

/// How linear layers add their bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasAdd {
    /// `candle_nn::Linear`: matmul, then a broadcast add. Candle runs broadcast operands through
    /// a strided-index kernel that costs more than the matmul at these shapes.
    Broadcast,
    /// The bias is stored as an extra row of the transposed weight and the input gets a column of
    /// ones, so the matmul adds it (accumulated in the matmul's f32 accumulator).
    Folded,
}

impl BiasAdd {
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "broadcast" => Ok(Self::Broadcast),
            "folded" => Ok(Self::Folded),
            _ => bail!("unknown bias mode {text:?} (broadcast or folded)"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Broadcast => "broadcast",
            Self::Folded => "folded",
        }
    }
}

/// How the encoder runs: dtypes, attention kernel and bias handling. None of these change the
/// architecture; only `precision` moves the numbers by more than rounding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub precision: Precision,
    pub attention: AttentionKernel,
    pub bias: BiasAdd,
}

impl Options {
    /// f32, fused attention, Candle's own `Linear`.
    pub const DEFAULT: Self =
        Self { precision: Precision::F32, attention: AttentionKernel::Fused, bias: BiasAdd::Broadcast };
}

/// The dtypes the encoder computes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Precision {
    /// Linear layers, attention and GELU.
    pub compute: DType,
    /// Residual stream, LayerNorms, position embeddings and class tokens.
    pub residual: DType,
}

impl Precision {
    pub const F32: Self = Self { compute: DType::F32, residual: DType::F32 };
    /// Everything in f16, like `model.half()`. Candle's LayerNorm and softmax kernels still
    /// reduce in f32 internally.
    pub const F16: Self = Self { compute: DType::F16, residual: DType::F16 };
    /// Everything in bf16, like `model.bfloat16()`.
    pub const BF16: Self = Self { compute: DType::BF16, residual: DType::BF16 };
    /// Like `torch.autocast(dtype=float16)`: matmuls and attention in f16, the residual stream
    /// and LayerNorms in f32.
    pub const F16_MIXED: Self = Self { compute: DType::F16, residual: DType::F32 };
    /// Like `torch.autocast(dtype=bfloat16)`, the eval framework's default.
    pub const BF16_MIXED: Self = Self { compute: DType::BF16, residual: DType::F32 };

    pub const ALL: [Self; 5] = [Self::F32, Self::F16, Self::BF16, Self::F16_MIXED, Self::BF16_MIXED];

    pub fn parse(text: &str) -> Result<Self> {
        Self::ALL
            .into_iter()
            .find(|p| p.name() == text)
            .with_context(|| format!("unknown precision {text:?} (f32, f16, bf16, f16-mixed, bf16-mixed)"))
    }

    pub fn name(&self) -> &'static str {
        match (self.compute, self.residual) {
            (DType::F32, DType::F32) => "f32",
            (DType::F16, DType::F16) => "f16",
            (DType::BF16, DType::BF16) => "bf16",
            (DType::F16, DType::F32) => "f16-mixed",
            (DType::BF16, DType::F32) => "bf16-mixed",
            _ => "custom",
        }
    }
}

/// In-mask patches and their patch-grid coordinates, uploaded to the device.
pub struct EncoderInput {
    /// f32 `[N, P]`, one flattened patch per row in `(px, py, pz)` order.
    pub patches: Tensor,
    /// u32 `[N]` per axis, the `gx`, `gy`, `gz` columns of the coordinates.
    pub coords: [Tensor; 3],
}

impl EncoderInput {
    /// Checks the coordinates against the patch grid, since an out-of-range index is not caught
    /// on the GPU.
    pub fn new(patches: &[f32], coords: &[[u32; 3]], arch: &Architecture, device: &Device) -> Result<Self> {
        let n = coords.len();
        let p = arch.patch_dim();
        ensure!(patches.len() == n * p, "{} patch values for {n} coordinates of {p} voxels", patches.len());
        let grid = arch.patch_grid();
        for coord in coords {
            for axis in 0..3 {
                ensure!(
                    (coord[axis] as usize) < grid[axis],
                    "coordinate {coord:?} is outside the grid {grid:?}"
                );
            }
        }
        let axis = |a: usize| -> Result<Tensor> {
            let ids: Vec<u32> = coords.iter().map(|c| c[a]).collect();
            Ok(Tensor::from_vec(ids, n, device)?)
        };
        Ok(Self {
            patches: Tensor::from_slice(patches, (n, p), device)?,
            coords: [axis(0)?, axis(1)?, axis(2)?],
        })
    }

    pub fn len(&self) -> usize {
        self.patches.dims()[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Encoder output after the final LayerNorm, in f32.
pub struct EncoderOutput {
    /// `[C, D]` class-token outputs.
    pub cls: Tensor,
    /// `[N, D]` patch-token outputs, in input order.
    pub tokens: Tensor,
}

/// The encoder half of nanobrain's `ViTMAE3D` (`nanobrain/model.py::ViTMAE3D`).
pub struct Encoder {
    arch: Architecture,
    options: Options,
    patch_embed: Dense,
    pos_embed: SeparablePosEmbed3D,
    cls_token: Option<Tensor>,
    blocks: Vec<Block>,
    norm: LayerNorm,
}

impl Encoder {
    /// Builds the encoder, casting each weight to the dtype its layer computes in.
    pub fn new(arch: &Architecture, weights: &Weights, options: Options) -> Result<Self> {
        arch.validate()?;
        let precision = options.precision;
        let blocks = (0..arch.depth)
            .map(|i| Block::new(arch, weights, &format!("blocks.{i}"), options))
            .collect::<Result<_>>()?;
        Ok(Self {
            arch: arch.clone(),
            options,
            patch_embed: Dense::new(weights, "patch_embed", precision.compute, options.bias)?,
            pos_embed: SeparablePosEmbed3D::new(weights, precision.residual)?,
            cls_token: weights.get_optional("cls_token", precision.residual)?,
            blocks,
            norm: layer_norm(weights, "norm", arch.norm_eps, precision.residual)?,
        })
    }

    pub fn architecture(&self) -> &Architecture {
        &self.arch
    }

    pub fn options(&self) -> Options {
        self.options
    }

    /// Mirrors `nanobrain/model.py::ViTMAE3D.forward_encoder` for one image (B = 1), split into
    /// class and patch tokens the way `ViTMAE3D.dense_embed` slices them.
    pub fn forward(&self, input: &EncoderInput) -> Result<EncoderOutput> {
        let precision = self.options.precision;
        let patches = input.patches.to_dtype(precision.compute)?.unsqueeze(0)?;
        let x = self.patch_embed.forward(&patches)?.to_dtype(precision.residual)?;
        let x = self.pos_embed.forward(&x, &input.coords)?;
        let mut x = match &self.cls_token {
            Some(cls) => Tensor::cat(&[cls, &x], 1)?,
            None => x,
        };
        for block in &self.blocks {
            x = block.forward(&x, precision, self.options.attention)?;
            memory::sample(x.device());
        }
        let x = self.norm.forward(&x)?.to_dtype(DType::F32)?.squeeze(0)?;
        let c = self.arch.class_tokens;
        Ok(EncoderOutput { cls: x.narrow(0, 0, c)?, tokens: x.narrow(0, c, input.len())? })
    }
}

/// `nanobrain/modules.py::SeparablePosEmbed3D`: one learned table per axis, summed.
struct SeparablePosEmbed3D {
    tables: [Tensor; 3],
}

impl SeparablePosEmbed3D {
    fn new(weights: &Weights, dtype: DType) -> Result<Self> {
        let table = |axis: &str| weights.get(&format!("pos_embed.weight_{axis}"), dtype);
        Ok(Self { tables: [table("x")?, table("y")?, table("z")?] })
    }

    /// `x + weight_x[gx] + weight_y[gy] + weight_z[gz]`, added one axis at a time in that order
    /// like `SeparablePosEmbed3D.forward` (the coordinates are already integer grid indices, so
    /// `floor(coord / unit)` with unit 1 is the identity).
    fn forward(&self, x: &Tensor, coords: &[Tensor; 3]) -> Result<Tensor> {
        let mut x = x.clone();
        for (table, ids) in self.tables.iter().zip(coords) {
            x = x.broadcast_add(&table.index_select(ids, 0)?.unsqueeze(0)?)?;
        }
        Ok(x)
    }
}

/// `nanobrain/modules.py::Block`: pre-norm attention and MLP, each added to the residual.
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    fn new(arch: &Architecture, weights: &Weights, prefix: &str, options: Options) -> Result<Self> {
        let Options { precision, bias, .. } = options;
        Ok(Self {
            norm1: layer_norm(weights, &format!("{prefix}.norm1"), arch.norm_eps, precision.residual)?,
            attn: Attention::new(arch, weights, &format!("{prefix}.attn"), precision.compute, bias)?,
            norm2: layer_norm(weights, &format!("{prefix}.norm2"), arch.norm_eps, precision.residual)?,
            mlp: Mlp::new(weights, &format!("{prefix}.mlp"), precision.compute, bias)?,
        })
    }

    /// `Block.forward` without context or drop path (identity at inference).
    fn forward(&self, x: &Tensor, precision: Precision, kernel: AttentionKernel) -> Result<Tensor> {
        let h = self.norm1.forward(x)?.to_dtype(precision.compute)?;
        let x = (x + self.attn.forward(&h, kernel)?.to_dtype(precision.residual)?)?;
        let h = self.norm2.forward(&x)?.to_dtype(precision.compute)?;
        Ok((&x + self.mlp.forward(&h)?.to_dtype(precision.residual)?)?)
    }
}

/// `nanobrain/modules.py::Attention`: separate q, k, v linears, multi-head attention, proj.
struct Attention {
    q: Dense,
    k: Dense,
    v: Dense,
    proj: Dense,
    heads: usize,
    /// `F.scaled_dot_product_attention`'s default scale, 1 / sqrt(head_dim).
    scale: f64,
}

impl Attention {
    fn new(
        arch: &Architecture,
        weights: &Weights,
        prefix: &str,
        dtype: DType,
        bias: BiasAdd,
    ) -> Result<Self> {
        let get = |name: &str| Dense::new(weights, &format!("{prefix}.{name}"), dtype, bias);
        Ok(Self {
            q: get("q")?,
            k: get("k")?,
            v: get("v")?,
            proj: get("proj")?,
            heads: arch.heads,
            scale: 1.0 / (arch.head_dim() as f64).sqrt(),
        })
    }

    /// `Attention.forward` for self-attention (no context, not causal).
    fn forward(&self, x: &Tensor, kernel: AttentionKernel) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        let to_heads = |y: Tensor| y.reshape((b, t, self.heads, d / self.heads))?.transpose(1, 2);
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let out = attend(&q, &k, &v, self.scale, kernel)?;
        memory::sample(out.device());
        let out = out.transpose(1, 2)?.reshape((b, t, d))?;
        self.proj.forward(&out)
    }
}

/// `F.scaled_dot_product_attention(q, k, v)` (not causal, no mask) over `[B, H, T, hd]` tensors
/// whose last dim is contiguous.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, kernel: AttentionKernel) -> Result<Tensor> {
    match kernel {
        // The fused kernel reads strided q/k/v directly, so no copies are needed.
        AttentionKernel::Fused => Ok(candle_nn::ops::sdpa(q, k, v, None, false, scale as f32, 1.0)?),
        AttentionKernel::Chunked { chunk } => {
            chunked_attention(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, scale, chunk)
        }
    }
}

/// softmax(q k^T * scale) v over `[B, H, T, hd]` tensors, `chunk` query rows at a time.
///
/// The same math as `F.scaled_dot_product_attention`'s reference path. Peak scratch memory is
/// two `[B, H, chunk, T]` score buffers instead of `[B, H, T, T]`.
pub fn chunked_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, chunk: usize) -> Result<Tensor> {
    let t = q.dim(2)?;
    // Scaling q (T x hd) is cheaper than scaling the scores (T x T).
    let q = (q * scale)?;
    let k_t = k.t()?;
    let mut outputs = Vec::with_capacity(t.div_ceil(chunk));
    for start in (0..t).step_by(chunk) {
        let rows = chunk.min(t - start);
        let scores = q.narrow(2, start, rows)?.matmul(&k_t)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        // scores and probs are both live here: the high-water mark of the pass
        memory::sample(probs.device());
        outputs.push(probs.matmul(v)?);
    }
    Ok(Tensor::cat(&outputs, 2)?)
}

/// `nanobrain/modules.py::Mlp`: fc1, exact (erf) GELU, fc2.
struct Mlp {
    fc1: Dense,
    fc2: Dense,
}

impl Mlp {
    fn new(weights: &Weights, prefix: &str, dtype: DType, bias: BiasAdd) -> Result<Self> {
        Ok(Self {
            fc1: Dense::new(weights, &format!("{prefix}.fc1"), dtype, bias)?,
            fc2: Dense::new(weights, &format!("{prefix}.fc2"), dtype, bias)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // nn.GELU() default approximate="none" is the erf form, not the tanh one. Candle's Metal
        // erf is the Abramowitz-Stegun 7.1.26 polynomial (|error| <= 1.5e-7), close to f32 rounding.
        let h = self.fc1.forward(x)?.gelu_erf()?;
        self.fc2.forward(&h)
    }
}

/// `nn.Linear` from `{prefix}.weight` (`[out, in]`) and an optional `{prefix}.bias`.
enum Dense {
    /// `candle_nn::Linear` (also used when there is no bias).
    Plain(Linear),
    /// `[in + 1, out]`: the transposed weight with the bias as its last row.
    Folded { weight_t: Tensor },
}

impl Dense {
    fn new(weights: &Weights, prefix: &str, dtype: DType, mode: BiasAdd) -> Result<Self> {
        let weight = weights.get(&format!("{prefix}.weight"), dtype)?;
        let bias = weights.get_optional(&format!("{prefix}.bias"), dtype)?;
        match (mode, bias) {
            (BiasAdd::Folded, Some(bias)) => {
                let weight_t = Tensor::cat(&[&weight.t()?, &bias.unsqueeze(0)?], 0)?.contiguous()?;
                Ok(Self::Folded { weight_t })
            }
            (_, bias) => Ok(Self::Plain(Linear::new(weight, bias))),
        }
    }

    /// `[B, T, in] -> [B, T, out]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Plain(linear) => Ok(linear.forward(x)?),
            Self::Folded { weight_t } => {
                let (b, t, k) = x.dims3()?;
                let rows = x.reshape((b * t, k))?;
                let ones = Tensor::ones((b * t, 1), x.dtype(), x.device())?;
                let y = Tensor::cat(&[&rows, &ones], 1)?.matmul(weight_t)?;
                Ok(y.reshape((b, t, ()))?)
            }
        }
    }
}

/// `modules.py::LayerNorm` (`nn.LayerNorm` with eps 1e-6 by default; the manifest sets it).
fn layer_norm(weights: &Weights, prefix: &str, eps: f64, dtype: DType) -> Result<LayerNorm> {
    let weight = weights.get(&format!("{prefix}.weight"), dtype)?;
    let bias = weights.get(&format!("{prefix}.bias"), dtype)?;
    Ok(LayerNorm::new(weight, bias, eps))
}

/// Mean over tokens, `[N, D] -> [D]`, in f32: the pooled embedding of
/// `nanobrain/model.py::ViTMAE3D.global_embed`.
pub fn mean_pool(tokens: &Tensor) -> Result<Tensor> {
    Ok(tokens.to_dtype(DType::F32)?.mean(D::Minus2)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_input(arch: &Architecture, n: usize, device: &Device) -> EncoderInput {
        let grid = arch.patch_grid();
        let coords: Vec<[u32; 3]> = (0..n)
            .map(|i| [(i % grid[0]) as u32, (i % grid[1]) as u32, ((i * 7) % grid[2]) as u32])
            .collect();
        let patches: Vec<f32> =
            (0..n * arch.patch_dim()).map(|i| ((i * 37 % 101) as f32 / 50.0) - 1.0).collect();
        EncoderInput::new(&patches, &coords, arch, device).unwrap()
    }

    #[test]
    fn parses_kernel_and_precision_names() {
        assert_eq!(AttentionKernel::parse("fused").unwrap(), AttentionKernel::Fused);
        assert_eq!(AttentionKernel::parse("chunked:512").unwrap(), AttentionKernel::Chunked { chunk: 512 });
        assert!(AttentionKernel::parse("chunked:0").is_err());
        for precision in Precision::ALL {
            assert_eq!(Precision::parse(precision.name()).unwrap(), precision);
        }
        for bias in [BiasAdd::Broadcast, BiasAdd::Folded] {
            assert_eq!(BiasAdd::parse(bias.name()).unwrap(), bias);
        }
    }

    #[test]
    fn rejects_coordinates_outside_the_grid() {
        let arch = Architecture::vit(64, 1, 2, 8);
        let patches = vec![0f32; arch.patch_dim()];
        assert!(EncoderInput::new(&patches, &[[24, 0, 0]], &arch, &Device::Cpu).is_err());
    }

    #[test]
    fn chunking_does_not_change_cpu_output() {
        let arch = Architecture::vit(64, 2, 2, 8);
        let weights = Weights::random(&arch, &Device::Cpu).unwrap();
        let input = tiny_input(&arch, 37, &Device::Cpu);
        let run = |chunk| {
            let options = Options { attention: AttentionKernel::Chunked { chunk }, ..Options::DEFAULT };
            let encoder = Encoder::new(&arch, &weights, options).unwrap();
            encoder.forward(&input).unwrap().tokens.to_vec2::<f32>().unwrap()
        };
        let whole = run(1000);
        let chunked = run(5);
        assert_eq!(whole.len(), 37);
        for (a, b) in whole.iter().flatten().zip(chunked.iter().flatten()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn fused_matches_chunked_on_metal() {
        let Ok(device) = Device::new_metal(0) else {
            return;
        };
        let arch = Architecture::vit(128, 2, 2, 8);
        let weights = Weights::random(&arch, &device).unwrap();
        let input = tiny_input(&arch, 300, &device);
        let run = |options| {
            let encoder = Encoder::new(&arch, &weights, options).unwrap();
            encoder.forward(&input).unwrap().tokens.to_vec2::<f32>().unwrap()
        };
        let fused = run(Options::DEFAULT);
        for other in [
            Options { attention: AttentionKernel::Chunked { chunk: 64 }, ..Options::DEFAULT },
            Options { bias: BiasAdd::Folded, ..Options::DEFAULT },
        ] {
            let tokens = run(other);
            let max_diff = fused
                .iter()
                .flatten()
                .zip(tokens.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(max_diff < 1e-4, "{other:?}: max diff {max_diff}");
        }
    }
}
