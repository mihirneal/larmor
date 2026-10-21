//! Scaled dot-product attention over long sequences.
//!
//! A patch-4 scan has about 60k tokens, and its full score matrix (6 heads × 60k × 60k f32) would
//! be about 86 GB. Both kernels here therefore split the *queries* into chunks: each chunk still
//! takes an exact softmax over all keys, so the result equals unchunked attention, but only one
//! chunk's scores exist at a time.

use burn::tensor::activation::softmax;
use burn::tensor::module::attention as burn_attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{FloatDType, Tensor};

/// Query chunks are a multiple of this many rows.
const ROW_ALIGN: usize = 64;

/// How each query chunk's attention is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKernel {
    /// Pick per device and precision, from measurements on an M3 Ultra (see [`Self::resolve`]).
    #[default]
    Auto,
    /// Burn's attention op (`burn::tensor::module::attention`). The CPU backend runs a flash
    /// kernel. GPU backends run flash kernels only when built with the `autotune` feature and
    /// otherwise materialize the chunk's scores in the compute dtype.
    Fused,
    /// Two matmuls and an exact softmax written out with tensor ops (see [`explicit_attention`]).
    Explicit,
}

impl AttentionKernel {
    /// Replace `Auto` with a concrete kernel.
    ///
    /// Measured on an M3 Ultra (ViT-S, patch 8, whole forward): in f32 the explicit kernel took
    /// 0.31 s against 0.47 s fused, so f32 on a GPU is explicit. In f16 the fused op's flash kernel
    /// was faster (0.20 s against 0.31 s), but it only exists with the `autotune` feature; without
    /// it the fused op materializes scores with a half-precision denominator. On the CPU the fused
    /// op is a flash kernel and about twice as fast as the explicit path.
    pub fn resolve(self, gpu: bool, compute: FloatDType) -> AttentionKernel {
        let half = compute != FloatDType::F32;
        match self {
            AttentionKernel::Auto if !gpu => AttentionKernel::Fused,
            AttentionKernel::Auto if half && cfg!(feature = "autotune") => AttentionKernel::Fused,
            AttentionKernel::Auto => AttentionKernel::Explicit,
            concrete => concrete,
        }
    }
}

/// Attention kernel and memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionOptions {
    /// Which kernel computes each chunk.
    pub kernel: AttentionKernel,
    /// Upper bound, in bytes, on one chunk's f32 score matrix (`heads × rows × keys × 4`). The
    /// explicit kernel holds about two such buffers at its peak. Capped at
    /// [`MAX_SCORE_BUDGET_BYTES`].
    pub score_budget_bytes: usize,
}

/// Largest per-chunk score budget. Metal refused an 8 GiB buffer on a 96 GB M3 Ultra, and
/// budgets past 2 GiB did not speed up patch 4.
pub const MAX_SCORE_BUDGET_BYTES: usize = 4 << 30;

impl Default for AttentionOptions {
    fn default() -> Self {
        Self { kernel: AttentionKernel::default(), score_budget_bytes: 2 << 30 }
    }
}

/// Query rows per chunk so that `heads × rows × keys` f32 scores fit in `budget_bytes`.
///
/// A multiple of 64, at least 64, and no more than `queries` rounded up to a multiple of 64.
pub fn query_chunk_rows(heads: usize, queries: usize, keys: usize, budget_bytes: usize) -> usize {
    let bytes_per_row = heads.max(1) * keys.max(1) * size_of::<f32>();
    let rows = (budget_bytes.min(MAX_SCORE_BUDGET_BYTES) / bytes_per_row) / ROW_ALIGN * ROW_ALIGN;
    rows.clamp(ROW_ALIGN, queries.max(1).next_multiple_of(ROW_ALIGN))
}

/// `softmax(q kᵀ / √d) v` for `[batch, heads, tokens, head_dim]` inputs, with no mask, computed in
/// query chunks.
///
/// Mirrors `torch.nn.functional.scaled_dot_product_attention(q, k, v)` as called by
/// `nanobrain/modules.py::Attention.forward`.
pub fn chunked_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, options: &AttentionOptions) -> Tensor<4> {
    let [batch, heads, queries, head_dim] = q.dims();
    let keys = k.dims()[2];
    let rows = query_chunk_rows(heads, queries, keys, options.score_budget_bytes);
    // Zero query rows pad the queries to whole chunks. They attend like any other query and their
    // outputs are dropped; the keys are untouched, so real rows are unaffected. Equal chunks keep
    // one kernel shape per layer (one compile, one autotune), and a multiple-of-64 row count lets
    // the explicit kernel's elementwise passes vectorize.
    let padding = queries.next_multiple_of(rows) - queries;
    let q = match padding {
        0 => q,
        _ => Tensor::cat(
            vec![q.clone(), Tensor::zeros([batch, heads, padding, head_dim], (&q.device(), q.dtype()))],
            2,
        ),
    };
    let attend = |q_chunk: Tensor<4>| match options.kernel {
        AttentionKernel::Fused => fused_attention(q_chunk, k.clone(), v.clone()),
        // `Encoder` resolves `Auto`; a direct caller gets the explicit path
        AttentionKernel::Explicit | AttentionKernel::Auto => {
            explicit_attention(q_chunk, k.clone(), v.clone())
        }
    };
    let out = match q.dims()[2] / rows {
        1 => attend(q),
        _ => Tensor::cat(q.split(rows, 2).into_iter().map(attend).collect(), 2),
    };
    match padding {
        0 => out,
        _ => out.narrow(2, 0, queries),
    }
}

/// Burn's attention op with its default scale, `1/√head_dim`.
fn fused_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    // an explicit `scale` would push GPU backends off the flash kernels, so leave it as the default
    let options = AttentionModuleOptions { scale: None, softcap: None, is_causal: false };
    burn_attention(q, k, v, None, None, options)
}

/// Attention from tensor ops: `softmax(s) v` with the scores laid out `sᵀ = k qᵀ / √d`.
///
/// The scores are `[keys, queries]` so that the per-query max and sum broadcast along a leading
/// axis: broadcasting along the last axis stops Burn's GPU kernels from vectorizing, which made the
/// `exp(s - max)` pass about three times slower. On long sequences these passes over the score
/// matrix are where the time goes.
///
/// In f32 the softmax denominator divides the `[queries, head_dim]` output instead of the scores,
/// which saves a read and a write of the score matrix. In half precision it cannot: the
/// unnormalized output of a near-uniform row over 60k keys is about 60k × |v|, past f16's 65504.
/// The denominator is always accumulated in f32.
pub fn explicit_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let compute: FloatDType = q.dtype().into();
    let scale = 1.0 / (q.dims()[3] as f64).sqrt();
    // Scaling q instead of the scores touches tokens × head_dim values rather than tokens × keys.
    // For head_dim 64 the scale is 1/8, a power of two, so this is bit-identical to scaling scores.
    let scores_t = k.matmul((q * scale).transpose()); // [batch, heads, keys, queries]
    let max = scores_t.clone().max_dim(2);
    let numerators = (scores_t - max).exp();
    let denominators = numerators.clone().cast(FloatDType::F32).sum_dim(2);
    if compute == FloatDType::F32 {
        numerators.transpose().matmul(v) / denominators.transpose()
    } else {
        let weights = (numerators.cast(FloatDType::F32) / denominators).cast(compute);
        weights.transpose().matmul(v)
    }
}

/// The textbook form, `softmax(q kᵀ / √d) v` with an f32 softmax; kept to check
/// [`explicit_attention`] against.
pub fn softmax_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let compute: FloatDType = q.dtype().into();
    let scale = 1.0 / (q.dims()[3] as f64).sqrt();
    let scores = (q * scale).matmul(k.transpose()).cast(FloatDType::F32);
    softmax(scores, 3).cast(compute).matmul(v)
}
