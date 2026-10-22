//! Scaled dot-product attention over long sequences.
//!
//! nanobrain calls `F.scaled_dot_product_attention(q, k, v)` with no mask. A patch-4 scan has
//! about 60k tokens, and the full score matrix of one layer (6 heads × 60k × 60k f32) would take
//! 85 GB, so neither kernel here builds it:
//!
//! - **Fused**: Candle's `sdpa` (MLX's "steel" attention on Metal) keeps a running softmax and
//!   never stores scores. It exists only on Metal, for head dims 32, 64, 72, 80, 96, 128 and 256
//!   (and 512 in half precision).
//! - **Chunked**: two matmuls and an exact softmax, a block of query rows at a time. Each block
//!   still takes its softmax over all keys, so the result equals unchunked attention, but only
//!   one block's scores exist at once. This runs on every device.

use std::fmt;
use std::str::FromStr;

use candle_core::{DType, Device, Tensor};

use crate::device::choice_of;
use crate::error::EncoderError;

/// Chunk sizes are rounded down to a multiple of this many query rows, when they can be.
const ROW_ALIGN: usize = 64;

/// The default [`AttentionOptions::score_budget_bytes`].
pub const DEFAULT_SCORE_BUDGET_BYTES: usize = 2 << 30;

/// How attention is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKernel {
    /// Fused where the device and head dim support it, chunked otherwise.
    #[default]
    Auto,
    /// Candle's fused `sdpa`. An error where it does not exist.
    Fused,
    /// Query-chunked matmul, softmax, matmul.
    Chunked,
}

impl AttentionKernel {
    /// Replace `Auto` with the kernel used for `head_dim`-wide heads in `dtype` on `device`, and
    /// check that an explicit `Fused` can run there.
    pub fn resolve(self, device: &Device, head_dim: usize, dtype: DType) -> Result<Self, EncoderError> {
        let fused_ok = fused_supported(device, head_dim, dtype);
        match self {
            AttentionKernel::Auto if fused_ok => Ok(AttentionKernel::Fused),
            AttentionKernel::Auto => Ok(AttentionKernel::Chunked),
            AttentionKernel::Fused if !fused_ok => Err(EncoderError::Device(format!(
                "fused attention needs Metal and a head dim of 32, 64, 72, 80, 96, 128 or 256 \
                 (512 in half precision); this is {} with head dim {head_dim} in {dtype:?}",
                choice_of(device)
            ))),
            concrete => Ok(concrete),
        }
    }
}

impl fmt::Display for AttentionKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AttentionKernel::Auto => "auto",
            AttentionKernel::Fused => "fused",
            AttentionKernel::Chunked => "chunked",
        })
    }
}

impl FromStr for AttentionKernel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(AttentionKernel::Auto),
            "fused" => Ok(AttentionKernel::Fused),
            "chunked" => Ok(AttentionKernel::Chunked),
            other => Err(format!("unknown attention kernel {other:?} (expected auto, fused or chunked)")),
        }
    }
}

/// Whether Candle's fused `sdpa` runs for this device, head dim and dtype (the checks in
/// `candle_nn::ops::Sdpa::metal_fwd`).
pub fn fused_supported(device: &Device, head_dim: usize, dtype: DType) -> bool {
    let head_dim_ok = match head_dim {
        32 | 64 | 72 | 80 | 96 | 128 | 256 => true,
        512 => dtype != DType::F32,
        _ => false,
    };
    device.is_metal() && head_dim_ok && matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
}

/// Attention kernel and the chunked kernel's memory budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionOptions {
    /// Which kernel computes attention.
    pub kernel: AttentionKernel,
    /// Query rows per chunk for the chunked kernel; `None` takes as many as fit the budget.
    pub chunk_rows: Option<usize>,
    /// The most bytes the chunked kernel's score buffers may hold at once: the scores and their
    /// softmax, `2 × heads × rows × tokens` values in the compute dtype. An encode whose chunks
    /// cannot fit is refused with [`EncoderError::MemoryBudget`].
    pub score_budget_bytes: usize,
}

impl Default for AttentionOptions {
    fn default() -> Self {
        Self {
            kernel: AttentionKernel::Auto,
            chunk_rows: None,
            score_budget_bytes: DEFAULT_SCORE_BUDGET_BYTES,
        }
    }
}

/// A resolved attention kernel with its chunk size, fixed for one encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionPlan {
    Fused,
    Chunked { rows: usize },
}

impl AttentionOptions {
    /// Bytes of score buffers the chunked kernel holds with `rows`-row chunks.
    pub fn score_bytes(heads: usize, rows: usize, tokens: usize, dtype: DType) -> usize {
        2 * heads * rows * tokens * dtype.size_in_bytes()
    }

    /// Query rows per chunk for `tokens` queries and keys, or an error when even the requested
    /// (or smallest) chunk exceeds the budget.
    ///
    /// Without `chunk_rows`, takes the most rows that fit, rounded down to a multiple of 64 when
    /// that is at least 64, and at most `tokens`.
    pub fn query_chunk_rows(&self, heads: usize, tokens: usize, dtype: DType) -> Result<usize, EncoderError> {
        let row_bytes = Self::score_bytes(heads, 1, tokens, dtype).max(1);
        let rows = match self.chunk_rows {
            Some(rows) => rows.clamp(1, tokens.max(1)),
            None => {
                let fit = self.score_budget_bytes / row_bytes;
                let aligned = if fit >= ROW_ALIGN { fit / ROW_ALIGN * ROW_ALIGN } else { fit };
                aligned.clamp(1, tokens.max(1))
            }
        };
        let needed_bytes = rows * row_bytes;
        if needed_bytes > self.score_budget_bytes {
            return Err(EncoderError::MemoryBudget {
                what: format!("chunked attention over {tokens} tokens in chunks of {rows} query rows"),
                needed_bytes,
                budget_bytes: self.score_budget_bytes,
            });
        }
        Ok(rows)
    }

    /// The plan for one encode of `tokens` tokens (class tokens included). `Auto` is planned as
    /// chunked; `Encoder` resolves it before this.
    pub(crate) fn plan(
        &self,
        heads: usize,
        tokens: usize,
        dtype: DType,
    ) -> Result<AttentionPlan, EncoderError> {
        match self.kernel {
            AttentionKernel::Fused => Ok(AttentionPlan::Fused),
            AttentionKernel::Chunked | AttentionKernel::Auto => {
                Ok(AttentionPlan::Chunked { rows: self.query_chunk_rows(heads, tokens, dtype)? })
            }
        }
    }
}

/// `softmax(q kᵀ · scale) v` for `[batch, heads, tokens, head_dim]` inputs with contiguous last
/// dims, by the plan's kernel.
///
/// Mirrors `torch.nn.functional.scaled_dot_product_attention(q, k, v)` as called by
/// `nanobrain/modules.py::Attention.forward`.
pub(crate) fn attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    plan: AttentionPlan,
) -> candle_core::Result<Tensor> {
    match plan {
        // reads strided q, k and v directly; no copies
        AttentionPlan::Fused => fused_attention(q, k, v, scale),
        AttentionPlan::Chunked { rows } => {
            chunked_attention(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, scale, rows)
        }
    }
}

/// `softmax(q kᵀ · scale) v` over `[batch, heads, tokens, head_dim]` tensors, `rows` query rows
/// at a time.
///
/// The softmax runs in the input dtype. Candle's Metal softmax kernel accumulates the sum in
/// f32; its CPU kernel sums in the input dtype, which is why `Encoder` runs only f32 on the CPU.
pub fn chunked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    rows: usize,
) -> candle_core::Result<Tensor> {
    let queries = q.dim(2)?;
    // Scaling q touches tokens × head_dim values rather than tokens × keys. For head_dim 64 the
    // scale is 1/8, a power of two, so this is bit-identical to scaling the scores.
    let q = (q * scale)?;
    let k_t = k.t()?;
    let rows = rows.max(1);
    let mut outputs = Vec::with_capacity(queries.div_ceil(rows));
    for start in (0..queries).step_by(rows) {
        let scores = q.narrow(2, start, rows.min(queries - start))?.matmul(&k_t)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        outputs.push(weights.matmul(v)?);
    }
    match outputs.len() {
        1 => Ok(outputs.remove(0)),
        _ => Tensor::cat(&outputs, 2),
    }
}

/// Candle's fused attention, `softmax(q kᵀ · scale) v`; Metal only.
pub fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> candle_core::Result<Tensor> {
    candle_nn::ops::sdpa(q, k, v, None, false, scale as f32, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(chunk_rows: Option<usize>, budget: usize) -> AttentionOptions {
        AttentionOptions { chunk_rows, score_budget_bytes: budget, ..AttentionOptions::default() }
    }

    #[test]
    fn chunk_rows_fit_the_budget() {
        // 2 buffers x 6 heads x 60k keys x 4 bytes = 2.88 MB per query row
        let rows = options(None, 1 << 30).query_chunk_rows(6, 60_000, DType::F32).unwrap();
        assert_eq!(rows, 320);
        // half precision fits twice the rows
        assert_eq!(options(None, 1 << 30).query_chunk_rows(6, 60_000, DType::F16).unwrap(), 704);
        // small problems are one chunk
        assert_eq!(options(None, 1 << 30).query_chunk_rows(6, 100, DType::F32).unwrap(), 100);
        // below 64 rows, no rounding
        assert_eq!(options(None, 10 * 2_880_000).query_chunk_rows(6, 60_000, DType::F32).unwrap(), 10);
    }

    #[test]
    fn over_budget_is_refused() {
        let err = options(None, 1000).query_chunk_rows(6, 60_000, DType::F32).unwrap_err();
        assert!(matches!(err, EncoderError::MemoryBudget { .. }), "{err}");
        // the full 60k x 60k matrix: 2 x 6 x 60k x 60k x 4 bytes = 172.8 GB
        let err = options(Some(60_000), 16 << 30).query_chunk_rows(6, 60_000, DType::F32).unwrap_err();
        assert!(err.to_string().contains("160.93 GiB"), "{err}");
        assert_eq!(options(Some(64), 1 << 30).query_chunk_rows(6, 60_000, DType::F32).unwrap(), 64);
    }

    #[test]
    fn fused_needs_metal_and_a_supported_head_dim() {
        assert!(!fused_supported(&Device::Cpu, 64, DType::F32));
        let kernel = AttentionKernel::Auto.resolve(&Device::Cpu, 64, DType::F32).unwrap();
        assert_eq!(kernel, AttentionKernel::Chunked);
        assert!(AttentionKernel::Fused.resolve(&Device::Cpu, 64, DType::F32).is_err());
        if let Ok(metal) = Device::new_metal(0) {
            assert!(fused_supported(&metal, 64, DType::F32));
            assert!(!fused_supported(&metal, 8, DType::F32));
            assert!(!fused_supported(&metal, 512, DType::F32));
            assert!(fused_supported(&metal, 512, DType::BF16));
        }
    }
}
