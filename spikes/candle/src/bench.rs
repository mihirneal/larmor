//! Wall-clock timing and GPU memory of encoder forward passes.

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use crate::memory;
use crate::model::{AttentionKernel, Encoder, EncoderInput, attend};

/// Timing of repeated forward passes on one input.
#[derive(Debug, Clone, Serialize)]
pub struct Timing {
    /// The first pass, including Metal pipeline compilation.
    pub cold_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub runs: usize,
    /// Highest Metal allocation sampled during the timed passes (weights, inputs, activations
    /// and Candle's buffer pool, see `memory`), or `None` off Metal.
    pub peak_gpu_bytes: Option<usize>,
}

/// Runs `warmup` untimed passes (the first is reported as `cold_ms`), then `runs` timed ones.
/// Each pass is timed until the device has finished all queued work.
pub fn time_forward(
    encoder: &Encoder,
    input: &EncoderInput,
    device: &Device,
    warmup: usize,
    runs: usize,
) -> Result<Timing> {
    let pass = || -> Result<f64> {
        device.synchronize()?;
        let start = Instant::now();
        let output = encoder.forward(input)?;
        device.synchronize()?;
        drop(output);
        Ok(start.elapsed().as_secs_f64() * 1e3)
    };
    repeat(pass, device, warmup, runs)
}

/// Times attention alone on random `[1, heads, tokens, head_dim]` q, k, v, the same way as
/// `time_forward` (memory is not sampled).
pub fn time_attention(
    device: &Device,
    shape: (usize, usize, usize),
    dtype: DType,
    kernel: AttentionKernel,
    warmup: usize,
    runs: usize,
) -> Result<Timing> {
    let (heads, tokens, head_dim) = shape;
    let random = || -> Result<Tensor> {
        Ok(Tensor::randn(0f32, 1.0, (1, heads, tokens, head_dim), device)?.to_dtype(dtype)?)
    };
    let (q, k, v) = (random()?, random()?, random()?);
    let scale = 1.0 / (head_dim as f64).sqrt();
    let pass = || -> Result<f64> {
        device.synchronize()?;
        let start = Instant::now();
        let out = attend(&q, &k, &v, scale, kernel)?;
        memory::sample(device);
        device.synchronize()?;
        drop(out);
        Ok(start.elapsed().as_secs_f64() * 1e3)
    };
    repeat(pass, device, warmup, runs)
}

/// Multiply-add FLOPs of softmax(Q K^T) V: two `tokens x tokens x head_dim` matmuls per head.
pub fn attention_flops(heads: usize, tokens: usize, head_dim: usize) -> f64 {
    4.0 * heads as f64 * (tokens as f64).powi(2) * head_dim as f64
}

/// `warmup` untimed passes (the first reported as cold), then `runs` timed ones, tracking the
/// memory peak over the timed ones.
fn repeat(pass: impl Fn() -> Result<f64>, device: &Device, warmup: usize, runs: usize) -> Result<Timing> {
    let mut cold_ms = None;
    for _ in 0..warmup.max(1) {
        let ms = pass()?;
        cold_ms.get_or_insert(ms);
    }
    memory::take_peak(device);
    let mut times = (0..runs.max(1)).map(|_| pass()).collect::<Result<Vec<_>>>()?;
    times.sort_by(f64::total_cmp);
    Ok(Timing {
        cold_ms: cold_ms.unwrap_or_default(),
        median_ms: median(&times),
        min_ms: times[0],
        max_ms: times[times.len() - 1],
        runs: times.len(),
        peak_gpu_bytes: memory::take_peak(device),
    })
}

/// Median of sorted values (mean of the middle two for an even count).
fn median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(&[1.0, 2.0, 9.0]), 2.0);
        assert_eq!(median(&[1.0, 2.0, 4.0, 9.0]), 3.0);
    }
}
