//! Attention alone on random inputs, fused against chunked, to see where encoder time goes.
//!
//! ```sh
//! cargo run --release -p larmor-model --example ops_bench -- metal f32 59360 2048
//! ```
//!
//! Arguments: device (`metal`, `cpu`, `cuda`), dtype (`f32`, `f16`, `bf16`), tokens, and chunk
//! rows for the chunked kernel. Heads and head dim are ViT-S's, 6 × 64.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use larmor_model::DeviceChoice;
use larmor_model::attention::{chunked_attention, fused_attention, fused_supported};

/// Median seconds of `runs` calls after two warm-up calls, waiting for the device each time.
fn time(device: &Device, runs: usize, f: impl Fn() -> candle_core::Result<Tensor>) -> f64 {
    for _ in 0..2 {
        f().unwrap();
        device.synchronize().unwrap();
    }
    let mut seconds: Vec<f64> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            let out = f().unwrap();
            device.synchronize().unwrap();
            drop(out);
            start.elapsed().as_secs_f64()
        })
        .collect();
    seconds.sort_by(f64::total_cmp);
    seconds[seconds.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let choice: DeviceChoice = args.get(1).map_or(DeviceChoice::Auto, |s| s.parse().unwrap());
    let dtype = match args.get(2).map(String::as_str) {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let tokens: usize = args.get(3).map_or(8151, |s| s.parse().unwrap());
    let rows: usize = args.get(4).map_or(2048, |s| s.parse().unwrap());
    let (heads, head_dim) = (6, 64);
    let device = choice.candle_device().unwrap();
    let random = || Tensor::randn(0f32, 1.0, (1, heads, tokens, head_dim), &device)?.to_dtype(dtype);
    let (q, k, v) = (random().unwrap(), random().unwrap(), random().unwrap());
    let scale = 1.0 / (head_dim as f64).sqrt();
    // two matmuls of tokens x tokens x head_dim per head, 2 flops per multiply-add
    let flops = 4.0 * (heads * tokens * tokens * head_dim) as f64;
    let runs = if tokens > 20_000 { 3 } else { 10 };
    println!("{} {dtype:?}, {heads} heads x {tokens} tokens x {head_dim}", choice.resolve());
    if fused_supported(&device, head_dim, dtype) {
        let seconds = time(&device, runs, || fused_attention(&q, &k, &v, scale));
        println!("fused            {:9.2} ms {:6.2} TFLOP/s", seconds * 1e3, flops / seconds / 1e12);
    }
    let seconds = time(&device, runs, || chunked_attention(&q, &k, &v, scale, rows));
    println!("chunked {rows:>6}   {:9.2} ms {:6.2} TFLOP/s", seconds * 1e3, flops / seconds / 1e12);
}
