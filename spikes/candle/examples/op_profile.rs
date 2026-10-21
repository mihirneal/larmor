//! Times the individual ops of one encoder block on Metal, to see where a forward pass goes.
//!
//! cargo run --release --example op_profile -- [tokens] [width] [f32|f16|bf16]

use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::Module;

fn time(device: &Device, label: &str, runs: usize, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    for _ in 0..2 {
        f()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..runs {
        f()?;
    }
    device.synchronize()?;
    let ms = start.elapsed().as_secs_f64() * 1e3 / runs as f64;
    println!("{label:<40} {ms:>8.3} ms");
    Ok(ms)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).map_or(Ok(8151), |s| s.parse())?;
    let d: usize = args.get(2).map_or(Ok(384), |s| s.parse())?;
    let dtype = match args.get(3).map(String::as_str) {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        _ => DType::F32,
    };
    let device = Device::new_metal(0)?;
    let x = Tensor::randn(0f32, 1.0, (1, t, d), &device)?.to_dtype(dtype)?;
    let w = Tensor::randn(0f32, 0.02, (d, d), &device)?.to_dtype(dtype)?;
    let w1 = Tensor::randn(0f32, 0.02, (4 * d, d), &device)?.to_dtype(dtype)?;
    let w2 = Tensor::randn(0f32, 0.02, (d, 4 * d), &device)?.to_dtype(dtype)?;
    let b = Tensor::zeros(d, dtype, &device)?;
    let b1 = Tensor::zeros(4 * d, dtype, &device)?;
    let ones = Tensor::ones(d, dtype, &device)?;
    let h = Tensor::randn(0f32, 1.0, (1, t, 4 * d), &device)?.to_dtype(dtype)?;
    let runs = 20;
    println!("tokens {t}, width {d}, {dtype:?}");
    let lin = candle_nn::Linear::new(w.clone(), Some(b.clone()));
    let lin1 = candle_nn::Linear::new(w1.clone(), Some(b1.clone()));
    let lin2 = candle_nn::Linear::new(w2.clone(), Some(b.clone()));
    let wt = w.t()?.contiguous()?;
    let w1t = w1.t()?.contiguous()?;
    let x2 = x.squeeze(0)?;
    let mut total = 0.0;
    total += 4.0 * time(&device, "Linear d->d (q, k, v, proj; x4 below)", runs, || Ok(lin.forward(&x)?))?;
    time(&device, "  matmul only, w.t() view", runs, || Ok(x2.matmul(&w.t()?)?))?;
    time(&device, "  matmul only, pre-transposed w", runs, || Ok(x2.matmul(&wt)?))?;
    time(&device, "  broadcast_add bias", runs, || Ok(x.broadcast_add(&b)?))?;
    time(&device, "  broadcast_add bias, 2D", runs, || Ok(x2.broadcast_add(&b)?))?;
    let b2 = b.unsqueeze(0)?;
    time(&device, "  broadcast_add bias [1,D], 2D", runs, || Ok(x2.broadcast_add(&b2)?))?;
    let ones_col = Tensor::ones((t, 1), dtype, &device)?;
    time(&device, "  bias via ones[T,1] @ b[1,D] + add", runs, || {
        Ok((x2.matmul(&wt)? + ones_col.matmul(&b2)?)?)
    })?;
    let w_aug = Tensor::cat(&[&wt, &b2], 0)?;
    time(&device, "  bias folded: cat([x,1]) @ [w;b]", runs, || {
        Ok(Tensor::cat(&[&x2, &ones_col], 1)?.matmul(&w_aug)?)
    })?;
    total += time(&device, "Linear fc1 d->4d", runs, || Ok(lin1.forward(&x)?))?;
    time(&device, "  matmul only, pre-transposed w1", runs, || Ok(x2.matmul(&w1t)?))?;
    total += time(&device, "Linear fc2 4d->d", runs, || Ok(lin2.forward(&h)?))?;
    total += time(&device, "gelu_erf on 4d", runs, || Ok(h.gelu_erf()?))?;
    total += 2.0
        * time(&device, "layer_norm (x2 below)", runs, || {
            Ok(candle_nn::ops::layer_norm(&x, &ones, &b, 1e-6)?)
        })?;
    total += 2.0 * time(&device, "residual add (x2 below)", runs, || Ok((&x + &x)?))?;
    let heads = d / 64;
    total += 3.0
        * time(&device, "to_heads transpose (view, x3 below)", runs, || {
            Ok(x.reshape((1, t, heads, 64))?.transpose(1, 2)?)
        })?;
    let o = x.reshape((1, t, heads, 64))?.transpose(1, 2)?.contiguous()?;
    total += time(&device, "from_heads transpose+reshape (copy)", runs, || {
        Ok(o.transpose(1, 2)?.reshape((1, t, d))?)
    })?;
    let o3 = o.squeeze(0)?;
    time(&device, "  3D [H,T,hd] -> [T,H,hd] copy", runs, || Ok(o3.transpose(0, 1)?.contiguous()?))?;
    time(&device, "  plain contiguous copy (same bytes)", runs, || Ok(x.copy()?))?;
    let q = x.reshape((1, t, heads, 64))?.transpose(1, 2)?;
    total +=
        time(&device, "sdpa fused", runs, || Ok(candle_nn::ops::sdpa(&q, &q, &q, None, false, 0.125, 1.0)?))?;
    println!("{:<40} {total:>8.3} ms  (x depth for the encoder)", "block total");
    Ok(())
}
