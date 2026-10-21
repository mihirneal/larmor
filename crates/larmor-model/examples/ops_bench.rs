//! Micro-benchmarks of the encoder's heavy ops on one device, to see where time goes.
//!
//! ```sh
//! cargo run --release -p larmor-model --features fusion,autotune --example ops_bench -- metal f32 59360 1472
//! ```

use std::time::Instant;

use burn::tensor::activation::softmax;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{DType, Device, Distribution, FloatDType, Tensor};
use larmor_model::DeviceChoice;
use larmor_model::attention::{explicit_attention, softmax_attention};

fn bench<const D: usize>(label: &str, flops: f64, device: &Device, f: impl Fn() -> Tensor<D>) {
    for _ in 0..2 {
        let out = f();
        let _ = out.into_data();
    }
    let runs = 5;
    let start = Instant::now();
    for _ in 0..runs {
        let out = f();
        device.sync().unwrap();
        drop(out);
    }
    let seconds = start.elapsed().as_secs_f64() / runs as f64;
    let tflops = if flops > 0.0 { format!(", {:.2} TFLOP/s", flops / seconds / 1e12) } else { String::new() };
    println!("{label:<48} {:8.2} ms{tflops}", seconds * 1e3);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let choice: DeviceChoice = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(DeviceChoice::Metal);
    let dtype = match args.get(2).map(String::as_str) {
        Some("f16") => FloatDType::F16,
        Some("bf16") => FloatDType::BF16,
        _ => FloatDType::F32,
    };
    let n: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(8151);
    let chunk: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(n);
    let device = choice.burn_device().unwrap();
    let (h, hd, d) = (6usize, 64usize, 384usize);
    let rand = |shape: [usize; 3]| {
        Tensor::<3>::random(shape, Distribution::Normal(0.0, 1.0), &device).cast(DType::from(dtype))
    };
    let q = rand([h, chunk, hd]);
    let k = rand([h, n, hd]);
    let v = rand([h, n, hd]);
    let kt_contig = k.clone().transpose().into_data();
    let kt = Tensor::<3>::from_data(kt_contig, (&device, DType::from(dtype)));
    let flops_qk = 2.0 * (h * chunk * n * hd) as f64;
    println!("device {choice} {dtype:?}, n {n}, chunk {chunk}");
    bench("q @ k^T (transposed view)", flops_qk, &device, || q.clone().matmul(k.clone().transpose()));
    bench("q @ kT (contiguous)", flops_qk, &device, || q.clone().matmul(kt.clone()));
    let scores = q.clone().matmul(kt.clone()).cast(FloatDType::F32);
    bench("softmax f32 [h, chunk, n]", 0.0, &device, || softmax(scores.clone(), 2));
    bench("max_dim f32 [h, chunk, n]", 0.0, &device, || scores.clone().max_dim(2));
    bench("sum_dim f32 [h, chunk, n]", 0.0, &device, || scores.clone().sum_dim(2));
    bench("exp f32 [h, chunk, n]", 0.0, &device, || scores.clone().exp());
    bench("exp(x - max) f32", 0.0, &device, || {
        let max = scores.clone().max_dim(2);
        (scores.clone() - max).exp()
    });
    let probs = softmax(scores.clone(), 2).cast(dtype);
    bench("p @ v", flops_qk, &device, || probs.clone().matmul(v.clone()));
    let q4 = q.clone().unsqueeze::<4>();
    let k4 = k.clone().unsqueeze::<4>();
    let v4 = v.clone().unsqueeze::<4>();
    let opts = AttentionModuleOptions { scale: None, softcap: None, is_causal: false };
    bench("fused attention op (chunk)", 2.0 * flops_qk, &device, || {
        attention(q4.clone(), k4.clone(), v4.clone(), None, None, opts)
    });
    bench("softmax_attention (chunk)", 2.0 * flops_qk, &device, || {
        softmax_attention(q4.clone(), k4.clone(), v4.clone())
    });
    bench("explicit_attention, deferred norm (chunk)", 2.0 * flops_qk, &device, || {
        explicit_attention(q4.clone(), k4.clone(), v4.clone())
    });
    bench("transposed-scores attention (chunk)", 2.0 * flops_qk, &device, || {
        transposed_attention(q4.clone(), k4.clone(), v4.clone())
    });
    bench("transposed, native exp, f32 sum", 2.0 * flops_qk, &device, || {
        transposed_attention_native(q4.clone(), k4.clone(), v4.clone(), true)
    });
    bench("transposed, native exp, native sum", 2.0 * flops_qk, &device, || {
        transposed_attention_native(q4.clone(), k4.clone(), v4.clone(), false)
    });
    // explicit_attention step by step, in the compute dtype
    let s = q.clone().matmul(kt.clone());
    bench("  step: s.max_dim", 0.0, &device, || s.clone().max_dim(2));
    let m = s.clone().max_dim(2).cast(FloatDType::F32);
    bench("  step: exp(s.f32 - m).cast", 0.0, &device, || {
        (s.clone().cast(FloatDType::F32) - m.clone()).exp().cast(dtype)
    });
    let e = (s.clone().cast(FloatDType::F32) - m.clone()).exp().cast(dtype);
    bench("  step: exp(s - m) in compute dtype", 0.0, &device, || (s.clone() - m.clone().cast(dtype)).exp());
    bench("  step: e.f32.sum_dim", 0.0, &device, || e.clone().cast(FloatDType::F32).sum_dim(2));
    bench("  step: e.sum_dim (compute dtype)", 0.0, &device, || e.clone().sum_dim(2));
    bench("  step: e @ v", flops_qk, &device, || e.clone().matmul(v.clone()));
    let x = rand([1, n, d]);
    let w1 = Tensor::<2>::random([d, 4 * d], Distribution::Normal(0.0, 0.02), &device).cast(dtype);
    bench("x @ fc1 [n,384]x[384,1536]", 2.0 * (n * d * 4 * d) as f64, &device, || {
        x.clone().matmul(w1.clone().unsqueeze::<3>())
    });
    let wq = Tensor::<2>::random([d, d], Distribution::Normal(0.0, 0.02), &device).cast(dtype);
    bench("x @ q [n,384]x[384,384]", 2.0 * (n * d * d) as f64, &device, || {
        x.clone().matmul(wq.clone().unsqueeze::<3>())
    });
}

/// Scores laid out [keys, queries] so the per-query max broadcasts along a non-last axis.
fn transposed_attention(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
    let compute: FloatDType = q.dtype().into();
    let scale = 1.0 / (q.dims()[3] as f64).sqrt();
    let q = q * scale;
    let scores_t = k.matmul(q.transpose()); // [b, h, keys, queries]
    let max = scores_t.clone().max_dim(2).cast(FloatDType::F32);
    let numerators = (scores_t.cast(FloatDType::F32) - max).exp().cast(compute);
    let denominators = numerators.clone().cast(FloatDType::F32).sum_dim(2); // [b, h, 1, queries]
    let out = numerators.transpose().matmul(v).cast(FloatDType::F32) / denominators.transpose();
    out.cast(compute)
}

/// Transposed scores with the exponentials in the compute dtype; `f32_sum` widens the denominator.
fn transposed_attention_native(q: Tensor<4>, k: Tensor<4>, v: Tensor<4>, f32_sum: bool) -> Tensor<4> {
    let scale = 1.0 / (q.dims()[3] as f64).sqrt();
    let compute: FloatDType = q.dtype().into();
    let scores_t = k.matmul((q * scale).transpose());
    let max = scores_t.clone().max_dim(2);
    let numerators = (scores_t - max).exp();
    let denominators = if f32_sum {
        numerators.clone().cast(FloatDType::F32).sum_dim(2)
    } else {
        numerators.clone().sum_dim(2).cast(FloatDType::F32)
    };
    let out = numerators.transpose().matmul(v).cast(FloatDType::F32) / denominators.transpose();
    out.cast(compute)
}
