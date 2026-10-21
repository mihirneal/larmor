//! Times the device work of one gradient evaluation of the segmentation solver at a realistic size (633k rows
//! of 384 features, 512 outputs), piece by piece, to choose the solver's layout.
//!
//! `cargo run --release -p larmor-probe --example solver_profile -- [--device metal] [--rows 633479]`

use candle_core::{DType, Device as CandleDevice, Tensor};
use larmor_probe::{Device, DeviceChoice};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut choice = DeviceChoice::Auto;
    let mut rows = 633_479usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => choice = args.next().expect("device").parse()?,
            "--rows" => rows = args.next().expect("rows").parse()?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let device = Device::open(choice)?;
    let dev: &CandleDevice = device.as_candle();
    let (d, k, chunk) = (385usize, 512usize, 1usize << 17);
    println!("{rows} rows on {device}");
    let n_chunks = rows.div_ceil(chunk);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for c in 0..n_chunks {
        let m = chunk.min(rows - c * chunk);
        xs.push((Tensor::randn(0f32, 1f32, (m, d), dev)?).contiguous()?);
        ys.push(Tensor::rand(0f32, 1f32, (m, k), dev)?.ge(0.97)?.to_dtype(DType::F32)?);
    }
    let theta = (Tensor::randn(0f32, 0.05f32, (d, k), dev)?).contiguous()?;
    device.synchronize()?;

    let time = |label: &str, f: &dyn Fn() -> anyhow::Result<Vec<Tensor>>| -> anyhow::Result<()> {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let start = Instant::now();
            let out = f()?;
            device.synchronize()?;
            best = best.min(start.elapsed().as_secs_f64());
            drop(out);
        }
        println!("{label:<44} {:8.1} ms", best * 1e3);
        Ok(())
    };
    time("logits X theta", &|| Ok(xs.iter().map(|x| x.matmul(&theta)).collect::<Result<Vec<_>, _>>()?))?;
    let logits: Vec<Tensor> = xs.iter().map(|x| x.matmul(&theta)).collect::<Result<_, _>>()?;
    time("sigmoid", &|| Ok(logits.iter().map(candle_nn::ops::sigmoid).collect::<Result<Vec<_>, _>>()?))?;
    let residuals: Vec<Tensor> =
        logits.iter().zip(&ys).map(|(z, y)| candle_nn::ops::sigmoid(z)?.sub(y)).collect::<Result<_, _>>()?;
    time("sigmoid - y", &|| {
        Ok(logits
            .iter()
            .zip(&ys)
            .map(|(z, y)| candle_nn::ops::sigmoid(z)?.sub(y))
            .collect::<Result<Vec<_>, _>>()?)
    })?;
    time("X' R, one product per chunk", &|| {
        Ok(xs.iter().zip(&residuals).map(|(x, r)| x.t()?.matmul(r)).collect::<Result<Vec<_>, _>>()?)
    })?;
    for block in [8192usize, 4096, 2048, 1024] {
        time(&format!("X' R, blocks of {block} rows, summed"), &|| {
            Ok(xs
                .iter()
                .zip(&residuals)
                .map(|(x, r)| {
                    let m = x.dim(0)?;
                    if m % block != 0 {
                        return x.t()?.matmul(r);
                    }
                    let xb = x.reshape((m / block, block, d))?.transpose(1, 2)?;
                    xb.matmul(&r.reshape((m / block, block, k))?)?.sum(0)
                })
                .collect::<Result<Vec<_>, _>>()?)
        })?;
    }
    time("whole gradient, blocks of 2048", &|| {
        Ok(xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| {
                let r = candle_nn::ops::sigmoid(&x.matmul(&theta)?)?.sub(y)?;
                let m = x.dim(0)?;
                if m % 2048 != 0 {
                    return x.t()?.matmul(&r);
                }
                let xb = x.reshape((m / 2048, 2048, d))?.transpose(1, 2)?;
                xb.matmul(&r.reshape((m / 2048, 2048, k))?)?.sum(0)
            })
            .collect::<Result<Vec<_>, _>>()?)
    })?;
    time("loss (softplus - y z, summed)", &|| {
        Ok(xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| {
                let z = x.matmul(&theta)?;
                let softplus = z.relu()?.add(&z.abs()?.neg()?.exp()?.affine(1.0, 1.0)?.log()?)?;
                softplus.sub(&y.mul(&z)?)?.sum_all()
            })
            .collect::<Result<Vec<_>, _>>()?)
    })?;
    Ok(())
}
