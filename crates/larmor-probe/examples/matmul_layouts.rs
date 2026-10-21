//! Times the pieces of one gradient evaluation of the segmentation solver on the default device. Used to pick
//! the device layout of the solver.

use burn::tensor::{DType, Device, Tensor, TensorCreationOptions, TensorData, Transaction, activation};
use std::time::Instant;

fn main() {
    let device = Device::default();
    let options = || TensorCreationOptions::new(device.clone()).with_dtype(DType::F32);
    let (n, d, k) = (131_072usize, 384usize, 512usize);
    let chunks = 6;
    let make = |seed: usize, rows: usize, cols: usize, scale: f32| -> Tensor<2> {
        let v: Vec<f32> =
            (0..rows * cols).map(|i| (((i * 7919 + seed) % 1000) as f32 / 1000.0 - 0.5) * scale).collect();
        Tensor::from_data(TensorData::new(v, [rows, cols]), options())
    };
    let xs: Vec<Tensor<2>> = (0..chunks).map(|c| make(c, n, d, 2.0)).collect();
    let ys: Vec<Tensor<2>> = (0..chunks).map(|c| make(c + 7, n, k, 0.0)).collect();
    let w = make(3, d, k, 0.01);
    let b = make(5, 1, k, 0.1);
    let sync = |t: Tensor<2>| {
        let _ = t.into_data();
    };
    for rep in 0..3 {
        let mut times = [0.0f64; 4];
        let start_all = Instant::now();
        for (x, y) in xs.iter().zip(&ys) {
            let t = Instant::now();
            let z = x.clone().matmul(w.clone());
            sync(z.clone().slice([0..1, 0..1]));
            times[0] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let r = activation::sigmoid(z + b.clone()) - y.clone();
            sync(r.clone().slice([0..1, 0..1]));
            times[1] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let g = x.clone().transpose().matmul(r.clone());
            sync(g.clone().slice([0..1, 0..1]));
            times[2] += t.elapsed().as_secs_f64();
            let t = Instant::now();
            sync(r.sum_dim(0));
            times[3] += t.elapsed().as_secs_f64();
        }
        println!(
            "rep {rep}: X W {:.3}s | sigmoid - y {:.3}s | X' R {:.3}s | sum {:.3}s | total {:.3}s",
            times[0],
            times[1],
            times[2],
            times[3],
            start_all.elapsed().as_secs_f64()
        );
    }
    // per chunk: both partial gradients read back right away
    for rep in 0..3 {
        let start = Instant::now();
        for (x, y) in xs.iter().zip(&ys) {
            let r = activation::sigmoid(x.clone().matmul(w.clone()) + b.clone()) - y.clone();
            let _ = x.clone().transpose().matmul(r.clone()).into_data();
            let _ = r.sum_dim(0).into_data();
        }
        println!("read per chunk rep {rep}: {:.3}s", start.elapsed().as_secs_f64());
    }
    // partial gradients summed on the device, one read at the end
    for rep in 0..3 {
        let start = Instant::now();
        let mut total_w: Option<Tensor<2>> = None;
        let mut total_b: Option<Tensor<2>> = None;
        for (x, y) in xs.iter().zip(&ys) {
            let r = activation::sigmoid(x.clone().matmul(w.clone()) + b.clone()) - y.clone();
            let g = x.clone().transpose().matmul(r.clone());
            let gb = r.sum_dim(0);
            total_w = Some(match total_w {
                Some(t) => t + g,
                None => g,
            });
            total_b = Some(match total_b {
                Some(t) => t + gb,
                None => gb,
            });
            sync(total_b.clone().unwrap());
        }
        let _ = total_w.unwrap().into_data();
        println!("device sum, sync per chunk rep {rep}: {:.3}s", start.elapsed().as_secs_f64());
    }
    // the solver's path: everything queued, one transaction
    for rep in 0..3 {
        let start = Instant::now();
        let mut transaction = Transaction::default();
        for (x, y) in xs.iter().zip(&ys) {
            let r = activation::sigmoid(x.clone().matmul(w.clone()) + b.clone()) - y.clone();
            transaction = transaction.register(x.clone().transpose().matmul(r.clone()));
            transaction = transaction.register(r.sum_dim(0));
        }
        let _ = transaction.execute();
        println!("transaction rep {rep}: {:.3}s", start.elapsed().as_secs_f64());
    }
}
