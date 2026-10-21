//! Seconds per gradient evaluation of the segmentation solver on the default device, for a few chunk sizes.
//!
//! `cargo run --release -p larmor-probe --example solver_profile -- [rows] [dim]`

use burn::tensor::Device;
use larmor_probe::Matrix;
use larmor_probe::segmentation::{SolverOptions, fit_logistic};

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse::<usize>().unwrap());
    let rows = args.next().unwrap_or(793_529);
    let dim = args.next().unwrap_or(384);
    let features: Vec<f32> =
        (0..rows * dim).map(|i| (((i * 7919) % 1000) as f32 / 1000.0 - 0.5) * 2.0).collect();
    // output j fires when feature j % dim is high, so the problem has signal and takes a while to solve
    let labels: Vec<u8> =
        (0..rows * 512).map(|i| u8::from(features[(i / 512) * dim + (i % 512) % dim] > 0.9)).collect();
    let features = Matrix::new(rows, dim, features);
    let labels = Matrix::new(rows, 512, labels);
    let device = Device::default();
    for chunk_rows in [1 << 17, 1 << 18, 1 << 20] {
        let options = SolverOptions { max_iter: 40, chunk_rows, ..Default::default() };
        let (_, stats) = fit_logistic(&features, &labels, 1e3, &options, None, &device);
        println!(
            "{rows} x {dim}, chunks of {chunk_rows}: setup {:.2}s, {} evaluations in {:.2}s ({:.3}s each), total {:.2}s",
            stats.setup_seconds,
            stats.evaluations,
            stats.solve_seconds,
            stats.solve_seconds / stats.evaluations as f64,
            stats.seconds
        );
    }
}
