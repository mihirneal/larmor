//! Times the segmentation probe at a realistic scale on synthetic lesion-like data.
//!
//! ```text
//! cargo run --release -p larmor-probe --example segmentation_bench -- fit [--set inner|outer]  # a fit per alpha
//! cargo run --release -p larmor-probe --example segmentation_bench -- cv         # 5 x 5 x 5 CV + deployable probe
//! cargo run --release -p larmor-probe --example segmentation_bench -- precision [--alpha 1000]  # f32 gradient error
//!     [--subjects 300] [--patches 6000] [--dim 384] [--ratio 10] [--positive 0.05] [--device auto|cpu|metal|cuda]
//! ```
//!
//! Each subject has `--patches` in-mask patches; a `--positive` fraction of them hold a lesion blob of random
//! extent. Features are a lesion signal (per-octant occupancy) plus shared factors and noise, so they are
//! correlated like real embeddings.

use larmor_probe::folds::kfold;
use larmor_probe::segmentation::{
    FitRecord, FitStats, PATCH_VOXELS, SegmentationOptions, SolverStop, SubjectPatches, TrainingSet,
    fit_segmentation, fit_segmentation_probe, probe_binary_segmentation, segmentation_folds,
};
use larmor_probe::{Device, DeviceChoice, Matrix};
use rayon::prelude::*;
use std::time::Instant;

struct Args {
    mode: String,
    subjects: usize,
    patches: usize,
    dim: usize,
    ratio: Option<f64>,
    positive: f64,
    /// For `fit`: which training set of outer fold 0.
    set: String,
    /// For `precision`: the penalty of the fit the gradient is checked at.
    alpha: f64,
    device: Device,
}

fn parse_args() -> Args {
    let mut args = Args {
        mode: "fit".into(),
        subjects: 300,
        patches: 6000,
        dim: 384,
        ratio: Some(10.0),
        positive: 0.05,
        set: "inner".into(),
        alpha: 1e3,
        device: Device::cpu(),
    };
    let mut choice = DeviceChoice::Auto;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().expect("missing value");
        match arg.as_str() {
            "--subjects" => args.subjects = value().parse().unwrap(),
            "--patches" => args.patches = value().parse().unwrap(),
            "--dim" => args.dim = value().parse().unwrap(),
            "--ratio" => args.ratio = value().parse().ok(),
            "--positive" => args.positive = value().parse().unwrap(),
            "--set" => args.set = value(),
            "--alpha" => args.alpha = value().parse().unwrap(),
            "--device" => choice = value().parse().unwrap(),
            mode => args.mode = mode.to_string(),
        }
    }
    args.device = Device::open(choice).unwrap();
    args
}

/// A small fast uniform generator (splitmix64) for synthetic data; the probe's own RNGs are not involved.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Shared across subjects: how lesions and anatomy show up in the features.
struct Embedding {
    octant_weights: Vec<f64>,
    factor_weights: Vec<f64>,
    offsets: Vec<f64>,
}

const FACTORS: usize = 8;

impl Embedding {
    fn new(dim: usize) -> Self {
        let mut rng = SplitMix(7);
        Self {
            octant_weights: (0..8 * dim).map(|_| rng.normal()).collect(),
            factor_weights: (0..FACTORS * dim).map(|_| rng.normal() * 0.7).collect(),
            offsets: (0..dim).map(|_| rng.normal() * 0.3).collect(),
        }
    }
}

fn make_subject(index: usize, args: &Args, embedding: &Embedding) -> SubjectPatches {
    let mut rng = SplitMix(1000 + index as u64);
    let (n, d) = (args.patches, args.dim);
    let mut labels = vec![0u8; n * PATCH_VOXELS];
    let mut features = vec![0.0f32; n * d];
    for p in 0..n {
        let label = &mut labels[p * PATCH_VOXELS..(p + 1) * PATCH_VOXELS];
        if rng.next() < args.positive {
            // a box of random extent anchored at a random corner of the 8x8x8 patch
            let extent: Vec<usize> = (0..3).map(|_| 1 + (rng.next() * 8.0) as usize % 8).collect();
            let flip: Vec<bool> = (0..3).map(|_| rng.next() < 0.5).collect();
            for x in 0..8 {
                for y in 0..8 {
                    for z in 0..8 {
                        let inside = [x, y, z]
                            .iter()
                            .zip(&extent)
                            .zip(&flip)
                            .all(|((&c, &e), &f)| if f { c >= 8 - e } else { c < e });
                        label[x * 64 + y * 8 + z] = u8::from(inside);
                    }
                }
            }
        }
        let mut octants = [0.0f64; 8];
        for (v, &l) in label.iter().enumerate() {
            let (x, y, z) = (v / 64, (v / 8) % 8, v % 8);
            octants[(x / 4) * 4 + (y / 4) * 2 + z / 4] += f64::from(l) / 64.0;
        }
        let factors: Vec<f64> = (0..FACTORS).map(|_| rng.normal()).collect();
        for j in 0..d {
            let mut value = embedding.offsets[j];
            for (o, &occupancy) in octants.iter().enumerate() {
                value += 2.0 * occupancy * embedding.octant_weights[o * d + j];
            }
            for (f, &factor) in factors.iter().enumerate() {
                value += factor * embedding.factor_weights[f * d + j];
            }
            value += 0.9 * rng.normal();
            // the cache stores f16 tokens
            features[p * d + j] = half::f16::from_f64(value).to_f32();
        }
    }
    SubjectPatches { features: Matrix::new(n, d, features), labels: Matrix::new(n, PATCH_VOXELS, labels) }
}

fn main() {
    let args = parse_args();
    let start = Instant::now();
    let embedding = Embedding::new(args.dim);
    let subjects: Vec<SubjectPatches> =
        (0..args.subjects).into_par_iter().map(|i| make_subject(i, &args, &embedding)).collect();
    println!(
        "{} subjects x {} patches x {} features, ratio {:?}, on {} (data made in {:.1}s)",
        args.subjects,
        args.patches,
        args.dim,
        args.ratio,
        args.device,
        start.elapsed().as_secs_f64()
    );
    match args.mode.as_str() {
        "fit" => bench_fits(&subjects, &args),
        "cv" => bench_cv(&subjects, &args),
        "precision" => gradient_precision(&subjects, &args),
        other => panic!("unknown mode {other}"),
    }
}

/// One cold-started fit per alpha on a training set of outer fold 0: its inner split 0 (`--set inner`, what
/// every inner CV fit sees) or all its training subjects (`--set outer`, the outer refit), plus predicting the
/// fold's test subjects.
fn bench_fits(subjects: &[SubjectPatches], args: &Args) {
    let folds = segmentation_folds(subjects.len()).unwrap();
    let ids: Vec<usize> = match args.set.as_str() {
        "outer" => folds[0].train.clone(),
        "inner" => {
            let inner = kfold(folds[0].train.len(), 5, Some(0)).unwrap();
            inner[0].train.iter().map(|&i| folds[0].train[i]).collect()
        }
        other => panic!("unknown --set {other}"),
    };
    let start = Instant::now();
    let training = TrainingSet::new(subjects, &ids, args.ratio);
    println!(
        "{} training set of outer fold 0: {} subjects, {} rows ({:.1}s to build)",
        args.set,
        ids.len(),
        training.features.rows(),
        start.elapsed().as_secs_f64()
    );
    let options = SegmentationOptions::default();
    // the first use of each device kernel compiles it; do that on one subject, outside the timings
    let start = Instant::now();
    let warm_up = TrainingSet::new(subjects, &ids[..1], args.ratio);
    fit_segmentation(&warm_up, 1e5, &options.solver, None, &args.device).unwrap();
    println!("warm-up fit on one subject: {:.1}s", start.elapsed().as_secs_f64());
    for &alpha in &options.alphas {
        let start = Instant::now();
        let (model, stats) = fit_segmentation(&training, alpha, &options.solver, None, &args.device).unwrap();
        let seconds = start.elapsed().as_secs_f64();
        println!("alpha {alpha:>7}: {seconds:6.1}s  {}", describe(&stats));
        if alpha == options.alphas[0] {
            let start = Instant::now();
            let voxels: usize = folds[0]
                .test
                .iter()
                .map(|&s| model.predict(&subjects[s].features, &args.device).unwrap().len())
                .sum();
            println!(
                "  predicting {} test subjects ({voxels} voxels): {:.1}s",
                folds[0].test.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
}

fn describe(stats: &FitStats) -> String {
    format!(
        "{} iterations, {} evaluations ({:.3}s each), stop {:?}, max grad {:.1e} | setup {:.2}s, gradients \
         {:.1}s ({:.3}s each), host L-BFGS {:.1}s",
        stats.iterations,
        stats.evaluations,
        stats.seconds / stats.evaluations as f64,
        stats.stop,
        stats.gradient_max,
        stats.setup_seconds,
        stats.gradient_seconds,
        stats.gradient_seconds / stats.evaluations as f64,
        stats.solve_seconds - stats.gradient_seconds,
    )
}

/// Where the time of a set of fits went.
fn summarize(label: &str, seconds: f64, fits: &[FitRecord]) {
    let sum = |f: fn(&FitStats) -> f64| fits.iter().map(|r| f(&r.stats)).sum::<f64>();
    let evaluations: usize = fits.iter().map(|r| r.stats.evaluations).sum();
    let iterations: usize = fits.iter().map(|r| r.stats.iterations).sum();
    let fit_seconds = sum(|s| s.seconds);
    let gradient_seconds = sum(|s| s.gradient_seconds);
    println!(
        "{label}: {seconds:.0}s total; {} fits {fit_seconds:.0}s ({iterations} iterations, {evaluations} \
         evaluations; gradients {gradient_seconds:.0}s = {:.3}s each, setup {:.0}s, host L-BFGS {:.0}s); \
         training sets, prediction and scoring {:.0}s",
        fits.len(),
        gradient_seconds / evaluations as f64,
        sum(|s| s.setup_seconds),
        sum(|s| s.solve_seconds) - gradient_seconds,
        seconds - fit_seconds,
    );
    let stops = |stop: SolverStop| fits.iter().filter(|r| r.stats.stop == stop).count();
    println!(
        "  stops: {} converged, {} stalled, {} at max_iter",
        stops(SolverStop::Converged),
        stops(SolverStop::Stalled),
        stops(SolverStop::MaxIter)
    );
}

/// The full probe: 5 outer folds x (5 inner splits x 5 alphas + 1 refit), then the deployable probe (5 inner
/// splits x 5 alphas over all subjects + 1 fit on all).
fn bench_cv(subjects: &[SubjectPatches], args: &Args) {
    let folds = segmentation_folds(subjects.len()).unwrap();
    let options =
        SegmentationOptions { max_negative_ratio: args.ratio, record_fits: true, ..Default::default() };
    let start = Instant::now();
    let cv = probe_binary_segmentation(subjects, &folds, &options, &args.device).unwrap();
    let cv_seconds = start.elapsed().as_secs_f64();
    summarize("5 x 5 x 5 CV", cv_seconds, &cv.fits);
    let report = &cv.report;
    println!(
        "  dice {:.4} {:?}  AP {:.4}  voxel AUROC {:.4}  alphas {:?}  thresholds {:?}",
        report.dice,
        report.dice_ci,
        report.average_precision,
        report.voxel_auroc,
        report.fold_alpha,
        report.fold_threshold
    );
    let start = Instant::now();
    let (model, selection, records) = fit_segmentation_probe(subjects, &options, &args.device).unwrap();
    let final_seconds = start.elapsed().as_secs_f64();
    summarize("deployable probe", final_seconds, &records);
    println!(
        "  alpha {} threshold {:?} (alpha id {}, threshold id {})",
        model.alpha, model.threshold, selection.alpha_id, selection.threshold_id
    );
    println!("CV plus deployable probe: {:.0}s", cv_seconds + final_seconds);
}

/// Float32 device gradient against a float64 CPU gradient at a fitted head (inner training set of outer fold 0).
fn gradient_precision(subjects: &[SubjectPatches], args: &Args) {
    use candle_core::{DType, Tensor};
    use larmor_probe::segmentation::solver::solver_gradient;
    let folds = segmentation_folds(subjects.len()).unwrap();
    let inner = kfold(folds[0].train.len(), 5, Some(0)).unwrap();
    let ids: Vec<usize> = inner[0].train.iter().map(|&i| folds[0].train[i]).collect();
    let training = TrainingSet::new(subjects, &ids, args.ratio);
    let alpha = args.alpha;
    let options = SegmentationOptions::default();
    let start = Instant::now();
    let (model, stats) = fit_segmentation(&training, alpha, &options.solver, None, &args.device).unwrap();
    println!(
        "{} rows, alpha {alpha}: fit {:.1}s, {stats:?}",
        training.features.rows(),
        start.elapsed().as_secs_f64()
    );
    let x = model.standardize(&training.features);
    let (n, d, k) = (x.rows(), x.cols(), PATCH_VOXELS);
    let start = Instant::now();
    let cpu = candle_core::Device::Cpu;
    let x64 = Tensor::from_slice(x.data(), (n, d), &cpu).unwrap().to_dtype(DType::F64).unwrap();
    let x64 = Tensor::cat(&[&x64, &Tensor::ones((n, 1), DType::F64, &cpu).unwrap()], 1).unwrap();
    let y64 = Tensor::from_slice(training.labels.data(), (n, k), &cpu).unwrap().to_dtype(DType::F64).unwrap();
    let mut theta: Vec<f32> = model.head.coef.data().to_vec();
    theta.extend_from_slice(&model.head.intercept);
    let theta64 = Tensor::from_slice(&theta, (d + 1, k), &cpu).unwrap().to_dtype(DType::F64).unwrap();
    let residual = candle_nn::ops::sigmoid(&x64.matmul(&theta64).unwrap()).unwrap().sub(&y64).unwrap();
    let raw = x64.t().unwrap().matmul(&residual).unwrap().flatten_all().unwrap().to_vec1::<f64>().unwrap();
    let penalty = 2.0 * alpha / n as f64;
    let exact: Vec<f64> = raw
        .iter()
        .enumerate()
        .map(|(i, g)| g / n as f64 + if i < d * k { penalty * f64::from(theta[i]) } else { 0.0 })
        .collect();
    let exact_max = exact.iter().fold(0.0f64, |m, g| m.max(g.abs()));
    println!("float64 gradient on the CPU: max {exact_max:.3e} ({:.1}s)", start.elapsed().as_secs_f64());
    for block_rows in [1usize << 17, 1 << 15, 1 << 13, 1 << 12, 1 << 11, 1 << 10] {
        let solver = larmor_probe::segmentation::SolverOptions { block_rows, ..options.solver.clone() };
        let ours = solver_gradient(&x, &training.labels, alpha, &model.head, &solver, &args.device).unwrap();
        let error = ours.iter().zip(&exact).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        let ours_max = ours.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        println!("block {block_rows:>6} rows: max |error| {error:.3e}, max |gradient| {ours_max:.3e}");
    }
}
