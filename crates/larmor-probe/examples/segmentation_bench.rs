//! Times the segmentation probe at a realistic scale on synthetic lesion-like data.
//!
//! ```text
//! cargo run --release -p larmor-probe --example segmentation_bench -- fit       # one outer-fold fit per alpha
//! cargo run --release -p larmor-probe --example segmentation_bench -- cv        # the full 5 x 5 x 5 CV
//!     [--subjects 300] [--patches 6000] [--dim 384] [--ratio 10] [--positive 0.05] [--device gpu|ndarray]
//! ```
//!
//! Each subject has `--patches` in-mask patches; a `--positive` fraction of them hold a lesion blob of random
//! extent. Features are a lesion signal (per-octant occupancy) plus shared factors and noise, so they are
//! correlated like real embeddings.

use burn::tensor::Device;
use larmor_probe::Matrix;
use larmor_probe::folds::kfold;
use larmor_probe::segmentation::{
    PATCH_VOXELS, SegmentationOptions, SubjectPatches, TrainingSet, fit_segmentation,
    probe_binary_segmentation, segmentation_folds,
};
use rayon::prelude::*;
use std::time::Instant;

struct Args {
    mode: String,
    subjects: usize,
    patches: usize,
    dim: usize,
    ratio: Option<f64>,
    positive: f64,
    device: Device,
}

#[allow(deprecated)]
fn parse_args() -> Args {
    let mut args = Args {
        mode: "fit".into(),
        subjects: 300,
        patches: 6000,
        dim: 384,
        ratio: Some(10.0),
        positive: 0.05,
        device: Device::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().expect("missing value");
        match arg.as_str() {
            "--subjects" => args.subjects = value().parse().unwrap(),
            "--patches" => args.patches = value().parse().unwrap(),
            "--dim" => args.dim = value().parse().unwrap(),
            "--ratio" => args.ratio = value().parse().ok(),
            "--positive" => args.positive = value().parse().unwrap(),
            "--device" => {
                args.device = if value() == "ndarray" { Device::ndarray() } else { Device::default() }
            }
            mode => args.mode = mode.to_string(),
        }
    }
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
        "{} subjects x {} patches x {} features, ratio {:?}, on {:?} (data made in {:.1}s)",
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
        other => panic!("unknown mode {other}"),
    }
}

/// One fit per alpha on outer fold 0's training subjects, cold-started, plus predicting its test subjects.
fn bench_fits(subjects: &[SubjectPatches], args: &Args) {
    let folds = segmentation_folds(subjects.len()).unwrap();
    let start = Instant::now();
    let training = TrainingSet::new(subjects, &folds[0].train, args.ratio);
    println!(
        "outer fold 0 training set: {} rows ({:.1}s to build)",
        training.features.rows(),
        start.elapsed().as_secs_f64()
    );
    let options = SegmentationOptions::default();
    for &alpha in &options.alphas {
        let start = Instant::now();
        let (model, stats) = fit_segmentation(&training, alpha, &options.solver, None, &args.device);
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "alpha {alpha:>7}: {seconds:6.1}s  {} iterations, {} evaluations ({:.3}s each), stop {:?}, max grad {:.1e}",
            stats.iterations,
            stats.evaluations,
            stats.seconds / stats.evaluations as f64,
            stats.stop,
            stats.gradient_max
        );
        if alpha == options.alphas[0] {
            let start = Instant::now();
            let voxels: usize =
                folds[0].test.iter().map(|&s| model.predict(&subjects[s].features, &args.device).len()).sum();
            println!(
                "  predicting {} test subjects ({voxels} voxels): {:.1}s",
                folds[0].test.len(),
                start.elapsed().as_secs_f64()
            );
        }
    }
    let inner = kfold(folds[0].train.len(), 5, Some(0)).unwrap();
    let ids: Vec<usize> = inner[0].train.iter().map(|&i| folds[0].train[i]).collect();
    println!("inner training sets: {} rows", TrainingSet::new(subjects, &ids, args.ratio).features.rows());
}

/// The full probe: 5 outer folds x (5 inner splits x 5 alphas + 1 refit).
fn bench_cv(subjects: &[SubjectPatches], args: &Args) {
    let folds = segmentation_folds(subjects.len()).unwrap();
    let options =
        SegmentationOptions { max_negative_ratio: args.ratio, record_fits: true, ..Default::default() };
    let start = Instant::now();
    let cv = probe_binary_segmentation(subjects, &folds, &options, &args.device).unwrap();
    let seconds = start.elapsed().as_secs_f64();
    let fit_seconds: f64 = cv.fits.iter().map(|f| f.stats.seconds).sum();
    let evaluations: usize = cv.fits.iter().map(|f| f.stats.evaluations).sum();
    println!(
        "5 x 5 x 5 CV: {seconds:.0}s total, {} fits taking {fit_seconds:.0}s ({evaluations} evaluations), the rest is prediction and scoring",
        cv.fits.len()
    );
    let report = &cv.report;
    println!(
        "dice {:.4} {:?}  AP {:.4}  voxel AUROC {:.4}  alphas {:?}  thresholds {:?}",
        report.dice,
        report.dice_ci,
        report.average_precision,
        report.voxel_auroc,
        report.fold_alpha,
        report.fold_threshold
    );
}
