//! `larmor bench`: per-stage timing of one scan on this machine, like `onset timing-test`.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Result, ensure};
use larmor_model::{DeviceChoice, Precision};
use larmor_volume::Nifti;
use serde::Serialize;

use crate::cache::{CacheEntry, EntryKey};
use crate::console::{bold, dim, secs};
use crate::hashing::sha256_hex;
use crate::machine::{Load, Machine};
use crate::model::{self, Model};
use crate::phantom;
use crate::pipeline::load_encoder;
use crate::run_info::{Encoding, RunInfo, StageSeconds, now};

/// Options of `larmor bench`.
pub struct BenchOptions {
    pub model: Option<String>,
    pub scan: Option<PathBuf>,
    pub runs: usize,
    pub device: DeviceChoice,
    pub precision: Precision,
    pub json: bool,
}

/// Seconds of one stage over the runs.
#[derive(Clone, Debug, Serialize)]
struct StageTiming {
    stage: &'static str,
    median: f64,
    min: f64,
    max: f64,
}

#[derive(Serialize)]
struct BenchReport {
    larmor_version: &'static str,
    machine: Machine,
    /// How busy the machine was just before the timing started (other processes, not this one).
    load: Load,
    model: String,
    model_source: String,
    scan: String,
    scan_source: String,
    device: String,
    precision: String,
    runs: usize,
    tokens: usize,
    encoder_load_seconds: f64,
    /// The first encode, including any kernel compilation and warm-up.
    encode_cold_seconds: f64,
    stages: Vec<StageTiming>,
    /// Median of the per-run sums of the stages (warm encode).
    total_median_seconds: f64,
}

pub fn cmd_bench(options: BenchOptions) -> Result<()> {
    ensure!(options.runs >= 2, "--runs must be at least 2 (the first encode is the cold one)");
    let machine = Machine::detect();
    // the phantom, random weights and the timed cache writes; removed when the bench ends
    let scratch_dir = tempfile::Builder::new().prefix("larmor-bench").tempdir()?;
    let scratch = scratch_dir.path();
    let (model, model_source) = choose_model(options.model.as_deref(), scratch)?;
    let (scan, scan_source) = choose_scan(options.scan.as_deref(), scratch)?;
    let load = Load::now();
    let start = Instant::now();
    let encoder = load_encoder(&model, options.device, options.precision)?;
    let encoder_load_seconds = start.elapsed().as_secs_f64();
    let (device, precision) = (encoder.options().device, encoder.options().precision);
    let progress = !options.json && std::io::stderr().is_terminal();
    if !options.json {
        println!("{}  {}", bold("larmor bench"), machine.describe());
        println!("  model {} ({model_source}) on {device} ({precision})", bold(model.id()));
        println!("  scan {} ({scan_source})", scan.display());
    }

    let mut runs: Vec<StageSeconds> = Vec::with_capacity(options.runs);
    let mut tokens = 0;
    let mut encode_cold_seconds = 0.0;
    for run in 0..options.runs {
        let mut s = StageSeconds::default();
        let start = Instant::now();
        let bytes = std::fs::read(&scan)?;
        let nifti = Nifti::from_bytes(&bytes)?;
        s.read = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let sha = sha256_hex(&bytes);
        s.hash = start.elapsed().as_secs_f64();
        let prepared = model.recipe.run(&nifti)?;
        let t = prepared.timings;
        (s.mask, s.conform, s.quantize, s.normalize) = (
            t.mask.as_secs_f64(),
            t.conform.as_secs_f64(),
            t.quantize.as_secs_f64(),
            t.normalize.as_secs_f64(),
        );
        let start = Instant::now();
        let patch_tokens = prepared.tokens(model.patch())?;
        s.tokens = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let encoded = encoder.encode(&patch_tokens.patches, &patch_tokens.coords)?;
        s.encode = start.elapsed().as_secs_f64();
        if run == 0 {
            encode_cold_seconds = s.encode;
        }
        tokens = encoded.n_tokens;
        let encoding = Encoding::new(&encoder, s.clone(), now());
        let info = RunInfo::new(&model, encoding, &scan, &sha, &nifti, &prepared, tokens, None)?;
        let key =
            EntryKey { scan_sha256: &sha, reference_sha256: None, precision: encoder.options().precision };
        let start = Instant::now();
        CacheEntry::write(
            &scratch.join("entry.safetensors"),
            &key,
            &encoded.tokens,
            &patch_tokens.coords,
            &encoded.pooled,
            &info,
        )?;
        s.cache_write = start.elapsed().as_secs_f64();
        s.total = s.read
            + s.hash
            + s.mask
            + s.conform
            + s.quantize
            + s.normalize
            + s.tokens
            + s.encode
            + s.cache_write;
        runs.push(s);
        if progress {
            eprint!("\r  run {}/{}", run + 1, options.runs);
        }
    }
    if progress {
        eprint!("\r\x1b[2K");
    }

    // the first run's encode is the cold one; the warm statistics leave it out
    let warm = &runs[1..];
    let stage = |name: &'static str, pick: fn(&StageSeconds) -> f64, from: &[StageSeconds]| {
        let mut values: Vec<f64> = from.iter().map(pick).collect();
        values.sort_by(f64::total_cmp);
        StageTiming { stage: name, median: median(&values), min: values[0], max: values[values.len() - 1] }
    };
    let mut stages = vec![
        stage("read", |s| s.read, &runs),
        stage("hash", |s| s.hash, &runs),
        stage("mask", |s| s.mask, &runs),
        stage("conform", |s| s.conform, &runs),
        stage("quantize", |s| s.quantize, &runs),
        stage("normalize", |s| s.normalize, &runs),
        stage("tokens", |s| s.tokens, &runs),
        stage("encode (warm)", |s| s.encode, warm),
        stage("cache write", |s| s.cache_write, &runs),
    ];
    let total = stage("total (warm)", |s| s.total, warm);
    let report = BenchReport {
        larmor_version: env!("CARGO_PKG_VERSION"),
        machine,
        load,
        model: model.id().to_string(),
        model_source,
        scan: scan.display().to_string(),
        scan_source,
        device: device.to_string(),
        precision: precision.to_string(),
        runs: options.runs,
        tokens,
        encoder_load_seconds,
        encode_cold_seconds,
        total_median_seconds: total.median,
        stages: {
            stages.push(total);
            stages
        },
    };
    if options.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_report(&report);
    }
    Ok(())
}

fn print_report(report: &BenchReport) {
    println!(
        "  {} tokens · {} runs · encoder load {} · first encode {}",
        report.tokens,
        report.runs,
        secs(report.encoder_load_seconds),
        secs(report.encode_cold_seconds)
    );
    println!("  {}", dim(&format!("{:<14} {:>10} {:>10} {:>10}", "stage", "median", "min", "max")));
    for s in &report.stages {
        let line = format!("{:<14} {:>10} {:>10} {:>10}", s.stage, secs(s.median), secs(s.min), secs(s.max));
        if s.stage.starts_with("total") { println!("  {}", bold(&line)) } else { println!("  {line}") }
    }
    println!("  {}", dim(&format!("machine load when the timing started: {}", report.load.describe())));
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}

/// `$LARMOR_FIXTURES`, else `./fixtures`, when it holds a fixture manifest.
fn fixtures_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("LARMOR_FIXTURES").map_or_else(|| PathBuf::from("fixtures"), PathBuf::from);
    dir.join("manifest.toml").is_file().then_some(dir)
}

/// `--model`, else the patch-8 reference model in the fixtures, else the first model with verified
/// weights in the model folders, else random ViT-S weights.
fn choose_model(arg: Option<&str>, scratch: &Path) -> Result<(Model, String)> {
    if let Some(arg) = arg {
        return Ok((Model::resolve(arg)?, "--model".into()));
    }
    if let Some(fixtures) = fixtures_dir() {
        let path = fixtures.join("models/nanobrain-vits-p8-ref.toml");
        if path.is_file() {
            return Ok((Model::load(&path)?, "fixture reference model".into()));
        }
    }
    for found in model::discover() {
        if let Ok(manifest) = found.manifest
            && manifest.verify_weights().is_ok()
        {
            return Ok((Model::new(manifest, found.path)?, "first installed model".into()));
        }
    }
    let path = phantom::random_vit_small(scratch, 8)?;
    Ok((Model::load(&path)?, "random weights, no model installed".into()))
}

/// `--scan`, else the first fixture scan, else the built-in phantom.
fn choose_scan(arg: Option<&Path>, scratch: &Path) -> Result<(PathBuf, String)> {
    if let Some(path) = arg {
        return Ok((path.to_path_buf(), "--scan".into()));
    }
    if let Some(fixtures) = fixtures_dir() {
        let text = std::fs::read_to_string(fixtures.join("manifest.toml"))?;
        let manifest: toml::Table = text.parse()?;
        let first = manifest
            .get("scan")
            .and_then(|s| s.as_array())
            .and_then(|a| a.first())
            .and_then(|s| s.get("file"))
            .and_then(|f| f.as_str());
        if let Some(file) = first {
            return Ok((fixtures.join(file), "first fixture scan".into()));
        }
    }
    let path = scratch.join("phantom.nii.gz");
    phantom::head_phantom().write(&path)?;
    Ok((path, "built-in phantom".into()))
}
