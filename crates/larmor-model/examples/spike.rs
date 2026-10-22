//! Phase 0 runtime spike: time the encoder, and check parity where a reference exists.
//!
//! ```sh
//! cargo run --release -p larmor-model --example spike -- \
//!     --devices auto --precision f32,f16,bf16 --models vits-p8,vits-p4,vitb-p8,vitl-p8 --wait-idle
//! cargo run --release -p larmor-model --example spike -- --devices cpu --precision f32 --models vits-p8
//! ```
//!
//! Models are `vit<size>-p<patch>`. Size `s` loads `fixtures/models/nanobrain-vits-p<patch>-ref`;
//! sizes `b` (768 wide, 12 deep, 12 heads) and `l` (1024, 24, 16) use random weights and report
//! timing only. Inputs (patches and coordinates) always come from the patch size's encode
//! reference for `--scan`, so token counts are those of a real scan.
//!
//! Each case runs `--warmup` untimed encodes (the first also compiles GPU kernels), then `--runs`
//! timed ones, and reports their median. A time is one whole `Encoder::encode`: upload, forward
//! pass and readback. Next to each time is the machine's load when the case started: GPU
//! utilization (from `ioreg`) and the 1-minute CPU load average. Memory is the peak of
//! the Metal allocation during the timed runs (weights, activations and Candle's buffer pool),
//! or the process's peak resident set on other devices.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use larmor_model::parity::{EncodeReference, compare};
use larmor_model::synthetic::random_safetensors;
use larmor_model::{
    Architecture, AttentionKernel, AttentionOptions, DeviceChoice, Encoded, Encoder, EncoderOptions,
    Manifest, Precision,
};

#[derive(Parser)]
struct Args {
    /// Fixtures directory.
    #[arg(long, env = "LARMOR_FIXTURES", default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures"))]
    fixtures: PathBuf,
    /// Comma-separated devices: auto, cpu or metal.
    #[arg(long, value_delimiter = ',', default_value = "auto")]
    devices: Vec<DeviceChoice>,
    /// Comma-separated precisions: f32, f16, bf16.
    #[arg(long, value_delimiter = ',', default_value = "f32,f16,bf16")]
    precision: Vec<Precision>,
    /// Comma-separated attention kernels: auto, fused, chunked.
    #[arg(long, value_delimiter = ',', default_value = "auto")]
    kernel: Vec<AttentionKernel>,
    /// Comma-separated models, e.g. vits-p8,vits-p4,vitb-p8,vitl-p8.
    #[arg(long, value_delimiter = ',', default_value = "vits-p8,vits-p4,vitb-p8,vitl-p8")]
    models: Vec<String>,
    /// Scan whose encode reference supplies the inputs.
    #[arg(long, default_value = "mni152-t1w")]
    scan: String,
    /// Untimed encodes before timing.
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    /// Timed encodes; the median is reported.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Fewer timed encodes for cases slower than 2 s (patch 4, the CPU).
    #[arg(long, default_value_t = 3)]
    slow_runs: usize,
    /// Budget for the chunked kernel's score buffers, in GiB.
    #[arg(long, default_value_t = 2.0)]
    score_budget_gib: f64,
    /// Query rows per chunk for the chunked kernel (default: as many as fit the budget).
    #[arg(long)]
    chunk_rows: Option<usize>,
    /// Before each case, wait until the GPU has been under 10% busy for 3 s (up to 5 minutes).
    #[arg(long)]
    wait_idle: bool,
}

/// One timed configuration, for the summary table.
struct Case {
    model: String,
    device: DeviceChoice,
    precision: Precision,
    kernel: AttentionKernel,
    tokens: usize,
    width: usize,
    median: f64,
    min: f64,
    max: f64,
    runs: usize,
    memory: String,
    load: String,
    parity: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut cases = Vec::new();
    for model in &args.models {
        let (size, patch) = parse_model(model)?;
        let reference_id = format!("nanobrain-vits-p{patch}-ref");
        let reference_path =
            args.fixtures.join(format!("reference/encode/{reference_id}/{}.safetensors", args.scan));
        let reference = EncodeReference::load(&reference_path)
            .with_context(|| format!("loading {}", reference_path.display()))?;
        let manifest = Manifest::load(args.fixtures.join(format!("models/{reference_id}.toml")))?;
        for &device in &args.devices {
            for &precision in &args.precision {
                for &kernel in &args.kernel {
                    let attention = AttentionOptions {
                        kernel,
                        chunk_rows: args.chunk_rows,
                        score_budget_bytes: (args.score_budget_gib * GIB) as usize,
                    };
                    let options = EncoderOptions { device, precision, attention };
                    match run_case(model, size, &manifest, &reference, options, &args) {
                        Ok(case) => cases.push(case),
                        Err(err) => println!("{model} {device} {precision} {kernel}: FAILED: {err:#}"),
                    }
                }
            }
        }
    }
    print_summary(&cases);
    Ok(())
}

const GIB: f64 = (1u64 << 30) as f64;

fn run_case(
    model: &str,
    size: char,
    manifest: &Manifest,
    reference: &EncodeReference,
    options: EncoderOptions,
    args: &Args,
) -> Result<Case> {
    let encoder = match size {
        's' => Encoder::load_with(manifest, options)?,
        _ => {
            let arch = scaled_architecture(&manifest.architecture, size);
            Encoder::from_safetensors(&arch, &random_safetensors(&arch, 0), options)?
        }
    };
    let resolved = encoder.options();
    let label = format!("{model} {} {} {}", resolved.device, resolved.precision, resolved.attention.kernel);
    let encode = || encoder.encode(&reference.patches, &reference.coords);
    // the first encode also compiles the GPU kernels
    let first = encode()?;
    let mut warm = first.seconds;
    for _ in 1..args.warmup {
        warm = encode()?.seconds;
    }
    let runs = if warm > 2.0 { args.slow_runs } else { args.runs }.max(1);
    if args.wait_idle {
        wait_for_idle_gpu();
    }
    let load = machine_load();
    let mut seconds = Vec::with_capacity(runs);
    let mut last = first.clone();
    let ((), peak_bytes) = while_sampling_memory(&encoder, || -> Result<()> {
        for _ in 0..runs {
            encoder.sync()?;
            last = encode()?;
            seconds.push(last.seconds);
        }
        Ok(())
    })?;
    seconds.sort_by(f64::total_cmp);
    let memory = match peak_bytes {
        Some(bytes) => format!("{:.2} GiB GPU peak", bytes as f64 / GIB),
        None => format!("{:.2} GiB RSS peak", max_rss_bytes() as f64 / GIB),
    };
    let parity = if size == 's' { parity_summary(&last, &first, reference) } else { "random weights".into() };
    let case = Case {
        model: model.to_string(),
        device: resolved.device,
        precision: resolved.precision,
        kernel: resolved.attention.kernel,
        tokens: last.n_tokens,
        width: last.width,
        median: median(&seconds),
        min: seconds[0],
        max: seconds[seconds.len() - 1],
        runs,
        memory,
        load,
        parity,
    };
    println!(
        "{label}: {} tokens, first {:.3} s, median {:.3} s (min {:.3}, max {:.3}, {runs} runs), {}, load {}\n  {}",
        case.tokens, first.seconds, case.median, case.min, case.max, case.memory, case.load, case.parity
    );
    encoder.release_cached_memory();
    Ok(case)
}

/// Parity against the reference's fp32 outputs, PyTorch's own bf16 noise floor, and the drift
/// between the first and the last encode.
fn parity_summary(last: &Encoded, first: &Encoded, reference: &EncodeReference) -> String {
    let parity = compare(last, reference);
    let verdict = if parity.passes(0.999) { "PASS" } else { "FAIL" };
    let floor = reference
        .bf16_noise_floor()
        .map(|f| format!(", torch bf16 floor 1-min {:.1e}", 1.0 - f.min_token_cosine))
        .unwrap_or_default();
    let drift =
        if last.tokens == first.tokens { ", runs identical".to_string() } else { ", runs differ".into() };
    format!(
        "{verdict}: 1-min token cos {:.1e}, 1-pooled {:.1e}, 1-min centered {:.1e}, max |d| {:.1e}{floor}{drift}",
        1.0 - parity.min_token_cosine,
        1.0 - parity.pooled_cosine,
        1.0 - parity.min_centered_cosine,
        parity.max_abs_diff
    )
}

fn print_summary(cases: &[Case]) {
    println!(
        "\n| model | device | precision | attention | tokens | width | median s | min s | max s | runs | memory | load at start | parity vs torch fp32 |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for c in cases {
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.3} | {:.3} | {:.3} | {} | {} | {} | {} |",
            c.model,
            c.device,
            c.precision,
            c.kernel,
            c.tokens,
            c.width,
            c.median,
            c.min,
            c.max,
            c.runs,
            c.memory,
            c.load,
            c.parity
        );
    }
}

/// Run `work` while a second thread polls the device's allocated bytes every millisecond, and
/// return the most it saw (`None` where the device does not report it). Candle's Metal backend
/// keeps every buffer a forward pass allocates until it next waits for the GPU, so the
/// allocation peaks just before each readback and polling catches it.
fn while_sampling_memory<T>(encoder: &Encoder, work: impl FnOnce() -> Result<T>) -> Result<(T, Option<u64>)> {
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        let sampler = scope.spawn(|| {
            let mut peak = encoder.device_memory_reserved();
            while !done.load(Ordering::Relaxed) {
                peak = peak.max(encoder.device_memory_reserved());
                thread::sleep(Duration::from_millis(1));
            }
            peak
        });
        let result = work();
        done.store(true, Ordering::Relaxed);
        let peak = sampler.join().expect("memory sampler panicked");
        result.map(|value| (value, peak))
    })
}

/// The ViT-S manifest's architecture widened to ViT-B or ViT-L.
fn scaled_architecture(base: &Architecture, size: char) -> Architecture {
    let (width, depth, heads) = match size {
        'b' => (768, 12, 12),
        'l' => (1024, 24, 16),
        _ => (base.width, base.depth, base.heads),
    };
    Architecture { width, depth, heads, ..base.clone() }
}

fn parse_model(model: &str) -> Result<(char, usize)> {
    let rest = model.strip_prefix("vit").context("model names look like vits-p8")?;
    let (size, patch) = rest.split_once("-p").context("model names look like vits-p8")?;
    let size = match size {
        "s" | "b" | "l" => size.chars().next().unwrap_or('s'),
        other => bail!("unknown ViT size {other:?} (s, b or l)"),
    };
    Ok((size, patch.parse().context("patch size")?))
}

/// Median of sorted values (mean of the middle two for an even count).
fn median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) { (sorted[mid - 1] + sorted[mid]) / 2.0 } else { sorted[mid] }
}

/// GPU utilization in percent, from the IOAccelerator's performance statistics.
fn gpu_utilization() -> Option<u32> {
    let output = Command::new("ioreg").args(["-r", "-d", "1", "-c", "IOAccelerator"]).output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let key = "\"Device Utilization %\"=";
    let start = text.find(key)? + key.len();
    let digits: String = text[start..].chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// Wait until the GPU has been under 10% busy for three one-second samples in a row, or five
/// minutes have passed.
fn wait_for_idle_gpu() {
    let deadline = Instant::now() + Duration::from_secs(300);
    let mut quiet = 0;
    while quiet < 3 && Instant::now() < deadline {
        match gpu_utilization() {
            Some(busy) if busy >= 10 => quiet = 0,
            Some(_) => quiet += 1,
            None => return,
        }
        thread::sleep(Duration::from_secs(1));
    }
}

/// `gpu N%, cpu L` where L is the 1-minute load average.
fn machine_load() -> String {
    let gpu = gpu_utilization().map_or("gpu ?".into(), |busy| format!("gpu {busy}%"));
    let mut loads = [0f64; 3];
    // SAFETY: getloadavg writes at most the given number of doubles into the buffer
    let filled = unsafe { libc::getloadavg(loads.as_mut_ptr(), 3) };
    let cpu = if filled >= 1 { format!("cpu {:.1}", loads[0]) } else { "cpu ?".into() };
    format!("{gpu}, {cpu}")
}

/// Peak resident set size of this process.
fn max_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes a complete rusage into the pointer it is given
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    // in bytes on macOS
    usage.ru_maxrss.max(0) as u64
}
