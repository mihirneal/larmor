//! `candle-spike`: parity and timing of nanobrain's encoder on Candle (Metal).
//!
//! ```text
//! candle-spike parity    --model M.toml --reference R.safetensors
//! candle-spike bench     --input R.safetensors (--model M.toml | --shape WIDTHxDEPTHxHEADS [--patch 8])
//!                        [--warmup 2] [--runs 5]
//! candle-spike attention --tokens T [--heads 6] [--head-dim 64] [--warmup 2] [--runs 5]
//! options (comma-separated lists, every combination is run):
//!   --precision f32,f16,bf16,f16-mixed,bf16-mixed   --attention fused,chunked:<rows>
//!   --bias broadcast,folded                         --json OUT.jsonl   --max-scratch-gb 16
//! ```
//!
//! `bench` re-patchifies the reference image when the model's patch size differs from the
//! reference's. `attention` times the attention kernel alone, in each precision's compute dtype.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use candle_core::Device;
use serde::Serialize;

use candle_spike::bench::{Timing, attention_flops, time_attention, time_forward};
use candle_spike::config::{Architecture, Manifest};
use candle_spike::memory;
use candle_spike::metal_fast_math;
use candle_spike::metrics::{Parity, compare};
use candle_spike::model::{AttentionKernel, BiasAdd, Encoder, EncoderInput, Options, Precision};
use candle_spike::reference::EncodeReference;
use candle_spike::tokens::Patches;
use candle_spike::weights::Weights;

const GIB: f64 = (1u64 << 30) as f64;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    let flags = Flags::parse(args)?;
    match command.as_str() {
        "parity" => run_parity(&flags),
        "bench" => run_bench(&flags),
        "attention" => run_attention(&flags),
        _ => bail!("usage: candle-spike parity|bench|attention [flags] (see the source header)"),
    }
}

fn run_parity(flags: &Flags) -> Result<()> {
    let manifest = Manifest::load(&flags.path("model")?)?;
    let reference = EncodeReference::load(&flags.path("reference")?)?;
    let scan = reference.metadata.get("scan").map(String::as_str).unwrap_or("?");
    println!(
        "parity {} on {} tokens of {scan} (fast math {})",
        manifest.id,
        reference.len(),
        metal_fast_math()
    );
    println!(
        "{:<32} {:>11} {:>11} {:>11} {:>11} {:>11} {:>11} {:>9}",
        "options", "min cos", "mean cos", "pooled", "min ctr", "mean ctr", "cls", "max |d|"
    );
    for options in flags.options()? {
        let mut record =
            Record::new("parity", &manifest.id, &manifest.architecture, reference.len(), options);
        let result =
            check_scratch(&manifest.architecture, reference.len(), options, flags.max_scratch_bytes()?)
                .and_then(|()| parity_one(&manifest, &reference, options));
        match result {
            Ok((parity, parity_bf16)) => {
                print_parity(options, &parity, "fp32 ref");
                print_parity(options, &parity_bf16, "bf16 ref");
                record.parity = Some(parity);
                record.parity_vs_bf16_reference = Some(parity_bf16);
            }
            Err(error) => record.fail(error),
        }
        flags.write_json(&record)?;
    }
    Ok(())
}

/// Encodes the reference's own patches and compares with its fp32 and bf16-autocast outputs.
fn parity_one(
    manifest: &Manifest,
    reference: &EncodeReference,
    options: Options,
) -> Result<(Parity, Parity)> {
    let arch = &manifest.architecture;
    let device = Device::new_metal(0)?;
    let weights = Weights::load(&manifest.weights_path, arch, &device)?;
    let encoder = Encoder::new(arch, &weights, options)?;
    let input = EncoderInput::new(&reference.patches.values, &reference.patches.coords, arch, &device)?;
    let output = encoder.forward(&input)?;
    let tokens = output.tokens.flatten_all()?.to_vec1::<f32>()?;
    let cls = output.cls.flatten_all()?.to_vec1::<f32>()?;
    let d = reference.dim;
    let parity = compare(&tokens, &reference.tokens, &reference.pooled, Some((&cls, &reference.cls)), d);
    let parity_bf16 = compare(&tokens, &reference.tokens_bf16, &reference.pooled_bf16, None, d);
    Ok((parity, parity_bf16))
}

fn print_parity(options: Options, p: &Parity, against: &str) {
    let cls = p.cls_cosine.map(|c| format!("{c:.9}")).unwrap_or_else(|| "-".into());
    println!(
        "{:<32} {:>11.9} {:>11.9} {:>11.9} {:>11.9} {:>11.9} {:>11} {:>9.2e}  ({against}, rel L2 {:.2e})",
        options_name(options),
        p.min_token_cosine,
        p.mean_token_cosine,
        p.pooled_cosine,
        p.min_centered_cosine,
        p.mean_centered_cosine,
        cls,
        p.max_abs_diff,
        p.relative_l2,
    );
}

fn run_bench(flags: &Flags) -> Result<()> {
    let reference = EncodeReference::load(&flags.path("input")?)?;
    let (model_id, arch, weights_path) = match flags.get("model") {
        Some(path) => {
            let manifest = Manifest::load(Path::new(path))?;
            (manifest.id, manifest.architecture, Some(manifest.weights_path))
        }
        None => {
            let shape = flags.get("shape").context("bench needs --model or --shape")?;
            let arch = parse_shape(shape, flags.usize_or("patch", 8)?)?;
            (format!("random-{shape}-p{}", arch.patch), arch, None)
        }
    };
    let patches = reference.patches_for(arch.patch)?;
    let tokens = patches.coords.len();
    let (warmup, runs) = (flags.usize_or("warmup", 2)?, flags.usize_or("runs", 5)?);
    println!(
        "bench {model_id}: {tokens} tokens (+{} cls), width {} depth {} heads {}, warmup {warmup}, runs {runs}, \
         fast math {}",
        arch.class_tokens,
        arch.width,
        arch.depth,
        arch.heads,
        metal_fast_math()
    );
    println!(
        "{:<32} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}",
        "options", "median ms", "min ms", "max ms", "cold ms", "peak GiB", "base GiB"
    );
    for options in flags.options()? {
        let mut record = Record::new("bench", &model_id, &arch, tokens, options);
        let result = check_scratch(&arch, tokens, options, flags.max_scratch_bytes()?)
            .and_then(|()| bench_one(&arch, weights_path.as_deref(), &patches, options, warmup, runs));
        match result {
            Ok((timing, base_bytes)) => {
                println!(
                    "{:<32} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>9} {:>9}",
                    options_name(options),
                    timing.median_ms,
                    timing.min_ms,
                    timing.max_ms,
                    timing.cold_ms,
                    gib(timing.peak_gpu_bytes),
                    gib(base_bytes),
                );
                record.timing = Some(timing);
                record.base_gpu_bytes = base_bytes;
            }
            Err(error) => record.fail(error),
        }
        flags.write_json(&record)?;
    }
    Ok(())
}

/// Times one configuration on a fresh Metal device, so its buffer pool and memory readings
/// start empty. Also returns the GPU bytes held once weights and inputs are loaded.
fn bench_one(
    arch: &Architecture,
    weights_path: Option<&Path>,
    patches: &Patches,
    options: Options,
    warmup: usize,
    runs: usize,
) -> Result<(Timing, Option<usize>)> {
    let device = Device::new_metal(0)?;
    let weights = match weights_path {
        Some(path) => Weights::load(path, arch, &device)?,
        None => Weights::random(arch, &device)?,
    };
    let encoder = Encoder::new(arch, &weights, options)?;
    drop(weights);
    let input = EncoderInput::new(&patches.values, &patches.coords, arch, &device)?;
    device.synchronize()?;
    let base_bytes = memory::allocated_bytes(&device);
    let timing = time_forward(&encoder, &input, &device, warmup, runs)?;
    Ok((timing, base_bytes))
}

fn run_attention(flags: &Flags) -> Result<()> {
    let tokens = flags.usize_or("tokens", 0)?;
    if tokens == 0 {
        bail!("attention needs --tokens");
    }
    let (heads, head_dim) = (flags.usize_or("heads", 6)?, flags.usize_or("head-dim", 64)?);
    let (warmup, runs) = (flags.usize_or("warmup", 2)?, flags.usize_or("runs", 5)?);
    let arch = Architecture { class_tokens: 0, ..Architecture::vit(heads * head_dim, 1, heads, 8) };
    let flops = attention_flops(heads, tokens, head_dim);
    println!(
        "attention: {heads} heads x {tokens} tokens x {head_dim} dims, {:.2} TFLOP per call",
        flops / 1e12
    );
    println!("{:<32} {:>10} {:>10} {:>9} {:>9}", "options", "median ms", "min ms", "TFLOP/s", "peak GiB");
    let mut seen = Vec::new();
    for options in flags.options()? {
        // Only the compute dtype and kernel matter here.
        let key = (options.precision.compute, options.attention);
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let mut record = Record::new("attention", "random-qkv", &arch, tokens, options);
        let result = check_scratch(&arch, tokens, options, flags.max_scratch_bytes()?).and_then(|()| {
            let device = Device::new_metal(0)?;
            time_attention(&device, (heads, tokens, head_dim), key.0, key.1, warmup, runs)
        });
        match result {
            Ok(timing) => {
                println!(
                    "{:<32} {:>10.2} {:>10.2} {:>9.2} {:>9}",
                    format!("{:?} {}", key.0, key.1.name()).to_lowercase(),
                    timing.median_ms,
                    timing.min_ms,
                    flops / (timing.median_ms * 1e-3) / 1e12,
                    gib(timing.peak_gpu_bytes),
                );
                record.timing = Some(timing);
            }
            Err(error) => record.fail(error),
        }
        flags.write_json(&record)?;
    }
    Ok(())
}

/// Refuses manual attention whose score buffers would exceed `max_bytes`, so a full
/// `tokens x tokens` matrix at patch 4 fails fast instead of exhausting unified memory on a
/// shared machine.
fn check_scratch(arch: &Architecture, tokens: usize, options: Options, max_bytes: usize) -> Result<()> {
    let AttentionKernel::Chunked { chunk } = options.attention else {
        return Ok(());
    };
    let t = tokens + arch.class_tokens;
    // the scores and the softmax probabilities are both live at once
    let bytes = 2 * arch.heads * chunk.min(t) * t * options.precision.compute.size_in_bytes();
    if bytes > max_bytes {
        bail!(
            "{} needs {:.1} GiB of score buffers at {t} tokens (limit {:.1} GiB, --max-scratch-gb)",
            options.attention.name(),
            bytes as f64 / GIB,
            max_bytes as f64 / GIB
        );
    }
    Ok(())
}

/// `768x12x12` = width x depth x heads.
fn parse_shape(text: &str, patch: usize) -> Result<Architecture> {
    let parts: Vec<usize> = text
        .split('x')
        .map(|s| s.parse::<usize>())
        .collect::<Result<_, _>>()
        .with_context(|| format!("bad --shape {text:?}, expected WIDTHxDEPTHxHEADS"))?;
    let [width, depth, heads] = parts[..] else {
        bail!("bad --shape {text:?}, expected WIDTHxDEPTHxHEADS");
    };
    let arch = Architecture::vit(width, depth, heads, patch);
    arch.validate()?;
    Ok(arch)
}

fn options_name(options: Options) -> String {
    format!("{} {} {}", options.precision.name(), options.attention.name(), options.bias.name())
}

fn gib(bytes: Option<usize>) -> String {
    bytes.map(|b| format!("{:.2}", b as f64 / GIB)).unwrap_or_else(|| "-".into())
}

/// One line of `--json` output.
#[derive(Serialize)]
struct Record {
    command: &'static str,
    model: String,
    arch: Architecture,
    tokens: usize,
    precision: &'static str,
    attention: String,
    bias: &'static str,
    fast_math: bool,
    parity: Option<Parity>,
    parity_vs_bf16_reference: Option<Parity>,
    timing: Option<Timing>,
    /// GPU bytes held before the first pass: weights, inputs and anything cast from them.
    base_gpu_bytes: Option<usize>,
    error: Option<String>,
}

impl Record {
    fn new(command: &'static str, model: &str, arch: &Architecture, tokens: usize, options: Options) -> Self {
        Self {
            command,
            model: model.to_string(),
            arch: arch.clone(),
            tokens,
            precision: options.precision.name(),
            attention: options.attention.name(),
            bias: options.bias.name(),
            fast_math: metal_fast_math(),
            parity: None,
            parity_vs_bf16_reference: None,
            timing: None,
            base_gpu_bytes: None,
            error: None,
        }
    }

    /// Prints and records an error, so one failing configuration does not stop the others.
    fn fail(&mut self, error: anyhow::Error) {
        let name = format!("{} {} {}", self.precision, self.attention, self.bias);
        println!("{name:<32} failed: {error:#}");
        self.error = Some(format!("{error:#}"));
    }
}

/// `--key value` flags.
struct Flags {
    values: HashMap<String, String>,
}

impl Flags {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut values = HashMap::new();
        while let Some(arg) = args.next() {
            let key = arg.strip_prefix("--").with_context(|| format!("unexpected argument {arg:?}"))?;
            let value = args.next().with_context(|| format!("--{key} needs a value"))?;
            values.insert(key.to_string(), value);
        }
        Ok(Self { values })
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }

    fn path(&self, key: &str) -> Result<PathBuf> {
        self.get(key).map(PathBuf::from).with_context(|| format!("--{key} is required"))
    }

    fn usize_or(&self, key: &str, default: usize) -> Result<usize> {
        self.get(key).map_or(Ok(default), |v| v.parse().with_context(|| format!("bad --{key} {v:?}")))
    }

    fn list<T>(&self, key: &str, default: &str, parse: impl Fn(&str) -> Result<T>) -> Result<Vec<T>> {
        self.get(key).unwrap_or(default).split(',').map(parse).collect()
    }

    /// Every combination of `--precision`, `--attention` and `--bias`.
    fn options(&self) -> Result<Vec<Options>> {
        let precisions = self.list("precision", "f32,f16,bf16,f16-mixed,bf16-mixed", Precision::parse)?;
        let kernels = self.list("attention", "fused,chunked:2048", AttentionKernel::parse)?;
        let biases = self.list("bias", "broadcast", BiasAdd::parse)?;
        let mut all = Vec::new();
        for &precision in &precisions {
            for &attention in &kernels {
                for &bias in &biases {
                    all.push(Options { precision, attention, bias });
                }
            }
        }
        Ok(all)
    }

    fn max_scratch_bytes(&self) -> Result<usize> {
        let gb: f64 =
            self.get("max-scratch-gb").map_or(Ok(16.0), |v| v.parse().context("bad --max-scratch-gb"))?;
        Ok((gb * GIB) as usize)
    }

    fn write_json(&self, record: &Record) -> Result<()> {
        let Some(path) = self.get("json") else {
            return Ok(());
        };
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", serde_json::to_string(record)?)?;
        Ok(())
    }
}
