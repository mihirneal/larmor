//! Phase 0 runtime spike: time the encoder forward pass, and check parity where a reference exists.
//!
//! ```sh
//! cargo run --release -p larmor-model --features fusion,autotune --example spike -- \
//!     --device metal --precision f32,f16 --models vits-p8,vits-p4,vitb-p8,vitl-p8
//! ```
//!
//! Models are `vit<size>-p<patch>`. Size `s` loads `fixtures/models/nanobrain-vits-p<patch>-ref`;
//! sizes `b` (768 wide, 12 deep, 12 heads) and `l` (1024, 24, 16) use random weights and report
//! timing only. Inputs (patches and coordinates) always come from the patch size's encode
//! reference, so token counts are those of a real scan.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use larmor_model::parity::{EncodeReference, compare, compare_parts};
use larmor_model::synthetic::random_safetensors;
use larmor_model::{
    Architecture, AttentionKernel, AttentionOptions, DeviceChoice, Encoder, EncoderOptions, Manifest,
    Precision,
};

#[derive(Parser)]
struct Args {
    /// Fixtures directory.
    #[arg(long, env = "LARMOR_FIXTURES", default_value = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures"))]
    fixtures: PathBuf,
    /// auto, cpu, metal, wgpu or cuda[:N].
    #[arg(long, default_value = "auto")]
    device: DeviceChoice,
    /// Comma-separated precisions: f32, f16, bf16.
    #[arg(long, value_delimiter = ',', default_value = "f32,f16")]
    precision: Vec<Precision>,
    /// Comma-separated attention kernels: auto, explicit, fused.
    #[arg(long, value_delimiter = ',', default_value = "auto")]
    kernel: Vec<String>,
    /// Comma-separated models, e.g. vits-p8,vits-p4,vitb-p8,vitl-p8.
    #[arg(long, value_delimiter = ',', default_value = "vits-p8,vits-p4,vitb-p8,vitl-p8")]
    models: Vec<String>,
    /// Scan whose encode reference supplies the inputs.
    #[arg(long, default_value = "mni152-t1w")]
    scan: String,
    /// Untimed runs before timing (the first also compiles kernels and autotunes).
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    /// Timed runs; the median is reported.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    /// Budget for one attention chunk's f32 score matrix, in GiB.
    #[arg(long, default_value_t = 2.0)]
    score_budget_gib: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = args.device.resolve();
    println!("device {device}; timings are wall clock on a shared machine, so preliminary");
    for model in &args.models {
        let (size, patch) = parse_model(model)?;
        let reference_id = format!("nanobrain-vits-p{patch}-ref");
        let reference_path =
            args.fixtures.join(format!("reference/encode/{reference_id}/{}.safetensors", args.scan));
        let reference = EncodeReference::load(&reference_path)
            .with_context(|| format!("loading {}", reference_path.display()))?;
        let manifest = Manifest::load(args.fixtures.join(format!("models/{reference_id}.toml")))?;
        for &precision in &args.precision {
            for kernel in &args.kernel {
                let attention = AttentionOptions {
                    kernel: parse_kernel(kernel)?,
                    score_budget_bytes: (args.score_budget_gib * (1u64 << 30) as f64) as usize,
                };
                let options = EncoderOptions { device, precision, attention };
                run_case(model, size, &manifest, &reference, options, &args)?;
            }
        }
    }
    Ok(())
}

fn run_case(
    model: &str,
    size: char,
    manifest: &Manifest,
    reference: &EncodeReference,
    options: EncoderOptions,
    args: &Args,
) -> Result<()> {
    let start = Instant::now();
    let encoder = match size {
        's' => Encoder::load_with(manifest, options)?,
        _ => {
            let arch = scaled_architecture(&manifest.architecture, size);
            Encoder::from_safetensors(&arch, &random_safetensors(&arch, 0), options)?
        }
    };
    let load_seconds = start.elapsed().as_secs_f64();
    let resolved = encoder.options();
    let case = format!("{model} {} {} {:?}", resolved.device, resolved.precision, resolved.attention.kernel);
    let first = encoder.encode(&reference.patches, &reference.coords)?;
    for _ in 1..args.warmup {
        encoder.encode(&reference.patches, &reference.coords)?;
    }
    let mut seconds = Vec::with_capacity(args.runs);
    let mut last = first.clone();
    for _ in 0..args.runs {
        last = encoder.encode(&reference.patches, &reference.coords)?;
        seconds.push(last.seconds);
    }
    seconds.sort_by(f64::total_cmp);
    let median = seconds.get(seconds.len() / 2).copied().unwrap_or(f64::NAN);
    let memory = match encoder.device_memory_reserved() {
        Some(bytes) => format!("device pool {:.1} GiB", bytes as f64 / (1u64 << 30) as f64),
        None => format!("process max RSS {:.1} GiB", max_rss_bytes() as f64 / (1u64 << 30) as f64),
    };
    println!(
        "{case}: {} tokens, width {}, load {load_seconds:.2}s, first {:.2}s, median {median:.3}s \
         (min {:.3}, max {:.3}, {} runs), {memory}",
        last.n_tokens,
        last.width,
        first.seconds,
        seconds.first().copied().unwrap_or(f64::NAN),
        seconds.last().copied().unwrap_or(f64::NAN),
        seconds.len(),
    );
    if size == 's' {
        let parity = compare(&last, reference);
        let verdict = if parity.passes(0.999) { "PASS" } else { "FAIL" };
        println!("  vs torch fp32 [{verdict}]: {parity}");
        if let (Some(tokens), Some(pooled)) = (&reference.tokens_bf16, &reference.pooled_bf16) {
            // how far torch's own bf16 autocast is from its fp32, for scale
            let torch_bf16 = compare_parts(tokens, &reference.cls, pooled, reference, reference.width);
            println!(
                "  torch bf16 autocast vs torch fp32: token cos min {:.6}, pooled {:.7}",
                torch_bf16.min_token_cosine, torch_bf16.pooled_cosine
            );
        }
        if first != last {
            let drift =
                compare_parts(&last.tokens, &last.cls, &last.pooled, &as_reference(&first), last.width);
            println!("  run-to-run: token cos min {:.9}", drift.min_token_cosine);
        }
    }
    encoder.release_cached_memory();
    Ok(())
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

fn parse_kernel(kernel: &str) -> Result<AttentionKernel> {
    match kernel {
        "auto" => Ok(AttentionKernel::Auto),
        "explicit" => Ok(AttentionKernel::Explicit),
        "fused" => Ok(AttentionKernel::Fused),
        other => bail!("unknown attention kernel {other:?} (auto, explicit or fused)"),
    }
}

/// Wrap an output as a reference so two runs can be compared.
fn as_reference(encoded: &larmor_model::Encoded) -> EncodeReference {
    EncodeReference {
        patches: vec![],
        coords: vec![],
        tokens: encoded.tokens.clone(),
        cls: encoded.cls.clone(),
        pooled: encoded.pooled.clone(),
        tokens_bf16: None,
        pooled_bf16: None,
        width: encoded.width,
    }
}

/// Peak resident set size of this process.
fn max_rss_bytes() -> u64 {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes a complete rusage into the pointer it is given
    let usage = unsafe {
        libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr());
        usage.assume_init()
    };
    let max_rss = usage.ru_maxrss.max(0) as u64;
    // macOS reports bytes, Linux kibibytes
    if cfg!(target_os = "macos") { max_rss } else { max_rss * 1024 }
}
