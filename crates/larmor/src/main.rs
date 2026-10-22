//! The `larmor` command-line program: MRI scans to foundation-model embeddings, and linear probes on
//! them. `docs/CLI.md` describes every command; `docs/DESIGN.md` is the contract it implements.

mod bench_cmd;
mod cache;
mod console;
mod device;
mod encode_cmd;
mod hashing;
mod labels;
mod locate;
mod machine;
mod model;
mod models_cmd;
mod outputs;
mod parity_cmd;
mod phantom;
mod pipeline;
mod probe_cmd;
mod run_info;
mod scan;
mod seg_labels;
mod site;
mod watch_cmd;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use larmor_model::{DeviceChoice, Precision};

use crate::console::red;
use crate::probe_cmd::Kind;

#[derive(Parser)]
#[command(name = "larmor", version, about = "MRI foundation-model embeddings and linear probes")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Preprocess and encode scans, cache their embeddings and write BIDS derivatives.
    Encode(EncodeArgs),
    /// List, verify and download model weights.
    Models {
        #[command(subcommand)]
        action: Option<ModelsCmd>,
    },
    /// Fit a linear probe on cached embeddings, or apply one to new scans.
    Probe {
        #[command(subcommand)]
        action: ProbeCmd,
    },
    /// Compare every preprocessing stage and embedding with the PyTorch reference fixtures.
    CheckParity(ParityArgs),
    /// Time each stage of one scan on this machine.
    Bench(BenchArgs),
    /// Watch the folders of a site file and encode scans as they arrive.
    Watch(WatchArgs),
}

/// Where the encoder runs.
#[derive(Args, Clone, Copy)]
struct ComputeArgs {
    /// auto, metal or cpu (auto: the Metal GPU when this Mac has one, else the CPU).
    #[arg(long, default_value = "auto")]
    device: DeviceChoice,
    /// Encoder precision: f32 (the parity target), or f16 or bf16 (GPU only, like PyTorch autocast).
    #[arg(long, default_value = "f32")]
    precision: Precision,
}

#[derive(Args)]
struct EncodeArgs {
    /// NIfTI files (.nii, .nii.gz) or folders to search recursively for NIfTI files and DICOM series.
    #[arg(required = true, value_name = "SCAN_OR_FOLDER")]
    inputs: Vec<PathBuf>,
    /// Model id (looked up in $LARMOR_MODELS, the user data folder, ./models) or path to a model.toml.
    #[arg(long)]
    model: String,
    /// Derivatives folder (default: <BIDS root>/derivatives/larmor, else ./derivatives/larmor).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Prepare every scan on this scan's head mask and grid: another modality of the same subject on
    /// the same voxel grid, as the eval framework prepares ADC and DWI on the FLAIR's mask.
    #[arg(long, value_name = "SCAN")]
    reference: Option<PathBuf>,
    #[command(flatten)]
    compute: ComputeArgs,
    /// Encode again even if the cache has the embedding, and replace outputs another scan wrote.
    #[arg(long)]
    force: bool,
    /// Also write the quantized model-grid image and its head mask as NIfTI.
    #[arg(long)]
    save_preprocessed: bool,
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List the model manifests found, with their weights status.
    List,
    /// Recompute the weights SHA-256 of one model (or all) and compare with the manifest.
    Verify {
        /// Model id or path to a model.toml (default: every model found).
        id: Option<String>,
    },
    /// Download a model's weights from its weights.url, then verify them.
    Download {
        /// Model id or path to a model.toml.
        id: String,
        /// Replace an existing weights file whose SHA-256 does not match the manifest.
        #[arg(long)]
        force: bool,
        /// Accept a plain http:// weights.url (https is required otherwise).
        #[arg(long)]
        allow_http: bool,
    },
}

#[derive(Subcommand)]
enum ProbeCmd {
    /// Fit a probe on cached embeddings with the eval framework's cross-validation.
    Fit(FitArgs),
    /// Predict with a saved probe.
    Apply(ApplyArgs),
}

#[derive(Args)]
struct FitArgs {
    /// Label table: a `scan` column (paths relative to the TSV), label columns, optional `fold`.
    labels: PathBuf,
    /// Model id or path to a model.toml.
    #[arg(long)]
    model: String,
    /// What the probe predicts.
    #[arg(long, value_enum)]
    kind: Kind,
    /// The label column to fit.
    #[arg(long)]
    label: String,
    /// Probe file to write; the weights go next to it as .safetensors.
    #[arg(long, default_value = "probe.toml")]
    out: PathBuf,
    #[command(flatten)]
    compute: ComputeArgs,
    /// Encode scans that are not cached yet instead of stopping.
    #[arg(long)]
    encode: bool,
    /// Segmentation: keep at most this many all-negative patches per positive one when fitting.
    #[arg(long, value_name = "RATIO")]
    max_negative_ratio: Option<f64>,
}

#[derive(Args)]
struct ApplyArgs {
    /// A probe.toml written by `larmor probe fit`.
    probe: PathBuf,
    /// NIfTI files, or folders to search for NIfTI files and DICOM series.
    #[arg(required = true, value_name = "SCAN_OR_FOLDER")]
    inputs: Vec<PathBuf>,
    /// Output folder (default: the derivatives folder, as for encode).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Use this model id or model.toml instead of the probe's model id (it must have the same weights).
    #[arg(long)]
    model: Option<String>,
    /// Prepare every scan on this scan's head mask and grid (see `larmor encode --reference`).
    #[arg(long, value_name = "SCAN")]
    reference: Option<PathBuf>,
    #[command(flatten)]
    compute: ComputeArgs,
}

#[derive(Args)]
struct ParityArgs {
    /// Fixtures folder with manifest.toml, scans/, models/ and reference/.
    fixtures: PathBuf,
    /// Only these models (comma-separated ids); default: every model with encode references.
    #[arg(long, value_delimiter = ',')]
    models: Option<Vec<String>>,
    #[command(flatten)]
    compute: ComputeArgs,
}

#[derive(Args)]
struct WatchArgs {
    /// The site file: the folders to watch, the output folder, the models (see examples/site.toml).
    #[arg(long, value_name = "SITE_TOML")]
    site: PathBuf,
    /// Encode the scans that are settled now (after one settle period), then exit.
    #[arg(long)]
    once: bool,
}

#[derive(Args)]
struct BenchArgs {
    /// Model id or path (default: the fixtures' patch-8 reference model, else an installed model,
    /// else random ViT-S weights).
    #[arg(long)]
    model: Option<String>,
    /// Scan to time (default: the first fixture scan, else a built-in phantom).
    #[arg(long)]
    scan: Option<PathBuf>,
    /// Number of runs; the first encode is reported separately as the cold one.
    #[arg(long, default_value_t = 5)]
    runs: usize,
    #[command(flatten)]
    compute: ComputeArgs,
    /// Print the report as JSON.
    #[arg(long)]
    json: bool,
}

fn main() {
    let cli = Cli::parse();
    let result = run(cli.cmd);
    if let Err(e) = result {
        eprintln!("{} {e:#}", red("error:"));
        std::process::exit(1);
    }
}

fn run(cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Encode(args) => encode_cmd::cmd_encode(encode_cmd::EncodeOptions {
            inputs: args.inputs,
            model: args.model,
            out: args.out,
            reference: args.reference,
            device: args.compute.device,
            precision: args.compute.precision,
            force: args.force,
            save_preprocessed: args.save_preprocessed,
        }),
        Cmd::Models { action } => match action.unwrap_or(ModelsCmd::List) {
            ModelsCmd::List => models_cmd::cmd_list(),
            ModelsCmd::Verify { id } => models_cmd::cmd_verify(id.as_deref()),
            ModelsCmd::Download { id, force, allow_http } => {
                models_cmd::cmd_download(&models_cmd::DownloadOptions { id, force, allow_http })
            }
        },
        Cmd::Probe { action: ProbeCmd::Fit(args) } => probe_cmd::cmd_probe_fit(probe_cmd::FitOptions {
            labels: args.labels,
            model: args.model,
            kind: args.kind,
            label: args.label,
            out: args.out,
            device: args.compute.device,
            precision: args.compute.precision,
            encode: args.encode,
            max_negative_ratio: args.max_negative_ratio,
        }),
        Cmd::Probe { action: ProbeCmd::Apply(args) } => probe_cmd::cmd_probe_apply(probe_cmd::ApplyOptions {
            probe: args.probe,
            inputs: args.inputs,
            out: args.out,
            model: args.model,
            reference: args.reference,
            device: args.compute.device,
            precision: args.compute.precision,
        }),
        Cmd::CheckParity(args) => parity_cmd::cmd_check_parity(parity_cmd::ParityOptions {
            fixtures: args.fixtures,
            models: args.models,
            device: args.compute.device,
            precision: args.compute.precision,
        }),
        Cmd::Watch(args) => {
            watch_cmd::cmd_watch(watch_cmd::WatchOptions { site: args.site, once: args.once })
        }
        Cmd::Bench(args) => bench_cmd::cmd_bench(bench_cmd::BenchOptions {
            model: args.model,
            scan: args.scan,
            runs: args.runs,
            device: args.compute.device,
            precision: args.compute.precision,
            json: args.json,
        }),
    }
}
