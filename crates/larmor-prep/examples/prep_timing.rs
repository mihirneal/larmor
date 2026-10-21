//! Times reading a scan and `Recipe::run` stage by stage, best of several runs.
//!
//! ```text
//! cargo run --release -p larmor-prep --example prep_timing -- scan.nii.gz [more.nii.gz ...]
//! cargo run --release -p larmor-prep --example prep_timing -- --fixtures [ROOT] [--runs N]
//! ```
//!
//! With `--fixtures` it times every scan in `ROOT/manifest.toml` (default
//! `/Users/mihir/Developer/larmor/fixtures`) and puts the Python seconds from each scan's
//! `reference/prep/<scan>/info.json` next to it. Those cover `threshold_mask`, `conform_image`
//! and `truncate_image`, including nibabel's lazy read of the file, but not the normalisation, so
//! the comparison column is Rust read + mask + conform + quantize.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use larmor_prep::{Recipe, Timings};
use larmor_volume::Nifti;

const DEFAULT_FIXTURES: &str = "/Users/mihir/Developer/larmor/fixtures";

/// One scan to time, with the Python reference seconds when there are any.
struct Job {
    name: String,
    path: PathBuf,
    python_s: Option<f64>,
}

/// The fastest of `runs` timings of each part (taken separately, so the parts can come from
/// different runs).
struct Best {
    read: Duration,
    stages: Timings,
}

fn time_scan(recipe: &Recipe, path: &Path, runs: usize) -> Result<Best, Box<dyn std::error::Error>> {
    let mut best: Option<Best> = None;
    for _ in 0..runs {
        let start = Instant::now();
        let bytes = std::fs::read(path)?;
        let scan = Nifti::from_bytes(&bytes)?;
        let read = start.elapsed();
        let t = recipe.run(&scan)?.timings;
        best = Some(match best {
            None => Best { read, stages: t },
            Some(b) => Best {
                read: b.read.min(read),
                stages: Timings {
                    mask: b.stages.mask.min(t.mask),
                    conform: b.stages.conform.min(t.conform),
                    quantize: b.stages.quantize.min(t.quantize),
                    normalize: b.stages.normalize.min(t.normalize),
                },
            },
        });
    }
    Ok(best.expect("at least one run"))
}

fn manifest_jobs(root: &Path) -> Result<Vec<Job>, Box<dyn std::error::Error>> {
    let manifest: toml::Table = std::fs::read_to_string(root.join("manifest.toml"))?.parse()?;
    let mut jobs = Vec::new();
    for scan in manifest["scan"].as_array().ok_or("manifest has no [[scan]]")? {
        let name = scan["name"].as_str().ok_or("scan without a name")?.to_owned();
        let path = root.join(scan["file"].as_str().ok_or("scan without a file")?);
        let info = root.join("reference/prep").join(&name).join("info.json");
        let python_s = std::fs::read_to_string(info).ok().and_then(|text| {
            let info: serde_json::Value = serde_json::from_str(&text).ok()?;
            Some(info["seconds"].as_object()?.values().filter_map(serde_json::Value::as_f64).sum())
        });
        jobs.push(Job { name, path, python_s });
    }
    Ok(jobs)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).peekable();
    let mut runs = 5;
    let mut jobs = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--runs" => runs = args.next().ok_or("--runs needs a number")?.parse()?,
            "--fixtures" => {
                let root = match args.next_if(|next| !next.starts_with("--")) {
                    Some(root) => PathBuf::from(root),
                    None => PathBuf::from(DEFAULT_FIXTURES),
                };
                jobs.extend(manifest_jobs(&root)?);
            }
            path => jobs.push(Job { name: path.to_owned(), path: PathBuf::from(path), python_s: None }),
        }
    }

    let recipe = Recipe::nanobrain_eval();
    // warm up the thread pool and the allocator so the first scan is not penalised
    if let Some(first) = jobs.first() {
        time_scan(&recipe, &first.path, 1)?;
    }
    println!("best of {runs} runs, seconds");
    println!(
        "| scan | read | mask | conform | quantize | normalize | read+mask+conform+quantize | Python | speedup |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for job in &jobs {
        let best = time_scan(&recipe, &job.path, runs)?;
        let t = best.stages;
        let comparable = (best.read + t.mask + t.conform + t.quantize).as_secs_f64();
        let (python, speedup) = match job.python_s {
            Some(p) => (format!("{p:.3}"), format!("{:.1}x", p / comparable)),
            None => ("-".into(), "-".into()),
        };
        println!(
            "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {comparable:.3} | {python} | {speedup} |",
            job.name,
            best.read.as_secs_f64(),
            t.mask.as_secs_f64(),
            t.conform.as_secs_f64(),
            t.quantize.as_secs_f64(),
            t.normalize.as_secs_f64(),
        );
    }
    Ok(())
}
