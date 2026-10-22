//! `larmor encode`: preprocess and encode scans through the cache and write BIDS derivatives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use larmor_model::{DeviceChoice, Precision};
use larmor_prep::Prepared;
use larmor_volume::{Nifti, VoxelData};

use crate::console::{bold, dim, green, red, secs};
use crate::locate;
use crate::model::Model;
use crate::outputs::{self, ScanOutputs};
use crate::pipeline::{Outcome, Pipeline, ScanResult};
use crate::run_info::CacheInfo;

/// Options of `larmor encode`.
pub struct EncodeOptions {
    pub inputs: Vec<PathBuf>,
    pub model: String,
    pub out: Option<PathBuf>,
    pub device: DeviceChoice,
    pub precision: Precision,
    pub force: bool,
    pub save_preprocessed: bool,
}

pub fn cmd_encode(options: EncodeOptions) -> Result<()> {
    let start = Instant::now();
    let model = Model::resolve(&options.model)?;
    let scans = locate::find_scans(&options.inputs)?;
    if scans.is_empty() {
        bail!("no .nii or .nii.gz scans found");
    }
    let destinations: Vec<(PathBuf, ScanOutputs)> = scans
        .iter()
        .map(|scan| {
            let root = options.out.clone().unwrap_or_else(|| locate::default_out_dir(scan));
            let outputs = ScanOutputs::new(&root, scan);
            (root, outputs)
        })
        .collect();
    check_unique_outputs(&scans, &destinations, model.id())?;

    let cache_root = locate::cache_root()?;
    println!(
        "{}  {} scan{} · model {} ({}) · cache {}",
        bold("larmor encode"),
        scans.len(),
        if scans.len() == 1 { "" } else { "s" },
        bold(model.id()),
        &model.weights_sha256()[..8],
        dim(&cache_root.display().to_string())
    );
    let mut pipeline = Pipeline::new(&model, options.device, options.precision, cache_root);
    pipeline.stager.force = options.force;
    let stager = pipeline.stager.clone();

    let (mut encoded, mut cached, mut failed) = (0, 0, 0);
    std::thread::scope(|scope| {
        // a worker reads and preprocesses the next scan while the encoder works on this one
        let (sender, staged_scans) = std::sync::mpsc::sync_channel(1);
        let all = &scans;
        scope.spawn(move || {
            for scan in all {
                if sender.send(stager.stage(scan, false)).is_err() {
                    break;
                }
            }
        });
        for ((scan, (root, outputs)), staged) in scans.iter().zip(&destinations).zip(staged_scans) {
            let result = staged.and_then(|staged| pipeline.finish(staged, false)).and_then(|result| {
                write_outputs(pipeline.model(), scan, root, outputs, &result, options.save_preprocessed)?;
                Ok(result)
            });
            match result {
                Ok(result) => {
                    println!("{}", status_line(&outputs.stem, &result));
                    match result.outcome {
                        Outcome::Encoded => encoded += 1,
                        Outcome::Cached => cached += 1,
                    }
                }
                Err(err) => {
                    println!("  {} {}  {err:#}", red("failed "), outputs.stem);
                    failed += 1;
                }
            }
        }
    });

    let roots: Vec<String> = {
        let mut roots: Vec<String> =
            destinations.iter().map(|(root, _)| root.display().to_string()).collect();
        roots.sort();
        roots.dedup();
        roots
    };
    println!(
        "{} encoded, {} cached, {} failed in {} · outputs in {}",
        encoded,
        cached,
        if failed > 0 { red(&failed.to_string()) } else { failed.to_string() },
        secs(start.elapsed().as_secs_f64()),
        roots.join(", ")
    );
    if failed > 0 {
        bail!("{failed} of {} scans failed", scans.len());
    }
    Ok(())
}

/// One scan's derivatives: the pooled embedding, run.json and, with `--save-preprocessed`, the
/// grid image and mask.
fn write_outputs(
    model: &Model,
    scan: &Path,
    root: &Path,
    outputs: &ScanOutputs,
    result: &ScanResult,
    save_preprocessed: bool,
) -> Result<()> {
    let model_id = model.id();
    outputs.create_dir()?;
    outputs::ensure_dataset_description(root)?;
    let pooled = &result.entry.pooled;
    outputs::write_npy_f32(&outputs.embedding(model_id), pooled, &[pooled.len()])?;
    let mut info = result.entry.info.clone();
    info.cache = Some(CacheInfo {
        path: result.cache_path.display().to_string(),
        hit: result.outcome == Outcome::Cached,
    });
    outputs::write_json(&outputs.run_json(model_id), &info)?;
    if save_preprocessed {
        match &result.prepared {
            Some(prepared) => write_preprocessed(prepared, outputs, model_id)?,
            None => write_preprocessed(&prepare_again(model, scan)?.1, outputs, model_id)?,
        }
    }
    Ok(())
}

/// Reads a scan whose embedding came from the cache and runs the recipe on it again (about 0.1 s).
pub fn prepare_again(model: &Model, scan: &Path) -> Result<(Nifti, Prepared)> {
    let nifti = Nifti::read(scan).with_context(|| format!("reading {}", scan.display()))?;
    let prepared = model.recipe.run(&nifti).with_context(|| format!("preprocessing {}", scan.display()))?;
    Ok((nifti, prepared))
}

/// `--save-preprocessed`: the quantized uint16 grid image and the uint8 grid mask, on the fit affine.
fn write_preprocessed(prepared: &Prepared, outputs: &ScanOutputs, model_id: &str) -> Result<()> {
    let quantized = Nifti::new(VoxelData::U16(prepared.quantized.clone().into_dyn()), &prepared.fit_affine);
    quantized.write(outputs.quantized(model_id))?;
    let mask = prepared.mask.mapv(u8::from).into_dyn();
    Nifti::new(VoxelData::U8(mask), &prepared.fit_affine).write(outputs.grid_mask(model_id))?;
    Ok(())
}

fn status_line(name: &str, result: &ScanResult) -> String {
    let info = &result.entry.info;
    match result.outcome {
        Outcome::Encoded => {
            let s = &info.seconds;
            let prep = s.mask + s.conform + s.quantize + s.normalize;
            format!(
                "  {} {name}  {} tokens · read {} · prep {} · encode {} · total {}",
                green("encoded"),
                info.tokens,
                secs(s.read + s.hash),
                secs(prep),
                secs(s.encode),
                secs(s.total)
            )
        }
        Outcome::Cached => format!(
            "  {} {name}  {} tokens · read {}",
            dim("cached "),
            info.tokens,
            secs(result.read_hash_seconds)
        ),
    }
}

/// Two scans with the same stem in folders that map to the same output folder would overwrite each
/// other's outputs.
fn check_unique_outputs(
    scans: &[PathBuf],
    destinations: &[(PathBuf, ScanOutputs)],
    model_id: &str,
) -> Result<()> {
    let mut seen: HashMap<PathBuf, &Path> = HashMap::new();
    for (scan, (_, outputs)) in scans.iter().zip(destinations) {
        let target = outputs.embedding(model_id);
        if let Some(other) = seen.insert(target.clone(), scan) {
            bail!(
                "{} and {} would both write {}; encode them separately with different --out folders",
                other.display(),
                scan.display(),
                target.display()
            );
        }
    }
    Ok(())
}
