//! `larmor encode`: preprocess and encode scans through the cache and write BIDS derivatives.

use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use larmor_model::{DeviceChoice, Precision};
use larmor_prep::Prepared;
use larmor_volume::{Nifti, VoxelData};

use crate::console::{bold, dim, green, red, secs};
use crate::locate;
use crate::model::Model;
use crate::outputs::{self, ScanOutputs, check_unique_outputs};
use crate::pipeline::{Outcome, Pipeline, ReadScan, ScanResult, panic_message};
use crate::run_info::{CacheInfo, RunInfo, absolute};

/// Options of `larmor encode`.
pub struct EncodeOptions {
    pub inputs: Vec<PathBuf>,
    pub model: String,
    pub out: Option<PathBuf>,
    /// Preprocess every scan on this scan's head mask and grid (another modality of the subject).
    pub reference: Option<PathBuf>,
    pub device: DeviceChoice,
    pub precision: Precision,
    pub force: bool,
    pub save_preprocessed: bool,
}

pub fn cmd_encode(options: EncodeOptions) -> Result<()> {
    let start = Instant::now();
    let model = Model::resolve(&options.model)?;
    let reference = options.reference.as_deref();
    if let Some(reference) = reference {
        check_reference(reference)?;
    }
    let found = locate::find_scans(&options.inputs, options.out.as_slice())?;
    let scans = found.scans;
    if scans.is_empty() {
        bail!("no .nii or .nii.gz scans found");
    }
    let destinations: Vec<(PathBuf, ScanOutputs)> = scans
        .iter()
        .map(|scan| {
            let root = options.out.clone().unwrap_or_else(|| locate::default_out_dir(scan));
            let outputs = ScanOutputs::new(&root, scan, found.root.as_deref());
            (root, outputs)
        })
        .collect();
    let targets: Vec<(&Path, PathBuf)> = scans
        .iter()
        .zip(&destinations)
        .map(|(scan, (_, o))| (scan.as_path(), o.embedding(model.id())))
        .collect();
    check_unique_outputs(&targets)?;

    let cache_root = locate::cache_root()?;
    let mut pipeline = Pipeline::new(&model, options.device, options.precision, cache_root);
    println!(
        "{}  {} scan{} · model {} ({}) · {} · cache {}{}",
        bold("larmor encode"),
        scans.len(),
        if scans.len() == 1 { "" } else { "s" },
        bold(model.id()),
        &model.encoder_key[..8],
        pipeline.precision(),
        dim(&pipeline.stager.cache_root.display().to_string()),
        reference.map_or_else(String::new, |r| format!(" · on the head mask of {}", r.display()))
    );
    pipeline.stager.force = options.force;
    let stager = pipeline.stager.clone();

    let (mut encoded, mut cached, mut failed) = (0, 0, 0);
    std::thread::scope(|scope| {
        // a worker reads and preprocesses the next scan while the encoder works on this one; a
        // panic there becomes that scan's error, so the other scans still go on
        let (sender, staged_scans) = std::sync::mpsc::sync_channel(1);
        let all = &scans;
        scope.spawn(move || {
            for scan in all {
                if sender.send(stager.stage_catching(scan, reference, false)).is_err() {
                    break;
                }
            }
        });
        for ((scan, (root, outputs)), staged) in scans.iter().zip(&destinations).zip(staged_scans) {
            // a panic while encoding becomes this scan's error too
            let finish = |staged| {
                panic::catch_unwind(AssertUnwindSafe(|| pipeline.finish(staged, false))).unwrap_or_else(
                    |payload| {
                        Err(anyhow!(
                            "internal error encoding {}: {}",
                            scan.display(),
                            panic_message(payload.as_ref())
                        ))
                    },
                )
            };
            let result = staged.and_then(finish).and_then(|result| {
                let write =
                    WriteOptions { save_preprocessed: options.save_preprocessed, force: options.force };
                write_outputs(pipeline.model(), scan, reference, root, outputs, &result, &write)?;
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

/// A `--reference` must be a readable NIfTI file.
pub fn check_reference(reference: &Path) -> Result<()> {
    if !reference.is_file() {
        bail!("the reference scan {} does not exist", reference.display());
    }
    if !locate::is_nifti(reference) {
        bail!("the reference scan {} is not a .nii or .nii.gz file", reference.display());
    }
    Ok(())
}

/// How [`write_outputs`] writes.
struct WriteOptions {
    save_preprocessed: bool,
    /// Replace outputs that another scan wrote.
    force: bool,
}

/// One scan's derivatives: the pooled embedding, run.json and, with `--save-preprocessed`, the
/// grid image and mask.
fn write_outputs(
    model: &Model,
    scan: &Path,
    reference: Option<&Path>,
    root: &Path,
    outputs: &ScanOutputs,
    result: &ScanResult,
    options: &WriteOptions,
) -> Result<()> {
    let model_id = model.id();
    let run_json = outputs.run_json(model_id);
    if !options.force {
        refuse_to_overwrite_another_scan(&run_json, scan, &result.entry.info.scan.sha256)?;
    }
    outputs.create_dir()?;
    outputs::ensure_dataset_description(root)?;
    let pooled = &result.entry.pooled;
    outputs::write_npy_f32(&outputs.embedding(model_id), pooled, &[pooled.len()])?;
    let cache =
        CacheInfo { path: result.cache_path.display().to_string(), hit: result.outcome == Outcome::Cached };
    outputs::write_json(&run_json, &result.entry.info.for_run(model, scan, cache))?;
    if options.save_preprocessed {
        match &result.prepared {
            Some(prepared) => write_preprocessed(prepared, outputs, model_id)?,
            None => write_preprocessed(&prepare_again(model, scan, reference)?.1, outputs, model_id)?,
        }
    }
    Ok(())
}

/// Two runs that put different scans with the same file name into the same output folder would
/// silently replace each other's derivatives; refuse instead. Outputs of the same scan file, or of
/// the same bytes moved elsewhere, are replaced as usual.
fn refuse_to_overwrite_another_scan(run_json: &Path, scan: &Path, sha256: &str) -> Result<()> {
    let Ok(text) = std::fs::read_to_string(run_json) else { return Ok(()) };
    let Ok(existing) = serde_json::from_str::<RunInfo>(&text) else { return Ok(()) };
    if existing.scan.path == absolute(scan) || existing.scan.sha256 == sha256 {
        return Ok(());
    }
    bail!(
        "{} holds the outputs of another scan ({}); refusing to replace them. Encode both scans in one run \
         (their outputs then mirror their folders), use another --out, or pass --force to replace them",
        run_json.display(),
        existing.scan.path
    )
}

/// Reads a scan whose embedding came from the cache and runs the recipe on it again (about 0.1 s),
/// on the head mask and grid of `reference` when one is given.
pub fn prepare_again(model: &Model, scan: &Path, reference: Option<&Path>) -> Result<(Nifti, Prepared)> {
    let target = ReadScan::read(scan)?;
    let nifti = target.parse()?;
    let reference = reference.map(ReadScan::read).transpose()?.filter(|r| r.sha256 != target.sha256);
    let prepared = match reference {
        None => model.recipe.run(&nifti),
        Some(reference) => model.recipe.run_with_reference(&nifti, &reference.parse()?),
    };
    let prepared = prepared.with_context(|| format!("preprocessing {}", scan.display()))?;
    Ok((nifti, prepared))
}

/// `--save-preprocessed`: the quantized uint16 grid image (on the affine it was conformed on, as
/// `prepare_images` stores it) and the uint8 grid mask (on the grid's affine).
fn write_preprocessed(prepared: &Prepared, outputs: &ScanOutputs, model_id: &str) -> Result<()> {
    let quantized = Nifti::new(VoxelData::U16(prepared.quantized.clone().into_dyn()), &prepared.image_affine);
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
