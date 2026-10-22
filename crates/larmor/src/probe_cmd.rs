//! `larmor probe fit` and `larmor probe apply`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::ValueEnum;
use larmor_model::{DeviceChoice, Precision};
use larmor_probe::Matrix;
use larmor_probe::fit::{
    Provenance, fit_classification_probe, fit_regression_probe, fit_segmentation_probe_file,
};
use larmor_probe::logistic::LogisticCvOptions;
use larmor_probe::probe_file::{EmbeddingSource, Probe, ProbeKind};
use larmor_probe::segmentation::{SegmentationOptions, SubjectPatches};
use larmor_volume::{Affine, Nifti, VoxelData};

use crate::cache::{self, CacheEntry};
use crate::console::{bold, dim, green, secs, with_ci, yellow};
use crate::device;
use crate::encode_cmd::prepare_again;
use crate::hashing::sha256_hex;
use crate::labels::{LabelColumn, LabelTable};
use crate::locate;
use crate::model::Model;
use crate::outputs::{self, ScanOutputs, tsv_cell};
use crate::pipeline::{Pipeline, ScanResult};
use crate::seg_labels::{assemble_patches, labels_on_grid, patch_rows};

/// `--kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Kind {
    Regression,
    Classification,
    Segmentation,
}

/// Options of `larmor probe fit`.
pub struct FitOptions {
    pub labels: PathBuf,
    pub model: String,
    pub kind: Kind,
    pub label: String,
    pub out: PathBuf,
    pub device: DeviceChoice,
    pub precision: Precision,
    pub encode: bool,
    pub max_negative_ratio: Option<f64>,
}

pub fn cmd_probe_fit(options: FitOptions) -> Result<()> {
    let start = Instant::now();
    let table = LabelTable::read(&options.labels)?;
    let column = table.column(&options.label)?;
    if column.skipped > 0 {
        println!("{} {} rows have no {} and are left out", dim("note:"), column.skipped, column.name);
    }
    let model = Model::resolve(&options.model)?;
    println!(
        "{}  {:?} probe on {} · {} scans · model {} ({})",
        bold("larmor probe fit"),
        options.kind,
        column.name,
        column.rows.len(),
        bold(model.id()),
        &model.weights_sha256()[..8]
    );

    let scans: Vec<PathBuf> = column.rows.iter().map(|row| row.scan.clone()).collect();
    let with_tokens = options.kind == Kind::Segmentation;
    let entries = cached_entries(&model, &scans, &options, with_tokens)?;

    let fit_start = Instant::now();
    let fit = FitInput { column: &column, table: &table, model: &model };
    let mut probe = match options.kind {
        Kind::Regression => fit_regression(&fit, &entries)?,
        Kind::Classification => fit_classification(&fit, &entries)?,
        Kind::Segmentation => fit_segmentation(&fit, entries, &options)?,
    };
    println!("  deployable probe refit on all {} scans: {}", probe.metadata.n_train, chosen(&probe));

    if let Some(dir) = options.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    probe.write(&options.out).with_context(|| format!("writing {}", options.out.display()))?;
    println!(
        "{} wrote {} and {} · fit {} · total {}",
        green("✓"),
        options.out.display(),
        options.out.with_extension("safetensors").display(),
        secs(fit_start.elapsed().as_secs_f64()),
        secs(start.elapsed().as_secs_f64())
    );
    Ok(())
}

/// What every probe kind is fitted from.
struct FitInput<'a> {
    column: &'a LabelColumn<'a>,
    table: &'a LabelTable,
    model: &'a Model,
}

impl FitInput<'_> {
    /// Where the embeddings and labels came from, for `probe.toml`.
    fn provenance(&self, label_names: Vec<String>) -> Provenance {
        Provenance {
            model_id: self.model.id().to_string(),
            model_sha256: self.model.weights_sha256(),
            preprocessing_version: self.model.prep_version.clone(),
            layer: "last".into(),
            label_names,
            label_table_sha256: self.table.sha256.clone(),
        }
    }

    fn folds(&self) -> Option<&[usize]> {
        self.column.folds.as_deref()
    }
}

/// `probe_regression` on the pooled embeddings, then the refit on all scans.
fn fit_regression(fit: &FitInput, entries: &[CacheEntry]) -> Result<Probe> {
    let targets = fit.column.numbers(&fit.table.path)?;
    let provenance = fit.provenance(vec![fit.column.name.clone()]);
    let (probe, cv) = fit_regression_probe(&pooled_matrix(entries), &targets, fit.folds(), &provenance)?;
    let r = &cv.report;
    println!(
        "  MAE {} · r {} · R² {}",
        with_ci(r.mae, r.mae_ci),
        with_ci(r.r, r.r_ci),
        with_ci(r.r2, r.r2_ci)
    );
    println!("  {}", dim(&format!("outer folds chose alpha {:?}", r.fold_alpha)));
    Ok(probe)
}

/// `probe_binary_classification` on the pooled embeddings, then the refit on all scans.
fn fit_classification(fit: &FitInput, entries: &[CacheEntry]) -> Result<Probe> {
    let [negative, positive] = fit.column.classes()?;
    let labels: Vec<bool> = fit.column.values.iter().map(|v| *v == positive).collect();
    let provenance = fit.provenance(vec![fit.column.name.clone(), negative.clone(), positive.clone()]);
    let options = LogisticCvOptions::default();
    let (probe, cv) =
        fit_classification_probe(&pooled_matrix(entries), &labels, fit.folds(), &options, &provenance)?;
    let r = &cv.report;
    println!(
        "  AUROC {} · balanced accuracy {}  ({positive} vs {negative})",
        with_ci(r.auroc, r.auroc_ci),
        with_ci(r.balanced_accuracy, r.balanced_accuracy_ci)
    );
    println!("  {}", dim(&format!("outer folds chose C {:?}", r.fold_c)));
    Ok(probe)
}

/// `probe_binary_segmentation` on the patch tokens, then the refit on all scans.
fn fit_segmentation(fit: &FitInput, entries: Vec<CacheEntry>, options: &FitOptions) -> Result<Probe> {
    let subjects = segmentation_subjects(entries, fit.column, &fit.table.path)?;
    let settings = SegmentationOptions {
        max_negative_ratio: options.max_negative_ratio,
        ..SegmentationOptions::default()
    };
    let device = device::segmentation_device(options.device)?;
    let provenance = fit.provenance(vec![fit.column.name.clone()]);
    let (probe, cv) = fit_segmentation_probe_file(&subjects, fit.folds(), &settings, &provenance, &device)?;
    let r = &cv.report;
    println!(
        "  Dice {} · average precision {} · voxel AUROC {}",
        with_ci(r.dice, r.dice_ci),
        with_ci(r.average_precision, r.average_precision_ci),
        with_ci(r.voxel_auroc, r.voxel_auroc_ci)
    );
    let choices = format!("outer folds chose alpha {:?}, threshold {:?}", r.fold_alpha, r.fold_threshold);
    println!("  {}", dim(&choices));
    Ok(probe)
}

/// The deployable probe's hyperparameters: `alpha 100`, `C 0.01`, `alpha 1000, threshold 0.3`.
fn chosen(probe: &Probe) -> String {
    let h = &probe.metadata.hyperparameters;
    let mut parts = Vec::new();
    if let Some(alpha) = h.alpha {
        parts.push(format!("alpha {alpha}"));
    }
    if let Some(c) = h.c {
        parts.push(format!("C {c}"));
    }
    if let Some(threshold) = h.threshold {
        parts.push(format!("threshold {threshold}"));
    }
    parts.join(", ")
}

/// The cache entries of `scans`, encoding missing ones with `--encode`, else an error listing them.
fn cached_entries(
    model: &Model,
    scans: &[PathBuf],
    options: &FitOptions,
    with_tokens: bool,
) -> Result<Vec<CacheEntry>> {
    let cache_root = locate::cache_root()?;
    let mut paths = Vec::with_capacity(scans.len());
    for scan in scans {
        let bytes = std::fs::read(scan).with_context(|| format!("reading {}", scan.display()))?;
        paths.push(cache::entry_path(&cache_root, model, &sha256_hex(&bytes)));
    }
    let missing: Vec<&PathBuf> =
        scans.iter().zip(&paths).filter(|(_, p)| !p.is_file()).map(|(s, _)| s).collect();
    if !missing.is_empty() && !options.encode {
        let list: Vec<String> = missing.iter().take(20).map(|p| format!("  {}", p.display())).collect();
        let more = if missing.len() > 20 {
            format!("\n  … and {} more", missing.len() - 20)
        } else {
            String::new()
        };
        bail!(
            "{} of {} scans are not encoded with {} yet:\n{}{more}\nRun `larmor encode <scans or folder> --model {}` first, or pass --encode.",
            missing.len(),
            scans.len(),
            model.id(),
            list.join("\n"),
            model.id()
        );
    }
    let mut pipeline = Pipeline::new(model, options.device, options.precision, cache_root);
    let mut entries = Vec::with_capacity(scans.len());
    for (scan, path) in scans.iter().zip(&paths) {
        let entry = if path.is_file() {
            CacheEntry::read(path, with_tokens)?
        } else {
            let entry = pipeline.run(scan, with_tokens)?.entry;
            println!("  {} {}  {} tokens", green("encoded"), locate::scan_stem(scan), entry.info.tokens);
            entry
        };
        entries.push(entry);
    }
    Ok(entries)
}

/// `[n_scans, D]` pooled embeddings.
fn pooled_matrix(entries: &[CacheEntry]) -> Matrix<f32> {
    let rows: Vec<Vec<f32>> = entries.iter().map(|e| e.pooled.clone()).collect();
    Matrix::from_rows(&rows)
}

/// Each scan's in-mask patch tokens and the voxel labels of those patches.
fn segmentation_subjects(
    entries: Vec<CacheEntry>,
    column: &LabelColumn,
    table: &Path,
) -> Result<Vec<SubjectPatches>> {
    let label_paths = column.paths(table);
    entries
        .into_iter()
        .zip(&label_paths)
        .map(|(entry, path)| {
            let grid = &entry.info.grid;
            let labels = labels_on_grid(path, grid.shape, &Affine(grid.fit_affine))?;
            let labels = patch_rows(&labels, &entry.coords, grid.patch);
            let (rows, width) = (entry.coords.len(), entry.width());
            Ok(SubjectPatches { features: Matrix::new(rows, width, entry.tokens), labels })
        })
        .collect()
}

/// Options of `larmor probe apply`.
pub struct ApplyOptions {
    pub probe: PathBuf,
    pub inputs: Vec<PathBuf>,
    pub out: Option<PathBuf>,
    pub model: Option<String>,
    pub device: DeviceChoice,
    pub precision: Precision,
}

pub fn cmd_probe_apply(options: ApplyOptions) -> Result<()> {
    let probe =
        Probe::read(&options.probe).with_context(|| format!("reading {}", options.probe.display()))?;
    let meta = &probe.metadata;
    let model = Model::resolve(options.model.as_deref().unwrap_or(&meta.model_id))?;
    let source = EmbeddingSource {
        model_sha256: model.weights_sha256(),
        preprocessing_version: model.prep_version.clone(),
        embed_dim: model.width(),
    };
    probe.check_embeddings(&source).with_context(|| {
        format!(
            "refusing to apply {} with model {} ({})",
            options.probe.display(),
            model.id(),
            model.path.display()
        )
    })?;
    let scans = locate::find_scans(&options.inputs)?;
    if scans.is_empty() {
        bail!("no .nii or .nii.gz scans found");
    }
    let name = probe_name(&options.probe, &probe);
    println!(
        "{}  {} ({:?}) · {} scan{} · model {}",
        bold("larmor probe apply"),
        name,
        meta.kind,
        scans.len(),
        if scans.len() == 1 { "" } else { "s" },
        bold(model.id())
    );
    let mut pipeline = Pipeline::new(&model, options.device, options.precision, locate::cache_root()?);
    match meta.kind {
        ProbeKind::Regression | ProbeKind::Classification => {
            apply_scan_level(&probe, &name, &scans, &mut pipeline, options.out.as_deref())
        }
        ProbeKind::Segmentation => {
            apply_segmentation(&probe, &name, &scans, &mut pipeline, options.out.as_deref(), options.device)
        }
    }
}

/// The name used in output files: the probe file's stem, or its label column when the file is just
/// `probe.toml`. Letters and digits only, as BIDS `desc-` labels need.
fn probe_name(path: &Path, probe: &Probe) -> String {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let raw = if stem == "probe" || stem.is_empty() {
        probe.metadata.label_names.first().cloned().unwrap_or(stem)
    } else {
        stem
    };
    let clean: String = raw.chars().filter(char::is_ascii_alphanumeric).collect();
    if clean.is_empty() { "probe".into() } else { clean }
}

/// Regression values or class probabilities for every scan, printed and written to
/// `<out>/<name>_predictions.tsv`.
fn apply_scan_level(
    probe: &Probe,
    name: &str,
    scans: &[PathBuf],
    pipeline: &mut Pipeline,
    out: Option<&Path>,
) -> Result<()> {
    let mut pooled = Vec::with_capacity(scans.len());
    for scan in scans {
        pooled.push(pipeline.run(scan, false)?.entry.pooled);
    }
    let predictions = probe.predict_scans(&Matrix::from_rows(&pooled))?;
    // label_names = [column, negative class, positive class] for classification
    let classes = match (probe.metadata.kind, probe.metadata.label_names.as_slice()) {
        (ProbeKind::Classification, [_, negative, positive, ..]) => Some((negative, positive)),
        (ProbeKind::Classification, _) => bail!("the probe's label_names do not name its two classes"),
        _ => None,
    };
    let mut tsv = String::from(if classes.is_some() {
        "scan\tprediction\tprobability\n"
    } else {
        "scan\tprediction\n"
    });
    for (scan, &value) in scans.iter().zip(&predictions) {
        let shown = tsv_cell(&scan.display().to_string());
        match classes {
            Some((negative, positive)) => {
                // sklearn's predict: the positive class when its probability is above 0.5
                let class = if value > 0.5 { positive } else { negative };
                println!("  {}  {}  p = {value:.4}", locate::scan_stem(scan), bold(class));
                let _ = writeln!(tsv, "{shown}\t{}\t{value}", tsv_cell(class));
            }
            None => {
                println!("  {}  {}", locate::scan_stem(scan), bold(&format!("{value:.4}")));
                let _ = writeln!(tsv, "{shown}\t{value}");
            }
        }
    }
    let out_root = out.map(Path::to_path_buf).unwrap_or_else(|| locate::default_out_dir(&scans[0]));
    let path = out_root.join(format!("{name}_predictions.tsv"));
    outputs::write_atomic(&path, tsv.as_bytes())?;
    println!("{} wrote {}", green("✓"), path.display());
    Ok(())
}

/// A native-space mask and a grid probability map for every scan.
fn apply_segmentation(
    probe: &Probe,
    name: &str,
    scans: &[PathBuf],
    pipeline: &mut Pipeline,
    out: Option<&Path>,
    device: DeviceChoice,
) -> Result<()> {
    let device = device::segmentation_device(device)?;
    for scan in scans {
        let start = Instant::now();
        let ScanResult { entry, nifti, prepared, .. } = pipeline.run(scan, true)?;
        let grid = &entry.info.grid;
        let features = Matrix::new(entry.coords.len(), entry.width(), entry.tokens);
        let (probabilities, threshold) = probe.predict_patches(&features, &device)?;
        let patch_voxels = grid.patch.pow(3);
        if probabilities.len() != entry.coords.len() * patch_voxels {
            bail!("the probe does not predict the {patch_voxels} voxels of this model's patches");
        }
        let probability = assemble_patches(&probabilities, &entry.coords, grid.patch, grid.shape, 0.0f32);
        // probe.py::dice_by_threshold: a voxel is in the mask when p >= threshold
        let grid_mask = probability.mapv(|p| u8::from(p >= threshold));

        let (nifti, prepared) = match (nifti, prepared) {
            (Some(nifti), Some(prepared)) => (nifti, prepared),
            _ => prepare_again(pipeline.model(), scan)?,
        };
        if prepared.fit_affine.0 != grid.fit_affine {
            eprintln!("{} {}: the grid differs from the cached one", yellow("warning:"), scan.display());
        }
        let native = prepared.to_native(grid_mask.view())?;
        let mask_ml = native.iter().filter(|&&v| v > 0).count() as f64
            * prepared.native_affine.determinant3().abs()
            / 1000.0;
        let mut mask_image = nifti.with_data(VoxelData::U8(native.into_dyn()));
        // the scan's display window means nothing for a 0/1 mask
        mask_image.header.cal_min = 0.0;
        mask_image.header.cal_max = 0.0;

        let out_root = out.map(Path::to_path_buf).unwrap_or_else(|| locate::default_out_dir(scan));
        let outputs = ScanOutputs::new(&out_root, scan);
        outputs.create_dir()?;
        outputs::ensure_dataset_description(&out_root)?;
        mask_image.write(outputs.probe_mask(name))?;
        Nifti::new(VoxelData::F32(probability.into_dyn()), &prepared.fit_affine)
            .write(outputs.probe_probseg(name))?;
        println!(
            "  {}  {mask_ml:.1} ml at threshold {threshold:.4} · {} · {}",
            locate::scan_stem(scan),
            secs(start.elapsed().as_secs_f64()),
            dim(&outputs.probe_mask(name).display().to_string())
        );
    }
    Ok(())
}
