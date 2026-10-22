//! `larmor probe fit` and `larmor probe apply`.

use std::collections::HashMap;
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
use larmor_probe::segmentation::{SegmentationOptions, SolverStop, SubjectPatches};
use larmor_volume::{Affine, Nifti, VoxelData};

use crate::cache::CacheEntry;
use crate::console::{bold, dim, green, red, secs, with_ci, yellow};
use crate::device;
use crate::encode_cmd::{check_reference, prepare_again};
use crate::hashing::sha256_file;
use crate::labels::{LabelColumn, LabelRow, LabelTable};
use crate::locate;
use crate::model::Model;
use crate::outputs::{self, ScanOutputs, check_unique_outputs, tsv_cell};
use crate::pipeline::{Pipeline, ScanResult, reference_key};
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
    let mut pipeline = Pipeline::new(&model, options.device, options.precision, locate::cache_root()?);
    println!(
        "{}  {:?} probe on {} · {} scans · model {} ({}) · {}",
        bold("larmor probe fit"),
        options.kind,
        column.name,
        column.rows.len(),
        bold(model.id()),
        &model.encoder_key[..8],
        pipeline.precision()
    );

    let with_tokens = options.kind == Kind::Segmentation;
    let entries = cached_entries(&mut pipeline, &table, &column, options.encode, with_tokens)?;

    let fit_start = Instant::now();
    let fit = FitInput { column: &column, table: &table, model: &model, precision: pipeline.precision() };
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
    precision: Precision,
}

impl FitInput<'_> {
    /// Where the embeddings and labels came from, for `probe.toml`.
    fn provenance(&self, label_names: Vec<String>) -> Provenance {
        Provenance {
            model_id: self.model.id().to_string(),
            model_sha256: self.model.weights_sha256(),
            encoder_key: self.model.encoder_key.clone(),
            precision: self.precision.to_string(),
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
    let max_iter = cv.stops.iter().filter(|&&s| s == SolverStop::MaxIter).count();
    if max_iter > 0 {
        println!(
            "  {} {max_iter} of {} segmentation fits reached the iteration limit before the gradient tolerance",
            yellow("warning:"),
            cv.stops.len()
        );
    }
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

/// A label row's scan and reference, hashed.
struct RowKey {
    scan_sha256: String,
    reference_sha256: Option<String>,
}

impl RowKey {
    /// The reference as the cache key sees it (none when the scan is its own reference).
    fn reference(&self) -> Option<&str> {
        reference_key(&self.scan_sha256, self.reference_sha256.as_deref())
    }
}

/// The cache entries of the column's scans, encoding missing (or unreadable) ones with `--encode`,
/// else an error listing them.
fn cached_entries(
    pipeline: &mut Pipeline,
    table: &LabelTable,
    column: &LabelColumn,
    encode: bool,
    with_tokens: bool,
) -> Result<Vec<CacheEntry>> {
    let model = pipeline.model();
    let mut keys = Vec::with_capacity(column.rows.len());
    for row in &column.rows {
        let hash = |path: &Path| sha256_file(path).with_context(|| format!("reading {}", path.display()));
        let reference_sha256 = row.reference.as_deref().map(hash).transpose()?;
        keys.push(RowKey { scan_sha256: hash(&row.scan)?, reference_sha256 });
    }
    check_duplicates(&table.path, &column.rows, &keys, column.folds.as_deref())?;

    let mut entries: Vec<Option<CacheEntry>> = Vec::with_capacity(keys.len());
    // rows (by index) without a usable entry, with the problem of an unreadable one
    let mut missing: Vec<(usize, Option<String>)> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        match pipeline.stager.lookup(&key.scan_sha256, key.reference(), with_tokens) {
            Ok(Some(entry)) => entries.push(Some(entry)),
            Ok(None) => {
                missing.push((i, None));
                entries.push(None);
            }
            Err(err) => {
                missing.push((i, Some(format!("{err:#}"))));
                entries.push(None);
            }
        }
    }
    if !missing.is_empty() && !encode {
        let list: Vec<String> = missing
            .iter()
            .take(20)
            .map(|(i, problem)| {
                let scan = column.rows[*i].scan.display();
                match problem {
                    None => format!("  {scan}"),
                    Some(problem) => format!("  {scan} (unreadable cache entry: {problem})"),
                }
            })
            .collect();
        let more = if missing.len() > 20 {
            format!("\n  … and {} more", missing.len() - 20)
        } else {
            String::new()
        };
        let other_precisions = other_precision_hint(pipeline, &missing, &keys);
        bail!(
            "{} of {} scans are not encoded with {} in {} yet:\n{}{more}\nRun `larmor encode <scans or folder> \
             --model {}` first (with --force for unreadable entries), or pass --encode.{other_precisions}",
            missing.len(),
            column.rows.len(),
            model.id(),
            pipeline.precision(),
            list.join("\n"),
            model.id()
        );
    }
    let mut out = Vec::with_capacity(entries.len());
    for (row, entry) in column.rows.iter().zip(entries) {
        let entry = match entry {
            Some(entry) => entry,
            // re-encodes an unreadable entry too
            None => {
                let entry = pipeline.run(&row.scan, row.reference.as_deref(), with_tokens)?.entry;
                println!(
                    "  {} {}  {} tokens",
                    green("encoded"),
                    locate::scan_stem(&row.scan),
                    entry.info.tokens
                );
                entry
            }
        };
        out.push(entry);
    }
    Ok(out)
}

/// A hint when the missing scans are cached in another precision.
fn other_precision_hint(pipeline: &Pipeline, missing: &[(usize, Option<String>)], keys: &[RowKey]) -> String {
    for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
        if precision == pipeline.precision() {
            continue;
        }
        let stager = pipeline.stager.with_precision(precision);
        let found = missing
            .iter()
            .filter(|(i, _)| stager.entry_path(&keys[*i].scan_sha256, keys[*i].reference()).is_file())
            .count();
        if found > 0 {
            return format!(
                "\n{found} of them are cached in {precision}; pass --precision {precision} to fit on those."
            );
        }
    }
    String::new()
}

/// Refuses a table that names the same scan (the same file, or files with the same bytes) on more
/// than one row: without a fold column, cross-validation could put copies on both sides of a split
/// and report leaked, inflated metrics. Copies that a `fold` column keeps in one fold are allowed.
fn check_duplicates(
    table: &Path,
    rows: &[&LabelRow],
    keys: &[RowKey],
    folds: Option<&[usize]>,
) -> Result<()> {
    let mut groups: HashMap<(&str, Option<&str>), Vec<usize>> = HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        groups.entry((key.scan_sha256.as_str(), key.reference())).or_default().push(i);
    }
    let mut duplicates: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    duplicates.sort();
    let mut problems = Vec::new();
    let mut allowed = 0;
    for group in &duplicates {
        if let Some(folds) = folds
            && group.iter().all(|&i| folds[i] == folds[group[0]])
        {
            allowed += group.len() - 1;
            continue;
        }
        let lines: Vec<String> = group.iter().map(|&i| rows[i].line.to_string()).collect();
        let canonical = |i: usize| std::fs::canonicalize(&rows[i].scan).ok();
        let same_file = group.iter().all(|&i| canonical(i) == canonical(group[0]));
        let what = if same_file {
            format!("the same scan ({})", rows[group[0]].scan.display())
        } else {
            let names: Vec<String> = group.iter().map(|&i| rows[i].scan.display().to_string()).collect();
            format!("files with identical bytes ({})", names.join(", "))
        };
        problems.push(format!("  lines {} name {what}", lines.join(", ")));
    }
    if allowed > 0 {
        println!(
            "{} {allowed} repeated scans are kept: the fold column puts every copy in one fold",
            dim("note:")
        );
    }
    if !problems.is_empty() {
        bail!(
            "{} names a scan more than once:\n{}\nCopies of a scan in different cross-validation folds leak between \
             training and test and inflate the metrics. Remove the repeats, or keep each scan's copies in one fold \
             with a `fold` column.",
            table.display(),
            problems.join("\n")
        );
    }
    Ok(())
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
    let (mut on_grid, mut in_patches) = (0usize, 0usize);
    let subjects = entries
        .into_iter()
        .zip(&label_paths)
        .map(|(entry, path)| {
            let grid = &entry.info.grid;
            let labels = labels_on_grid(path, grid.shape, &Affine(grid.fit_affine))?;
            on_grid += labels.iter().filter(|&&v| v > 0).count();
            let labels = patch_rows(&labels, &entry.coords, grid.patch);
            in_patches += labels.data().iter().filter(|&&v| v > 0).count();
            let (rows, width) = (entry.coords.len(), entry.width());
            Ok(SubjectPatches { features: Matrix::new(rows, width, entry.tokens), labels })
        })
        .collect::<Result<Vec<_>>>()?;
    if on_grid > in_patches {
        println!(
            "{} {} of {on_grid} labelled voxels on the model grid lie outside the in-mask patches and are not \
             used (as in the eval framework)",
            dim("note:"),
            on_grid - in_patches
        );
    }
    Ok(subjects)
}

/// Options of `larmor probe apply`.
pub struct ApplyOptions {
    pub probe: PathBuf,
    pub inputs: Vec<PathBuf>,
    pub out: Option<PathBuf>,
    pub model: Option<String>,
    /// Prepare every scan on this scan's head mask and grid.
    pub reference: Option<PathBuf>,
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
        encoder_key: model.encoder_key.clone(),
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
    if let Some(reference) = &options.reference {
        check_reference(reference)?;
    }
    let found = locate::find_scans(&options.inputs, options.out.as_slice())?;
    let scans = found.scans;
    if scans.is_empty() {
        bail!("no .nii or .nii.gz scans found");
    }
    let name = probe_name(&options.probe, &probe);
    let mut pipeline = Pipeline::new(&model, options.device, options.precision, locate::cache_root()?);
    println!(
        "{}  {} ({:?}) · {} scan{} · model {} · {}",
        bold("larmor probe apply"),
        name,
        meta.kind,
        scans.len(),
        if scans.len() == 1 { "" } else { "s" },
        bold(model.id()),
        pipeline.precision()
    );
    if let Some(trained) = &meta.precision
        && *trained != pipeline.precision().to_string()
    {
        eprintln!(
            "{} the probe was fit on {trained} embeddings; these are {} (pass --precision {trained} to match)",
            yellow("warning:"),
            pipeline.precision()
        );
    }
    let scans = Scans { paths: &scans, root: found.root.as_deref(), reference: options.reference.as_deref() };
    let failed = match meta.kind {
        ProbeKind::Regression | ProbeKind::Classification => {
            apply_scan_level(&probe, &name, &scans, &mut pipeline, options.out.as_deref())?
        }
        ProbeKind::Segmentation => {
            apply_segmentation(&probe, &name, &scans, &mut pipeline, options.out.as_deref(), options.device)?
        }
    };
    if failed > 0 {
        bail!("{failed} of {} scans failed", scans.paths.len());
    }
    Ok(())
}

/// The scans `probe apply` predicts on.
struct Scans<'a> {
    paths: &'a [PathBuf],
    /// The inputs' common folder, for output paths.
    root: Option<&'a Path>,
    reference: Option<&'a Path>,
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
/// `<out>/<name>_predictions.tsv`. A scan that fails is reported and left out; returns how many
/// failed.
fn apply_scan_level(
    probe: &Probe,
    name: &str,
    scans: &Scans,
    pipeline: &mut Pipeline,
    out: Option<&Path>,
) -> Result<usize> {
    // label_names = [column, negative class, positive class] for classification
    let classes = match (probe.metadata.kind, probe.metadata.label_names.as_slice()) {
        (ProbeKind::Classification, [_, negative, positive, ..]) => Some((negative, positive)),
        (ProbeKind::Classification, _) => bail!("the probe's label_names do not name its two classes"),
        _ => None,
    };
    let mut done: Vec<&PathBuf> = Vec::with_capacity(scans.paths.len());
    let mut pooled = Vec::with_capacity(scans.paths.len());
    let mut failed = 0;
    for scan in scans.paths {
        match pipeline.run(scan, scans.reference, false) {
            Ok(result) => {
                done.push(scan);
                pooled.push(result.entry.pooled);
            }
            Err(err) => {
                println!("  {} {}  {err:#}", red("failed "), locate::scan_stem(scan));
                failed += 1;
            }
        }
    }
    if done.is_empty() {
        return Ok(failed);
    }
    let predictions = probe.predict_scans(&Matrix::from_rows(&pooled))?;
    let mut tsv = String::from(if classes.is_some() {
        "scan\tprediction\tprobability\n"
    } else {
        "scan\tprediction\n"
    });
    for (scan, &value) in done.iter().zip(&predictions) {
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
    let out_root = out.map(Path::to_path_buf).unwrap_or_else(|| locate::default_out_dir(&scans.paths[0]));
    let path = out_root.join(format!("{name}_predictions.tsv"));
    outputs::write_atomic(&path, tsv.as_bytes())?;
    println!("{} wrote {}", green("✓"), path.display());
    Ok(failed)
}

/// A native-space mask and a grid probability map for every scan. Two scans that would write the
/// same files stop the run before anything is written; a scan that fails is reported and the others
/// go on. Returns how many failed.
fn apply_segmentation(
    probe: &Probe,
    name: &str,
    scans: &Scans,
    pipeline: &mut Pipeline,
    out: Option<&Path>,
    device: DeviceChoice,
) -> Result<usize> {
    let device = device::segmentation_device(device)?;
    let destinations: Vec<(PathBuf, ScanOutputs)> = scans
        .paths
        .iter()
        .map(|scan| {
            let out_root = out.map(Path::to_path_buf).unwrap_or_else(|| locate::default_out_dir(scan));
            let outputs = ScanOutputs::new(&out_root, scan, scans.root);
            (out_root, outputs)
        })
        .collect();
    let targets: Vec<(&Path, PathBuf)> = scans
        .paths
        .iter()
        .zip(&destinations)
        .map(|(scan, (_, o))| (scan.as_path(), o.probe_mask(name)))
        .collect();
    check_unique_outputs(&targets)?;

    let mut failed = 0;
    for (scan, (out_root, outputs)) in scans.paths.iter().zip(&destinations) {
        let start = Instant::now();
        let result = segment_one(probe, name, scan, scans.reference, pipeline, &device, out_root, outputs);
        match result {
            Ok((mask_ml, threshold)) => println!(
                "  {}  {mask_ml:.1} ml at threshold {threshold:.4} · {} · {}",
                locate::scan_stem(scan),
                secs(start.elapsed().as_secs_f64()),
                dim(&outputs.probe_mask(name).display().to_string())
            ),
            Err(err) => {
                println!("  {} {}  {err:#}", red("failed "), locate::scan_stem(scan));
                failed += 1;
            }
        }
    }
    Ok(failed)
}

/// One scan's segmentation: the mask volume (ml) and the threshold used.
#[allow(clippy::too_many_arguments)]
fn segment_one(
    probe: &Probe,
    name: &str,
    scan: &Path,
    reference: Option<&Path>,
    pipeline: &mut Pipeline,
    device: &larmor_probe::Device,
    out_root: &Path,
    outputs: &ScanOutputs,
) -> Result<(f64, f32)> {
    let ScanResult { entry, nifti, prepared, .. } = pipeline.run(scan, reference, true)?;
    let grid = &entry.info.grid;
    let features = Matrix::new(entry.coords.len(), entry.width(), entry.tokens);
    let (probabilities, threshold) = probe.predict_patches(&features, device)?;
    let patch_voxels = grid.patch.pow(3);
    if probabilities.len() != entry.coords.len() * patch_voxels {
        bail!("the probe does not predict the {patch_voxels} voxels of this model's patches");
    }
    let probability = assemble_patches(&probabilities, &entry.coords, grid.patch, grid.shape, 0.0f32);
    // probe.py::dice_by_threshold: a voxel is in the mask when p >= threshold
    let grid_mask = probability.mapv(|p| u8::from(p >= threshold));

    let (nifti, prepared) = match (nifti, prepared) {
        (Some(nifti), Some(prepared)) => (nifti, prepared),
        _ => prepare_again(pipeline.model(), scan, reference)?,
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

    outputs.create_dir()?;
    outputs::ensure_dataset_description(out_root)?;
    mask_image.write(outputs.probe_mask(name))?;
    Nifti::new(VoxelData::F32(probability.into_dyn()), &prepared.fit_affine)
        .write(outputs.probe_probseg(name))?;
    Ok((mask_ml, threshold))
}
