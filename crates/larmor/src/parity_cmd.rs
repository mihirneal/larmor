//! `larmor check-parity`: every preprocessing stage and every embedding against the PyTorch
//! reference fixtures (`docs/DESIGN.md`, "Reference fixtures" and "Gates").
//!
//! The comparisons are those of `larmor-prep`'s `tests/fixtures.rs` and `larmor-model`'s parity
//! module, run on every scan in `<fixtures>/manifest.toml`.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use larmor_model::Encoder;
use larmor_model::parity::{EncodeReference, cosine, row_cosines};
use larmor_prep::{Prepared, Recipe};
use larmor_volume::ndarray::{Array3, Ix3};
use larmor_volume::{Nifti, VoxelData};
use serde::Deserialize;
use serde_json::Value;

use crate::console::{bold, dim, green, red, secs};
use crate::device::{DeviceArg, PrecisionArg};
use crate::hashing::sha256_hex;
use crate::model::Model;
use crate::pipeline::load_encoder;

/// Largest share of voxels on which a mask may disagree with the reference (0.01%).
const MAX_MASK_DISAGREE: f64 = 1e-4;
/// Smallest share of in-mask voxels whose quantized value must be exact (99.99%).
const MIN_QUANT_EXACT: f64 = 0.9999;
/// Largest conform error, relative to the value range of the reference conform image.
const MAX_CONFORM_REL: f64 = 1e-5;
/// Smallest token and pooled cosine against the fp32 reference.
const MIN_COSINE: f64 = 0.999;

/// Options of `larmor check-parity`.
pub struct ParityOptions {
    pub fixtures: PathBuf,
    /// Only these model ids; all with an encode reference when `None`.
    pub models: Option<Vec<String>>,
    pub device: DeviceArg,
    pub precision: PrecisionArg,
}

/// `<fixtures>/manifest.toml` (only the fields used here).
#[derive(Deserialize)]
struct FixtureManifest {
    scan: Vec<FixtureScan>,
}

#[derive(Deserialize)]
struct FixtureScan {
    name: String,
    /// Relative to the fixtures folder.
    file: String,
    sha256: String,
}

/// How one scan's preprocessing compares.
#[derive(Default)]
struct PrepReport {
    head_disagree: usize,
    head_voxels: usize,
    fit_disagree: usize,
    fit_voxels: usize,
    conform_rel: f64,
    conform_exact: f64,
    quant_exact: f64,
    quant_max_diff: i32,
    /// vmin, vmax, centroid, fit affine and fwhm all equal to the reference.
    scalars_exact: bool,
}

/// How one model's embedding of one scan compares.
struct EncodeReport {
    model: String,
    tokens: usize,
    reference_tokens: usize,
    coords_equal: bool,
    min_cosine: f64,
    mean_cosine: f64,
    min_centered: f64,
    pooled_cosine: f64,
    max_abs_diff: f64,
    /// PyTorch's own bf16-vs-fp32 cosines on this scan, from the reference metadata.
    bf16_min_cosine: Option<f64>,
    bf16_pooled_cosine: Option<f64>,
    seconds: f64,
}

#[derive(Default)]
struct ScanReport {
    name: String,
    prep: Option<PrepReport>,
    encodes: Vec<EncodeReport>,
    failures: Vec<String>,
    seconds: f64,
}

impl ScanReport {
    fn gate(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            self.failures.push(what());
        }
    }
}

/// A reference model and, once loaded, its encoder.
struct ReferenceModel {
    model: Model,
    encoder: Option<Encoder>,
}

pub fn cmd_check_parity(options: ParityOptions) -> Result<()> {
    let start = Instant::now();
    let root = &options.fixtures;
    let manifest_path = root.join("manifest.toml");
    let text = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let manifest: FixtureManifest =
        toml::from_str(&text).with_context(|| format!("parsing {}", manifest_path.display()))?;
    ensure!(!manifest.scan.is_empty(), "{} lists no scans", manifest_path.display());
    let mut models = reference_models(root, options.models.as_deref())?;
    // make_reference.py's prep references always use the nanobrain-eval recipe
    let prep_recipe = Recipe::nanobrain_eval();

    println!(
        "{}  {} scans · models {} · device {} · {}",
        bold("larmor check-parity"),
        manifest.scan.len(),
        if models.is_empty() {
            "none".to_string()
        } else {
            models.keys().cloned().collect::<Vec<_>>().join(", ")
        },
        options.device.resolve(),
        options.precision
    );
    print_header();
    let mut reports = Vec::new();
    for scan in &manifest.scan {
        let report = check_scan(root, scan, &prep_recipe, &mut models, &options);
        print_rows(&report);
        reports.push(report);
    }
    let failed = print_summary(&reports, &models, start.elapsed().as_secs_f64());
    if failed > 0 {
        bail!("{failed} of {} scans failed a parity gate", reports.len());
    }
    Ok(())
}

/// Every model with a folder in `reference/encode/`, or the ones `--models` names.
fn reference_models(root: &Path, only: Option<&[String]>) -> Result<BTreeMap<String, ReferenceModel>> {
    let encode_dir = root.join("reference/encode");
    let mut ids: Vec<String> = match std::fs::read_dir(&encode_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(_) => Vec::new(),
    };
    ids.sort();
    if let Some(only) = only {
        for id in only {
            ensure!(ids.contains(id), "no encode references for {id} in {}", encode_dir.display());
        }
        ids.retain(|id| only.contains(id));
    }
    let mut models = BTreeMap::new();
    for id in ids {
        let path = root.join("models").join(format!("{id}.toml"));
        let model = Model::load(&path).with_context(|| format!("reference model {id}"))?;
        models.insert(id, ReferenceModel { model, encoder: None });
    }
    Ok(models)
}

fn check_scan(
    root: &Path,
    scan: &FixtureScan,
    prep_recipe: &Recipe,
    models: &mut BTreeMap<String, ReferenceModel>,
    options: &ParityOptions,
) -> ScanReport {
    let start = Instant::now();
    let mut report = ScanReport { name: scan.name.clone(), ..ScanReport::default() };
    if let Err(err) = check_scan_inner(root, scan, prep_recipe, models, options, &mut report) {
        report.failures.push(format!("{err:#}"));
    }
    report.seconds = start.elapsed().as_secs_f64();
    report
}

fn check_scan_inner(
    root: &Path,
    scan: &FixtureScan,
    prep_recipe: &Recipe,
    models: &mut BTreeMap<String, ReferenceModel>,
    options: &ParityOptions,
    report: &mut ScanReport,
) -> Result<()> {
    let path = root.join(&scan.file);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let sha = sha256_hex(&bytes);
    report.gate(sha == scan.sha256, || format!("scan sha256 {sha} vs manifest {}", scan.sha256));
    let nifti = Nifti::from_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))?;

    let dir = root.join("reference/prep").join(&scan.name);
    let (prepared, references) = rayon::join(|| prep_recipe.run(&nifti), || PrepReferences::read(&dir));
    let prepared = prepared.context("preprocessing")?;
    let references = references?;
    report.gate(references.info["scan_sha256"].as_str() == Some(sha.as_str()), || {
        "prep reference was made from another file".into()
    });
    report.prep = Some(compare_prep(&references, &prepared, report));

    for (id, reference_model) in models.iter_mut() {
        let reference_path =
            root.join("reference/encode").join(id).join(format!("{}.safetensors", scan.name));
        if !reference_path.is_file() {
            continue;
        }
        let encode =
            compare_encode(reference_model, &reference_path, &nifti, &prepared, prep_recipe, options)
                .with_context(|| id.clone())?;
        report.gate(encode.coords_equal, || {
            format!(
                "{id}: {} tokens vs reference {} (or different coords)",
                encode.tokens, encode.reference_tokens
            )
        });
        // tokens on different grids cannot be compared one to one
        if encode.coords_equal {
            report.gate(encode.min_cosine >= MIN_COSINE, || {
                format!("{id}: min token cosine {:.6}", encode.min_cosine)
            });
            report.gate(encode.pooled_cosine >= MIN_COSINE, || {
                format!("{id}: pooled cosine {:.6}", encode.pooled_cosine)
            });
        }
        report.encodes.push(encode);
    }
    Ok(())
}

/// The files in `reference/prep/<scan>/`.
struct PrepReferences {
    head_mask: Array3<u8>,
    fit_mask: Array3<u8>,
    conform: Array3<f32>,
    quantized: Array3<u16>,
    info: Value,
}

impl PrepReferences {
    /// Reads the four images in parallel (each gunzip is single-threaded).
    fn read(dir: &Path) -> Result<PrepReferences> {
        let info_path = dir.join("info.json");
        let info_text = std::fs::read_to_string(&info_path)
            .with_context(|| format!("no prep reference at {}", info_path.display()))?;
        let info: Value = serde_json::from_str(&info_text)?;
        let ((head, fit), (conform, quantized)) = rayon::join(
            || {
                rayon::join(
                    || read_u8(&dir.join("head_mask.nii.gz")),
                    || read_u8(&dir.join("fit_mask.nii.gz")),
                )
            },
            || {
                rayon::join(
                    || read_f32(&dir.join("conform.nii.gz")),
                    || read_u16(&dir.join("quantized.nii.gz")),
                )
            },
        );
        Ok(PrepReferences {
            head_mask: head?,
            fit_mask: fit?,
            conform: conform?,
            quantized: quantized?,
            info,
        })
    }
}

fn read_nifti(path: &Path) -> Result<Nifti> {
    Nifti::read(path).with_context(|| format!("reading {}", path.display()))
}

fn read_u8(path: &Path) -> Result<Array3<u8>> {
    match read_nifti(path)?.data {
        VoxelData::U8(a) => Ok(a.into_dimensionality::<Ix3>()?),
        other => bail!("{}: expected uint8, got {}", path.display(), other.dtype().name()),
    }
}

fn read_u16(path: &Path) -> Result<Array3<u16>> {
    match read_nifti(path)?.data {
        VoxelData::U16(a) => Ok(a.into_dimensionality::<Ix3>()?),
        other => bail!("{}: expected uint16, got {}", path.display(), other.dtype().name()),
    }
}

fn read_f32(path: &Path) -> Result<Array3<f32>> {
    Ok(read_nifti(path)?.fdata_f32().into_dimensionality::<Ix3>()?)
}

fn mask_disagreements(ours: &Array3<bool>, reference: &Array3<u8>) -> usize {
    ours.iter().zip(reference).filter(|&(&a, &b)| a != (b > 0)).count()
}

fn compare_prep(refs: &PrepReferences, prepared: &Prepared, report: &mut ScanReport) -> PrepReport {
    let mut prep = PrepReport::default();
    let info = &refs.info;

    prep.head_voxels = refs.head_mask.len();
    if prepared.native_mask.shape() == refs.head_mask.shape() {
        prep.head_disagree = mask_disagreements(&prepared.native_mask, &refs.head_mask);
    } else {
        prep.head_disagree = prep.head_voxels;
        report.failures.push(format!(
            "head mask shape {:?} vs {:?}",
            prepared.native_mask.shape(),
            refs.head_mask.shape()
        ));
    }
    let share = prep.head_disagree as f64 / prep.head_voxels.max(1) as f64;
    report
        .gate(share <= MAX_MASK_DISAGREE, || format!("head mask: {:.4}% of voxels disagree", 100.0 * share));

    prep.fit_voxels = refs.fit_mask.len();
    let fit_in_mask = refs.fit_mask.iter().filter(|&&v| v > 0).count();
    if prepared.mask.shape() == refs.fit_mask.shape() {
        prep.fit_disagree = mask_disagreements(&prepared.mask, &refs.fit_mask);
    } else {
        prep.fit_disagree = prep.fit_voxels;
        report.failures.push(format!(
            "fit mask shape {:?} vs {:?}",
            prepared.mask.shape(),
            refs.fit_mask.shape()
        ));
    }
    let share = prep.fit_disagree as f64 / prep.fit_voxels.max(1) as f64;
    report.gate(share <= MAX_MASK_DISAGREE, || format!("fit mask: {:.4}% of voxels disagree", 100.0 * share));

    if prepared.conformed.shape() == refs.conform.shape() {
        let (lo, hi) =
            refs.conform.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        let max_diff =
            prepared.conformed.iter().zip(&refs.conform).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        prep.conform_rel = f64::from(max_diff) / f64::from(hi - lo).max(f64::MIN_POSITIVE);
        let same =
            prepared.conformed.iter().zip(&refs.conform).filter(|(a, b)| a.to_bits() == b.to_bits()).count();
        prep.conform_exact = same as f64 / refs.conform.len() as f64;
    } else {
        prep.conform_rel = f64::INFINITY;
    }
    let rel = prep.conform_rel;
    report.gate(rel <= MAX_CONFORM_REL, || format!("conform relative error {rel:.2e}"));

    // outside the mask both sides are 0 when the masks agree, so the share is over the reference's
    // in-mask voxels, the stricter denominator
    let mut mismatches = refs.quantized.len();
    if prepared.quantized.shape() == refs.quantized.shape() {
        mismatches = 0;
        for (&a, &b) in prepared.quantized.iter().zip(&refs.quantized) {
            let diff = (i32::from(a) - i32::from(b)).abs();
            mismatches += usize::from(diff != 0);
            prep.quant_max_diff = prep.quant_max_diff.max(diff);
        }
    }
    prep.quant_exact = 1.0 - mismatches as f64 / fit_in_mask.max(1) as f64;
    let exact = prep.quant_exact;
    report.gate(exact >= MIN_QUANT_EXACT, || {
        format!("quantized exact on {:.4}% of in-mask voxels", 100.0 * exact)
    });

    let head_count = refs.head_mask.iter().filter(|&&v| v > 0).count() as u64;
    report
        .gate(info["head_mask_voxels"].as_u64() == Some(head_count), || "info.json head_mask_voxels".into());
    report.gate(info["fit_mask_voxels"].as_u64() == Some(fit_in_mask as u64), || {
        "info.json fit_mask_voxels".into()
    });

    let before = report.failures.len();
    let (vmin, vmax) = (info["vmin"].as_f64(), info["vmax"].as_f64());
    report.gate(vmin == Some(prepared.vmin), || format!("vmin {} vs reference {vmin:?}", prepared.vmin));
    report.gate(vmax == Some(prepared.vmax), || format!("vmax {} vs reference {vmax:?}", prepared.vmax));
    let centroid: Option<[i64; 3]> = serde_json::from_value(info["centroid"].clone()).ok();
    report.gate(centroid == Some(prepared.centroid), || {
        format!("centroid {:?} vs {centroid:?}", prepared.centroid)
    });
    let fit_affine: Option<[[f64; 4]; 4]> = serde_json::from_value(info["fit_affine"].clone()).ok();
    report.gate(fit_affine == Some(prepared.fit_affine.0), || {
        format!("fit affine {:?} vs {fit_affine:?}", prepared.fit_affine.0)
    });
    // info.json has null when conform_image did not smooth; Larmor reports zeros then
    let fwhm: [f64; 3] = serde_json::from_value(info["fwhm"].clone()).unwrap_or([0.0; 3]);
    report.gate(fwhm == prepared.fwhm, || format!("fwhm {:?} vs {fwhm:?}", prepared.fwhm));
    prep.scalars_exact = report.failures.len() == before;
    prep
}

fn compare_encode(
    reference_model: &mut ReferenceModel,
    path: &Path,
    nifti: &Nifti,
    prepared: &Prepared,
    prep_recipe: &Recipe,
    options: &ParityOptions,
) -> Result<EncodeReport> {
    let model = &reference_model.model;
    let patch = model.patch();
    let own_prepared;
    let prepared = if model.recipe == *prep_recipe {
        prepared
    } else {
        own_prepared = model.recipe.run(nifti)?;
        &own_prepared
    };
    let tokens = prepared.tokens(patch)?;
    if reference_model.encoder.is_none() {
        reference_model.encoder = Some(load_encoder(model, options.device, options.precision)?.0);
    }
    let encoder = reference_model.encoder.as_ref().expect("loaded above");
    let reference = EncodeReference::load(path)?;
    let metadata = safetensors_metadata(path)?;
    let floor = |key: &str| metadata.get(key).and_then(|v| v.parse::<f64>().ok());
    let mut report = EncodeReport {
        model: model.id().to_string(),
        tokens: tokens.len(),
        reference_tokens: reference.n_tokens(),
        coords_equal: tokens.coords == reference.coords,
        min_cosine: f64::NAN,
        mean_cosine: f64::NAN,
        min_centered: f64::NAN,
        pooled_cosine: f64::NAN,
        max_abs_diff: f64::NAN,
        bf16_min_cosine: floor("bf16_min_token_cosine"),
        bf16_pooled_cosine: floor("bf16_pooled_cosine"),
        seconds: 0.0,
    };
    if !report.coords_equal {
        return Ok(report);
    }
    let start = Instant::now();
    let encoded = encoder.encode(&tokens.patches, &tokens.coords)?;
    report.seconds = start.elapsed().as_secs_f64();
    let width = encoded.width;
    ensure!(width == reference.width, "width {width} vs reference {}", reference.width);
    let cosines = row_cosines(&encoded.tokens, &reference.tokens, width);
    report.min_cosine = cosines.iter().copied().fold(f64::INFINITY, f64::min);
    report.mean_cosine = cosines.iter().sum::<f64>() / cosines.len().max(1) as f64;
    let centered =
        row_cosines(&centered_rows(&encoded.tokens, width), &centered_rows(&reference.tokens, width), width);
    report.min_centered = centered.iter().copied().fold(f64::INFINITY, f64::min);
    report.pooled_cosine = cosine(&encoded.pooled, &reference.pooled);
    report.max_abs_diff = encoded
        .tokens
        .iter()
        .zip(&reference.tokens)
        .fold(0f64, |m, (&a, &b)| m.max((f64::from(a) - f64::from(b)).abs()));
    Ok(report)
}

/// Rows minus their column mean, so a model that outputs near-constant tokens cannot score a high
/// cosine.
fn centered_rows(values: &[f32], width: usize) -> Vec<f32> {
    let rows = (values.len() / width).max(1) as f64;
    let mut mean = vec![0f64; width];
    for row in values.chunks_exact(width) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += f64::from(v) / rows;
        }
    }
    values
        .chunks_exact(width)
        .flat_map(|row| row.iter().zip(&mean).map(|(&v, &m)| (f64::from(v) - m) as f32))
        .collect()
}

/// The `__metadata__` of a safetensors file, read from its header alone.
fn safetensors_metadata(path: &Path) -> Result<HashMap<String, String>> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    ensure!(len < 1 << 27, "{}: implausible safetensors header", path.display());
    let mut header = vec![0u8; len as usize];
    file.read_exact(&mut header)?;
    let header: Value = serde_json::from_slice(&header)?;
    Ok(serde_json::from_value(header["__metadata__"].clone()).unwrap_or_default())
}

fn print_header() {
    println!(
        "{}",
        dim(&format!(
            "{:<26} {:>8} {:>8} {:>9} {:>9}  {:<24} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}  {}",
            "scan",
            "head Δ",
            "fit Δ",
            "conform",
            "quant",
            "model",
            "tokens",
            "1-cos min",
            "1-cos avg",
            "1-ctr min",
            "1-pooled",
            "max |Δ|",
            "time",
            "result"
        ))
    );
}

fn print_rows(report: &ScanReport) {
    let prep = report.prep.as_ref().map_or_else(
        || format!("{:>8} {:>8} {:>9} {:>9}", "-", "-", "-", "-"),
        |p| {
            format!(
                "{:>8} {:>8} {:>9.1e} {:>8.4}%",
                p.head_disagree,
                p.fit_disagree,
                p.conform_rel,
                100.0 * p.quant_exact
            )
        },
    );
    let result = if report.failures.is_empty() { green("pass") } else { red("FAIL") };
    let blank = " ".repeat(26 + 1 + 8 + 1 + 8 + 1 + 9 + 1 + 9);
    if report.encodes.is_empty() {
        println!(
            "{:<26} {prep}  {:<24} {:>6} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}  {result}",
            report.name,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            secs(report.seconds)
        );
    }
    for (i, e) in report.encodes.iter().enumerate() {
        let lead = if i == 0 { format!("{:<26} {prep}", report.name) } else { blank.clone() };
        let tokens = if e.coords_equal {
            e.tokens.to_string()
        } else {
            format!("{}≠{}", e.tokens, e.reference_tokens)
        };
        let time = if i == 0 { secs(report.seconds) } else { String::new() };
        let shown = if i + 1 == report.encodes.len() { result.clone() } else { String::new() };
        println!(
            "{lead}  {:<24} {:>6} {:>9.1e} {:>9.1e} {:>9.1e} {:>9.1e} {:>8.1e} {:>8}  {shown}",
            e.model,
            tokens,
            1.0 - e.min_cosine,
            1.0 - e.mean_cosine,
            1.0 - e.min_centered,
            1.0 - e.pooled_cosine,
            e.max_abs_diff,
            time
        );
    }
    for failure in &report.failures {
        println!("    {} {failure}", red("✗"));
    }
}

/// Prints the gates with their worst values; returns the number of failing scans.
fn print_summary(reports: &[ScanReport], models: &BTreeMap<String, ReferenceModel>, seconds: f64) -> usize {
    let preps: Vec<&PrepReport> = reports.iter().filter_map(|r| r.prep.as_ref()).collect();
    let worst = |f: &dyn Fn(&PrepReport) -> f64, max: bool| {
        let values = preps.iter().map(|p| f(p));
        if max { values.fold(f64::NEG_INFINITY, f64::max) } else { values.fold(f64::INFINITY, f64::min) }
    };
    let head = worst(&|p| p.head_disagree as f64 / p.head_voxels.max(1) as f64, true);
    let fit = worst(&|p| p.fit_disagree as f64 / p.fit_voxels.max(1) as f64, true);
    let conform = worst(&|p| p.conform_rel, true);
    let conform_exact = worst(&|p| p.conform_exact, false);
    let quant = worst(&|p| p.quant_exact, false);
    let quant_diff = preps.iter().map(|p| p.quant_max_diff).max().unwrap_or(0);
    let scalars = preps.iter().filter(|p| p.scalars_exact).count();

    println!("\n{}", bold("Gates (worst scan)"));
    let row = |name: &str, value: String, limit: &str, ok: bool| {
        println!("  {name:<44} {value:>28}   {limit:<10} {}", if ok { green("pass") } else { red("FAIL") });
    };
    let have_prep = preps.len() == reports.len() && !preps.is_empty();
    row(
        "head mask disagreement",
        format!("{:.4}%", 100.0 * head),
        "≤ 0.01%",
        have_prep && head <= MAX_MASK_DISAGREE,
    );
    row(
        "fit mask disagreement",
        format!("{:.4}%", 100.0 * fit),
        "≤ 0.01%",
        have_prep && fit <= MAX_MASK_DISAGREE,
    );
    row(
        "conform relative error",
        format!("{conform:.2e} ({:.2}% exact)", 100.0 * conform_exact),
        "≤ 1e-5",
        have_prep && conform <= MAX_CONFORM_REL,
    );
    row(
        "quantized exact (in-mask voxels)",
        format!("{:.4}% (max Δ {quant_diff})", 100.0 * quant),
        "≥ 99.99%",
        have_prep && quant >= MIN_QUANT_EXACT,
    );
    row(
        "vmin, vmax, centroid, fit affine, fwhm exact",
        format!("{scalars}/{} scans", reports.len()),
        "all",
        scalars == reports.len(),
    );
    for id in models.keys() {
        let encodes: Vec<&EncodeReport> =
            reports.iter().flat_map(|r| &r.encodes).filter(|e| e.model == *id).collect();
        if encodes.is_empty() {
            continue;
        }
        let aligned = encodes.iter().filter(|e| e.coords_equal).count();
        let min_token = encodes.iter().map(|e| e.min_cosine).fold(f64::INFINITY, nan_min);
        let pooled = encodes.iter().map(|e| e.pooled_cosine).fold(f64::INFINITY, nan_min);
        let mean_time = encodes.iter().map(|e| e.seconds).sum::<f64>() / encodes.len() as f64;
        row(
            &format!("{id}: token grid equal"),
            format!("{aligned}/{} scans", encodes.len()),
            "all",
            aligned == encodes.len(),
        );
        row(&format!("{id}: min token cosine"), cosine_text(min_token), "≥ 0.999", min_token >= MIN_COSINE);
        row(&format!("{id}: pooled cosine"), cosine_text(pooled), "≥ 0.999", pooled >= MIN_COSINE);
        let floor_token = encodes.iter().filter_map(|e| e.bf16_min_cosine).fold(f64::INFINITY, f64::min);
        let floor_pooled = encodes.iter().filter_map(|e| e.bf16_pooled_cosine).fold(f64::INFINITY, f64::min);
        if floor_token.is_finite() {
            println!(
                "  {}",
                dim(&format!(
                    "{id}: PyTorch's own bf16 vs fp32 (noise floor): min token cosine {}, pooled {}; encode {} per scan",
                    cosine_text(floor_token),
                    cosine_text(floor_pooled),
                    secs(mean_time)
                ))
            );
        }
    }
    let failed = reports.iter().filter(|r| !r.failures.is_empty()).count();
    let encodes: usize = reports.iter().map(|r| r.encodes.len()).sum();
    if failed == 0 {
        println!(
            "{} {} scans, {encodes} encodes: every gate passes ({})",
            green("✓"),
            reports.len(),
            secs(seconds)
        );
    } else {
        println!("{} {failed} of {} scans failed ({})", red("✗"), reports.len(), secs(seconds));
        for r in reports.iter().filter(|r| !r.failures.is_empty()) {
            println!("  {}: {}", r.name, r.failures.join("; "));
        }
    }
    failed
}

/// A cosine near 1 with its distance from 1: `0.999999 (1-cos 1.7e-12)`.
fn cosine_text(cosine: f64) -> String {
    format!("{cosine:.6} (1-cos {:.1e})", 1.0 - cosine)
}

/// The minimum, with NaN (a comparison that could not run) winning.
fn nan_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) }
}
