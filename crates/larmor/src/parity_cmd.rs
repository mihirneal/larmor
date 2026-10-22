//! `larmor check-parity`: every preprocessing stage and every embedding against the PyTorch
//! reference fixtures (`docs/DESIGN.md`, "Reference fixtures" and "Gates").
//!
//! The comparisons are those of `larmor-prep`'s `tests/fixtures.rs` and `larmor-model`'s parity
//! module, run on every scan in `<fixtures>/manifest.toml`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use larmor_model::parity::{EncodeReference, compare};
use larmor_model::{DeviceChoice, Encoder, Precision};
use larmor_prep::{Prepared, Recipe};
use larmor_volume::ndarray::{Array3, Ix3};
use larmor_volume::{Nifti, VoxelData};
use serde::Deserialize;
use serde_json::Value;

use crate::console::{bold, dim, green, red, secs};
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
    pub device: DeviceChoice,
    pub precision: Precision,
}

/// `<fixtures>/manifest.toml` (only the fields used here).
#[derive(Deserialize)]
struct FixtureManifest {
    scan: Vec<FixtureScan>,
    /// Target scans prepared on a reference scan's head mask and grid.
    #[serde(default)]
    pair: Vec<FixturePair>,
    /// DICOM series and what Larmor must make of them.
    #[serde(default)]
    dicom: Vec<FixtureDicom>,
}

/// A `[[dicom]]`: the DICOM files under `folder`.
#[derive(Deserialize)]
struct FixtureDicom {
    name: String,
    /// Relative to the fixtures folder.
    folder: String,
    /// `series`: converts bit-exact to the `[[scan]]` named `scan` (dcm2niix's NIfTI of it) and passes
    /// that scan's recipe and encode references; `refuse`: refused with a message containing
    /// `refuse`; `pixels`: every file decodes to the same samples, except the transfer syntaxes in
    /// `unsupported`, which are refused.
    kind: String,
    /// SHA-256 of the files' SHA-256s, sorted (as `tools/fetch_fixtures.py` pins them).
    files_sha256: String,
    #[serde(default)]
    scan: Option<String>,
    #[serde(default)]
    refuse: Option<String>,
    #[serde(default)]
    unsupported: Vec<String>,
}

/// A `[[pair]]`: `target` prepared on `reference`'s head mask and grid, as `prepare_images` does
/// with `reference=` (both are `[[scan]]` names).
#[derive(Deserialize)]
struct FixturePair {
    name: String,
    target: String,
    reference: String,
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
    /// Smallest token cosine after subtracting the reference's mean token from both sides.
    min_centered: f64,
    pooled_cosine: f64,
    max_abs_diff: f64,
    /// PyTorch's own bf16-vs-fp32 cosines on this scan, from the reference metadata.
    bf16_min_cosine: Option<f64>,
    bf16_pooled_cosine: Option<f64>,
    seconds: f64,
}

/// What a report row checked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Checked {
    /// A NIfTI scan through the recipe and the encoders.
    #[default]
    Scan,
    /// A target on a reference scan's mask, gated bit-exact.
    Pair,
    /// A DICOM series converted, then through the recipe and the encoders.
    Dicom,
    /// A DICOM series that must be refused.
    DicomRefusal,
    /// DICOM files in several transfer syntaxes that must decode alike.
    DicomPixels,
}

impl Checked {
    /// Rows that run the recipe and so count in the preprocessing gates.
    fn has_prep(self) -> bool {
        matches!(self, Checked::Scan | Checked::Pair | Checked::Dicom)
    }
}

#[derive(Default)]
struct ScanReport {
    name: String,
    checked: Checked,
    prep: Option<PrepReport>,
    encodes: Vec<EncodeReport>,
    /// A DICOM series: converted voxels, dtype, affine, zooms and scaling equal dcm2niix's.
    conversion_exact: Option<bool>,
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

/// A reference model and its encoder.
struct ReferenceModel {
    model: Model,
    encoder: Encoder,
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
    let models = reference_models(root, options.models.as_deref(), options.device, options.precision)?;
    // make_reference.py's prep references always use the nanobrain-eval recipe
    let prep_recipe = Recipe::nanobrain_eval();

    let encoding = models.values().next().map_or_else(
        || "no encoder".to_string(),
        |m| format!("device {} · {}", m.encoder.options().device, m.encoder.options().precision),
    );
    println!(
        "{}  {} scans · models {} · {encoding}",
        bold("larmor check-parity"),
        manifest.scan.len(),
        if models.is_empty() {
            "none".to_string()
        } else {
            models.keys().cloned().collect::<Vec<_>>().join(", ")
        },
    );
    print_header();
    let mut reports = Vec::new();
    for scan in &manifest.scan {
        let report = check_scan(root, scan, &prep_recipe, &models);
        print_rows(&report);
        reports.push(report);
    }
    for pair in &manifest.pair {
        let report = check_pair(root, pair, &manifest.scan, &prep_recipe, &models);
        print_rows(&report);
        reports.push(report);
    }
    for dicom in &manifest.dicom {
        let report = check_dicom(root, dicom, &manifest.scan, &prep_recipe, &models);
        print_rows(&report);
        reports.push(report);
    }
    let failed = print_summary(&reports, &models, start.elapsed().as_secs_f64());
    if failed > 0 {
        bail!("{failed} of {} {} failed a parity gate", reports.len(), what_was_checked(&reports));
    }
    Ok(())
}

/// `scans`, `scans and pairs`, or `fixtures` when DICOM sets were checked too.
fn what_was_checked(reports: &[ScanReport]) -> &'static str {
    let has = |checked: Checked| reports.iter().any(|r| r.checked == checked);
    if has(Checked::Dicom) || has(Checked::DicomRefusal) || has(Checked::DicomPixels) {
        "fixtures"
    } else if has(Checked::Pair) {
        "scans and pairs"
    } else {
        "scans"
    }
}

fn check_dicom(
    root: &Path,
    dicom: &FixtureDicom,
    scans: &[FixtureScan],
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
) -> ScanReport {
    let start = Instant::now();
    let checked = match dicom.kind.as_str() {
        "refuse" => Checked::DicomRefusal,
        "pixels" => Checked::DicomPixels,
        _ => Checked::Dicom,
    };
    let mut report = ScanReport { name: dicom.name.clone(), checked, ..ScanReport::default() };
    if let Err(err) = check_dicom_inner(root, dicom, scans, prep_recipe, models, &mut report) {
        report.failures.push(format!("{err:#}"));
    }
    report.seconds = start.elapsed().as_secs_f64();
    report
}

/// Every file below `dir`, sorted, with its bytes.
fn read_tree(dir: &Path) -> Result<Vec<(PathBuf, Vec<u8>)>> {
    let mut paths = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(folder) = stack.pop() {
        for entry in std::fs::read_dir(&folder).with_context(|| format!("reading {}", folder.display()))? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                paths.push(path);
            }
        }
    }
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            Ok((path, bytes))
        })
        .collect()
}

/// `tools/fetch_fixtures.py::files_sha256`: the SHA-256 of the files' SHA-256s, sorted.
fn files_sha256(files: &[(PathBuf, Vec<u8>)]) -> String {
    let mut digests: Vec<String> = files.iter().map(|(_, bytes)| sha256_hex(bytes)).collect();
    digests.sort();
    sha256_hex(digests.iter().map(|d| format!("{d}\n")).collect::<String>().as_bytes())
}

fn check_dicom_inner(
    root: &Path,
    dicom: &FixtureDicom,
    scans: &[FixtureScan],
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
    report: &mut ScanReport,
) -> Result<()> {
    let files = read_tree(&root.join(&dicom.folder))?;
    let sha = files_sha256(&files);
    report
        .gate(sha == dicom.files_sha256, || format!("files_sha256 {sha} vs manifest {}", dicom.files_sha256));
    match report.checked {
        Checked::DicomRefusal => {
            let expected = dicom.refuse.as_deref().unwrap_or_default();
            match larmor_dicom::convert_files(&files) {
                Ok(_) => report.failures.push(format!("converted, but should be refused ({expected})")),
                Err(err) => report.gate(err.to_string().contains(expected), || {
                    format!("refused with \"{err}\", not a message containing \"{expected}\"")
                }),
            }
            Ok(())
        }
        Checked::DicomPixels => {
            let mut first: Option<(&Path, Vec<u8>)> = None;
            for (path, bytes) in &files {
                let header = larmor_dicom::SliceHeader::from_bytes(path, bytes)?;
                let decoded = larmor_dicom::decode_pixels(path, bytes);
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if dicom.unsupported.contains(&header.transfer_syntax) {
                    report.gate(decoded.is_err(), || {
                        format!("{name}: {} decoded but should be refused", header.transfer_syntax)
                    });
                    continue;
                }
                match (decoded, &first) {
                    (Err(err), _) => report.failures.push(format!("{name}: {err}")),
                    (Ok(samples), None) => first = Some((path, samples)),
                    (Ok(samples), Some((first_path, expected))) => report.gate(samples == *expected, || {
                        format!(
                            "{name} ({}) decodes to other samples than {}",
                            header.transfer_syntax,
                            first_path.display()
                        )
                    }),
                }
            }
            Ok(())
        }
        _ => {
            let scan_name = dicom.scan.as_deref().context("a [[dicom]] series needs `scan`")?;
            let scan = scans
                .iter()
                .find(|s| s.name == scan_name)
                .with_context(|| format!("no [[scan]] named {scan_name}"))?;
            let converted = larmor_dicom::convert_files(&files)?;
            let reference_path = root.join(&scan.file);
            let reference = read_nifti(&reference_path)?;
            let exact = compare_conversion(&converted.nifti, &reference, report)?;
            report.conversion_exact = Some(exact);
            check_prepared(root, scan_name, &converted.nifti, None, prep_recipe, models, report)
        }
    }
}

/// The converted image against dcm2niix's NIfTI as nibabel loads it: shape, dtype, voxels, affine,
/// zooms and scale factors all equal.
fn compare_conversion(ours: &Nifti, dcm2niix: &Nifti, report: &mut ScanReport) -> Result<bool> {
    let before = report.failures.len();
    report.gate(ours.shape() == dcm2niix.shape(), || {
        format!("shape {:?} vs dcm2niix {:?}", ours.shape(), dcm2niix.shape())
    });
    report.gate(ours.data.dtype() == dcm2niix.data.dtype(), || {
        format!("dtype {} vs dcm2niix {}", ours.data.dtype().name(), dcm2niix.data.dtype().name())
    });
    let (a, b) = (ours.affine()?, dcm2niix.affine()?);
    report.gate(a == b, || format!("affine {:?} vs dcm2niix {:?}", a.0, b.0));
    report.gate(ours.zooms() == dcm2niix.zooms(), || {
        format!("zooms {:?} vs dcm2niix {:?}", ours.zooms(), dcm2niix.zooms())
    });
    let scaling = |n: &Nifti| n.header.scaling();
    report.gate(scaling(ours) == scaling(dcm2niix), || {
        format!("scaling {:?} vs dcm2niix {:?}", scaling(ours), scaling(dcm2niix))
    });
    if ours.shape() == dcm2niix.shape()
        && ours.data.dtype() == dcm2niix.data.dtype()
        && ours.data != dcm2niix.data
    {
        let (x, y) = (ours.data.to_f64(), dcm2niix.data.to_f64());
        let differ = x.iter().zip(y.iter()).filter(|(p, q)| p != q).count();
        report.failures.push(format!("{differ} voxels differ from dcm2niix's"));
    }
    Ok(report.failures.len() == before)
}

fn check_pair(
    root: &Path,
    pair: &FixturePair,
    scans: &[FixtureScan],
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
) -> ScanReport {
    let start = Instant::now();
    let mut report = ScanReport { name: pair.name.clone(), checked: Checked::Pair, ..ScanReport::default() };
    if let Err(err) = check_pair_inner(root, pair, scans, prep_recipe, models, &mut report) {
        report.failures.push(format!("{err:#}"));
    }
    report.seconds = start.elapsed().as_secs_f64();
    report
}

/// A pair's preprocessing must equal `prepare_images` with `reference=` bit for bit: fit mask,
/// quantized volume, grid and image affines, and (through the encode references) the token grid.
fn check_pair_inner(
    root: &Path,
    pair: &FixturePair,
    scans: &[FixtureScan],
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
    report: &mut ScanReport,
) -> Result<()> {
    let read = |name: &str, report: &mut ScanReport| -> Result<(Nifti, String)> {
        let scan =
            scans.iter().find(|s| s.name == name).with_context(|| format!("no [[scan]] named {name}"))?;
        let path = root.join(&scan.file);
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        let sha = sha256_hex(&bytes);
        report.gate(sha == scan.sha256, || format!("{name}: sha256 {sha} vs manifest {}", scan.sha256));
        Ok((Nifti::from_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))?, sha))
    };
    let (target, target_sha) = read(&pair.target, report)?;
    let (reference, reference_sha) = read(&pair.reference, report)?;
    let dir = root.join("reference/prep").join(&pair.name);
    let (prepared, references) =
        rayon::join(|| prep_recipe.run_with_reference(&target, &reference), || PrepReferences::read(&dir));
    let prepared = prepared.context("preprocessing on the reference's mask")?;
    let references = references?;
    let info = &references.info;
    report.gate(
        info["scan_sha256"].as_str() == Some(target_sha.as_str())
            && info["reference_sha256"].as_str() == Some(reference_sha.as_str()),
        || "prep reference was made from other files".into(),
    );
    let prep = compare_prep(&references, &prepared, report);
    report.gate(prep.fit_disagree == 0, || {
        format!("fit mask: {} voxels disagree (must be 0)", prep.fit_disagree)
    });
    report.gate(prep.quant_exact == 1.0, || "quantized volume is not bit-exact".into());
    let image_affine: Option<[[f64; 4]; 4]> = serde_json::from_value(info["image_affine"].clone()).ok();
    report.gate(image_affine == Some(prepared.image_affine.0), || {
        format!("image affine {:?} vs {image_affine:?}", prepared.image_affine.0)
    });
    report.prep = Some(prep);
    for (id, reference_model) in models {
        let reference_path =
            root.join("reference/encode").join(id).join(format!("{}.safetensors", pair.name));
        if !reference_path.is_file() {
            continue;
        }
        let encode = compare_encode(
            reference_model,
            &reference_path,
            &target,
            Some(&reference),
            &prepared,
            prep_recipe,
        )
        .with_context(|| id.clone())?;
        gate_encode(report, id, &encode);
        report.encodes.push(encode);
    }
    Ok(())
}

/// The token-grid and cosine gates of one encode comparison.
fn gate_encode(report: &mut ScanReport, id: &str, encode: &EncodeReport) {
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
}

/// Every model with a folder in `reference/encode/`, or the ones `--models` names, with its encoder.
fn reference_models(
    root: &Path,
    only: Option<&[String]>,
    device: DeviceChoice,
    precision: Precision,
) -> Result<BTreeMap<String, ReferenceModel>> {
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
        let encoder = load_encoder(&model, device, precision)?;
        models.insert(id, ReferenceModel { model, encoder });
    }
    Ok(models)
}

fn check_scan(
    root: &Path,
    scan: &FixtureScan,
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
) -> ScanReport {
    let start = Instant::now();
    let mut report = ScanReport { name: scan.name.clone(), ..ScanReport::default() };
    if let Err(err) = check_scan_inner(root, scan, prep_recipe, models, &mut report) {
        report.failures.push(format!("{err:#}"));
    }
    report.seconds = start.elapsed().as_secs_f64();
    report
}

fn check_scan_inner(
    root: &Path,
    scan: &FixtureScan,
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
    report: &mut ScanReport,
) -> Result<()> {
    let path = root.join(&scan.file);
    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let sha = sha256_hex(&bytes);
    report.gate(sha == scan.sha256, || format!("scan sha256 {sha} vs manifest {}", scan.sha256));
    let nifti = Nifti::from_bytes(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    check_prepared(root, &scan.name, &nifti, Some(&sha), prep_recipe, models, report)
}

/// The recipe on `nifti` against `reference/prep/<name>/` (made from the file with SHA-256 `sha`,
/// when given), then each model's encode against `reference/encode/<model>/<name>.safetensors`.
fn check_prepared(
    root: &Path,
    name: &str,
    nifti: &Nifti,
    sha: Option<&str>,
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
    report: &mut ScanReport,
) -> Result<()> {
    let dir = root.join("reference/prep").join(name);
    let (prepared, references) = rayon::join(|| prep_recipe.run(nifti), || PrepReferences::read(&dir));
    let prepared = prepared.context("preprocessing")?;
    let references = references?;
    if let Some(sha) = sha {
        report.gate(references.info["scan_sha256"].as_str() == Some(sha), || {
            "prep reference was made from another file".into()
        });
    }
    report.prep = Some(compare_prep(&references, &prepared, report));

    for (id, reference_model) in models {
        let reference_path = root.join("reference/encode").join(id).join(format!("{name}.safetensors"));
        if !reference_path.is_file() {
            continue;
        }
        let encode = compare_encode(reference_model, &reference_path, nifti, None, &prepared, prep_recipe)
            .with_context(|| id.clone())?;
        gate_encode(report, id, &encode);
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
    reference_model: &ReferenceModel,
    path: &Path,
    nifti: &Nifti,
    reference_nifti: Option<&Nifti>,
    prepared: &Prepared,
    prep_recipe: &Recipe,
) -> Result<EncodeReport> {
    let model = &reference_model.model;
    let own_prepared;
    let prepared = if model.recipe == *prep_recipe {
        prepared
    } else {
        own_prepared = match reference_nifti {
            None => model.recipe.run(nifti)?,
            Some(reference) => model.recipe.run_with_reference(nifti, reference)?,
        };
        &own_prepared
    };
    let tokens = prepared.tokens(model.patch())?;
    let reference = EncodeReference::load(path)?;
    let floor = reference.bf16_noise_floor();
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
        bf16_min_cosine: floor.map(|f| f.min_token_cosine),
        bf16_pooled_cosine: floor.map(|f| f.pooled_cosine),
        seconds: 0.0,
    };
    if !report.coords_equal {
        return Ok(report);
    }
    let start = Instant::now();
    let encoded = reference_model.encoder.encode(&tokens.patches, &tokens.coords)?;
    report.seconds = start.elapsed().as_secs_f64();
    ensure!(encoded.width == reference.width, "width {} vs reference {}", encoded.width, reference.width);
    let parity = compare(&encoded, &reference);
    report.min_cosine = parity.min_token_cosine;
    report.mean_cosine = parity.mean_token_cosine;
    report.min_centered = parity.min_centered_cosine;
    report.pooled_cosine = parity.pooled_cosine;
    report.max_abs_diff = parity.max_abs_diff;
    Ok(report)
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

/// Prints the gates with their worst values and the scans they came from; returns the number of
/// failing scans.
fn print_summary(reports: &[ScanReport], models: &BTreeMap<String, ReferenceModel>, seconds: f64) -> usize {
    let preps: Vec<(&str, &PrepReport)> =
        reports.iter().filter_map(|r| Some((r.name.as_str(), r.prep.as_ref()?))).collect();
    let prep_worst = |f: fn(&PrepReport) -> f64, worse: Worse| {
        worst(&preps.iter().map(|(n, p)| (*n, f(p))).collect::<Vec<_>>(), worse)
    };
    let head = prep_worst(|p| p.head_disagree as f64 / p.head_voxels.max(1) as f64, Worse::Larger);
    let fit = prep_worst(|p| p.fit_disagree as f64 / p.fit_voxels.max(1) as f64, Worse::Larger);
    let conform = prep_worst(|p| p.conform_rel, Worse::Larger);
    let conform_exact = prep_worst(|p| p.conform_exact, Worse::Smaller);
    let quant = prep_worst(|p| p.quant_exact, Worse::Smaller);
    let quant_diff = preps.iter().map(|(_, p)| p.quant_max_diff).max().unwrap_or(0);
    let with_prep: Vec<&ScanReport> = reports.iter().filter(|r| r.checked.has_prep()).collect();
    let not_exact: Vec<&str> = with_prep
        .iter()
        .filter(|r| !r.prep.as_ref().is_some_and(|p| p.scalars_exact))
        .map(|r| r.name.as_str())
        .collect();

    println!("\n{}", bold("Gates"));
    println!(
        "  {}",
        dim(&format!("{:<44} {:>28}  {:<26} {:<10}", "gate", "worst value", "worst scan", "limit"))
    );
    let row = |name: &str, value: String, scan: &str, limit: &str, ok: bool| {
        let status = if ok { green("pass") } else { red("FAIL") };
        println!("  {name:<44} {value:>28}  {scan:<26} {limit:<10} {status}");
    };
    let have_prep = preps.len() == with_prep.len() && !preps.is_empty();
    row(
        "head mask disagreement",
        format!("{:.4}%", 100.0 * head.value),
        head.scan,
        "≤ 0.01%",
        have_prep && head.value <= MAX_MASK_DISAGREE,
    );
    row(
        "fit mask disagreement",
        format!("{:.4}%", 100.0 * fit.value),
        fit.scan,
        "≤ 0.01%",
        have_prep && fit.value <= MAX_MASK_DISAGREE,
    );
    row(
        "conform relative error",
        format!("{:.2e} ({:.2}% exact)", conform.value, 100.0 * conform_exact.value),
        conform.scan,
        "≤ 1e-5",
        have_prep && conform.value <= MAX_CONFORM_REL,
    );
    row(
        "quantized exact (in-mask voxels)",
        format!("{:.4}% (max Δ {quant_diff})", 100.0 * quant.value),
        quant.scan,
        "≥ 99.99%",
        have_prep && quant.value >= MIN_QUANT_EXACT,
    );
    row(
        "vmin, vmax, centroid, fit affine, fwhm exact",
        format!("{}/{} scans", with_prep.len() - not_exact.len(), with_prep.len()),
        not_exact.first().copied().unwrap_or("-"),
        "all",
        not_exact.is_empty(),
    );
    let pairs: Vec<&ScanReport> = reports.iter().filter(|r| r.checked == Checked::Pair).collect();
    if !pairs.is_empty() {
        let inexact: Vec<&str> =
            pairs.iter().filter(|r| !r.failures.is_empty()).map(|r| r.name.as_str()).collect();
        row(
            "pairs on a reference mask bit-exact",
            format!("{}/{} pairs", pairs.len() - inexact.len(), pairs.len()),
            inexact.first().copied().unwrap_or("-"),
            "all",
            inexact.is_empty(),
        );
    }
    let kind_row = |checked: Checked, gate: &str, unit: &str| {
        let rows: Vec<&ScanReport> = reports.iter().filter(|r| r.checked == checked).collect();
        if rows.is_empty() {
            return;
        }
        let ok = |r: &ScanReport| match checked {
            Checked::Dicom => r.conversion_exact == Some(true),
            _ => r.failures.is_empty(),
        };
        let bad: Vec<&str> = rows.iter().filter(|r| !ok(r)).map(|r| r.name.as_str()).collect();
        row(
            gate,
            format!("{}/{} {unit}", rows.len() - bad.len(), rows.len()),
            bad.first().copied().unwrap_or("-"),
            "all",
            bad.is_empty(),
        );
    };
    kind_row(Checked::Dicom, "DICOM to NIfTI bit-exact with dcm2niix", "series");
    kind_row(Checked::DicomRefusal, "DICOM series refused with the reason", "series");
    kind_row(Checked::DicomPixels, "DICOM decoders agree on every syntax", "sets");
    for id in models.keys() {
        let encodes: Vec<(&str, &EncodeReport)> = reports
            .iter()
            .flat_map(|r| r.encodes.iter().map(move |e| (r.name.as_str(), e)))
            .filter(|(_, e)| e.model == *id)
            .collect();
        if encodes.is_empty() {
            continue;
        }
        let encode_worst = |f: fn(&EncodeReport) -> f64, worse: Worse| {
            worst(&encodes.iter().map(|(n, e)| (*n, f(e))).collect::<Vec<_>>(), worse)
        };
        let misaligned: Vec<&str> =
            encodes.iter().filter(|(_, e)| !e.coords_equal).map(|(n, _)| *n).collect();
        row(
            &format!("{id}: token grid equal"),
            format!("{}/{} scans", encodes.len() - misaligned.len(), encodes.len()),
            misaligned.first().copied().unwrap_or("-"),
            "all",
            misaligned.is_empty(),
        );
        let min_token = encode_worst(|e| e.min_cosine, Worse::Smaller);
        row(
            &format!("{id}: min token cosine"),
            cosine_text(min_token.value),
            min_token.scan,
            "≥ 0.999",
            min_token.value >= MIN_COSINE,
        );
        let pooled = encode_worst(|e| e.pooled_cosine, Worse::Smaller);
        row(
            &format!("{id}: pooled cosine"),
            cosine_text(pooled.value),
            pooled.scan,
            "≥ 0.999",
            pooled.value >= MIN_COSINE,
        );

        let centered = encode_worst(|e| e.min_centered, Worse::Smaller);
        let max_diff = encode_worst(|e| e.max_abs_diff, Worse::Larger);
        let mean_time = encodes.iter().map(|(_, e)| e.seconds).sum::<f64>() / encodes.len() as f64;
        println!(
            "  {}",
            dim(&format!(
                "{id}: also min centered cosine {} ({}), max |Δ| {:.2e} ({}); encode {} per scan",
                cosine_text(centered.value),
                centered.scan,
                max_diff.value,
                max_diff.scan,
                secs(mean_time)
            ))
        );
        let floor_token = encodes.iter().filter_map(|(_, e)| e.bf16_min_cosine).fold(f64::INFINITY, f64::min);
        let floor_pooled =
            encodes.iter().filter_map(|(_, e)| e.bf16_pooled_cosine).fold(f64::INFINITY, f64::min);
        if floor_token.is_finite() {
            println!(
                "  {}",
                dim(&format!(
                    "{id}: PyTorch's own bf16 vs fp32 (noise floor): min token cosine {}, pooled {}",
                    cosine_text(floor_token),
                    cosine_text(floor_pooled),
                ))
            );
        }
    }
    let failed = reports.iter().filter(|r| !r.failures.is_empty()).count();
    let encodes: usize = reports.iter().map(|r| r.encodes.len()).sum();
    let count = |checked: Checked| reports.iter().filter(|r| r.checked == checked).count();
    let n_pairs = count(Checked::Pair);
    let n_dicom = count(Checked::Dicom) + count(Checked::DicomRefusal) + count(Checked::DicomPixels);
    let plural = |n: usize, one: &str, many: &str| match n {
        0 => String::new(),
        1 => format!("1 {one}, "),
        n => format!("{n} {many}, "),
    };
    let others = plural(n_pairs, "pair", "pairs") + &plural(n_dicom, "DICOM set", "DICOM sets");
    if failed == 0 {
        println!(
            "{} {} scans, {others}{encodes} encodes: every gate passes ({})",
            green("✓"),
            count(Checked::Scan),
            secs(seconds)
        );
    } else {
        let what = what_was_checked(reports);
        println!("{} {failed} of {} {what} failed ({})", red("✗"), reports.len(), secs(seconds));
        for r in reports.iter().filter(|r| !r.failures.is_empty()) {
            println!("  {}: {}", r.name, r.failures.join("; "));
        }
    }
    failed
}

/// Which direction of a metric is worse.
#[derive(Clone, Copy)]
enum Worse {
    Larger,
    Smaller,
}

/// The worst value of a metric over the scans, and the scan it came from.
#[derive(Debug, PartialEq)]
struct Worst<'a> {
    /// The scan, `all scans` when every scan has the same value, or `-` when there are none.
    scan: &'a str,
    value: f64,
}

/// The worst of `(scan, value)` pairs. NaN (a comparison that could not run) is worst of all.
fn worst<'a>(values: &[(&'a str, f64)], worse: Worse) -> Worst<'a> {
    let is_worse = |a: f64, b: f64| {
        a.is_nan()
            || (!b.is_nan()
                && match worse {
                    Worse::Larger => a > b,
                    Worse::Smaller => a < b,
                })
    };
    let Some(&(scan, value)) = values.iter().reduce(|w, v| if is_worse(v.1, w.1) { v } else { w }) else {
        return Worst { scan: "-", value: f64::NAN };
    };
    let tie = values.len() > 1 && values.iter().all(|&(_, v)| v == value);
    Worst { scan: if tie { "all scans" } else { scan }, value }
}

/// A cosine near 1 with its distance from 1: `0.999999 (1-cos 1.7e-12)`.
fn cosine_text(cosine: f64) -> String {
    format!("{cosine:.6} (1-cos {:.1e})", 1.0 - cosine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_names_the_extreme_scan() {
        let values = [("a", 0.5), ("b", 0.9), ("c", 0.1)];
        assert_eq!(worst(&values, Worse::Larger), Worst { scan: "b", value: 0.9 });
        assert_eq!(worst(&values, Worse::Smaller), Worst { scan: "c", value: 0.1 });
        assert_eq!(worst(&[("a", f64::NAN), ("b", 0.1)], Worse::Smaller).scan, "a");
        assert_eq!(worst(&[("a", 0.1), ("b", f64::NAN)], Worse::Smaller).scan, "b");
        assert_eq!(worst(&[("a", 0.0), ("b", 0.0)], Worse::Larger), Worst { scan: "all scans", value: 0.0 });
        assert_eq!(worst(&[("a", 0.0)], Worse::Larger).scan, "a");
        assert_eq!(worst(&[], Worse::Larger).scan, "-");
    }
}
