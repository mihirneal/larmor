//! Stage-by-stage parity with the Python reference fixtures that `tools/make_reference.py`
//! writes under `$LARMOR_FIXTURES` (default `/Users/mihir/Developer/larmor/fixtures`).
//!
//! Every scan in `fixtures/manifest.toml` is checked. Its file must hash to the manifest's sha256,
//! its prep reference (`reference/prep/<scan>/`) must exist, and so must an encode reference for
//! each model in its `encode` list. A model's patch size and recipe come from its own
//! `fixtures/models/<id>.toml`, so a new reference model needs no change here.
//!
//! These read full-size scans, so they are ignored by default:
//!
//! ```text
//! cargo test --release -p larmor-prep --test fixtures -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use larmor_prep::{Prepared, Recipe};
use larmor_volume::ndarray::{Array3, ArrayD, Ix3};
use larmor_volume::{Nifti, VoxelData};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Largest share of voxels on which a mask may disagree with the reference (0.01%).
const MAX_MASK_DISAGREE: f64 = 1e-4;
/// Smallest share of in-mask voxels whose quantized value must be exact (99.99%).
const MIN_QUANT_EXACT: f64 = 0.9999;
/// Largest conform error, relative to the value range of the reference conform image.
const MAX_CONFORM_REL: f64 = 1e-5;
/// Largest absolute difference allowed in the normalized image and its patches.
const MAX_IMAGE_ABS: f64 = 1e-5;

fn fixtures_root() -> PathBuf {
    std::env::var_os("LARMOR_FIXTURES")
        .map_or_else(|| PathBuf::from("/Users/mihir/Developer/larmor/fixtures"), PathBuf::from)
}

/// `fixtures/manifest.toml`, as `tools/fetch_fixtures.py` writes it (only the fields used here).
#[derive(Deserialize)]
struct Manifest {
    scan: Vec<ManifestScan>,
}

#[derive(Deserialize)]
struct ManifestScan {
    name: String,
    /// Relative to the fixtures root.
    file: String,
    sha256: String,
    /// Ids of the models with an encode reference for this scan.
    #[serde(default)]
    encode: Vec<String>,
}

fn load_manifest(root: &Path) -> Manifest {
    let path = root.join("manifest.toml");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// What the tests need from a reference model's `model.toml`.
struct ReferenceModel {
    patch: usize,
    recipe: Recipe,
}

fn load_model(root: &Path, id: &str) -> ReferenceModel {
    let path = root.join("models").join(format!("{id}.toml"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let doc: toml::Table = text.parse().unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let patch = doc["architecture"]["patch"].as_integer().expect("architecture.patch") as usize;
    let steps: Vec<toml::Table> = doc["preprocessing"]["steps"]
        .as_array()
        .expect("preprocessing.steps")
        .iter()
        .map(|step| step.as_table().expect("a step is a table").clone())
        .collect();
    let recipe = Recipe::from_steps(&steps).unwrap_or_else(|e| panic!("{id}: {e}"));
    ReferenceModel { patch, recipe }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A short hash of every array `Recipe::run` returns, to show the output does not depend on the
/// thread count (compare runs with different `RAYON_NUM_THREADS`).
fn output_digest(prepared: &Prepared) -> String {
    let mut hasher = Sha256::new();
    for v in prepared.image.iter().chain(prepared.conformed.iter()) {
        hasher.update(v.to_bits().to_le_bytes());
    }
    for v in &prepared.quantized {
        hasher.update(v.to_le_bytes());
    }
    for &v in prepared.mask.iter().chain(prepared.native_mask.iter()) {
        hasher.update([v as u8]);
    }
    hasher.finalize().iter().take(6).map(|b| format!("{b:02x}")).collect()
}

fn read_u8(path: &Path) -> Array3<u8> {
    let img = Nifti::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match img.data {
        VoxelData::U8(a) => a.into_dimensionality().unwrap(),
        other => panic!("{}: expected uint8, got {}", path.display(), other.dtype().name()),
    }
}

fn read_f32(path: &Path) -> Array3<f32> {
    let img = Nifti::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    img.fdata_f32().into_dimensionality().unwrap()
}

fn read_u16(path: &Path) -> Array3<u16> {
    let img = Nifti::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    match img.data {
        VoxelData::U16(a) => a.into_dimensionality().unwrap(),
        other => panic!("{}: expected uint16, got {}", path.display(), other.dtype().name()),
    }
}

fn tensor<'a>(st: &'a SafeTensors<'a>, name: &str, dtype: Dtype) -> (&'a [u8], Vec<usize>) {
    let t = st.tensor(name).unwrap_or_else(|_| panic!("encode reference lacks {name}"));
    assert_eq!(t.dtype(), dtype, "{name} dtype");
    (t.data(), t.shape().to_vec())
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn max_abs_diff<'a>(a: impl IntoIterator<Item = &'a f32>, b: impl IntoIterator<Item = &'a f32>) -> f32 {
    a.into_iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn bit_exact_fraction(a: &Array3<f32>, b: &Array3<f32>) -> f64 {
    let same = a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    same as f64 / a.len() as f64
}

/// How one encode reference compares (one model, one scan).
struct EncodeReport {
    patch: usize,
    tokens: usize,
    reference_tokens: usize,
    coords_equal: bool,
    mask_disagree: usize,
    image_max_abs: f32,
    image_exact: f64,
    /// `None` when the coordinates differ, so the patches cannot be lined up.
    patches_max_abs: Option<f32>,
}

/// Everything measured for one scan, and the gates it failed.
#[derive(Default)]
struct ScanReport {
    name: String,
    head_disagree: usize,
    head_voxels: usize,
    fit_disagree: usize,
    fit_voxels: usize,
    fit_in_mask: usize,
    quant_mismatch: usize,
    quant_max_diff: i32,
    conform_rel: f64,
    conform_exact: f64,
    encodes: BTreeMap<String, EncodeReport>,
    read_s: f64,
    prep_s: f64,
    normalize_s: f64,
    python_s: f64,
    digest: String,
    failures: Vec<String>,
}

impl ScanReport {
    fn gate(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            self.failures.push(what());
        }
    }

    /// False (and a recorded failure) when the shapes differ, so the arrays cannot be compared.
    fn same_shape(&mut self, what: &str, ours: &[usize], reference: &[usize]) -> bool {
        self.gate(ours == reference, || format!("{what} shape {ours:?} vs reference {reference:?}"));
        ours == reference
    }
}

fn mask_disagreements(ours: &Array3<bool>, reference: &Array3<u8>) -> usize {
    ours.iter().zip(reference).filter(|&(&a, &b)| a != (b > 0)).count()
}

fn compare_prep(dir: &Path, info: &Value, prepared: &Prepared, report: &mut ScanReport) {
    let head = read_u8(&dir.join("head_mask.nii.gz"));
    report.head_voxels = head.len();
    if report.same_shape("head mask", prepared.native_mask.shape(), head.shape()) {
        report.head_disagree = mask_disagreements(&prepared.native_mask, &head);
    }
    let (count, share) = (report.head_disagree, report.head_disagree as f64 / report.head_voxels as f64);
    report.gate(share <= MAX_MASK_DISAGREE, || {
        format!("head mask: {count} voxels ({:.4}%) disagree", 100.0 * share)
    });
    let head_count = head.iter().filter(|&&v| v > 0).count() as u64;
    report
        .gate(info["head_mask_voxels"].as_u64() == Some(head_count), || "info.json head_mask_voxels".into());

    let fit = read_u8(&dir.join("fit_mask.nii.gz"));
    report.fit_voxels = fit.len();
    report.fit_in_mask = fit.iter().filter(|&&v| v > 0).count();
    if report.same_shape("fit mask", prepared.mask.shape(), fit.shape()) {
        report.fit_disagree = mask_disagreements(&prepared.mask, &fit);
    }
    let (count, share) = (report.fit_disagree, report.fit_disagree as f64 / report.fit_voxels as f64);
    report.gate(share <= MAX_MASK_DISAGREE, || {
        format!("fit mask: {count} voxels ({:.4}%) disagree", 100.0 * share)
    });
    report.gate(info["fit_mask_voxels"].as_u64() == Some(report.fit_in_mask as u64), || {
        "info.json fit_mask_voxels".into()
    });

    let conform = read_f32(&dir.join("conform.nii.gz"));
    if report.same_shape("conform", prepared.conformed.shape(), conform.shape()) {
        let (lo, hi) =
            conform.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| (l.min(v), h.max(v)));
        report.conform_rel = max_abs_diff(&prepared.conformed, &conform) as f64 / (hi - lo) as f64;
        report.conform_exact = bit_exact_fraction(&prepared.conformed, &conform);
    } else {
        report.conform_rel = f64::INFINITY;
    }
    let rel = report.conform_rel;
    report.gate(rel <= MAX_CONFORM_REL, || format!("conform relative error {rel:e}"));

    // outside the mask both sides are 0 whenever the masks agree, so the share is taken over the
    // reference's in-mask voxels, the stricter denominator
    let quant = read_u16(&dir.join("quantized.nii.gz"));
    if report.same_shape("quantized", prepared.quantized.shape(), quant.shape()) {
        for (&a, &b) in prepared.quantized.iter().zip(&quant) {
            let diff = (a as i32 - b as i32).abs();
            report.quant_mismatch += usize::from(diff != 0);
            report.quant_max_diff = report.quant_max_diff.max(diff);
        }
    } else {
        report.quant_mismatch = quant.len();
    }
    let exact = 1.0 - report.quant_mismatch as f64 / report.fit_in_mask.max(1) as f64;
    report.gate(exact >= MIN_QUANT_EXACT, || {
        format!("quantized exact on {:.4}% of in-mask voxels", 100.0 * exact)
    });

    // info.json is parsed with serde_json's float_roundtrip, so float64 values compare exactly
    let (vmin, vmax) = (info["vmin"].as_f64().unwrap(), info["vmax"].as_f64().unwrap());
    report.gate(prepared.vmin == vmin, || format!("vmin {} vs reference {vmin}", prepared.vmin));
    report.gate(prepared.vmax == vmax, || format!("vmax {} vs reference {vmax}", prepared.vmax));

    let centroid: Vec<i64> =
        info["centroid"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    report
        .gate(centroid == prepared.centroid, || format!("centroid {:?} vs {centroid:?}", prepared.centroid));

    let reference_affine: Vec<Vec<f64>> = serde_json::from_value(info["fit_affine"].clone()).unwrap();
    let same_affine = (0..4).all(|i| (0..4).all(|j| prepared.fit_affine.0[i][j] == reference_affine[i][j]));
    report.gate(same_affine, || format!("fit affine {:?} vs {reference_affine:?}", prepared.fit_affine.0));

    // info.json has null when conform_image did not smooth; Larmor reports zeros then
    let reference_fwhm: [f64; 3] = serde_json::from_value(info["fwhm"].clone()).unwrap_or([0.0; 3]);
    report
        .gate(prepared.fwhm == reference_fwhm, || format!("fwhm {:?} vs {reference_fwhm:?}", prepared.fwhm));
}

fn compare_encode(path: &Path, patch: usize, prepared: &Prepared) -> EncodeReport {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let st = SafeTensors::deserialize(&bytes).unwrap();

    let (image, shape) = tensor(&st, "image", Dtype::F32);
    let image = ArrayD::from_shape_vec(shape, f32s(image)).unwrap().into_dimensionality::<Ix3>().unwrap();
    let (mask, _) = tensor(&st, "mask", Dtype::U8);
    let mask_disagree = prepared.mask.iter().zip(mask).filter(|&(&a, &b)| a != (b > 0)).count();

    let tokens = prepared.tokens(patch).unwrap();
    let (coords, _) = tensor(&st, "coords", Dtype::I32);
    let coords: Vec<i32> = coords.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
    let coords: Vec<[u16; 3]> = coords.as_chunks::<3>().0.iter().map(|c| c.map(|v| v as u16)).collect();
    let coords_equal = coords == tokens.coords;
    let patches_max_abs = coords_equal.then(|| {
        let (patches, _) = tensor(&st, "patches", Dtype::F32);
        max_abs_diff(&tokens.patches, &f32s(patches))
    });

    EncodeReport {
        patch,
        tokens: tokens.len(),
        reference_tokens: coords.len(),
        coords_equal,
        mask_disagree,
        image_max_abs: max_abs_diff(&prepared.image, &image),
        image_exact: bit_exact_fraction(&prepared.image, &image),
        patches_max_abs,
    }
}

fn check_encode(report: &mut ScanReport, model: &str, encode: &EncodeReport) {
    let p = encode.patch;
    report.gate(encode.coords_equal, || {
        format!(
            "{model}: {} p{p} tokens vs reference {}, or different coords",
            encode.tokens, encode.reference_tokens
        )
    });
    let disagree = encode.mask_disagree as f64 / report.fit_voxels.max(1) as f64;
    report.gate(disagree <= MAX_MASK_DISAGREE, || {
        format!("{model}: encode mask {} voxels", encode.mask_disagree)
    });
    report.gate(encode.image_max_abs as f64 <= MAX_IMAGE_ABS, || {
        format!("{model}: image max abs diff {:e}", encode.image_max_abs)
    });
    if let Some(diff) = encode.patches_max_abs {
        report.gate(diff as f64 <= MAX_IMAGE_ABS, || format!("{model}: patches max abs diff {diff:e}"));
    }
}

/// Runs and compares one manifest scan.
fn check_scan(
    root: &Path,
    scan: &ManifestScan,
    prep_recipe: &Recipe,
    models: &BTreeMap<String, ReferenceModel>,
) -> ScanReport {
    let mut report = ScanReport { name: scan.name.clone(), ..ScanReport::default() };
    let dir = root.join("reference/prep").join(&scan.name);
    let Ok(info_text) = std::fs::read_to_string(dir.join("info.json")) else {
        report.failures.push(format!("no prep reference at {}", dir.display()));
        return report;
    };
    let info: Value = serde_json::from_str(&info_text).unwrap();

    let path = root.join(&scan.file);
    let start = Instant::now();
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let nifti = Nifti::from_bytes(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    report.read_s = start.elapsed().as_secs_f64();
    let sha = sha256_hex(&bytes);
    report.gate(sha == scan.sha256, || format!("scan sha256 {sha} vs manifest {}", scan.sha256));
    report.gate(info["scan_sha256"].as_str() == Some(sha.as_str()), || {
        "prep reference made from another file".into()
    });

    let prepared = match prep_recipe.run(&nifti) {
        Ok(prepared) => prepared,
        Err(e) => {
            report.failures.push(format!("failed to preprocess: {e}"));
            return report;
        }
    };
    let t = prepared.timings;
    report.prep_s = (t.mask + t.conform + t.quantize).as_secs_f64();
    report.normalize_s = t.normalize.as_secs_f64();
    // the Python seconds cover threshold_mask, conform_image and truncate_image (no normalize)
    report.python_s =
        info["seconds"].as_object().map_or(f64::NAN, |s| s.values().filter_map(Value::as_f64).sum());
    report.digest = output_digest(&prepared);
    compare_prep(&dir, &info, &prepared, &mut report);

    for id in &scan.encode {
        let path = root.join("reference/encode").join(id).join(format!("{}.safetensors", scan.name));
        if !path.exists() {
            report.failures.push(format!("{id}: listed in the manifest but {} is missing", path.display()));
            continue;
        }
        let model = &models[id];
        let encode = if model.recipe == *prep_recipe {
            compare_encode(&path, model.patch, &prepared)
        } else {
            compare_encode(&path, model.patch, &model.recipe.run(&nifti).unwrap())
        };
        check_encode(&mut report, id, &encode);
        report.encodes.insert(id.clone(), encode);
    }
    report
}

fn print_table(reports: &[ScanReport]) {
    println!(
        "\n| scan | head mask disagree | fit mask disagree | quantized mismatches (max diff) | conform rel err | tokens (coords equal) | image max diff | Rust read s | Rust prep s | Rust normalize s | Python s | digest | result |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in reports {
        let tokens: Vec<String> = r
            .encodes
            .values()
            .map(|e| format!("p{} {} {}", e.patch, e.tokens, if e.coords_equal { "yes" } else { "NO" }))
            .collect();
        let image = r.encodes.values().map(|e| e.image_max_abs).fold(0.0f32, f32::max);
        let exact = r.encodes.values().map(|e| e.image_exact).fold(1.0f64, f64::min);
        println!(
            "| {} | {} | {} | {} ({}) | {:.1e} | {} | {:.1e} ({:.1}% exact) | {:.3} | {:.3} | {:.3} | {:.3} | {} | {} |",
            r.name,
            r.head_disagree,
            r.fit_disagree,
            r.quant_mismatch,
            r.quant_max_diff,
            r.conform_rel,
            tokens.join(", "),
            image,
            100.0 * exact,
            r.read_s,
            r.prep_s,
            r.normalize_s,
            r.python_s,
            r.digest,
            if r.failures.is_empty() { "pass" } else { "FAIL" },
        );
    }
}

#[test]
#[ignore = "needs the reference fixtures; run with --ignored --release"]
fn prep_matches_python_reference_fixtures() {
    let root = fixtures_root();
    let manifest = load_manifest(&root);
    assert!(!manifest.scan.is_empty(), "no scans in {}/manifest.toml", root.display());

    let mut models = BTreeMap::new();
    for id in manifest.scan.iter().flat_map(|scan| &scan.encode) {
        if !models.contains_key(id) {
            models.insert(id.clone(), load_model(&root, id));
        }
    }
    // make_reference.py's prep references always use the nanobrain-eval recipe
    let prep_recipe = Recipe::nanobrain_eval();

    let mut reports = Vec::new();
    for scan in &manifest.scan {
        let report = check_scan(&root, scan, &prep_recipe, &models);
        println!("{}: {}", report.name, if report.failures.is_empty() { "pass" } else { "FAIL" });
        reports.push(report);
    }
    print_table(&reports);

    let listed: Vec<&str> = manifest.scan.iter().map(|scan| scan.name.as_str()).collect();
    if let Ok(entries) = std::fs::read_dir(root.join("reference/prep")) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !listed.contains(&name.as_str()) {
                println!("note: reference/prep/{name} is not in the manifest and was not checked");
            }
        }
    }

    let failures: Vec<String> =
        reports.iter().flat_map(|r| r.failures.iter().map(move |f| format!("{}: {f}", r.name))).collect();
    assert!(failures.is_empty(), "parity gates failed:\n{}", failures.join("\n"));
}
