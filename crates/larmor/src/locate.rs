//! Where things are: model manifests, the embedding cache, input scans, and the BIDS layout of the
//! outputs.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Directories searched for `model.toml` files by id, in order: each entry of `$LARMOR_MODELS`
/// (colon-separated), the user data directory (`~/Library/Application Support/larmor/models` on
/// macOS, `~/.local/share/larmor/models` on Linux), then `./models`. A folder named twice is
/// searched once.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(list) = std::env::var_os("LARMOR_MODELS") {
        dirs.extend(std::env::split_paths(&list).filter(|p| !p.as_os_str().is_empty()));
    }
    if let Some(data) = dirs::data_dir() {
        dirs.push(data.join("larmor").join("models"));
    }
    dirs.push(PathBuf::from("models"));
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|dir| seen.insert(std::fs::canonicalize(dir).unwrap_or_else(|_| dir.clone())));
    dirs
}

/// The embedding cache root: `$LARMOR_CACHE`, else `~/Library/Caches/larmor` on macOS
/// (`$XDG_CACHE_HOME/larmor` or `~/.cache/larmor` on Linux).
pub fn cache_root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("LARMOR_CACHE").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let base = dirs::cache_dir().context("no cache directory on this system; set LARMOR_CACHE")?;
    Ok(base.join("larmor"))
}

/// True for `.nii` and `.nii.gz` file names (any case).
pub fn is_nifti(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    name.ends_with(".nii") || name.ends_with(".nii.gz")
}

/// Image formats larmor does not read, by file name ending, with what they are.
const OTHER_FORMATS: [(&str, &str); 6] = [
    (".hdr", "two-file NIfTI or Analyze"),
    (".img", "two-file NIfTI or Analyze"),
    (".nii.bz2", "bzip2-compressed NIfTI"),
    (".nii.zst", "zstd-compressed NIfTI"),
    (".mgz", "FreeSurfer MGH"),
    (".mgh", "FreeSurfer MGH"),
];

/// What an image file larmor cannot read is, if its name says so.
fn other_format(path: &Path) -> Option<&'static str> {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    OTHER_FORMATS.iter().find(|(ending, _)| name.ends_with(ending)).map(|(_, what)| *what)
}

/// The file name without `.nii` / `.nii.gz`: `sub-01_T1w` for `sub-01_T1w.nii.gz`.
pub fn scan_stem(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let lower = name.to_ascii_lowercase();
    for ext in [".nii.gz", ".nii"] {
        if lower.ends_with(ext) {
            return name[..name.len() - ext.len()].to_string();
        }
    }
    name
}

/// BIDS suffixes of label images rather than scans: masks, segmentations and ROIs.
const LABEL_SUFFIXES: [&str; 5] = ["mask", "dseg", "probseg", "roi", "defacemask"];

/// True when a file name's BIDS suffix (the part after the last `_`) marks a label image, like
/// `sub-01_desc-lesion_mask.nii.gz`.
pub fn is_label_image(path: &Path) -> bool {
    let stem = scan_stem(path).to_ascii_lowercase();
    stem.rsplit_once('_').is_some_and(|(_, suffix)| LABEL_SUFFIXES.contains(&suffix))
}

/// True for the grid images `larmor encode --save-preprocessed` writes
/// (`<stem>_model-<id>_desc-quantized.nii.gz`; its head mask is a label image).
pub fn is_larmor_output(path: &Path) -> bool {
    let stem = scan_stem(path);
    stem.contains("_model-") && stem.ends_with("_desc-quantized")
}

/// BIDS datatype folders that hold no anatomical scans (fMRI, diffusion, field maps, perfusion,
/// PET, electrophysiology, behaviour, microscopy...). Inside a BIDS dataset, a folder search does
/// not enter them; naming one on the command line does.
const NON_ANATOMICAL_DATATYPES: [&str; 13] =
    ["func", "dwi", "fmap", "perf", "pet", "meg", "eeg", "ieeg", "beh", "micr", "nirs", "motion", "mrs"];

/// The scans named on the command line and where their outputs mirror from.
#[derive(Debug)]
pub struct FoundScans {
    pub scans: Vec<PathBuf>,
    /// The deepest folder holding every input (a folder input itself, a file input's folder).
    /// Outputs of scans outside `sub-XX/` folders mirror their path below it.
    pub root: Option<PathBuf>,
}

/// What a folder search left out, for the notes it prints.
#[derive(Default)]
struct Skipped {
    other_formats: Vec<PathBuf>,
    not_3d: Vec<(PathBuf, Vec<usize>)>,
    datatype_folders: Vec<PathBuf>,
}

/// The NIfTI files named on the command line: files as given, folders searched recursively.
///
/// Folder searches skip hidden folders, `derivatives/`, the folders in `skip` (the run's output
/// folders) and any folder whose `dataset_description.json` says it is a derivative dataset; inside
/// a BIDS dataset they skip non-anatomical datatype folders (`func/`, `dwi/`, `fmap/`, `perf/`...).
/// Files are skipped when they are label images (`_mask`, `_dseg`, `_probseg`, `_roi`,
/// `_defacemask`), larmor's own grid images, or (from the header alone) not 3-D; a note says what
/// was left out. Sorted within each folder; duplicates removed.
pub fn find_scans(inputs: &[PathBuf], skip: &[PathBuf]) -> Result<FoundScans> {
    let skip: Vec<PathBuf> = skip.iter().filter_map(|dir| std::fs::canonicalize(dir).ok()).collect();
    let mut scans = Vec::new();
    let mut skipped = Skipped::default();
    for input in inputs {
        if input.is_dir() {
            let before = scans.len();
            let in_bids = is_bids_dataset(input) || bids_root(input).is_some();
            walk(input, in_bids, &skip, &mut scans, &mut skipped)
                .with_context(|| format!("reading {}", input.display()))?;
            if scans.len() == before {
                eprintln!("note: no .nii or .nii.gz scans under {}", input.display());
            }
        } else if input.is_file() {
            if let Some(what) = other_format(input) {
                bail!("{} is {what}, which larmor does not read; convert it to .nii.gz", input.display());
            }
            if !is_nifti(input) {
                bail!("{} is not a .nii or .nii.gz file", input.display());
            }
            scans.push(input.clone());
        } else {
            bail!("{} does not exist", input.display());
        }
    }
    print_skipped(&skipped);
    let mut seen = std::collections::HashSet::new();
    scans.retain(|scan| seen.insert(std::fs::canonicalize(scan).unwrap_or_else(|_| scan.clone())));
    Ok(FoundScans { scans, root: input_root(inputs) })
}

fn print_skipped(skipped: &Skipped) {
    let list = |paths: &mut dyn Iterator<Item = String>, total: usize| {
        let mut shown: Vec<String> = paths.take(3).collect();
        if total > 3 {
            shown.push(format!("and {} more", total - 3));
        }
        shown.join(", ")
    };
    if !skipped.not_3d.is_empty() {
        let n = skipped.not_3d.len();
        let names =
            list(&mut skipped.not_3d.iter().map(|(p, shape)| format!("{} {shape:?}", p.display())), n);
        eprintln!(
            "note: skipped {n} image{} that {} not 3-D (4-D series such as fMRI or DWI are not anatomical scans): {names}",
            if n == 1 { "" } else { "s" },
            if n == 1 { "is" } else { "are" }
        );
    }
    if !skipped.other_formats.is_empty() {
        let n = skipped.other_formats.len();
        let names = list(&mut skipped.other_formats.iter().map(|p| p.display().to_string()), n);
        eprintln!(
            "note: skipped {n} file{} in formats larmor does not read (.hdr/.img, .nii.bz2, .nii.zst, .mgz); convert \
             them to .nii.gz: {names}",
            if n == 1 { "" } else { "s" }
        );
    }
    if !skipped.datatype_folders.is_empty() {
        let n = skipped.datatype_folders.len();
        let names = list(&mut skipped.datatype_folders.iter().map(|p| p.display().to_string()), n);
        eprintln!(
            "note: skipped {n} non-anatomical BIDS folder{} (name a folder or file to include it): {names}",
            if n == 1 { "" } else { "s" }
        );
    }
}

/// True when `dir` holds a `dataset_description.json` (it is a BIDS dataset root).
fn is_bids_dataset(dir: &Path) -> bool {
    dir.join("dataset_description.json").is_file()
}

/// True when `dir` holds a `dataset_description.json` with `"DatasetType": "derivative"`.
fn is_derivative_dataset(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("dataset_description.json")) else { return false };
    serde_json::from_str::<serde_json::Value>(&text)
        .is_ok_and(|v| v["DatasetType"].as_str().is_some_and(|t| t.eq_ignore_ascii_case("derivative")))
}

fn walk(
    dir: &Path,
    in_bids: bool,
    skip: &[PathBuf],
    out: &mut Vec<PathBuf>,
    skipped: &mut Skipped,
) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> =
        std::fs::read_dir(dir)?.map(|entry| entry.map(|e| e.path())).collect::<Result<_, _>>()?;
    entries.sort();
    for path in entries {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            if name == "derivatives"
                || std::fs::canonicalize(&path).is_ok_and(|p| skip.contains(&p))
                || is_derivative_dataset(&path)
            {
                continue;
            }
            if in_bids && NON_ANATOMICAL_DATATYPES.contains(&name.as_str()) {
                skipped.datatype_folders.push(path);
                continue;
            }
            walk(&path, in_bids || is_bids_dataset(&path), skip, out, skipped)?;
        } else if is_nifti(&path) {
            if is_label_image(&path) || is_larmor_output(&path) {
                continue;
            }
            // from the header alone; an unreadable header is left for the run to report
            match larmor_volume::NiftiHeader::read(&path) {
                Ok(header) if header.shape().len() != 3 => skipped.not_3d.push((path, header.shape())),
                _ => out.push(path),
            }
        } else if other_format(&path).is_some() {
            skipped.other_formats.push(path);
        }
    }
    Ok(())
}

/// The deepest folder that holds every input: a folder input itself, a file input's folder.
pub fn input_root(inputs: &[PathBuf]) -> Option<PathBuf> {
    let folders: Vec<PathBuf> = inputs
        .iter()
        .filter_map(|input| {
            let absolute = std::path::absolute(input).ok()?;
            if input.is_dir() { Some(absolute) } else { absolute.parent().map(Path::to_path_buf) }
        })
        .collect();
    let first = folders.first()?;
    let mut root: Vec<Component> = first.components().collect();
    for folder in &folders[1..] {
        let common = root.iter().zip(folder.components()).take_while(|(a, b)| **a == *b).count();
        root.truncate(common);
    }
    (!root.is_empty()).then(|| root.iter().collect())
}

/// The root of the BIDS dataset holding `path`: the nearest ancestor folder with a
/// `dataset_description.json`, if any.
pub fn bids_root(path: &Path) -> Option<PathBuf> {
    let absolute = std::path::absolute(path).ok()?;
    absolute
        .ancestors()
        .skip(1)
        .find(|dir| dir.join("dataset_description.json").is_file())
        .map(Path::to_path_buf)
}

/// The derivatives folder for a scan when `--out` is not given: `<BIDS root>/derivatives/larmor`
/// inside a BIDS dataset, otherwise `./derivatives/larmor`.
pub fn default_out_dir(scan: &Path) -> PathBuf {
    match bids_root(scan) {
        Some(root) => root.join("derivatives").join("larmor"),
        None => PathBuf::from("derivatives").join("larmor"),
    }
}

/// Where a scan's outputs go below the derivatives folder: the `sub-XX/[ses-YY/]<datatype>/` part
/// of its path when it has one (everything from the `sub-` folder to the file's folder); else its
/// folder's path below `root` (the inputs' common folder, see [`input_root`]), so
/// `subj01/T1.nii.gz` and `subj02/T1.nii.gz` get `subj01/` and `subj02/`; and for a scan directly
/// in `root`, a folder named after the file stem.
pub fn output_subdir(scan: &Path, root: Option<&Path>) -> PathBuf {
    let parent = scan.parent().unwrap_or(Path::new(""));
    let parts: Vec<&str> = parent
        .components()
        .filter_map(|c| match c {
            Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect();
    if let Some(start) = parts.iter().rposition(|part| part.starts_with("sub-")) {
        return parts[start..].iter().collect();
    }
    let below_root = root.and_then(|root| {
        let folder = std::path::absolute(scan).ok()?.parent()?.to_path_buf();
        let relative = folder.strip_prefix(root).ok()?.to_path_buf();
        (!relative.as_os_str().is_empty()).then_some(relative)
    });
    below_root.unwrap_or_else(|| PathBuf::from(scan_stem(scan)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_drop_nifti_extensions() {
        assert_eq!(scan_stem(Path::new("a/sub-01_T1w.nii.gz")), "sub-01_T1w");
        assert_eq!(scan_stem(Path::new("scan.NII")), "scan");
        assert!(is_nifti(Path::new("x/y.Nii.Gz")));
        assert!(!is_nifti(Path::new("x/y.json")));
        assert!(is_label_image(Path::new("sub-01_desc-lesion_mask.nii.gz")));
        assert!(is_label_image(Path::new("sub-01_label-WM_probseg.nii")));
        assert!(!is_label_image(Path::new("sub-01_T1w.nii.gz")));
        assert!(!is_label_image(Path::new("masked.nii.gz")));
    }

    #[test]
    fn bids_paths_are_mirrored() {
        let scan = Path::new("/data/ds/sub-01/ses-02/anat/sub-01_ses-02_T1w.nii.gz");
        assert_eq!(output_subdir(scan, None), PathBuf::from("sub-01/ses-02/anat"));
        assert_eq!(output_subdir(scan, Some(Path::new("/data"))), PathBuf::from("sub-01/ses-02/anat"));
        let scan = Path::new("/data/ds/sub-7/anat/sub-7_T2w.nii");
        assert_eq!(output_subdir(scan, None), PathBuf::from("sub-7/anat"));
        assert_eq!(output_subdir(Path::new("/tmp/scans/brain.nii.gz"), None), PathBuf::from("brain"));
    }

    #[test]
    fn non_bids_paths_mirror_the_folders_below_the_inputs() {
        // HCP: 100307/T1w/T1w_acpc_dc_restore.nii.gz under the input root '.'
        let cwd = std::path::absolute(".").unwrap();
        let scan = Path::new("100307/T1w/T1w_acpc_dc_restore.nii.gz");
        assert_eq!(output_subdir(scan, Some(&cwd)), PathBuf::from("100307/T1w"));
        let root = Path::new("/data/study");
        assert_eq!(
            output_subdir(Path::new("/data/study/subj01/T1.nii.gz"), Some(root)),
            PathBuf::from("subj01")
        );
        assert_eq!(
            output_subdir(Path::new("/data/study/subj02/T1.nii.gz"), Some(root)),
            PathBuf::from("subj02")
        );
        // directly in the root, or outside it: the stem
        assert_eq!(output_subdir(Path::new("/data/study/T1.nii.gz"), Some(root)), PathBuf::from("T1"));
        assert_eq!(output_subdir(Path::new("/elsewhere/T1.nii.gz"), Some(root)), PathBuf::from("T1"));
    }

    #[test]
    fn the_input_root_is_the_deepest_common_folder() {
        let dir = tempfile::tempdir().unwrap();
        let root = std::path::absolute(dir.path()).unwrap();
        for sub in ["a/x", "b"] {
            std::fs::create_dir_all(root.join(sub)).unwrap();
        }
        std::fs::write(root.join("a/x/scan.nii"), b"").unwrap();
        assert_eq!(input_root(&[root.join("a"), root.join("b")]), Some(root.clone()));
        assert_eq!(input_root(&[root.join("a/x/scan.nii")]), Some(root.join("a/x")));
        assert_eq!(input_root(&[root.join("a")]), Some(root.join("a")));
        assert_eq!(input_root(&[]), None);
    }

    fn write_image(path: &Path, shape: &[usize]) {
        use larmor_volume::ndarray::{ArrayD, IxDyn};
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let data = larmor_volume::VoxelData::U8(ArrayD::zeros(IxDyn(shape)));
        larmor_volume::Nifti::new(data, &larmor_volume::Affine::IDENTITY).write(path).unwrap();
    }

    #[test]
    fn a_bids_folder_search_keeps_anatomical_3d_scans_only() {
        let dir = tempfile::tempdir().unwrap();
        let ds = dir.path().join("ds");
        std::fs::create_dir_all(&ds).unwrap();
        std::fs::write(ds.join("dataset_description.json"), r#"{"Name": "x", "BIDSVersion": "1.10.0"}"#)
            .unwrap();
        write_image(&ds.join("sub-01/anat/sub-01_T1w.nii.gz"), &[4, 4, 4]);
        write_image(&ds.join("sub-01/func/sub-01_task-rest_bold.nii.gz"), &[4, 4, 4, 5]);
        write_image(&ds.join("sub-01/func/sub-01_task-rest_sbref.nii.gz"), &[4, 4, 4]);
        write_image(&ds.join("sub-01/fmap/sub-01_phasediff.nii.gz"), &[4, 4, 4]);
        // a 4-D image outside the datatype folders is skipped on its header
        write_image(&ds.join("sub-01/anat/sub-01_inplaneT2.nii.gz"), &[4, 4, 4, 2]);
        std::fs::write(ds.join("sub-01/anat/sub-01_T2w.hdr"), b"").unwrap();
        let found = find_scans(std::slice::from_ref(&ds), &[]).unwrap();
        assert_eq!(found.scans, [ds.join("sub-01/anat/sub-01_T1w.nii.gz")]);
        // naming the folder includes it
        let found = find_scans(&[ds.join("sub-01/func")], &[]).unwrap();
        assert_eq!(found.scans, [ds.join("sub-01/func/sub-01_task-rest_sbref.nii.gz")]);
        let err = find_scans(&[ds.join("sub-01/anat/sub-01_T2w.hdr")], &[]).unwrap_err().to_string();
        assert!(err.contains("two-file NIfTI or Analyze") && err.contains(".nii.gz"), "{err}");
    }

    #[test]
    fn output_folders_and_larmor_outputs_are_not_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        write_image(&data.join("T1.nii.gz"), &[4, 4, 4]);
        write_image(&data.join("larmor/T1/T1_model-m_desc-quantized.nii.gz"), &[4, 4, 4]);
        write_image(&data.join("out2/T1/T1_model-m_desc-quantized.nii.gz"), &[4, 4, 4]);
        write_image(&data.join("out2/other.nii.gz"), &[4, 4, 4]);
        write_image(&data.join("deriv/x.nii.gz"), &[4, 4, 4]);
        std::fs::write(data.join("deriv/dataset_description.json"), r#"{"DatasetType": "derivative"}"#)
            .unwrap();
        let found = find_scans(std::slice::from_ref(&data), &[data.join("larmor")]).unwrap();
        assert_eq!(found.scans, [data.join("T1.nii.gz"), data.join("out2/other.nii.gz")]);
    }
}
