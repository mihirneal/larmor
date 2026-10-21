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

/// The NIfTI files named on the command line: files as given, folders searched recursively.
/// Folder searches skip hidden folders, `derivatives/` (outputs, not inputs) and label images
/// (`_mask`, `_dseg`, `_probseg`, `_roi`, `_defacemask`). Sorted within each folder; duplicates
/// removed.
pub fn find_scans(inputs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut scans = Vec::new();
    for input in inputs {
        if input.is_dir() {
            let before = scans.len();
            walk(input, &mut scans).with_context(|| format!("reading {}", input.display()))?;
            if scans.len() == before {
                eprintln!("note: no .nii or .nii.gz files under {}", input.display());
            }
        } else if input.is_file() {
            if !is_nifti(input) {
                bail!("{} is not a .nii or .nii.gz file", input.display());
            }
            scans.push(input.clone());
        } else {
            bail!("{} does not exist", input.display());
        }
    }
    let mut seen = std::collections::HashSet::new();
    scans.retain(|scan| seen.insert(std::fs::canonicalize(scan).unwrap_or_else(|_| scan.clone())));
    Ok(scans)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> =
        std::fs::read_dir(dir)?.map(|entry| entry.map(|e| e.path())).collect::<Result<_, _>>()?;
    entries.sort();
    for path in entries {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if path.is_dir() {
            if !name.starts_with('.') && name != "derivatives" {
                walk(&path, out)?;
            }
        } else if is_nifti(&path) && !name.starts_with('.') && !is_label_image(&path) {
            out.push(path);
        }
    }
    Ok(())
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
/// of its path when it has one (everything from the `sub-` folder to the file's folder), else a
/// folder named after the file stem.
pub fn output_subdir(scan: &Path) -> PathBuf {
    let parent = scan.parent().unwrap_or(Path::new(""));
    let parts: Vec<&str> = parent
        .components()
        .filter_map(|c| match c {
            Component::Normal(part) => part.to_str(),
            _ => None,
        })
        .collect();
    match parts.iter().rposition(|part| part.starts_with("sub-")) {
        Some(start) => parts[start..].iter().collect(),
        None => PathBuf::from(scan_stem(scan)),
    }
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
        assert_eq!(output_subdir(scan), PathBuf::from("sub-01/ses-02/anat"));
        let scan = Path::new("/data/ds/sub-7/anat/sub-7_T2w.nii");
        assert_eq!(output_subdir(scan), PathBuf::from("sub-7/anat"));
        assert_eq!(output_subdir(Path::new("/tmp/scans/brain.nii.gz")), PathBuf::from("brain"));
    }
}
