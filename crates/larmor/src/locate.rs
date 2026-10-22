//! Where things are: model manifests, the embedding cache, input scans, and the BIDS layout of the
//! outputs.

use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

use crate::scan::Scan;

/// Directories searched for `model.toml` files by id, in order: each entry of `$LARMOR_MODELS`
/// (colon-separated), the user's `~/Library/Application Support/larmor/models`, then `./models`.
/// A folder named twice is searched once.
pub fn model_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(list) = std::env::var_os("LARMOR_MODELS") {
        dirs.extend(std::env::split_paths(&list).filter(|p| !p.as_os_str().is_empty()));
    }
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join("Library/Application Support/larmor/models"));
    }
    dirs.push(PathBuf::from("models"));
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|dir| seen.insert(std::fs::canonicalize(dir).unwrap_or_else(|_| dir.clone())));
    dirs
}

/// The embedding cache root: `$LARMOR_CACHE`, else `~/Library/Caches/larmor`.
pub fn cache_root() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("LARMOR_CACHE").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(dir));
    }
    let home = dirs::home_dir().context("no home folder for the cache; set LARMOR_CACHE")?;
    Ok(home.join("Library/Caches/larmor"))
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
    pub scans: Vec<Scan>,
    /// The deepest folder holding every input (a folder input itself, a file input's folder).
    /// Outputs of scans outside `sub-XX/` folders mirror their path below it.
    pub root: Option<PathBuf>,
}

/// What a folder search left out, for the notes it prints.
#[derive(Default)]
pub struct Skipped {
    other_formats: Vec<PathBuf>,
    not_3d: Vec<(PathBuf, Vec<usize>)>,
    datatype_folders: Vec<PathBuf>,
    /// DICOM series that are not one 3-D volume: `folder (reason)`.
    pub dicom_series: Vec<String>,
    /// DICOM files without an image (reports, presentation states) or unreadable, with the reason.
    dicom_files: Vec<(PathBuf, String)>,
}

/// A file a folder walk found, from its name alone.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Candidate {
    /// A `.nii` / `.nii.gz` file that is not a label image or a larmor output.
    Nifti(PathBuf),
    /// A file that may be DICOM: its first bytes and header tell.
    MaybeDicom(PathBuf),
}

impl Candidate {
    pub fn path(&self) -> &Path {
        match self {
            Candidate::Nifti(path) | Candidate::MaybeDicom(path) => path,
        }
    }
}

/// What a candidate file turned out to be, from its header (no voxels are read).
#[derive(Clone, Debug)]
pub enum Analysis {
    /// A 3-D NIfTI image (or one whose header the run will report as unreadable).
    Nifti,
    /// A NIfTI image that is not 3-D.
    NotThreeD(Vec<usize>),
    /// One DICOM image.
    Dicom(Box<larmor_dicom::SliceHeader>),
    /// A DICOM file without an image (a report, presentation state...) or unreadable.
    DicomWithoutImage(String),
    /// Not DICOM.
    Other,
}

/// Reads the header of a candidate file.
pub fn analyze(candidate: &Candidate) -> Analysis {
    match candidate {
        // an unreadable header is left for the run to report
        Candidate::Nifti(path) => match larmor_volume::NiftiHeader::read(path) {
            Ok(header) if header.shape().len() != 3 => Analysis::NotThreeD(header.shape()),
            _ => Analysis::Nifti,
        },
        Candidate::MaybeDicom(path) if larmor_dicom::is_dicom_file(path) => {
            match larmor_dicom::SliceHeader::read(path) {
                Ok(header) if header.rows > 0 && header.columns > 0 => Analysis::Dicom(Box::new(header)),
                Ok(_) => Analysis::DicomWithoutImage("no image".into()),
                Err(larmor_dicom::DicomError::Missing { what, .. }) => {
                    Analysis::DicomWithoutImage(format!("no {what}"))
                }
                Err(err) => Analysis::DicomWithoutImage(err.to_string()),
            }
        }
        Candidate::MaybeDicom(_) => Analysis::Other,
    }
}

/// File name endings that are never DICOM, so a folder search does not open them (on Box Drive and
/// other cloud folders, opening a file downloads it).
const NOT_DICOM: [&str; 24] = [
    ".json",
    ".tsv",
    ".csv",
    ".txt",
    ".md",
    ".html",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bval",
    ".bvec",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".zst",
    ".npy",
    ".npz",
    ".safetensors",
    ".toml",
    ".log",
];

fn may_be_dicom(path: &Path) -> bool {
    let name = path.file_name().map(|n| n.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    name != "dicomdir" && !NOT_DICOM.iter().any(|ending| name.ends_with(ending))
}

/// The scans named on the command line: NIfTI files as given, folders searched recursively for
/// NIfTI files and DICOM series.
///
/// Folder searches skip hidden folders, `derivatives/`, the folders in `skip` (the run's output
/// folders) and any folder whose `dataset_description.json` says it is a derivative dataset; inside
/// a BIDS dataset they skip non-anatomical datatype folders (`func/`, `dwi/`, `fmap/`, `perf/`...)
/// and `sourcedata/`. NIfTI files are skipped when they are label images (`_mask`, `_dseg`,
/// `_probseg`, `_roi`, `_defacemask`), larmor's own grid images, or (from the header alone) not 3-D.
/// DICOM files are grouped into series by SeriesInstanceUID (from their headers, without pixel
/// data); a series that is not one 3-D volume (a localizer, a 4-D series...) is skipped. A note says
/// what was left out. Sorted within each folder; duplicates removed.
pub fn find_scans(inputs: &[PathBuf], skip: &[PathBuf]) -> Result<FoundScans> {
    let skip: Vec<PathBuf> = skip.iter().filter_map(|dir| std::fs::canonicalize(dir).ok()).collect();
    let mut scans = Vec::new();
    let mut skipped = Skipped::default();
    for input in inputs {
        if input.is_dir() {
            let found = candidates(input, &skip, &mut skipped)
                .with_context(|| format!("reading {}", input.display()))?;
            let analyzed: Vec<(Candidate, Analysis)> =
                found.into_par_iter().map(|c| (c.clone(), analyze(&c))).collect();
            let before = scans.len();
            scans.extend(assemble(analyzed, &mut skipped));
            if scans.len() == before {
                eprintln!("note: no .nii, .nii.gz or DICOM scans under {}", input.display());
            }
        } else if input.is_file() {
            if let Some(what) = other_format(input) {
                bail!("{} is {what}, which larmor does not read; convert it to .nii.gz", input.display());
            }
            if larmor_dicom::is_dicom_file(input) {
                bail!(
                    "{} is one DICOM file; name the folder of its series (larmor reads a DICOM series from its \
                     folder)",
                    input.display()
                );
            }
            if !is_nifti(input) {
                bail!("{} is not a .nii or .nii.gz file, or a folder of DICOM files", input.display());
            }
            scans.push(Scan::Nifti(input.clone()));
        } else {
            bail!("{} does not exist", input.display());
        }
    }
    print_skipped(&skipped);
    let mut seen = std::collections::HashSet::new();
    scans.retain(|scan| {
        let key = match scan {
            Scan::Nifti(path) => std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()),
            Scan::Dicom(dicom) => PathBuf::from(format!("dicom:{}", dicom.series.uid)),
        };
        seen.insert(key)
    });
    Ok(FoundScans { scans, root: input_root(inputs) })
}

/// The candidate files below `dir`, in walk order (sorted within each folder), by name alone; see
/// [`find_scans`] for what is skipped.
pub fn candidates(dir: &Path, skip: &[PathBuf], skipped: &mut Skipped) -> std::io::Result<Vec<Candidate>> {
    let mut found = Vec::new();
    let in_bids = is_bids_dataset(dir) || bids_root(dir).is_some();
    walk(dir, in_bids, skip, &mut found, skipped)?;
    Ok(found)
}

/// The scans among analysed candidates, in walk order: 3-D NIfTI files, and DICOM series (grouped by
/// SeriesInstanceUID) at the place of their first file. What is left out goes to `skipped`.
pub fn assemble(analyzed: Vec<(Candidate, Analysis)>, skipped: &mut Skipped) -> Vec<Scan> {
    let mut placed: Vec<(usize, Scan)> = Vec::new();
    let mut position_of = std::collections::HashMap::new();
    let mut images = Vec::new();
    for (i, (candidate, analysis)) in analyzed.into_iter().enumerate() {
        let path = candidate.path().to_path_buf();
        match analysis {
            Analysis::Nifti => placed.push((i, Scan::Nifti(path))),
            Analysis::NotThreeD(shape) => skipped.not_3d.push((path, shape)),
            Analysis::Dicom(header) => {
                position_of.insert(path, i);
                images.push(*header);
            }
            Analysis::DicomWithoutImage(why) => skipped.dicom_files.push((path, why)),
            Analysis::Other => {}
        }
    }
    let mut usable = Vec::new();
    for series in larmor_dicom::group_series(images) {
        match series.check() {
            Ok(()) => usable.push(series),
            Err(err) => skipped.dicom_series.push(format!("{} ({err})", series.folder().display())),
        }
    }
    // series that share a folder are told apart by SeriesNumber, and by the end of their UID when
    // their numbers repeat (or are missing)
    let mut numbers: std::collections::HashMap<PathBuf, Vec<Option<i32>>> = std::collections::HashMap::new();
    for series in &usable {
        numbers.entry(series.folder()).or_default().push(series.number());
    }
    for series in usable {
        let first =
            series.headers.iter().filter_map(|h| position_of.get(&h.path)).min().copied().unwrap_or(0);
        let in_folder = &numbers[&series.folder()];
        let id = (in_folder.len() > 1).then(|| {
            let tail: String = series.uid.chars().skip(series.uid.len().saturating_sub(6)).collect();
            match series.number() {
                Some(n) if in_folder.iter().filter(|&&m| m == Some(n)).count() == 1 => n.to_string(),
                Some(n) => format!("{n}-{tail}"),
                None => tail,
            }
        });
        placed.push((first, Scan::dicom(series, id)));
    }
    placed.sort_by_key(|(i, _)| *i);
    placed.into_iter().map(|(_, scan)| scan).collect()
}

fn print_skipped(skipped: &Skipped) {
    for note in skipped.notes() {
        eprintln!("note: {note}");
    }
}

impl Skipped {
    /// One line per kind of thing left out (every skipped DICOM series on its own line, since each
    /// reason is worth reading).
    pub fn notes(&self) -> Vec<String> {
        let list = |paths: &mut dyn Iterator<Item = String>, total: usize| {
            let mut shown: Vec<String> = paths.take(3).collect();
            if total > 3 {
                shown.push(format!("and {} more", total - 3));
            }
            shown.join(", ")
        };
        let plural = |n: usize| if n == 1 { "" } else { "s" };
        let mut notes = Vec::new();
        if !self.not_3d.is_empty() {
            let n = self.not_3d.len();
            let names =
                list(&mut self.not_3d.iter().map(|(p, shape)| format!("{} {shape:?}", p.display())), n);
            notes.push(format!(
                "skipped {n} image{} that {} not 3-D (4-D series such as fMRI or DWI are not anatomical scans): \
                 {names}",
                plural(n),
                if n == 1 { "is" } else { "are" }
            ));
        }
        if !self.other_formats.is_empty() {
            let n = self.other_formats.len();
            let names = list(&mut self.other_formats.iter().map(|p| p.display().to_string()), n);
            notes.push(format!(
                "skipped {n} file{} in formats larmor does not read (.hdr/.img, .nii.bz2, .nii.zst, .mgz); \
                 convert them to .nii.gz: {names}",
                plural(n)
            ));
        }
        if !self.datatype_folders.is_empty() {
            let n = self.datatype_folders.len();
            let names = list(&mut self.datatype_folders.iter().map(|p| p.display().to_string()), n);
            notes.push(format!(
                "skipped {n} non-anatomical BIDS folder{} or sourcedata/ (name a folder or file to include \
                 it): {names}",
                plural(n)
            ));
        }
        for series in &self.dicom_series {
            notes.push(format!("skipped the DICOM series in {series}"));
        }
        if !self.dicom_files.is_empty() {
            let n = self.dicom_files.len();
            let names =
                list(&mut self.dicom_files.iter().map(|(p, why)| format!("{} ({why})", p.display())), n);
            notes
                .push(format!("skipped {n} DICOM file{} without an image or unreadable: {names}", plural(n)));
        }
        notes
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

/// Name endings of files that are still being written or synced (browsers, Box Drive, rsync).
const PARTIAL: [&str; 5] = [".part", ".partial", ".tmp", ".download", ".crdownload"];

fn walk(
    dir: &Path,
    in_bids: bool,
    skip: &[PathBuf],
    out: &mut Vec<Candidate>,
    skipped: &mut Skipped,
) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> =
        std::fs::read_dir(dir)?.map(|entry| entry.map(|e| e.path())).collect::<Result<_, _>>()?;
    entries.sort();
    for path in entries {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let lower = name.to_ascii_lowercase();
        if name.starts_with('.') || name.starts_with("~$") || PARTIAL.iter().any(|e| lower.ends_with(e)) {
            continue;
        }
        if path.is_dir() {
            if name == "derivatives"
                || std::fs::canonicalize(&path).is_ok_and(|p| skip.contains(&p))
                || is_derivative_dataset(&path)
            {
                continue;
            }
            // BIDS sourcedata/ holds what the dataset was converted from (DICOM, say)
            if in_bids && (NON_ANATOMICAL_DATATYPES.contains(&name.as_str()) || name == "sourcedata") {
                skipped.datatype_folders.push(path);
                continue;
            }
            walk(&path, in_bids || is_bids_dataset(&path), skip, out, skipped)?;
        } else if is_nifti(&path) {
            if !(is_label_image(&path) || is_larmor_output(&path)) {
                out.push(Candidate::Nifti(path));
            }
        } else if other_format(&path).is_some() {
            skipped.other_formats.push(path);
        } else if may_be_dicom(&path) {
            out.push(Candidate::MaybeDicom(path));
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
/// in `root`, a folder named after its stem. A DICOM series counts as a file named after its
/// folder, in that folder's parent ([`Scan::location`]).
pub fn output_subdir(scan: &Scan, root: Option<&Path>) -> PathBuf {
    let location = scan.location();
    let path = location.as_path();
    let parent = path.parent().unwrap_or(Path::new(""));
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
        let folder = std::path::absolute(path).ok()?.parent()?.to_path_buf();
        let relative = folder.strip_prefix(root).ok()?.to_path_buf();
        (!relative.as_os_str().is_empty()).then_some(relative)
    });
    below_root.unwrap_or_else(|| PathBuf::from(scan.stem()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nifti(path: &Path) -> Scan {
        Scan::Nifti(path.to_path_buf())
    }

    fn paths(found: &FoundScans) -> Vec<PathBuf> {
        found.scans.iter().map(|s| s.path().to_path_buf()).collect()
    }

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
        assert_eq!(output_subdir(&nifti(scan), None), PathBuf::from("sub-01/ses-02/anat"));
        assert_eq!(
            output_subdir(&nifti(scan), Some(Path::new("/data"))),
            PathBuf::from("sub-01/ses-02/anat")
        );
        let scan = Path::new("/data/ds/sub-7/anat/sub-7_T2w.nii");
        assert_eq!(output_subdir(&nifti(scan), None), PathBuf::from("sub-7/anat"));
        assert_eq!(output_subdir(&nifti(Path::new("/tmp/scans/brain.nii.gz")), None), PathBuf::from("brain"));
    }

    #[test]
    fn non_bids_paths_mirror_the_folders_below_the_inputs() {
        // HCP: 100307/T1w/T1w_acpc_dc_restore.nii.gz under the input root '.'
        let cwd = std::path::absolute(".").unwrap();
        let scan = Path::new("100307/T1w/T1w_acpc_dc_restore.nii.gz");
        assert_eq!(output_subdir(&nifti(scan), Some(&cwd)), PathBuf::from("100307/T1w"));
        let root = Path::new("/data/study");
        assert_eq!(
            output_subdir(&nifti(Path::new("/data/study/subj01/T1.nii.gz")), Some(root)),
            PathBuf::from("subj01")
        );
        assert_eq!(
            output_subdir(&nifti(Path::new("/data/study/subj02/T1.nii.gz")), Some(root)),
            PathBuf::from("subj02")
        );
        // directly in the root, or outside it: the stem
        assert_eq!(
            output_subdir(&nifti(Path::new("/data/study/T1.nii.gz")), Some(root)),
            PathBuf::from("T1")
        );
        assert_eq!(output_subdir(&nifti(Path::new("/elsewhere/T1.nii.gz")), Some(root)), PathBuf::from("T1"));
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
        assert_eq!(paths(&found), [ds.join("sub-01/anat/sub-01_T1w.nii.gz")]);
        // naming the folder includes it
        let found = find_scans(&[ds.join("sub-01/func")], &[]).unwrap();
        assert_eq!(paths(&found), [ds.join("sub-01/func/sub-01_task-rest_sbref.nii.gz")]);
        let err = find_scans(&[ds.join("sub-01/anat/sub-01_T2w.hdr")], &[]).unwrap_err().to_string();
        assert!(err.contains("two-file NIfTI or Analyze") && err.contains(".nii.gz"), "{err}");
    }

    #[test]
    fn series_sharing_a_folder_and_a_number_get_distinct_names() {
        use larmor_dicom::synthetic::{SeriesSpec, write_series};
        let dir = tempfile::tempdir().unwrap();
        let volume = larmor_volume::ndarray::Array3::from_shape_fn((6, 6, 4), |(x, y, z)| (x * y + z) as i16);
        let a = SeriesSpec { series_uid: "2.25.1000001".into(), ..SeriesSpec::axial([0.0; 3], [1.0; 3]) };
        write_series(&dir.path().join("flat"), &volume, &a).unwrap();
        // a second series with the same SeriesNumber, written next to the first under other names
        let b = SeriesSpec { series_uid: "2.25.2000002".into(), ..a.clone() };
        let tmp = dir.path().join("tmp");
        for (i, file) in write_series(&tmp, &volume, &b).unwrap().iter().enumerate() {
            std::fs::rename(file, dir.path().join(format!("flat/b_{i}.dcm"))).unwrap();
        }
        let found = find_scans(&[dir.path().join("flat")], &[]).unwrap();
        let stems: Vec<String> = found.scans.iter().map(Scan::stem).collect();
        assert_eq!(stems, ["flat_series-3-000002", "flat_series-3-000001"]);
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
        assert_eq!(paths(&found), [data.join("T1.nii.gz"), data.join("out2/other.nii.gz")]);
    }
}
