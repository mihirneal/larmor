//! Files written to the BIDS derivatives folder: per-scan names, `.npy` arrays, JSON sidecars and
//! the derivative `dataset_description.json`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::locate;
use crate::scan::Scan;

/// The output files of one scan: `<out>/<sub-XX/[ses-YY/]datatype, folders below the inputs, or
/// stem>/<stem>_...` (see [`locate::output_subdir`]).
#[derive(Clone, Debug)]
pub struct ScanOutputs {
    pub dir: PathBuf,
    pub stem: String,
}

impl ScanOutputs {
    /// `input_root` is the inputs' common folder ([`locate::FoundScans::root`]).
    pub fn new(out_root: &Path, scan: &Scan, input_root: Option<&Path>) -> ScanOutputs {
        ScanOutputs { dir: out_root.join(locate::output_subdir(scan, input_root)), stem: scan.stem() }
    }

    fn file(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}_{suffix}", self.stem))
    }

    /// `<stem>_model-<id>_embedding.npy`: the pooled embedding, float32 `[D]`.
    pub fn embedding(&self, model_id: &str) -> PathBuf {
        self.file(&format!("model-{model_id}_embedding.npy"))
    }

    /// `<stem>_model-<id>_run.json`.
    pub fn run_json(&self, model_id: &str) -> PathBuf {
        self.file(&format!("model-{model_id}_run.json"))
    }

    /// `<stem>_model-<id>_desc-quantized.nii.gz`: the uint16 model-grid image (`--save-preprocessed`).
    pub fn quantized(&self, model_id: &str) -> PathBuf {
        self.file(&format!("model-{model_id}_desc-quantized.nii.gz"))
    }

    /// `<stem>_model-<id>_desc-head_mask.nii.gz`: the uint8 model-grid head mask (`--save-preprocessed`).
    pub fn grid_mask(&self, model_id: &str) -> PathBuf {
        self.file(&format!("model-{model_id}_desc-head_mask.nii.gz"))
    }

    /// `<stem>_desc-<name>_mask.nii.gz`: a segmentation probe's mask in the scan's own space.
    pub fn probe_mask(&self, probe: &str) -> PathBuf {
        self.file(&format!("desc-{probe}_mask.nii.gz"))
    }

    /// `<stem>_desc-<name>_probseg.nii.gz`: a segmentation probe's probabilities on the model grid.
    pub fn probe_probseg(&self, probe: &str) -> PathBuf {
        self.file(&format!("desc-{probe}_probseg.nii.gz"))
    }

    pub fn create_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.dir).with_context(|| format!("creating {}", self.dir.display()))
    }
}

/// Writes a float32 array in numpy's `.npy` format (version 1.0, little-endian, C order), so
/// Python reads it with `np.load`.
pub fn write_npy_f32(path: &Path, values: &[f32], shape: &[usize]) -> Result<()> {
    let dims: Vec<String> = shape.iter().map(usize::to_string).collect();
    // a 1-tuple needs its trailing comma: (384,)
    let shape_text =
        if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_text}, }}");
    // magic (6) + version (2) + header length (2) + header, padded with spaces to a multiple of 64
    // bytes and ended by a newline
    let unpadded = 10 + header.len() + 1;
    header.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 4 * values.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    write_atomic(path, &bytes)
}

/// Pretty JSON with a trailing newline, written atomically.
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    write_atomic(path, (serde_json::to_string_pretty(value)? + "\n").as_bytes())
}

/// Writes through a synced, uniquely named temporary file and a rename, so readers (and an
/// interrupted run) never see half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    larmor_volume::write_atomic(path, bytes).with_context(|| format!("writing {}", path.display()))
}

/// Fails before anything is written when two scans would write the same output file. Paths are
/// compared case-insensitively: on macOS's default file system `Brain` and `brain` are the same
/// file.
pub fn check_unique_outputs(targets: &[(&Path, PathBuf)]) -> Result<()> {
    let mut seen: HashMap<String, &Path> = HashMap::new();
    for (scan, target) in targets {
        let folded = target.to_string_lossy().to_lowercase();
        if let Some(other) = seen.insert(folded, scan) {
            bail!(
                "{} and {} would both write {} (file names that differ only in case are the same file on \
                 macOS); pass their folders together so the outputs mirror them, or use different --out folders",
                other.display(),
                scan.display(),
                target.display()
            );
        }
    }
    Ok(())
}

/// Writes the BIDS derivative `dataset_description.json` at the derivatives root, unless one is
/// already there.
pub fn ensure_dataset_description(out_root: &Path) -> Result<()> {
    let path = out_root.join("dataset_description.json");
    if path.exists() {
        return Ok(());
    }
    let description = serde_json::json!({
        "Name": "Larmor foundation-model embeddings",
        "BIDSVersion": "1.10.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{
            "Name": "larmor",
            "Version": env!("CARGO_PKG_VERSION"),
            "CodeURL": env!("CARGO_PKG_REPOSITORY"),
        }],
    });
    write_json(&path, &description)
}

/// TSV cells cannot hold tabs or newlines.
pub fn tsv_cell(text: &str) -> String {
    if text.is_empty() { "n/a".into() } else { text.replace(['\t', '\n', '\r'], " ") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_differing_only_in_case_collide() {
        let (a, b) = (Path::new("a/Brain.nii.gz"), Path::new("b/brain.nii.gz"));
        let targets =
            [(a, PathBuf::from("out/Brain/Brain_x.npy")), (b, PathBuf::from("out/brain/brain_x.npy"))];
        let err = check_unique_outputs(&targets).unwrap_err().to_string();
        assert!(err.contains("a/Brain.nii.gz and b/brain.nii.gz"), "{err}");
        let distinct = [(a, PathBuf::from("out/a/Brain_x.npy")), (b, PathBuf::from("out/b/brain_x.npy"))];
        check_unique_outputs(&distinct).unwrap();
    }

    #[test]
    fn npy_header_is_aligned_and_parsable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.npy");
        write_npy_f32(&path, &[1.0, 2.5, -3.0], &[3]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
        assert!(header.contains("'shape': (3,)") && header.ends_with('\n'), "{header}");
        assert_eq!(bytes.len(), 10 + header_len + 12);
        assert_eq!(f32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap()), -3.0);
    }
}
