//! Files written to the BIDS derivatives folder: per-scan names, `.npy` arrays, JSON sidecars and
//! the derivative `dataset_description.json`.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::locate;

/// The output files of one scan: `<out>/<sub-XX/[ses-YY/]datatype or stem>/<stem>_...`.
#[derive(Clone, Debug)]
pub struct ScanOutputs {
    pub dir: PathBuf,
    pub stem: String,
}

impl ScanOutputs {
    pub fn new(out_root: &Path, scan: &Path) -> ScanOutputs {
        ScanOutputs { dir: out_root.join(locate::output_subdir(scan)), stem: locate::scan_stem(scan) }
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

/// Writes through a temporary file and a rename, so readers never see half a file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    let mut file = std::fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
    file.write_all(bytes).with_context(|| format!("writing {}", tmp.display()))?;
    drop(file);
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
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
