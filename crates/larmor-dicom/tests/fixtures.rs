//! The DICOM fixtures (`[[dicom]]` in `$LARMOR_FIXTURES/manifest.toml`, default
//! `/Users/mihir/Developer/larmor/fixtures`): every series converts to dcm2niix's NIfTI of it bit
//! for bit, what dcm2niix cannot make one volume of is refused, and every transfer syntax decodes
//! to the same samples. `larmor check-parity` runs the same checks and the recipe after them.
//!
//! ```text
//! cargo test --release -p larmor-dicom --test fixtures -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use larmor_dicom::{DicomError, SliceHeader, convert_files, decode_pixels};
use larmor_volume::Nifti;

fn fixtures_root() -> PathBuf {
    std::env::var_os("LARMOR_FIXTURES")
        .map_or_else(|| PathBuf::from("/Users/mihir/Developer/larmor/fixtures"), PathBuf::from)
}

fn files(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap();
                out.push((path, bytes));
            }
        }
    }
    out.sort();
    out
}

#[test]
#[ignore = "needs the reference fixtures; run with --ignored --release"]
fn dicom_fixtures_convert_like_dcm2niix() {
    let root = fixtures_root();
    let manifest: toml::Table = std::fs::read_to_string(root.join("manifest.toml")).unwrap().parse().unwrap();
    let scans = manifest["scan"].as_array().unwrap();
    let entries = manifest["dicom"].as_array().expect("the manifest lists [[dicom]] sets");
    for entry in entries {
        let field = |key: &str| entry[key].as_str().unwrap_or_default().to_string();
        let name = field("name");
        let files = files(&root.join(field("folder")));
        match field("kind").as_str() {
            "series" => {
                let scan = scans.iter().find(|s| s["name"].as_str() == Some(&field("scan"))).unwrap();
                let reference = Nifti::read(root.join(scan["file"].as_str().unwrap())).unwrap();
                let converted = convert_files(&files).unwrap_or_else(|e| panic!("{name}: {e}"));
                let ours = &converted.nifti;
                assert_eq!(ours.shape(), reference.shape(), "{name}");
                assert_eq!(ours.data, reference.data, "{name}: voxels");
                assert_eq!(ours.affine().unwrap(), reference.affine().unwrap(), "{name}: affine");
                assert_eq!(ours.zooms(), reference.zooms(), "{name}: zooms");
                assert_eq!(ours.header.scaling(), reference.header.scaling(), "{name}: scaling");
                println!("{name}: {:?} {} bit-exact", ours.shape(), ours.data.dtype().name());
            }
            "refuse" => match convert_files(&files) {
                Err(DicomError::Unsupported(message)) => {
                    assert!(message.contains(&field("refuse")), "{name}: {message}");
                    println!("{name}: refused ({message})");
                }
                other => {
                    panic!("{name}: expected a refusal, got {:?}", other.map(|c| c.nifti.shape().to_vec()))
                }
            },
            _ => {
                let unsupported: Vec<String> = entry["unsupported"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();
                let mut first: Option<Vec<u8>> = None;
                for (path, bytes) in &files {
                    let header = SliceHeader::from_bytes(path, bytes).unwrap();
                    let decoded = decode_pixels(path, bytes);
                    if unsupported.contains(&header.transfer_syntax) {
                        assert!(decoded.is_err(), "{}: should be refused", path.display());
                        continue;
                    }
                    let samples = decoded.unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                    match &first {
                        None => first = Some(samples),
                        Some(expected) => assert_eq!(&samples, expected, "{}", path.display()),
                    }
                }
                println!("{name}: {} files decode alike", files.len());
            }
        }
    }
}
