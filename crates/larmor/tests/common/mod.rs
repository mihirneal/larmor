//! Helpers for the CLI tests: a temporary workspace, synthetic head scans and a tiny model.
//!
//! Every test runs the built `larmor` binary in its own temporary folder with its own cache,
//! model folder and `HOME`, so nothing outside the folder is read or written.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use larmor_model::{Manifest, synthetic};
use larmor_volume::ndarray::{Array3, ArrayD, Ix3};
use larmor_volume::{Affine, Nifti, VoxelData};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Shape of the synthetic scans: 4 mm voxels over a 160 x 192 x 160 mm field of view.
pub const SCAN_SHAPE: [usize; 3] = [40, 48, 40];
pub const VOXEL_MM: f64 = 4.0;

/// A temporary folder the binary runs in.
pub struct Workspace {
    dir: tempfile::TempDir,
}

impl Workspace {
    pub fn new() -> Workspace {
        Workspace { dir: tempfile::tempdir().expect("temp dir") }
    }

    pub fn root(&self) -> &Path {
        self.dir.path()
    }

    pub fn path(&self, relative: &str) -> PathBuf {
        self.dir.path().join(relative)
    }

    pub fn cache(&self) -> PathBuf {
        self.path("cache")
    }

    /// `larmor <args>` in this folder, with its own cache, models folder and home.
    pub fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_larmor"));
        command
            .args(args)
            .current_dir(self.root())
            .env("LARMOR_CACHE", self.cache())
            .env("LARMOR_MODELS", self.path("models"))
            .env("HOME", self.path("home"))
            .env("NO_COLOR", "1")
            .env_remove("LARMOR_FIXTURES")
            .env_remove("HF_TOKEN");
        command
    }

    pub fn run(&self, args: &[&str]) -> Output {
        self.command(args).output().expect("run larmor")
    }

    /// Runs and asserts success; returns stdout.
    pub fn ok(&self, args: &[&str]) -> String {
        let output = self.run(args);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "larmor {args:?} failed\nstdout:\n{stdout}\nstderr:\n{stderr}");
        stdout
    }

    /// Runs and asserts failure; returns stdout and stderr together.
    pub fn fails(&self, args: &[&str]) -> String {
        let output = self.run(args);
        let text =
            format!("{}{}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        assert!(!output.status.success(), "larmor {args:?} should have failed:\n{text}");
        text
    }

    /// Writes the tiny model into `models/` and returns its weights SHA-256.
    pub fn tiny_model(&self, id: &str, seed: u64) -> String {
        write_tiny_model(&self.path("models"), id, seed)
    }
}

/// How a synthetic subject differs from the others.
#[derive(Clone, Copy, Debug)]
pub struct Subject {
    /// Head size factor around 1.
    pub scale: f64,
    /// Brightness of the lesion.
    pub lesion_intensity: f64,
    /// Lesion centre in mm from the head centre.
    pub lesion_offset: [f64; 3],
    /// Lesion radius in mm.
    pub lesion_radius: f64,
}

impl Subject {
    /// Subject `i` of a deterministic series: sizes, lesion positions and, for odd `i`, bright
    /// lesions ("patients").
    pub fn nth(i: usize) -> Subject {
        let t = i as f64;
        Subject {
            scale: 0.9 + 0.025 * t,
            lesion_intensity: if i % 2 == 1 { 1700.0 } else { 1150.0 },
            lesion_offset: [12.0 * (t * 1.3).sin(), 14.0 * (t * 0.7).cos(), 8.0 * (t * 0.9).sin()],
            lesion_radius: 12.0 + (i % 3) as f64 * 2.0,
        }
    }

    pub fn is_patient(&self) -> bool {
        self.lesion_intensity > 1500.0
    }
}

/// The voxel-to-world affine of the synthetic scans (RAS+, centred on the origin).
pub fn scan_affine() -> Affine {
    let c = SCAN_SHAPE.map(|n| -(n as f64) * VOXEL_MM / 2.0);
    Affine::from_matvec([[VOXEL_MM, 0.0, 0.0], [0.0, VOXEL_MM, 0.0], [0.0, 0.0, VOXEL_MM]], c)
}

/// World coordinates (mm) of voxel `(x, y, z)`.
fn world(x: usize, y: usize, z: usize) -> [f64; 3] {
    scan_affine().apply([x as f64, y as f64, z as f64])
}

fn in_lesion(subject: &Subject, p: [f64; 3]) -> bool {
    let d: f64 = (0..3).map(|i| (p[i] - subject.lesion_offset[i]).powi(2)).sum::<f64>().sqrt();
    d < subject.lesion_radius
}

/// A head-like int16 image: scalp, CSF, cortex, white matter, a lesion and deterministic noise.
pub fn head_image(subject: &Subject) -> Array3<i16> {
    let radii = [56.0, 68.0, 60.0].map(|r| r * subject.scale);
    let mut state = 0x2545_F491_4F6C_DD1Du64 ^ subject.scale.to_bits();
    Array3::from_shape_fn(SCAN_SHAPE, |(x, y, z)| {
        let p = world(x, y, z);
        let r = (0..3).map(|i| (p[i] / radii[i]).powi(2)).sum::<f64>().sqrt();
        let tissue = if r < 0.6 {
            900.0
        } else if r < 0.84 {
            600.0 + 60.0 * ((p[0] * 0.3).sin() + (p[1] * 0.25).cos())
        } else if r < 0.9 {
            200.0
        } else if r < 1.0 {
            450.0
        } else {
            0.0
        };
        let value = if r < 0.84 && in_lesion(subject, p) { subject.lesion_intensity } else { tissue };
        // xorshift noise, a few percent of the tissue contrast
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let noise = (state >> 40) as f64 / (1u64 << 24) as f64 * 40.0 - 20.0;
        (value + noise).max(0.0) as i16
    })
}

/// The subject's lesion as a uint8 0/1 image on the scan's grid.
pub fn lesion_image(subject: &Subject) -> Array3<u8> {
    let radii = [56.0, 68.0, 60.0].map(|r| r * subject.scale);
    Array3::from_shape_fn(SCAN_SHAPE, |(x, y, z)| {
        let p = world(x, y, z);
        let r = (0..3).map(|i| (p[i] / radii[i]).powi(2)).sum::<f64>().sqrt();
        u8::from(r < 0.84 && in_lesion(subject, p))
    })
}

pub fn write_scan(path: &Path, subject: &Subject) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    Nifti::new(VoxelData::I16(head_image(subject).into_dyn()), &scan_affine()).write(path).unwrap();
}

pub fn write_lesion(path: &Path, subject: &Subject) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    Nifti::new(VoxelData::U8(lesion_image(subject).into_dyn()), &scan_affine()).write(path).unwrap();
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A `model.toml` for a one-block ViT of width 32 on nanobrain's grid.
pub fn tiny_manifest_text(id: &str, sha: &str) -> String {
    format!(
        r#"id = "{id}"
family = "vit3d"
description = "tiny random ViT for the CLI tests"

[architecture]
grid = [192, 240, 192]
patch = 8
width = 32
depth = 1
heads = 2
mlp_ratio = 4.0
class_tokens = 1
pos_embed = "separable"
qkv = "separate"
qkv_bias = true
proj_bias = true
norm_eps = 1e-6
activation = "gelu"

[readout]
tokens = "last"
pool = "mean"

[weights]
file = "{id}.safetensors"
sha256 = "{sha}"
url = ""
naming = "nanobrain"

[[preprocessing.steps]]
op = "threshold_mask"
resolution = 2.0
sigma = 4.0
remove_islands = true
fill_holes = true

[[preprocessing.steps]]
op = "conform"
min_voxel_size = 1.0
fov = [192.0, 240.0, 192.0]
top_margin = 5.0
fixed_grid = true
order = 1

[[preprocessing.steps]]
op = "quantize"
low = 0.005
high = 0.995

[[preprocessing.steps]]
op = "normalize"
method = "max_zscore"
std_floor = 1e-6
"#
    )
}

/// Writes `<dir>/<id>.toml` and random weights `<dir>/<id>.safetensors`; returns their SHA-256.
pub fn write_tiny_model(dir: &Path, id: &str, seed: u64) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let draft = Manifest::parse(&tiny_manifest_text(id, &"0".repeat(64)), dir).unwrap();
    let weights = synthetic::random_safetensors(&draft.architecture, seed);
    std::fs::write(dir.join(format!("{id}.safetensors")), &weights).unwrap();
    let sha = sha256_hex(&weights);
    std::fs::write(dir.join(format!("{id}.toml")), tiny_manifest_text(id, &sha)).unwrap();
    sha
}

/// Reads a float32 `.npy` written by larmor: (shape, values).
pub fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00", "{}", path.display());
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    assert!(header.contains("'descr': '<f4'") && header.contains("'fortran_order': False"), "{header}");
    let shape_text = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> =
        shape_text.split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().unwrap()).collect();
    let values = bytes[10 + header_len..].as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
    (shape, values)
}

pub fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
}

pub fn read_nifti(path: &Path) -> Nifti {
    Nifti::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn as3<T: Clone>(array: &ArrayD<T>) -> Array3<T> {
    array.clone().into_dimensionality::<Ix3>().unwrap()
}

/// Files below `dir`, relative and sorted.
pub fn files_below(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p.strip_prefix(dir).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}
