//! Loading the reference files written by `tools/reference/probe_reference.py`.

#![allow(dead_code)]

use larmor_probe::{Device, DeviceChoice};
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The CPU by default; `LARMOR_TEST_DEVICE=gpu` (or `metal`, `cuda`, `cuda:N`, `auto`) runs the segmentation
/// tests on that device instead.
pub fn test_device() -> Device {
    let choice = match std::env::var("LARMOR_TEST_DEVICE") {
        Ok(name) => name.parse().unwrap_or_else(|e| panic!("LARMOR_TEST_DEVICE: {e}")),
        Err(_) => DeviceChoice::Cpu,
    };
    Device::open(choice).unwrap_or_else(|e| panic!("{e}"))
}

pub fn test_data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

pub fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/reference/probes").join(name)
}

/// A `<name>.safetensors` + `<name>.json` pair.
pub struct Reference {
    bytes: Vec<u8>,
    pub info: Value,
}

impl Reference {
    pub fn load(stem: &Path) -> Self {
        let bytes = std::fs::read(stem.with_extension("safetensors"))
            .unwrap_or_else(|e| panic!("reading {}: {e}", stem.display()));
        let info =
            serde_json::from_str(&std::fs::read_to_string(stem.with_extension("json")).unwrap()).unwrap();
        Self { bytes, info }
    }

    fn raw(&self, name: &str) -> (Dtype, Vec<usize>, Vec<u8>) {
        let tensors = SafeTensors::deserialize(&self.bytes).unwrap();
        let view = tensors.tensor(name).unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        (view.dtype(), view.shape().to_vec(), view.data().to_vec())
    }

    pub fn has(&self, name: &str) -> bool {
        SafeTensors::deserialize(&self.bytes).unwrap().tensor(name).is_ok()
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.raw(name).1
    }

    pub fn f64s(&self, name: &str) -> Vec<f64> {
        let (dtype, _, data) = self.raw(name);
        match dtype {
            Dtype::F64 => data.as_chunks::<8>().0.iter().map(|&c| f64::from_le_bytes(c)).collect(),
            Dtype::F32 => data.as_chunks::<4>().0.iter().map(|&c| f64::from(f32::from_le_bytes(c))).collect(),
            other => panic!("{name} is {other:?}, not a float"),
        }
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        let (dtype, _, data) = self.raw(name);
        match dtype {
            Dtype::F32 => data.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect(),
            Dtype::F16 => {
                data.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c).to_f32()).collect()
            }
            other => panic!("{name} is {other:?}, not f32"),
        }
    }

    pub fn f16s(&self, name: &str) -> Vec<half::f16> {
        let (dtype, _, data) = self.raw(name);
        assert_eq!(dtype, Dtype::F16, "{name}");
        data.as_chunks::<2>().0.iter().map(|&c| half::f16::from_le_bytes(c)).collect()
    }

    pub fn i64s(&self, name: &str) -> Vec<i64> {
        let (dtype, _, data) = self.raw(name);
        match dtype {
            Dtype::I64 => data.as_chunks::<8>().0.iter().map(|&c| i64::from_le_bytes(c)).collect(),
            Dtype::I32 => data.as_chunks::<4>().0.iter().map(|&c| i64::from(i32::from_le_bytes(c))).collect(),
            Dtype::U8 => data.iter().map(|&b| i64::from(b)).collect(),
            other => panic!("{name} is {other:?}, not an integer"),
        }
    }

    pub fn u8s(&self, name: &str) -> Vec<u8> {
        let (dtype, _, data) = self.raw(name);
        assert_eq!(dtype, Dtype::U8, "{name}");
        data
    }
}

pub fn json_f64s(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "lengths differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| if x.is_nan() && y.is_nan() { 0.0 } else { (x - y).abs() })
        .fold(0.0, f64::max)
}

/// Prints a compared quantity that is reported but not asserted, next to the goal it is measured against.
pub fn report(quantity: &str, diff: f64, goal: f64) {
    let status = if diff <= goal { "within goal" } else { "beyond goal (reported)" };
    println!("PARITY | {quantity:<58} | {diff:>11.3e} | {goal:>8.0e} | {status}");
}

/// Prints a compared quantity and asserts it is within tolerance.
pub fn check(quantity: &str, diff: f64, tolerance: f64) {
    let status = if diff <= tolerance { "ok" } else { "FAIL" };
    println!("PARITY | {quantity:<58} | {diff:>11.3e} | {tolerance:>8.0e} | {status}");
    assert!(diff <= tolerance, "{quantity}: max difference {diff:e} > {tolerance:e}");
}
