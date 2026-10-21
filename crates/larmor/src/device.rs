//! `--device` and `--precision`: parsing, resolving `auto`, and naming them in outputs.
//!
//! Only `larmor_model`'s `DeviceChoice` and `Precision` variants are used, so the CLI does not depend
//! on which backend larmor-model runs on.

use std::fmt;
use std::str::FromStr;

use larmor_model::{DeviceChoice, Precision};

/// What `--device` accepts: `auto`, `cpu`, `metal`, `cuda` or `cuda:N`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceArg(pub DeviceChoice);

impl FromStr for DeviceArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let lower = s.to_ascii_lowercase();
        let choice = match lower.as_str() {
            "auto" => DeviceChoice::Auto,
            "cpu" => DeviceChoice::Cpu,
            "metal" => DeviceChoice::Metal,
            "cuda" => DeviceChoice::Cuda(0),
            _ => match lower.strip_prefix("cuda:").map(str::parse) {
                Some(Ok(index)) => DeviceChoice::Cuda(index),
                _ => return Err(format!("unknown device {s:?} (expected auto, cpu, metal, cuda or cuda:N)")),
            },
        };
        Ok(DeviceArg(choice))
    }
}

impl fmt::Display for DeviceArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            DeviceChoice::Cpu => f.write_str("cpu"),
            DeviceChoice::Metal => f.write_str("metal"),
            DeviceChoice::Cuda(index) => write!(f, "cuda:{index}"),
            DeviceChoice::Auto => f.write_str("auto"),
            // larmor-model may know devices the CLI never asks for
            #[allow(unreachable_patterns)]
            other => write!(f, "{other:?}"),
        }
    }
}

impl DeviceArg {
    /// The device `auto` means on this build: Metal on macOS, CUDA when built with the `cuda`
    /// feature, otherwise the CPU. Other choices are returned unchanged.
    pub fn resolve(self) -> DeviceArg {
        if self.0 != DeviceChoice::Auto {
            return self;
        }
        if cfg!(target_os = "macos") {
            DeviceArg(DeviceChoice::Metal)
        } else if cfg!(feature = "cuda") {
            DeviceArg(DeviceChoice::Cuda(0))
        } else {
            DeviceArg(DeviceChoice::Cpu)
        }
    }
}

/// What `--precision` accepts: `f32` (the parity target), `f16` or `bf16`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrecisionArg(pub Precision);

impl FromStr for PrecisionArg {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(PrecisionArg(Precision::F32)),
            "f16" | "fp16" => Ok(PrecisionArg(Precision::F16)),
            "bf16" => Ok(PrecisionArg(Precision::Bf16)),
            _ => Err(format!("unknown precision {s:?} (expected f32, f16 or bf16)")),
        }
    }
}

impl fmt::Display for PrecisionArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self.0 {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
        })
    }
}

/// The device larmor-probe's segmentation solver runs on.
///
/// API gap: larmor-probe takes its backend's own device type and has no constructor from a
/// [`DeviceChoice`], so the CLI cannot honour `--device` for segmentation fits without naming the
/// backend crate. Until it does, this returns the backend's default device (the GPU when one is
/// compiled in). It is the one place to change when larmor-probe gains such a constructor.
pub fn segmentation_device<D: Default>(_requested: DeviceArg) -> D {
    D::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_parse_and_print() {
        for text in ["auto", "cpu", "metal", "cuda:2"] {
            assert_eq!(text.parse::<DeviceArg>().unwrap().to_string(), text);
        }
        assert_eq!("CUDA".parse::<DeviceArg>().unwrap().0, DeviceChoice::Cuda(0));
        assert!("wgpu".parse::<DeviceArg>().is_err());
        assert!("cuda:x".parse::<DeviceArg>().is_err());
    }

    #[test]
    fn precisions_parse_and_print() {
        for text in ["f32", "f16", "bf16"] {
            assert_eq!(text.parse::<PrecisionArg>().unwrap().to_string(), text);
        }
        assert!("int8".parse::<PrecisionArg>().is_err());
    }

    #[test]
    fn auto_resolves_to_a_concrete_device() {
        assert_ne!(DeviceArg(DeviceChoice::Auto).resolve().0, DeviceChoice::Auto);
        assert_eq!(DeviceArg(DeviceChoice::Cpu).resolve().0, DeviceChoice::Cpu);
    }
}
