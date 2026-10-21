//! Picking a Burn device and a compute precision.
//!
//! Burn 0.22 puts the backend in the [`Device`] rather than in a type parameter, so choosing a
//! backend is a runtime decision among the ones compiled in by Cargo features:
//!
//! | Feature | Device | Notes |
//! |---|---|---|
//! | `metal` (default) | [`DeviceChoice::Metal`] | wgpu runtime, kernels compiled to native MSL |
//! | `wgpu` | [`DeviceChoice::Wgpu`] | wgpu runtime, WGSL (on macOS this also runs on Metal) |
//! | `cuda` | [`DeviceChoice::Cuda`] | Linux/NVIDIA |
//! | `flex` (default) | [`DeviceChoice::Cpu`] | pure-Rust CPU backend, used by the tests |

use std::fmt;
use std::str::FromStr;

#[allow(unused_imports)] // which ones are used depends on the enabled backends
use burn::tensor::{DType, Device, DeviceKind, DeviceType, FloatDType};

use crate::error::EncoderError;

/// Which compute device runs the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceChoice {
    /// The best compiled-in device present on this machine: CUDA, then Metal, then wgpu, then CPU.
    #[default]
    Auto,
    /// The CPU (Burn's Flex backend).
    Cpu,
    /// Apple GPU through Metal (MSL kernels).
    Metal,
    /// Any GPU through wgpu's portable shader path.
    Wgpu,
    /// NVIDIA GPU by index.
    Cuda(usize),
}

/// Numeric precision of the encoder's matrix multiplies.
///
/// `F32` is the parity target. The half precisions mirror PyTorch autocast, which is how
/// nanobrain's eval framework runs: linears and attention run in the half type while the residual
/// stream, LayerNorms and position embeddings stay in f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Everything in f32.
    #[default]
    F32,
    /// Linears and attention in IEEE half precision.
    F16,
    /// Linears and attention in bfloat16.
    Bf16,
}

impl Precision {
    /// The dtype linears and attention compute in.
    pub fn compute_dtype(self) -> FloatDType {
        match self {
            Precision::F32 => FloatDType::F32,
            Precision::F16 => FloatDType::F16,
            Precision::Bf16 => FloatDType::BF16,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
        })
    }
}

impl FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "f32" | "fp32" => Ok(Precision::F32),
            "f16" | "fp16" => Ok(Precision::F16),
            "bf16" => Ok(Precision::Bf16),
            other => Err(format!("unknown precision {other:?} (expected f32, f16 or bf16)")),
        }
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceChoice::Auto => f.write_str("auto"),
            DeviceChoice::Cpu => f.write_str("cpu"),
            DeviceChoice::Metal => f.write_str("metal"),
            DeviceChoice::Wgpu => f.write_str("wgpu"),
            DeviceChoice::Cuda(index) => write!(f, "cuda:{index}"),
        }
    }
}

impl FromStr for DeviceChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_ascii_lowercase();
        match s.as_str() {
            "auto" => Ok(DeviceChoice::Auto),
            "cpu" => Ok(DeviceChoice::Cpu),
            "metal" => Ok(DeviceChoice::Metal),
            "wgpu" => Ok(DeviceChoice::Wgpu),
            "cuda" => Ok(DeviceChoice::Cuda(0)),
            _ => match s.strip_prefix("cuda:").map(str::parse) {
                Some(Ok(index)) => Ok(DeviceChoice::Cuda(index)),
                _ => Err(format!("unknown device {s:?} (expected auto, cpu, metal, wgpu or cuda[:N])")),
            },
        }
    }
}

impl DeviceChoice {
    /// Resolve `Auto` to a concrete choice. Returns the first compiled-in GPU that this machine
    /// has, or `Cpu`.
    pub fn resolve(self) -> DeviceChoice {
        if self != DeviceChoice::Auto {
            return self;
        }
        #[cfg(feature = "cuda")]
        if !Device::enumerate(DeviceType::Cuda).is_empty() {
            return DeviceChoice::Cuda(0);
        }
        #[cfg(feature = "metal")]
        if !Device::enumerate(DeviceType::Metal).is_empty() {
            return DeviceChoice::Metal;
        }
        #[cfg(feature = "wgpu")]
        if !Device::enumerate(DeviceType::Wgpu).is_empty() {
            return DeviceChoice::Wgpu;
        }
        DeviceChoice::Cpu
    }

    /// The Burn device for this choice, or an error naming the Cargo feature it needs.
    pub fn burn_device(self) -> Result<Device, EncoderError> {
        match self.resolve() {
            DeviceChoice::Auto => unreachable!("resolve never returns Auto"),
            DeviceChoice::Cpu => cpu_device(),
            DeviceChoice::Metal => metal_device(),
            DeviceChoice::Wgpu => wgpu_device(),
            DeviceChoice::Cuda(index) => cuda_device(index),
        }
    }

    /// Whether this is a GPU choice (after resolving `Auto`).
    pub fn is_gpu(self) -> bool {
        !matches!(self.resolve(), DeviceChoice::Cpu)
    }
}

fn not_compiled(feature: &str) -> EncoderError {
    EncoderError::Device(format!("larmor-model was built without the {feature:?} feature"))
}

fn cpu_device() -> Result<Device, EncoderError> {
    #[cfg(feature = "flex")]
    return Ok(Device::flex());
    #[allow(unreachable_code)]
    Err(not_compiled("flex"))
}

fn metal_device() -> Result<Device, EncoderError> {
    #[cfg(feature = "metal")]
    {
        if Device::enumerate(DeviceType::Metal).is_empty() {
            return Err(EncoderError::Device("no Metal GPU found".into()));
        }
        return Ok(Device::metal(DeviceKind::DefaultDevice));
    }
    #[allow(unreachable_code)]
    Err(not_compiled("metal"))
}

fn wgpu_device() -> Result<Device, EncoderError> {
    #[cfg(feature = "wgpu")]
    {
        if Device::enumerate(DeviceType::Wgpu).is_empty() {
            return Err(EncoderError::Device("no wgpu adapter found".into()));
        }
        return Ok(Device::wgpu(DeviceKind::DefaultDevice));
    }
    #[allow(unreachable_code)]
    Err(not_compiled("wgpu"))
}

fn cuda_device(index: usize) -> Result<Device, EncoderError> {
    #[cfg(feature = "cuda")]
    {
        let count = Device::enumerate(DeviceType::Cuda).len();
        if index >= count {
            return Err(EncoderError::Device(format!("CUDA device {index} not found ({count} present)")));
        }
        return Ok(Device::cuda(index));
    }
    #[allow(unreachable_code)]
    {
        let _ = index;
        Err(not_compiled("cuda"))
    }
}
