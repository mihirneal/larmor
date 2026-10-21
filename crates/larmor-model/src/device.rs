//! Picking a Candle device and a compute precision.
//!
//! Which devices exist depends on the build:
//!
//! | Device | Built | Attention kernel |
//! |---|---|---|
//! | [`DeviceChoice::Cpu`] | always | query-chunked |
//! | [`DeviceChoice::Metal`] | on macOS | fused `sdpa` (query-chunked for head dims it lacks) |
//! | [`DeviceChoice::Cuda`] | with the `cuda` feature | query-chunked |

use std::fmt;
use std::panic;
use std::str::FromStr;

use candle_core::{DType, Device};

use crate::error::EncoderError;

/// Which compute device runs the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceChoice {
    /// The best device this build and machine have: CUDA, then Metal, then the CPU.
    #[default]
    Auto,
    /// The CPU.
    Cpu,
    /// The Apple GPU, through Metal.
    Metal,
    /// An NVIDIA GPU, by index.
    Cuda(usize),
}

/// Numeric precision of the encoder.
///
/// `F32` is the parity target and the default, and the only precision on the CPU. The half
/// precisions work like PyTorch autocast, which is how nanobrain's eval framework runs the model
/// (`torch.autocast(dtype=bfloat16)`): the linears, attention and GELU run in the half type
/// (matmuls accumulating in f32), while the residual stream, LayerNorms, position embeddings and
/// class tokens stay in f32.
///
/// This mixed form was chosen over running everything in the half type (`model.half()`) from
/// measurements on an M3 Ultra (ViT-S, Metal, fused attention). Speed is the same within noise:
/// at patch 8 both take 0.14-0.155 s per scan, at patch 4 both 4.09-4.23 s. Parity is better
/// mixed. Against PyTorch fp32 at patch 4, pure bf16 reaches `1 - min token cosine` 1.7e-4 and
/// `1 - min centered cosine` 1.6e-3, against 6.8e-5 and 6.2e-4 mixed; mixed bf16 lands where
/// PyTorch's own bf16 autocast does (6.0e-5). Mixed f16 is also closer (1.0e-6 against 1.8e-6),
/// and an f32 residual stream cannot overflow f16's ±65504 range between blocks.
///
/// Half precision needs a GPU (Metal or CUDA). Candle's CPU backend has no bf16 matmul, and on
/// Apple silicon its f16 matmuls and softmax accumulate in f16: ViT-S patch 8 then reaches a min
/// token cosine of only 0.87 against PyTorch fp32.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// Everything in f32.
    #[default]
    F32,
    /// Linears, attention and GELU in IEEE half precision; the rest in f32. GPU only.
    F16,
    /// Linears, attention and GELU in bfloat16; the rest in f32. GPU only.
    Bf16,
}

impl Precision {
    /// The dtype linears, attention and GELU compute in.
    pub fn compute_dtype(self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::Bf16 => DType::BF16,
        }
    }

    /// The dtype of the residual stream, LayerNorms, position embeddings and class tokens: f32 in
    /// every precision, as under autocast.
    pub fn residual_dtype(self) -> DType {
        DType::F32
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
            "cuda" => Ok(DeviceChoice::Cuda(0)),
            _ => match s.strip_prefix("cuda:").map(str::parse) {
                Some(Ok(index)) => Ok(DeviceChoice::Cuda(index)),
                _ => Err(format!("unknown device {s:?} (expected auto, cpu, metal or cuda[:N])")),
            },
        }
    }
}

impl DeviceChoice {
    /// Resolve `Auto` to the device [`DeviceChoice::open`] would pick. Probes the hardware, so
    /// it costs a device initialization.
    pub fn resolve(self) -> DeviceChoice {
        match self.open() {
            Ok((choice, _)) => choice,
            Err(_) => self,
        }
    }

    /// The Candle device for this choice, or an error saying why it is unavailable.
    pub fn candle_device(self) -> Result<Device, EncoderError> {
        self.open().map(|(_, device)| device)
    }

    /// Open the device, resolving `Auto` to the first of CUDA, Metal and CPU that works.
    pub fn open(self) -> Result<(DeviceChoice, Device), EncoderError> {
        match self {
            DeviceChoice::Auto => {
                let gpu = [DeviceChoice::Cuda(0), DeviceChoice::Metal]
                    .into_iter()
                    .find_map(|choice| choice.open().ok());
                Ok(gpu.unwrap_or((DeviceChoice::Cpu, Device::Cpu)))
            }
            DeviceChoice::Cpu => Ok((self, Device::Cpu)),
            DeviceChoice::Metal => match Device::new_metal(0) {
                Ok(device) => Ok((self, device)),
                Err(_) if !cfg!(target_os = "macos") => Err(EncoderError::Device(
                    "Metal is only available in macOS builds of larmor-model".into(),
                )),
                Err(err) => Err(EncoderError::Device(format!("no Metal GPU: {err}"))),
            },
            // cudarc panics, rather than returning an error, when it cannot load the CUDA libraries
            DeviceChoice::Cuda(index) => match panic::catch_unwind(|| Device::new_cuda(index)) {
                Ok(Ok(device)) => Ok((self, device)),
                Ok(Err(_)) if !cfg!(feature = "cuda") => {
                    Err(EncoderError::Device("larmor-model was built without the \"cuda\" feature".into()))
                }
                Ok(Err(err)) => Err(EncoderError::Device(format!("CUDA device {index}: {err}"))),
                Err(_) => Err(EncoderError::Device(format!(
                    "CUDA device {index}: the CUDA driver libraries could not be loaded"
                ))),
            },
        }
    }

    /// Whether this is a GPU choice (after resolving `Auto`).
    pub fn is_gpu(self) -> bool {
        !matches!(self.resolve(), DeviceChoice::Cpu)
    }
}

/// The [`DeviceChoice`] a Candle device corresponds to.
pub(crate) fn choice_of(device: &Device) -> DeviceChoice {
    match device.location() {
        candle_core::DeviceLocation::Cpu => DeviceChoice::Cpu,
        candle_core::DeviceLocation::Metal { .. } => DeviceChoice::Metal,
        candle_core::DeviceLocation::Cuda { gpu_id } => DeviceChoice::Cuda(gpu_id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for choice in [DeviceChoice::Auto, DeviceChoice::Cpu, DeviceChoice::Metal, DeviceChoice::Cuda(1)] {
            assert_eq!(choice.to_string().parse::<DeviceChoice>(), Ok(choice));
        }
        for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
            assert_eq!(precision.to_string().parse::<Precision>(), Ok(precision));
        }
        assert_eq!("cuda".parse::<DeviceChoice>(), Ok(DeviceChoice::Cuda(0)));
        assert!("wgpu".parse::<DeviceChoice>().is_err());
    }

    #[test]
    fn cpu_always_opens() {
        let (choice, device) = DeviceChoice::Cpu.open().unwrap();
        assert_eq!(choice, DeviceChoice::Cpu);
        assert!(device.is_cpu());
        assert_ne!(DeviceChoice::Auto.resolve(), DeviceChoice::Auto);
    }
}
