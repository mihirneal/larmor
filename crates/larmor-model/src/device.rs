//! Picking a Candle device and a compute precision.
//!
//! Larmor runs on Apple silicon Macs. The encoder runs on the GPU through Metal, or on the CPU: the
//! fallback where Metal is missing (CI runners may not expose it) and what the tests use.
//!
//! | Device | Attention kernel |
//! |---|---|
//! | [`DeviceChoice::Metal`] | fused `sdpa` (query-chunked for head dims it lacks) |
//! | [`DeviceChoice::Cpu`] | query-chunked |

use std::fmt;
use std::str::FromStr;

use candle_core::{DType, Device};

use crate::error::EncoderError;

/// Which compute device runs the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DeviceChoice {
    /// Metal when this Mac has it, else the CPU.
    #[default]
    Auto,
    /// The CPU.
    Cpu,
    /// The Apple GPU, through Metal.
    Metal,
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
/// Half precision needs the GPU (Metal). Candle's CPU backend has no bf16 matmul, and on
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
        }
    }
}

impl FromStr for DeviceChoice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(DeviceChoice::Auto),
            "cpu" => Ok(DeviceChoice::Cpu),
            "metal" => Ok(DeviceChoice::Metal),
            other => Err(format!("unknown device {other:?} (expected auto, metal or cpu)")),
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

    /// Open the device, resolving `Auto` to Metal if it opens, else the CPU.
    pub fn open(self) -> Result<(DeviceChoice, Device), EncoderError> {
        match self {
            DeviceChoice::Auto => Ok(DeviceChoice::Metal.open().unwrap_or((DeviceChoice::Cpu, Device::Cpu))),
            DeviceChoice::Cpu => Ok((self, Device::Cpu)),
            DeviceChoice::Metal => match Device::new_metal(0) {
                Ok(device) => Ok((self, device)),
                Err(err) => Err(EncoderError::Device(format!("no Metal GPU: {err}"))),
            },
        }
    }

    /// Whether this is a GPU choice (after resolving `Auto`).
    pub fn is_gpu(self) -> bool {
        !matches!(self.resolve(), DeviceChoice::Cpu)
    }
}

/// The [`DeviceChoice`] a Candle device corresponds to. Larmor opens only Metal and the CPU.
pub(crate) fn choice_of(device: &Device) -> DeviceChoice {
    if device.is_cpu() { DeviceChoice::Cpu } else { DeviceChoice::Metal }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for choice in [DeviceChoice::Auto, DeviceChoice::Cpu, DeviceChoice::Metal] {
            assert_eq!(choice.to_string().parse::<DeviceChoice>(), Ok(choice));
        }
        for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
            assert_eq!(precision.to_string().parse::<Precision>(), Ok(precision));
        }
        assert_eq!("Metal".parse::<DeviceChoice>(), Ok(DeviceChoice::Metal));
        assert!("cuda".parse::<DeviceChoice>().is_err());
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
