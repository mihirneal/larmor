//! The compute device the segmentation probe runs on (Candle: Metal on macOS, CUDA on Linux, or the CPU).
//!
//! Only the segmentation probe uses a device; the scan-level probes are small and run on the CPU in float64.

use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Which device to run on, as a user picks it (`--device auto|cpu|metal|cuda|cuda:N|gpu`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// CUDA device 0 when built with the `cuda` feature, else Metal when this build has it (macOS), else
    /// the CPU (the same order as `larmor_model::DeviceChoice`).
    #[default]
    Auto,
    Cpu,
    Metal,
    /// A CUDA device by index; needs the `cuda` feature.
    Cuda(usize),
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceChoice::Auto => write!(f, "auto"),
            DeviceChoice::Cpu => write!(f, "cpu"),
            DeviceChoice::Metal => write!(f, "metal"),
            DeviceChoice::Cuda(index) => write!(f, "cuda:{index}"),
        }
    }
}

impl FromStr for DeviceChoice {
    type Err = DeviceError;

    /// Parses `auto`, `cpu`, `metal`, `cuda` (device 0), `cuda:N`, or `gpu` (Metal on macOS, else CUDA 0).
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let lower = text.trim().to_ascii_lowercase();
        match lower.as_str() {
            "auto" => Ok(DeviceChoice::Auto),
            "cpu" => Ok(DeviceChoice::Cpu),
            "metal" => Ok(DeviceChoice::Metal),
            "cuda" => Ok(DeviceChoice::Cuda(0)),
            "gpu" if cfg!(target_os = "macos") => Ok(DeviceChoice::Metal),
            "gpu" => Ok(DeviceChoice::Cuda(0)),
            _ => lower
                .strip_prefix("cuda:")
                .and_then(|index| index.parse().ok())
                .map(DeviceChoice::Cuda)
                .ok_or_else(|| DeviceError::UnknownChoice(text.to_string())),
        }
    }
}

/// Why a device could not be opened or a computation on it failed.
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("unknown device {0:?} (expected auto, cpu, metal, cuda or cuda:N)")]
    UnknownChoice(String),
    #[error("cannot open device {choice}: {reason}")]
    Unavailable { choice: DeviceChoice, reason: String },
    #[error("computation on the device failed: {0}")]
    Compute(#[from] candle_core::Error),
}

/// An opened compute device. Cloning it is cheap and shares the device.
#[derive(Clone, Debug)]
pub struct Device {
    inner: candle_core::Device,
}

impl Device {
    /// Opens the chosen device. `Auto` never fails: it falls back to the CPU.
    pub fn open(choice: DeviceChoice) -> Result<Self, DeviceError> {
        let opened = match choice {
            DeviceChoice::Auto => return Ok(Self::auto()),
            DeviceChoice::Cpu => Ok(candle_core::Device::Cpu),
            DeviceChoice::Metal => open_metal(),
            DeviceChoice::Cuda(index) => open_cuda(index),
        };
        opened.map(|inner| Self { inner }).map_err(|reason| DeviceError::Unavailable { choice, reason })
    }

    /// The CPU.
    pub fn cpu() -> Self {
        Self { inner: candle_core::Device::Cpu }
    }

    fn auto() -> Self {
        [DeviceChoice::Cuda(0), DeviceChoice::Metal]
            .into_iter()
            .find_map(|choice| Self::open(choice).ok())
            .unwrap_or_else(Self::cpu)
    }

    /// Wraps a Candle device opened elsewhere (for example the encoder's), so both share it.
    pub fn from_candle(inner: candle_core::Device) -> Self {
        Self { inner }
    }

    /// The underlying Candle device.
    pub fn as_candle(&self) -> &candle_core::Device {
        &self.inner
    }

    /// Whether this is a GPU (Metal or CUDA).
    pub fn is_gpu(&self) -> bool {
        !self.inner.is_cpu()
    }

    /// Waits until all queued work on the device has finished (for timings).
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(self.inner.synchronize()?)
    }
}

fn open_metal() -> Result<candle_core::Device, String> {
    if !candle_core::utils::metal_is_available() {
        return Err("Metal is only built on macOS".into());
    }
    candle_core::Device::new_metal(0).map_err(|e| e.to_string())
}

fn open_cuda(index: usize) -> Result<candle_core::Device, String> {
    if !candle_core::utils::cuda_is_available() {
        return Err("larmor-probe was built without the \"cuda\" feature".into());
    }
    // cudarc panics, rather than returning an error, when it cannot load the CUDA libraries
    match std::panic::catch_unwind(|| candle_core::Device::new_cuda(index)) {
        Ok(opened) => opened.map_err(|e| e.to_string()),
        Err(_) => Err("the CUDA driver libraries could not be loaded".into()),
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.inner.location() {
            candle_core::DeviceLocation::Cpu => write!(f, "cpu"),
            candle_core::DeviceLocation::Metal { .. } => write!(f, "metal"),
            candle_core::DeviceLocation::Cuda { gpu_id } => write!(f, "cuda:{gpu_id}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_choices() {
        assert_eq!("auto".parse::<DeviceChoice>().unwrap(), DeviceChoice::Auto);
        assert_eq!("CPU".parse::<DeviceChoice>().unwrap(), DeviceChoice::Cpu);
        assert_eq!("cuda".parse::<DeviceChoice>().unwrap(), DeviceChoice::Cuda(0));
        assert_eq!("cuda:2".parse::<DeviceChoice>().unwrap(), DeviceChoice::Cuda(2));
        assert!("tpu".parse::<DeviceChoice>().is_err());
        assert_eq!(DeviceChoice::Cuda(1).to_string(), "cuda:1");
    }

    #[test]
    fn cpu_and_auto_always_open() {
        assert_eq!(Device::open(DeviceChoice::Cpu).unwrap().to_string(), "cpu");
        let auto = Device::open(DeviceChoice::Auto).unwrap();
        assert_eq!(auto.is_gpu(), auto.to_string() != "cpu");
    }
}
