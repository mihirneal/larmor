//! The compute device the segmentation probe runs on (Candle: the Apple GPU through Metal, or the CPU).
//!
//! Only the segmentation probe uses a device; the scan-level probes are small and run on the CPU in float64.

use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// Which device to run on, as a user picks it (`--device auto|cpu|metal|gpu`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// Metal when this Mac has it, else the CPU (as `larmor_model::DeviceChoice` picks).
    #[default]
    Auto,
    Cpu,
    Metal,
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceChoice::Auto => write!(f, "auto"),
            DeviceChoice::Cpu => write!(f, "cpu"),
            DeviceChoice::Metal => write!(f, "metal"),
        }
    }
}

impl FromStr for DeviceChoice {
    type Err = DeviceError;

    /// Parses `auto`, `cpu`, or `metal` (also `gpu`).
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(DeviceChoice::Auto),
            "cpu" => Ok(DeviceChoice::Cpu),
            "metal" | "gpu" => Ok(DeviceChoice::Metal),
            _ => Err(DeviceError::UnknownChoice(text.to_string())),
        }
    }
}

/// Why a device could not be opened or a computation on it failed.
#[derive(Debug, Error)]
pub enum DeviceError {
    #[error("unknown device {0:?} (expected auto, metal or cpu)")]
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
            DeviceChoice::Metal => candle_core::Device::new_metal(0).map_err(|e| e.to_string()),
        };
        opened.map(|inner| Self { inner }).map_err(|reason| DeviceError::Unavailable { choice, reason })
    }

    /// The CPU.
    pub fn cpu() -> Self {
        Self { inner: candle_core::Device::Cpu }
    }

    fn auto() -> Self {
        Self::open(DeviceChoice::Metal).unwrap_or_else(|_| Self::cpu())
    }

    /// Wraps a Candle device opened elsewhere (for example the encoder's), so both share it.
    pub fn from_candle(inner: candle_core::Device) -> Self {
        Self { inner }
    }

    /// The underlying Candle device.
    pub fn as_candle(&self) -> &candle_core::Device {
        &self.inner
    }

    /// Whether this is the GPU (Metal).
    pub fn is_gpu(&self) -> bool {
        !self.inner.is_cpu()
    }

    /// Waits until all queued work on the device has finished (for timings).
    pub fn synchronize(&self) -> Result<(), DeviceError> {
        Ok(self.inner.synchronize()?)
    }
}

impl fmt::Display for Device {
    /// `cpu` or `metal`, the only devices Larmor opens.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.inner.is_cpu() { "cpu" } else { "metal" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_choices() {
        assert_eq!("auto".parse::<DeviceChoice>().unwrap(), DeviceChoice::Auto);
        assert_eq!("CPU".parse::<DeviceChoice>().unwrap(), DeviceChoice::Cpu);
        assert_eq!("metal".parse::<DeviceChoice>().unwrap(), DeviceChoice::Metal);
        assert_eq!("gpu".parse::<DeviceChoice>().unwrap(), DeviceChoice::Metal);
        assert!("cuda".parse::<DeviceChoice>().is_err());
        assert!("tpu".parse::<DeviceChoice>().is_err());
        assert_eq!(DeviceChoice::Metal.to_string(), "metal");
    }

    #[test]
    fn cpu_and_auto_always_open() {
        assert_eq!(Device::open(DeviceChoice::Cpu).unwrap().to_string(), "cpu");
        let auto = Device::open(DeviceChoice::Auto).unwrap();
        assert_eq!(auto.is_gpu(), auto.to_string() != "cpu");
    }
}
