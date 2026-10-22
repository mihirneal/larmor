//! `--device` for the segmentation probe, which picks its device through larmor-probe.
//!
//! `--device` and `--precision` parse straight into `larmor_model`'s [`DeviceChoice`] and
//! `Precision` (`auto`, `cpu`, `metal`, `cuda`, `cuda:N`; `f32`, `f16`, `bf16`). The encoder opens
//! its own device from them; the segmentation probe gets its device here.

use anyhow::{Context, Result};
use larmor_model::DeviceChoice;

/// The device the segmentation probe fits and predicts on: the one `--device` names. `auto` picks
/// CUDA, then Metal, then the CPU, the same order as the encoder.
pub fn segmentation_device(choice: DeviceChoice) -> Result<larmor_probe::Device> {
    let probe_choice = match choice {
        DeviceChoice::Auto => larmor_probe::DeviceChoice::Auto,
        DeviceChoice::Cpu => larmor_probe::DeviceChoice::Cpu,
        DeviceChoice::Metal => larmor_probe::DeviceChoice::Metal,
        DeviceChoice::Cuda(index) => larmor_probe::DeviceChoice::Cuda(index),
    };
    larmor_probe::Device::open(probe_choice).context("opening the segmentation probe's device")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmentation_device_follows_the_choice() {
        assert_eq!(segmentation_device(DeviceChoice::Cpu).unwrap().to_string(), "cpu");
        let auto = segmentation_device(DeviceChoice::Auto).unwrap();
        assert_eq!(auto.is_gpu(), auto.to_string() != "cpu");
    }
}
