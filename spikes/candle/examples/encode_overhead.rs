//! Splits one ViT-S forward pass into CPU encode time (queuing kernels) and GPU drain time.
//!
//! cargo run --release --example encode_overhead -- MODEL.toml REFERENCE.safetensors

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::Device;
use candle_spike::config::Manifest;
use candle_spike::model::{AttentionKernel, BiasAdd, Encoder, EncoderInput, Options, Precision};
use candle_spike::reference::EncodeReference;
use candle_spike::weights::Weights;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let manifest = Manifest::load(Path::new(&args[1]))?;
    let reference = EncodeReference::load(Path::new(&args[2]))?;
    let arch = &manifest.architecture;
    let device = Device::new_metal(0)?;
    let weights = Weights::load(&manifest.weights_path, arch, &device)?;
    let patches = reference.patches_for(arch.patch)?;
    let input = EncoderInput::new(&patches.values, &patches.coords, arch, &device)?;
    for precision in [Precision::F32, Precision::F16] {
        for bias in [BiasAdd::Broadcast, BiasAdd::Folded] {
            let options = Options { precision, attention: AttentionKernel::Fused, bias };
            let encoder = Encoder::new(arch, &weights, options)?;
            for run in 0..4 {
                device.synchronize()?;
                let start = Instant::now();
                let out = encoder.forward(&input)?;
                let encoded = start.elapsed().as_secs_f64() * 1e3;
                device.synchronize()?;
                let total = start.elapsed().as_secs_f64() * 1e3;
                drop(out);
                if run > 0 {
                    println!(
                        "{} {}: encode {encoded:.1} ms, total {total:.1} ms",
                        precision.name(),
                        bias.name()
                    );
                }
            }
        }
    }
    Ok(())
}
