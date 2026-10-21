//! The encoder as the CLI sees it: load a manifest on a device, encode one scan's patches.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use burn::tensor::{Device, Int, Tensor, TensorData};

use crate::attention::AttentionOptions;
use crate::device::{DeviceChoice, Precision};
use crate::error::EncoderError;
use crate::manifest::{Architecture, Manifest};
use crate::vit::VisionTransformer;
use crate::weights::WeightReader;

/// Everything that decides how (not what) the encoder computes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncoderOptions {
    /// Compute device.
    pub device: DeviceChoice,
    /// Matmul precision; `F32` is the parity target.
    pub precision: Precision,
    /// Attention kernel and memory budget.
    pub attention: AttentionOptions,
}

/// A loaded encoder on a device, ready to embed scans.
#[derive(Debug)]
pub struct Encoder {
    model: VisionTransformer,
    arch: Architecture,
    device: Device,
    options: EncoderOptions,
}

/// One scan's encoder output, all in f32 and row-major.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// `[n_tokens, width]`: the final-norm output of each in-mask patch, class tokens dropped.
    pub tokens: Vec<f32>,
    /// `[class_tokens, width]`: the class-token outputs.
    pub cls: Vec<f32>,
    /// `[width]`: the mean of `tokens`.
    pub pooled: Vec<f32>,
    /// Number of patch tokens.
    pub n_tokens: usize,
    /// Embedding width.
    pub width: usize,
    /// Wall time of `encode`: upload, forward pass and readback.
    pub seconds: f64,
}

impl Encoder {
    /// Load the manifest's weights (checking their SHA-256) onto `device`.
    ///
    /// Mirrors `tools/make_reference.py::load_model`.
    pub fn load(
        manifest: &Manifest,
        device: DeviceChoice,
        precision: Precision,
    ) -> Result<Self, EncoderError> {
        Self::load_with(manifest, EncoderOptions { device, precision, ..EncoderOptions::default() })
    }

    /// [`Encoder::load`] with every option.
    pub fn load_with(manifest: &Manifest, options: EncoderOptions) -> Result<Self, EncoderError> {
        let bytes = manifest.read_verified_weights()?;
        Self::from_safetensors(&manifest.architecture, &bytes, options)
    }

    /// Build an encoder from safetensors bytes in `naming = "nanobrain"`, without a manifest or a
    /// checksum. For tests and benchmarks; real models go through [`Encoder::load`].
    pub fn from_safetensors(
        arch: &Architecture,
        bytes: &[u8],
        options: EncoderOptions,
    ) -> Result<Self, EncoderError> {
        let device_choice = options.device.resolve();
        let compute = options.precision.compute_dtype();
        let kernel = options.attention.kernel.resolve(device_choice.is_gpu(), compute);
        let attention = AttentionOptions { kernel, ..options.attention };
        let options = EncoderOptions { device: device_choice, attention, ..options };
        let device = device_choice.burn_device()?;
        if !device.supports_dtype(compute) {
            return Err(EncoderError::Device(format!(
                "{} does not support {}",
                options.device, options.precision
            )));
        }
        let mut weights = WeightReader::new(bytes, &device)?;
        let model = VisionTransformer::load(arch, &mut weights, compute, options.attention)?;
        weights.finish()?;
        Ok(Self { model, arch: arch.clone(), device, options })
    }

    /// Encode one scan.
    ///
    /// `patches` is `[N, p³]` row-major, the normalized image patchified as
    /// `model.py::patchify3d` does, in-mask patches only. `coords` are their `(gx, gy, gz)` patch
    /// grid positions. Mirrors `model.py::ViTMAE3D.forward_encoder` followed by the class-token
    /// split and mean in `dense_embed` / `global_embed`.
    pub fn encode(&self, patches: &[f32], coords: &[[u16; 3]]) -> Result<Encoded, EncoderError> {
        self.check_input(patches, coords)?;
        let start = Instant::now();
        // Burn panics rather than returning errors when a device runs out of memory or a kernel
        // fails; turn that into an error so one bad scan does not end a batch.
        let mut values = panic::catch_unwind(AssertUnwindSafe(|| self.forward(patches, coords)))
            .map_err(|payload| EncoderError::Backend(panic_message(payload.as_ref())))??;
        let width = self.arch.width;
        let tokens = values.split_off(self.arch.class_tokens * width);
        let pooled = mean_rows(&tokens, width);
        let n_tokens = coords.len();
        Ok(Encoded { tokens, cls: values, pooled, n_tokens, width, seconds: start.elapsed().as_secs_f64() })
    }

    /// Upload, run and read back: `[class_tokens + N, width]` row-major.
    fn forward(&self, patches: &[f32], coords: &[[u16; 3]]) -> Result<Vec<f32>, EncoderError> {
        let n_tokens = coords.len();
        let patch_dim = self.arch.patch_dim();
        let patches =
            Tensor::<2>::from_data(TensorData::new(patches.to_vec(), [n_tokens, patch_dim]), &self.device);
        let coords = [0, 1, 2].map(|axis| {
            let ids: Vec<i32> = coords.iter().map(|c| i32::from(c[axis])).collect();
            Tensor::<1, Int>::from_data(TensorData::new(ids, [n_tokens]), &self.device)
        });
        let output = self.model.forward(patches, coords);
        output
            .try_into_data()
            .map_err(|err| EncoderError::Backend(format!("{err:?}")))?
            .try_into_vec::<f32>()
            .map_err(|err| EncoderError::Backend(format!("{err:?}")))
    }

    /// The architecture this encoder was built for.
    pub fn architecture(&self) -> &Architecture {
        &self.arch
    }

    /// The options in effect, with `Auto` device and attention kernel resolved to what is used.
    pub fn options(&self) -> EncoderOptions {
        self.options
    }

    /// Wait for all queued device work; useful before timing.
    pub fn sync(&self) -> Result<(), EncoderError> {
        self.device.sync().map_err(|err| EncoderError::Backend(format!("{err:?}")))
    }

    /// Bytes the device allocator holds (in use plus cached pages), where the backend reports it.
    /// Pages are kept for reuse, so after an encode this is close to that encode's peak.
    pub fn device_memory_reserved(&self) -> Option<u64> {
        self.device.memory_pool_usage().map(|usage| usage.bytes_reserved)
    }

    /// Ask the device allocator to return cached pages.
    pub fn release_cached_memory(&self) {
        self.device.memory_cleanup();
    }

    fn check_input(&self, patches: &[f32], coords: &[[u16; 3]]) -> Result<(), EncoderError> {
        let patch_dim = self.arch.patch_dim();
        if coords.is_empty() {
            return Err(EncoderError::Input("no patches (the mask is empty)".into()));
        }
        if patches.len() != coords.len() * patch_dim {
            return Err(EncoderError::Input(format!(
                "{} patch values for {} coordinates; expected {} values per patch",
                patches.len(),
                coords.len(),
                patch_dim
            )));
        }
        let grid = self.arch.patch_grid();
        if let Some(bad) = coords.iter().find(|c| (0..3).any(|axis| usize::from(c[axis]) >= grid[axis])) {
            return Err(EncoderError::Input(format!(
                "coordinate {bad:?} is outside the patch grid {grid:?}"
            )));
        }
        Ok(())
    }
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic");
    format!("the backend panicked: {message}")
}

/// Column means of a row-major `[rows, width]` matrix, accumulated in f64.
fn mean_rows(values: &[f32], width: usize) -> Vec<f32> {
    let rows = values.len() / width;
    let mut sums = vec![0f64; width];
    for row in values.chunks_exact(width) {
        for (sum, &value) in sums.iter_mut().zip(row) {
            *sum += f64::from(value);
        }
    }
    sums.into_iter().map(|sum| (sum / rows as f64) as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::mean_rows;

    #[test]
    fn mean_rows_averages_columns() {
        assert_eq!(mean_rows(&[1.0, 2.0, 3.0, 6.0], 2), vec![2.0, 4.0]);
    }
}
