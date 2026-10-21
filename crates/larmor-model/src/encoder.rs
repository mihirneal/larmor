//! The encoder as the CLI sees it: load a manifest on a device, encode one scan's patches.

use std::any::Any;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use candle_core::{Device, Tensor};

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
    /// Numeric precision; `F32` is the parity target.
    pub precision: Precision,
    /// Attention kernel and memory budget.
    pub attention: AttentionOptions,
}

/// A loaded encoder on a device, ready to embed scans.
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
        let (device_choice, device) = options.device.open()?;
        if device.is_cpu() && options.precision != Precision::F32 {
            return Err(EncoderError::Device(format!(
                "{} needs a GPU; use f32 on the CPU (Candle's CPU backend has no bf16 matmul, and its \
                 f16 matmuls and softmax accumulate in f16, which misses the parity gate)",
                options.precision
            )));
        }
        let kernel =
            options.attention.kernel.resolve(&device, arch.head_dim(), options.precision.compute_dtype())?;
        let attention = AttentionOptions { kernel, ..options.attention };
        let options = EncoderOptions { device: device_choice, attention, ..options };
        let mut weights = WeightReader::new(bytes, &device)?;
        let model = VisionTransformer::load(arch, &mut weights, options.precision)?;
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
        // Candle reports failures as errors, but a few paths (assertions in its Metal backend)
        // panic instead; turn those into errors too so one bad scan does not end a batch.
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
        let sequence = n_tokens + self.arch.class_tokens;
        let plan =
            self.options.attention.plan(self.arch.heads, sequence, self.options.precision.compute_dtype())?;
        let patches = Tensor::from_slice(patches, (n_tokens, self.arch.patch_dim()), &self.device)?;
        let column = |axis: usize| {
            let ids: Vec<u32> = coords.iter().map(|c| u32::from(c[axis])).collect();
            Tensor::from_vec(ids, n_tokens, &self.device)
        };
        let coords = [column(0)?, column(1)?, column(2)?];
        let output = self.model.forward(&patches, &coords, plan)?;
        Ok(output.flatten_all()?.to_vec1::<f32>()?)
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
        Ok(self.device.synchronize()?)
    }

    /// Bytes the device holds for this process right now, where the backend reports it (Metal):
    /// the weights, plus during an encode its activations and the buffers Candle keeps for reuse.
    /// Candle frees unused buffers whenever it waits for the GPU, so after an encode returns this
    /// is about the weights alone; poll it during an encode to see the peak.
    pub fn device_memory_reserved(&self) -> Option<u64> {
        match &self.device {
            #[cfg(target_os = "macos")]
            Device::Metal(metal) => Some(metal.current_allocated_size() as u64),
            _ => None,
        }
    }

    /// Ask the device to free buffers it keeps for reuse. Candle's Metal backend frees them
    /// whenever it waits for the GPU, so this is [`Encoder::sync`] with errors ignored.
    pub fn release_cached_memory(&self) {
        let _ = self.device.synchronize();
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
        // the GPU would read out of bounds rather than fail, so check here
        let grid = self.arch.patch_grid();
        if let Some(bad) = coords.iter().find(|c| (0..3).any(|axis| usize::from(c[axis]) >= grid[axis])) {
            return Err(EncoderError::Input(format!(
                "coordinate {bad:?} is outside the patch grid {grid:?}"
            )));
        }
        Ok(())
    }
}

impl fmt::Debug for Encoder {
    // Candle's tensors print their values, so list only what identifies the encoder.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoder")
            .field("arch", &self.arch)
            .field("options", &self.options)
            .finish_non_exhaustive()
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
