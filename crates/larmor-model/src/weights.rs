//! Reading encoder tensors out of a safetensors file (`naming = "nanobrain"`).

use std::collections::BTreeSet;

use candle_core::{DType, Device, Tensor};
use safetensors::{Dtype, SafeTensors};

use crate::error::EncoderError;

/// The tensors of one safetensors file, handed out by name to the layer builders.
///
/// Every tensor must be taken exactly once: names left over at the end mean the file and the
/// architecture disagree (a misnamed or extra layer), which `finish` reports as an error, like
/// PyTorch's `load_state_dict(strict=True)`.
pub(crate) struct WeightReader<'a> {
    file: SafeTensors<'a>,
    unused: BTreeSet<String>,
    device: Device,
}

impl<'a> WeightReader<'a> {
    pub(crate) fn new(bytes: &'a [u8], device: &Device) -> Result<Self, EncoderError> {
        let file = SafeTensors::deserialize(bytes)?;
        let unused = file.names().into_iter().map(str::to_string).collect();
        Ok(Self { file, unused, device: device.clone() })
    }

    /// The f32 values of `name`, checked against the expected shape.
    pub(crate) fn values(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, EncoderError> {
        let view = self.file.tensor(name).map_err(|_| EncoderError::MissingTensor(name.to_string()))?;
        if view.shape() != shape {
            return Err(bad(name, format!("shape {:?}, expected {shape:?}", view.shape())));
        }
        if view.dtype() != Dtype::F32 {
            return Err(bad(name, format!("dtype {:?}, expected F32", view.dtype())));
        }
        self.unused.remove(name);
        Ok(f32s_from_le_bytes(view.data()))
    }

    /// `name` as a device tensor of the given dtype.
    pub(crate) fn tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, EncoderError> {
        let values = self.values(name, shape)?;
        self.upload(values, shape, dtype)
    }

    /// One or more PyTorch `nn.Linear` layers with the same input, as a single `[in (+ 1), out]`
    /// matrix for `x @ W` (`out` sums the layers' outputs, in order).
    ///
    /// Each weight `[out, in]` is transposed; with `bias`, each layer's bias becomes the last row,
    /// so appending a column of ones to `x` adds the bias inside the matmul. Candle adds a
    /// separate bias through a broadcast kernel that costs more than the matmul at these shapes;
    /// folding it in made ViT-S about 20% faster on Metal.
    pub(crate) fn linear(
        &mut self,
        prefixes: &[String],
        d_in: usize,
        d_out: usize,
        bias: bool,
        dtype: DType,
    ) -> Result<Tensor, EncoderError> {
        let width = prefixes.len() * d_out;
        let rows = d_in + usize::from(bias);
        let mut folded = vec![0f32; rows * width];
        for (layer, prefix) in prefixes.iter().enumerate() {
            let weight = self.values(&format!("{prefix}.weight"), &[d_out, d_in])?;
            for (out, weight_row) in weight.chunks_exact(d_in).enumerate() {
                for (row, &value) in weight_row.iter().enumerate() {
                    folded[row * width + layer * d_out + out] = value;
                }
            }
            if bias {
                let values = self.values(&format!("{prefix}.bias"), &[d_out])?;
                let start = d_in * width + layer * d_out;
                folded[start..start + d_out].copy_from_slice(&values);
            }
        }
        self.upload(folded, &[rows, width], dtype)
    }

    /// Error if any tensor in the file was never taken.
    pub(crate) fn finish(self) -> Result<(), EncoderError> {
        if self.unused.is_empty() {
            Ok(())
        } else {
            Err(EncoderError::UnexpectedTensors(self.unused.into_iter().collect()))
        }
    }

    fn upload(&self, values: Vec<f32>, shape: &[usize], dtype: DType) -> Result<Tensor, EncoderError> {
        Ok(Tensor::from_vec(values, shape, &self.device)?.to_dtype(dtype)?)
    }
}

/// Decode little-endian f32s. safetensors data is not necessarily 4-byte aligned, so no casting.
pub(crate) fn f32s_from_le_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// Decode little-endian i32s.
pub(crate) fn i32s_from_le_bytes(bytes: &[u8]) -> Vec<i32> {
    bytes.as_chunks::<4>().0.iter().map(|&b| i32::from_le_bytes(b)).collect()
}

fn bad(name: &str, reason: String) -> EncoderError {
    EncoderError::BadTensor { name: name.to_string(), reason }
}
