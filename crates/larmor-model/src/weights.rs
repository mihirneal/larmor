//! Reading encoder tensors out of a safetensors file (`naming = "nanobrain"`).

use std::collections::BTreeSet;

use burn::tensor::{Device, FloatDType, Tensor, TensorData};
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
    pub(crate) fn tensor<const D: usize>(
        &mut self,
        name: &str,
        shape: [usize; D],
        dtype: FloatDType,
    ) -> Result<Tensor<D>, EncoderError> {
        let values = self.values(name, &shape)?;
        Ok(self.upload(values, shape, dtype))
    }

    /// A PyTorch `nn.Linear` weight (`[out, in]`) transposed to Burn's `[in, out]` layout.
    pub(crate) fn linear_weight(
        &mut self,
        name: &str,
        d_in: usize,
        d_out: usize,
        dtype: FloatDType,
    ) -> Result<Tensor<2>, EncoderError> {
        let torch_layout = self.values(name, &[d_out, d_in])?;
        let mut burn_layout = vec![0.0; torch_layout.len()];
        for (row, out_row) in torch_layout.chunks_exact(d_in).enumerate() {
            for (col, &value) in out_row.iter().enumerate() {
                burn_layout[col * d_out + row] = value;
            }
        }
        Ok(self.upload(burn_layout, [d_in, d_out], dtype))
    }

    /// Error if any tensor in the file was never taken.
    pub(crate) fn finish(self) -> Result<(), EncoderError> {
        if self.unused.is_empty() {
            Ok(())
        } else {
            Err(EncoderError::UnexpectedTensors(self.unused.into_iter().collect()))
        }
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    fn upload<const D: usize>(&self, values: Vec<f32>, shape: [usize; D], dtype: FloatDType) -> Tensor<D> {
        let data = TensorData::new(values, shape);
        // persistent allocations keep weights out of the pools that activations churn through
        self.device
            .memory_persistent_allocations(data, |data| Tensor::from_data(data, (&self.device, dtype.into())))
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
