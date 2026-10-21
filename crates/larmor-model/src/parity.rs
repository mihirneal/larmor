//! Comparing encoder outputs with the PyTorch references in
//! `fixtures/reference/encode/<model_id>/<scan>.safetensors` (layout in `docs/DESIGN.md`).

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use safetensors::{Dtype, SafeTensors};

use crate::encoder::Encoded;
use crate::error::EncoderError;
use crate::weights::{f32s_from_le_bytes, i32s_from_le_bytes};

/// The tensors of one encode reference, as written by `tools/make_reference.py::run_encode`.
#[derive(Debug, Clone)]
pub struct EncodeReference {
    /// `[N, p³]` model input patches.
    pub patches: Vec<f32>,
    /// `[N]` patch grid coordinates.
    pub coords: Vec<[u16; 3]>,
    /// `[N, D]` fp32 encoder tokens.
    pub tokens: Vec<f32>,
    /// `[C, D]` fp32 class-token outputs.
    pub cls: Vec<f32>,
    /// `[D]` fp32 pooled embedding.
    pub pooled: Vec<f32>,
    /// `[N, D]` tokens under bf16 autocast, if present.
    pub tokens_bf16: Option<Vec<f32>>,
    /// `[D]` pooled under bf16 autocast, if present.
    pub pooled_bf16: Option<Vec<f32>>,
    /// Embedding width `D`.
    pub width: usize,
    /// The file's `__metadata__`: model id and sha256, scan, library versions, timings and the
    /// bf16 noise floor (see [`EncodeReference::bf16_noise_floor`]).
    pub metadata: BTreeMap<String, String>,
}

/// How far PyTorch's own bf16 autocast lands from its fp32, as stored in a reference's metadata.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseFloor {
    /// Smallest per-token cosine of `tokens_bf16` against `tokens` (`bf16_min_token_cosine`).
    pub min_token_cosine: f64,
    /// Cosine of `pooled_bf16` against `pooled` (`bf16_pooled_cosine`).
    pub pooled_cosine: f64,
}

impl EncodeReference {
    /// Read a reference file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, EncoderError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|source| EncoderError::Read { path: path.to_path_buf(), source })?;
        let (_, header) = SafeTensors::read_metadata(&bytes)?;
        let metadata = header.metadata().clone().unwrap_or_default().into_iter().collect();
        let file = SafeTensors::deserialize(&bytes)?;
        let (tokens, token_shape) = f32_tensor(&file, "tokens")?;
        let width = token_shape.get(1).copied().unwrap_or(0);
        // out-of-range values become u16::MAX, which `Encoder::encode` rejects as off the grid
        let to_u16 = |v: i32| u16::try_from(v).unwrap_or(u16::MAX);
        let coords = i32_tensor(&file, "coords")?.as_chunks::<3>().0.iter().map(|c| c.map(to_u16)).collect();
        let optional = |name| f32_tensor(&file, name).ok().map(|(values, _)| values);
        Ok(Self {
            patches: f32_tensor(&file, "patches")?.0,
            coords,
            tokens,
            cls: f32_tensor(&file, "cls")?.0,
            pooled: f32_tensor(&file, "pooled")?.0,
            tokens_bf16: optional("tokens_bf16"),
            pooled_bf16: optional("pooled_bf16"),
            width,
            metadata,
        })
    }

    /// Number of tokens.
    pub fn n_tokens(&self) -> usize {
        self.coords.len()
    }

    /// PyTorch's bf16-autocast agreement with its own fp32, from the metadata that
    /// `tools/make_reference.py::run_encode` writes. `None` if the file lacks it.
    pub fn bf16_noise_floor(&self) -> Option<NoiseFloor> {
        let number = |key: &str| self.metadata.get(key)?.parse::<f64>().ok();
        Some(NoiseFloor {
            min_token_cosine: number("bf16_min_token_cosine")?,
            pooled_cosine: number("bf16_pooled_cosine")?,
        })
    }
}

/// How close an encoder output is to a reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parity {
    /// Smallest per-token cosine similarity.
    pub min_token_cosine: f64,
    /// Mean per-token cosine similarity.
    pub mean_token_cosine: f64,
    /// Smallest per-token cosine after subtracting the reference's mean token from both sides.
    /// Raw cosines are inflated by the component every token shares; this one measures the
    /// token-specific part, so a model that outputs near-constant tokens scores high on the plain
    /// cosine but not here.
    pub min_centered_cosine: f64,
    /// Mean centered cosine.
    pub mean_centered_cosine: f64,
    /// Cosine of the pooled embeddings.
    pub pooled_cosine: f64,
    /// Smallest class-token cosine (1 when there are none).
    pub min_cls_cosine: f64,
    /// Largest absolute token difference.
    pub max_abs_diff: f64,
    /// Largest absolute token value in the reference, for scale.
    pub max_abs_reference: f64,
}

impl Parity {
    /// Phase 0 gate: min token cosine and pooled cosine at least 0.999.
    pub fn passes(&self, threshold: f64) -> bool {
        self.min_token_cosine >= threshold && self.pooled_cosine >= threshold
    }
}

impl fmt::Display for Parity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "token cos min {:.8} mean {:.8} | centered min {:.8} mean {:.8} | pooled {:.8} | cls min {:.8} | \
             max |diff| {:.2e} (ref max {:.2})",
            self.min_token_cosine,
            self.mean_token_cosine,
            self.min_centered_cosine,
            self.mean_centered_cosine,
            self.pooled_cosine,
            self.min_cls_cosine,
            self.max_abs_diff,
            self.max_abs_reference
        )
    }
}

/// Compare `encoded` with the reference's fp32 outputs.
pub fn compare(encoded: &Encoded, reference: &EncodeReference) -> Parity {
    compare_parts(&encoded.tokens, &encoded.cls, &encoded.pooled, reference, encoded.width)
}

/// Compare raw token, class-token and pooled values with a reference.
pub fn compare_parts(
    tokens: &[f32],
    cls: &[f32],
    pooled: &[f32],
    reference: &EncodeReference,
    width: usize,
) -> Parity {
    let token_cos = row_cosines(tokens, &reference.tokens, width);
    let centered = centered_row_cosines(tokens, &reference.tokens, width);
    let cls_cos = row_cosines(cls, &reference.cls, width);
    let max_abs = |values: &[f32]| values.iter().fold(0f64, |m, &v| m.max(f64::from(v).abs()));
    let max_abs_diff = tokens
        .iter()
        .zip(&reference.tokens)
        .fold(0f64, |m, (&a, &b)| m.max((f64::from(a) - f64::from(b)).abs()));
    Parity {
        min_token_cosine: min(&token_cos),
        mean_token_cosine: mean(&token_cos),
        min_centered_cosine: min(&centered),
        mean_centered_cosine: mean(&centered),
        pooled_cosine: cosine(pooled, &reference.pooled),
        min_cls_cosine: if cls_cos.is_empty() { 1.0 } else { min(&cls_cos) },
        max_abs_diff,
        max_abs_reference: max_abs(&reference.tokens),
    }
}

/// Cosine similarity of two vectors, in f64.
pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    cosine_of(a.iter().zip(b).map(|(&x, &y)| (f64::from(x), f64::from(y))))
}

/// Cosine similarity of paired components.
fn cosine_of(pairs: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut dot, mut aa, mut bb) = (0f64, 0f64, 0f64);
    for (x, y) in pairs {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa.sqrt() * bb.sqrt()).max(f64::MIN_POSITIVE)
}

/// Per-row cosine similarities of two row-major `[rows, width]` matrices.
pub fn row_cosines(a: &[f32], b: &[f32], width: usize) -> Vec<f64> {
    a.chunks_exact(width).zip(b.chunks_exact(width)).map(|(x, y)| cosine(x, y)).collect()
}

/// Per-row cosines of `a − m` and `reference − m`, where `m` is the reference's mean row.
fn centered_row_cosines(a: &[f32], reference: &[f32], width: usize) -> Vec<f64> {
    let mean = mean_row(reference, width);
    let centered = |v: f32, m: f64| f64::from(v) - m;
    a.chunks_exact(width)
        .zip(reference.chunks_exact(width))
        .map(|(x, y)| {
            cosine_of(x.iter().zip(y).zip(&mean).map(|((&x, &y), &m)| (centered(x, m), centered(y, m))))
        })
        .collect()
}

/// The mean row of a row-major `[rows, width]` matrix, in f64.
fn mean_row(values: &[f32], width: usize) -> Vec<f64> {
    let rows = (values.len() / width).max(1) as f64;
    let mut mean = vec![0f64; width];
    for row in values.chunks_exact(width) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += f64::from(v);
        }
    }
    mean.iter_mut().for_each(|m| *m /= rows);
    mean
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, f64::min)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn f32_tensor(file: &SafeTensors, name: &str) -> Result<(Vec<f32>, Vec<usize>), EncoderError> {
    let view = file.tensor(name).map_err(|_| EncoderError::MissingTensor(name.to_string()))?;
    if view.dtype() != Dtype::F32 {
        return Err(EncoderError::BadTensor {
            name: name.into(),
            reason: format!("dtype {:?}", view.dtype()),
        });
    }
    Ok((f32s_from_le_bytes(view.data()), view.shape().to_vec()))
}

fn i32_tensor(file: &SafeTensors, name: &str) -> Result<Vec<i32>, EncoderError> {
    let view = file.tensor(name).map_err(|_| EncoderError::MissingTensor(name.to_string()))?;
    if view.dtype() != Dtype::I32 {
        return Err(EncoderError::BadTensor {
            name: name.into(),
            reason: format!("dtype {:?}", view.dtype()),
        });
    }
    Ok(i32s_from_le_bytes(view.data()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_of_parallel_and_orthogonal_vectors() {
        assert!((cosine(&[1.0, 2.0], &[2.0, 4.0]) - 1.0).abs() < 1e-12);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).abs() < 1e-12);
    }

    #[test]
    fn mean_row_averages_rows() {
        // rows [5, 6] and [7, 4] have mean [6, 5]
        assert_eq!(mean_row(&[5.0, 6.0, 7.0, 4.0], 2), vec![6.0, 5.0]);
    }

    #[test]
    fn centering_exposes_token_specific_error() {
        // both tokens share a large offset, so the raw cosine stays high while the parts that
        // differ between tokens point the opposite way
        let reference = [100.0, 1.0, 100.0, -1.0];
        let swapped = [100.0, -1.0, 100.0, 1.0];
        assert!(row_cosines(&swapped, &reference, 2).iter().all(|&c| c > 0.999));
        let centered = centered_row_cosines(&swapped, &reference, 2);
        assert!(centered.iter().all(|&c| (c + 1.0).abs() < 1e-12), "{centered:?}");
    }
}
