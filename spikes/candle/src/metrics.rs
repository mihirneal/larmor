//! Parity metrics between Candle and PyTorch embeddings, computed in f64.

use serde::Serialize;

/// How closely candidate embeddings match a reference.
#[derive(Debug, Clone, Serialize)]
pub struct Parity {
    pub tokens: usize,
    /// Per-token cosine similarity.
    pub min_token_cosine: f64,
    pub mean_token_cosine: f64,
    /// Per-token cosine after subtracting the reference's mean token from both sides. Raw
    /// cosines are inflated by the component every token shares; this one measures the
    /// token-specific part. The candidate's own pooling error stays in its residuals.
    pub min_centered_cosine: f64,
    pub mean_centered_cosine: f64,
    /// Cosine between the mean of the candidate tokens and the reference `pooled`.
    pub pooled_cosine: f64,
    pub cls_cosine: Option<f64>,
    pub max_abs_diff: f64,
    /// `||candidate - reference|| / ||reference||` over all tokens.
    pub relative_l2: f64,
}

/// Compares `[N, D]` token matrices, the pooled embedding and (when given) class tokens.
pub fn compare(
    tokens: &[f32],
    reference_tokens: &[f32],
    reference_pooled: &[f32],
    cls: Option<(&[f32], &[f32])>,
    dim: usize,
) -> Parity {
    assert_eq!(tokens.len(), reference_tokens.len(), "token matrices differ in size");
    let n = tokens.len() / dim;
    let reference_mean = column_mean(reference_tokens, dim);

    let mut min_cos = f64::INFINITY;
    let mut sum_cos = 0.0;
    let mut min_centered = f64::INFINITY;
    let mut sum_centered = 0.0;
    let (mut diff_sq, mut ref_sq, mut max_abs) = (0.0, 0.0, 0.0f64);
    let mut centered_a = vec![0.0; dim];
    let mut centered_b = vec![0.0; dim];
    for (a, b) in tokens.chunks_exact(dim).zip(reference_tokens.chunks_exact(dim)) {
        let cos = cosine(a.iter().map(|&v| v as f64), b.iter().map(|&v| v as f64));
        min_cos = min_cos.min(cos);
        sum_cos += cos;
        for j in 0..dim {
            centered_a[j] = a[j] as f64 - reference_mean[j];
            centered_b[j] = b[j] as f64 - reference_mean[j];
            let diff = a[j] as f64 - b[j] as f64;
            diff_sq += diff * diff;
            ref_sq += (b[j] as f64).powi(2);
            max_abs = max_abs.max(diff.abs());
        }
        let centered = cosine(centered_a.iter().copied(), centered_b.iter().copied());
        min_centered = min_centered.min(centered);
        sum_centered += centered;
    }
    let pooled = column_mean(tokens, dim);
    Parity {
        tokens: n,
        min_token_cosine: min_cos,
        mean_token_cosine: sum_cos / n as f64,
        min_centered_cosine: min_centered,
        mean_centered_cosine: sum_centered / n as f64,
        pooled_cosine: cosine(pooled.iter().copied(), reference_pooled.iter().map(|&v| v as f64)),
        cls_cosine: cls.map(|(a, b)| cosine(a.iter().map(|&v| v as f64), b.iter().map(|&v| v as f64))),
        max_abs_diff: max_abs,
        relative_l2: (diff_sq / ref_sq).sqrt(),
    }
}

/// Mean over rows of an `[N, D]` row-major matrix.
pub fn column_mean(values: &[f32], dim: usize) -> Vec<f64> {
    let n = values.len() / dim;
    let mut mean = vec![0.0; dim];
    for row in values.chunks_exact(dim) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    mean.iter_mut().for_each(|m| *m /= n as f64);
    mean
}

fn cosine(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.zip(b) {
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    dot / (aa.sqrt() * bb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_tokens_are_perfect() {
        let tokens = [1.0f32, 2.0, 3.0, -1.0, 0.5, 2.0];
        let pooled = [0.0f32, 1.25, 2.5];
        let parity = compare(&tokens, &tokens, &pooled, None, 3);
        assert_eq!(parity.tokens, 2);
        assert!((parity.min_token_cosine - 1.0).abs() < 1e-12);
        assert!((parity.min_centered_cosine - 1.0).abs() < 1e-12);
        assert!((parity.pooled_cosine - 1.0).abs() < 1e-12);
        assert_eq!(parity.max_abs_diff, 0.0);
    }

    #[test]
    fn centering_exposes_token_specific_error() {
        // Both tokens share a large offset, so raw cosine stays high while the residuals point
        // in opposite directions.
        let reference = [100.0f32, 1.0, 100.0, -1.0];
        let swapped = [100.0f32, -1.0, 100.0, 1.0];
        let parity = compare(&swapped, &reference, &[100.0, 0.0], None, 2);
        assert!(parity.min_token_cosine > 0.999);
        assert!(parity.max_abs_diff == 2.0);
        assert!((parity.min_centered_cosine + 1.0).abs() < 1e-12);
    }
}
