//! Linear probes on foundation-model embeddings, with the exact numbers of nanobrain's eval framework
//! (`nanobrain/eval/probe.py` with scikit-learn 1.8.0).
//!
//! - [`regression`]: `probe_regression` (StandardScaler + RidgeCV with efficient leave-one-out).
//! - [`classification`]: `probe_binary_classification` (StandardScaler + LogisticRegressionCV).
//! - [`segmentation`]: `probe_binary_segmentation` (per-voxel L2 logistic heads on patch tokens, on a Burn
//!   device).
//! - [`fit`]: what `larmor probe fit` runs: the eval-identical CV plus a deployable probe refit on
//!   everything.
//! - [`probe_file`]: `probe.toml` + `probe.safetensors`, and the check that refuses mismatched embeddings.
//!
//! Folds, bootstrap resamples and negative subsampling reproduce numpy's and torch's generators ([`rng`]), so
//! the numbers match without a fold column. Nothing here depends on a particular encoder: the embedding width
//! and the number of voxels per patch come from the data.

pub mod bootstrap;
pub mod classification;
pub mod fit;
pub mod folds;
pub mod lbfgsb;
pub mod logistic;
pub mod matrix;
pub mod metrics;
pub mod numpy;
pub mod probe_file;
pub mod regression;
pub mod ridge;
pub mod rng;
pub mod scaler;
pub mod segmentation;

pub use matrix::Matrix;
