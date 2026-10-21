//! `model.toml` manifests, safetensors weights and the 3D ViT encoder, on Candle.
//!
//! Mirrors `nanobrain.model.ViTMAE3D` (encoder half) and `nanobrain.modules` at nanobrain commit
//! `a1bfdd1`. The encoder is built from the manifest's `[architecture]` alone, so a new model
//! with supported values is a new `model.toml`, not new code.
//!
//! Devices: the CPU always; Metal (Apple GPUs) in macOS builds; CUDA with the `cuda` feature.
//! Attention uses Candle's fused `sdpa` kernel on Metal and query-chunked attention elsewhere
//! (see [`attention`]). Typical use:
//!
//! ```no_run
//! use larmor_model::{DeviceChoice, Encoder, Manifest, Precision};
//!
//! let manifest = Manifest::load("models/nanobrain-vits-p8.toml")?;
//! let encoder = Encoder::load(&manifest, DeviceChoice::Auto, Precision::F32)?;
//! # let (patches, coords): (Vec<f32>, Vec<[u16; 3]>) = (vec![], vec![]);
//! let encoded = encoder.encode(&patches, &coords)?;
//! println!("{} tokens, pooled[0] = {}", encoded.n_tokens, encoded.pooled[0]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod attention;
pub mod device;
pub mod encoder;
pub mod error;
pub mod manifest;
pub mod parity;
pub mod synthetic;
mod vit;
mod weights;

pub use attention::{AttentionKernel, AttentionOptions};
pub use device::{DeviceChoice, Precision};
pub use encoder::{Encoded, Encoder, EncoderOptions};
pub use error::{EncoderError, ManifestError};
pub use manifest::{Architecture, Manifest, Preprocessing, Readout, Weights, sha256_hex};
