//! Comparison spike: nanobrain's 3D ViT encoder (`nanobrain/model.py::ViTMAE3D.forward_encoder`)
//! on Candle with the Metal backend, checked against the PyTorch reference fixtures and timed.
//!
//! Not production code; it exists so the owner can choose between Burn and Candle for
//! `larmor-model`.

pub mod bench;
pub mod config;
pub mod memory;
pub mod metrics;
pub mod model;
pub mod reference;
pub mod tokens;
pub mod weights;

/// Whether Candle compiles its Metal kernels with fast math (`CANDLE_METAL_ENABLE_FAST_MATH`,
/// default on). It is read when kernels are first compiled, so set it before starting.
pub fn metal_fast_math() -> bool {
    match std::env::var("CANDLE_METAL_ENABLE_FAST_MATH") {
        Ok(value) => matches!(value.as_str(), "true" | "t" | "yes" | "y" | "1"),
        Err(_) => true,
    }
}
