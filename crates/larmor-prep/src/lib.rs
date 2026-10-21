//! The `nanobrain-eval` preprocessing recipe: from a raw NIfTI scan to the normalised image and
//! patch tokens nanobrain's encoder sees, matching the Python reference stage by stage.
//!
//! ```no_run
//! use larmor_prep::Recipe;
//! use larmor_volume::Nifti;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let manifest: toml::Table = std::fs::read_to_string("model.toml")?.parse()?;
//! let steps: Vec<toml::Table> = manifest["preprocessing"]["steps"]
//!     .as_array()
//!     .unwrap()
//!     .iter()
//!     .map(|step| step.as_table().unwrap().clone())
//!     .collect();
//! let recipe = Recipe::from_steps(&steps)?;
//!
//! let scan = Nifti::read("sub-01_T1w.nii.gz")?;
//! let prepared = recipe.run(&scan)?;
//! let tokens = prepared.tokens(8)?;
//! println!("{} tokens, cache key part {}", tokens.len(), &recipe.version_hash()[..8]);
//! # Ok(())
//! # }
//! ```
//!
//! | Step | Here | nanobrain |
//! |---|---|---|
//! | 1 | [`threshold_mask`] | `utils/preprocessing.py::threshold_mask` |
//! | 2–4 | [`conform_image`] | `eval/preprocessing.py::conform_image` (fixed grid) |
//! | 5 | in [`Recipe::run`] | `eval/prepare.py::prepare_images` (`resample_from_to(mask, conformed, order=0)`) |
//! | 6 | [`truncate_image`] | `eval/prepare.py::truncate_image` |
//! | 7 | [`max_zscore`] | `model.py::ViTMAE3D.dense_embed` |
//! | 8 | [`Prepared::tokens`] | `model.py::patchify3d` and `make_coord_grid` |
//!
//! A volume predicted on the model grid (a segmentation probe's output, say) goes back to the
//! scan's own voxels with [`Prepared::to_native`]. [`Recipe::version_hash`] and [`IMPL_VERSION`]
//! identify the preprocessing in cache keys.

#![warn(missing_docs)]

mod conform;
mod error;
mod mask;
mod normalize;
mod prepared;
mod quantize;
mod recipe;
mod scan;
mod tokens;

pub use conform::{Conformed, conform_image, rescale_affine};
pub use error::PrepError;
pub use mask::threshold_mask;
pub use normalize::max_zscore;
pub use prepared::{Prepared, Timings};
pub use quantize::{Quantized, truncate_image};
pub use recipe::{
    Conform, IMPL_VERSION, Normalize, NormalizeMethod, Quantize, Recipe, RecipeError, Step, ThresholdMask,
};
pub use tokens::{Tokens, tokens};
