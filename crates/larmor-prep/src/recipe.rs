//! The preprocessing recipe from a model manifest's `[[preprocessing.steps]]` tables.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

/// Version of this crate's numerics, part of the cache key. Bump it whenever a change could move
/// any output (mask, conform, quantize, normalize or tokens) for any input.
pub const IMPL_VERSION: &str = "1";

/// A problem with the `[[preprocessing.steps]]` of a manifest.
#[derive(Debug, Error, PartialEq)]
pub enum RecipeError {
    /// A step table could not be parsed (unknown op, unknown key, wrong type).
    #[error("preprocessing step {index}: {message}")]
    Step {
        /// Position of the step, from 1.
        index: usize,
        /// What was wrong.
        message: String,
    },
    /// The steps are valid on their own but not in the order the recipe needs.
    #[error(
        "preprocessing steps must be threshold_mask, conform, quantize, normalize (in that order); got {0}"
    )]
    Order(String),
    /// A value is outside what this implementation supports.
    #[error("preprocessing step `{op}`: {message}")]
    Value {
        /// The step's `op`.
        op: &'static str,
        /// What was wrong.
        message: String,
    },
}

/// `op = "threshold_mask"`: the head mask, from nanobrain's `threshold_mask`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ThresholdMask {
    /// Isotropic voxel size (mm) the mask is computed at.
    pub resolution: f64,
    /// Gaussian smoothing sigma (mm).
    pub sigma: f64,
    /// Keep only the largest 6-connected component.
    pub remove_islands: bool,
    /// Fill enclosed holes.
    pub fill_holes: bool,
}

impl Default for ThresholdMask {
    /// nanobrain's defaults: `resolution=2.0, sigma=4.0, remove_islands=True, fill_holes=True`.
    fn default() -> Self {
        ThresholdMask { resolution: 2.0, sigma: 4.0, remove_islands: true, fill_holes: true }
    }
}

/// `op = "conform"`: crop and resample to the fixed model grid, from `conform_image`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Conform {
    /// Output voxel size (mm); finer input axes are anti-aliased first.
    pub min_voxel_size: f64,
    /// Field of view (mm) per axis; the grid is `ceil(fov / min_voxel_size)` voxels.
    pub fov: [f64; 3],
    /// Margin (mm) kept above the top of the head mask.
    pub top_margin: f64,
    /// Resample onto exactly `fov` (zero padded). Only `true` is supported.
    pub fixed_grid: bool,
    /// Spline order of the final resampling: 0 (nearest) or 1 (linear).
    pub order: i64,
}

impl Default for Conform {
    /// The values `eval/prepare.py::prepare_images` passes: `max_fov=GRID_FOV=(192, 240, 192)`,
    /// `fixed_grid=True`, and `conform_image`'s own defaults for the rest.
    fn default() -> Self {
        Conform {
            min_voxel_size: 1.0,
            fov: [192.0, 240.0, 192.0],
            top_margin: 5.0,
            fixed_grid: true,
            order: 1,
        }
    }
}

/// `op = "quantize"`: clip to in-mask quantiles and scale to uint16, from `truncate_image`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Quantize {
    /// Lower quantile (`VMIN_QUANTILE`).
    pub low: f64,
    /// Upper quantile (`VMAX_QUANTILE`).
    pub high: f64,
}

impl Default for Quantize {
    fn default() -> Self {
        Quantize { low: 0.005, high: 0.995 }
    }
}

/// How the model input is normalised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormalizeMethod {
    /// `ViTMAE3D.dense_embed`: divide in-mask values by their max, then z-score them.
    MaxZscore,
}

/// `op = "normalize"`: the model-input normalisation in `ViTMAE3D.dense_embed`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Normalize {
    /// The normalisation.
    pub method: NormalizeMethod,
    /// Lower bound for the standard deviation (`clamp_min`).
    pub std_floor: f64,
}

impl Default for Normalize {
    fn default() -> Self {
        Normalize { method: NormalizeMethod::MaxZscore, std_floor: 1e-6 }
    }
}

/// One `[[preprocessing.steps]]` table, tagged by its `op` key.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Step {
    /// `op = "threshold_mask"`.
    ThresholdMask(ThresholdMask),
    /// `op = "conform"`.
    Conform(Conform),
    /// `op = "quantize"`.
    Quantize(Quantize),
    /// `op = "normalize"`.
    Normalize(Normalize),
}

impl Step {
    /// The `op` name.
    pub fn op(&self) -> &'static str {
        match self {
            Step::ThresholdMask(_) => "threshold_mask",
            Step::Conform(_) => "conform",
            Step::Quantize(_) => "quantize",
            Step::Normalize(_) => "normalize",
        }
    }
}

/// The nanobrain-eval preprocessing recipe: what `eval/prepare.py::prepare_images` and
/// `ViTMAE3D.dense_embed` do to a scan before the encoder sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct Recipe {
    /// Head-mask step.
    pub mask: ThresholdMask,
    /// Conform step.
    pub conform: Conform,
    /// Quantize step.
    pub quantize: Quantize,
    /// Normalize step.
    pub normalize: Normalize,
}

impl Default for Recipe {
    fn default() -> Self {
        Recipe::nanobrain_eval()
    }
}

impl Recipe {
    /// The recipe nanobrain's eval framework uses, with every value at its default.
    pub fn nanobrain_eval() -> Recipe {
        Recipe {
            mask: ThresholdMask::default(),
            conform: Conform::default(),
            quantize: Quantize::default(),
            normalize: Normalize::default(),
        }
    }

    /// Parses and validates the `[[preprocessing.steps]]` tables of a `model.toml`.
    ///
    /// Keys left out take nanobrain's defaults. Unknown ops or keys, steps out of order, and
    /// values this implementation does not support are errors that name the step.
    pub fn from_steps(tables: &[toml::Table]) -> Result<Recipe, RecipeError> {
        let steps = tables
            .iter()
            .enumerate()
            .map(|(i, table)| {
                toml::Value::Table(table.clone()).try_into::<Step>().map_err(|e| RecipeError::Step {
                    index: i + 1,
                    message: e.to_string().split_whitespace().collect::<Vec<_>>().join(" "),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let recipe = match steps.as_slice() {
            [
                Step::ThresholdMask(mask),
                Step::Conform(conform),
                Step::Quantize(quantize),
                Step::Normalize(normalize),
            ] => Recipe {
                mask: mask.clone(),
                conform: conform.clone(),
                quantize: quantize.clone(),
                normalize: normalize.clone(),
            },
            _ => return Err(RecipeError::Order(steps.iter().map(Step::op).collect::<Vec<_>>().join(", "))),
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// The steps as `[[preprocessing.steps]]` would list them, every value explicit.
    pub fn steps(&self) -> Vec<Step> {
        vec![
            Step::ThresholdMask(self.mask.clone()),
            Step::Conform(self.conform.clone()),
            Step::Quantize(self.quantize.clone()),
            Step::Normalize(self.normalize.clone()),
        ]
    }

    /// Checks the values this implementation can run.
    pub fn validate(&self) -> Result<(), RecipeError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        let fail = |op, message: &str| Err(RecipeError::Value { op, message: message.to_string() });
        if !positive(self.mask.resolution) || !positive(self.mask.sigma) {
            return fail("threshold_mask", "resolution and sigma must be positive (mm)");
        }
        let c = &self.conform;
        if !positive(c.min_voxel_size) || !c.fov.iter().all(|&f| positive(f)) {
            return fail("conform", "min_voxel_size and fov must be positive (mm)");
        }
        if !c.top_margin.is_finite() {
            return fail("conform", "top_margin must be finite");
        }
        if !c.fixed_grid {
            return fail("conform", "only fixed_grid = true is supported (the model needs a fixed grid)");
        }
        if !(0..=1).contains(&c.order) {
            return fail(
                "conform",
                "order must be 0 (nearest) or 1 (linear); higher spline orders are not implemented",
            );
        }
        let q = &self.quantize;
        if !(0.0 <= q.low && q.low < q.high && q.high <= 1.0) {
            return fail("quantize", "need 0 <= low < high <= 1");
        }
        if !positive(self.normalize.std_floor) {
            return fail("normalize", "std_floor must be positive");
        }
        Ok(())
    }

    /// The model grid this recipe produces, `ceil(fov / min_voxel_size)` per axis.
    ///
    /// Mirrors `new_shape` in `conform_image` (fixed-grid branch).
    pub fn grid_shape(&self) -> [usize; 3] {
        self.conform.fov.map(|w| (w / self.conform.min_voxel_size).ceil() as usize)
    }

    /// The preprocessing version for cache keys: SHA-256 (hex) of the canonical JSON
    /// `{"impl_version": IMPL_VERSION, "steps": [...]}`, where the steps carry every value
    /// (defaults filled in) and object keys are sorted.
    pub fn version_hash(&self) -> String {
        let doc = serde_json::json!({ "impl_version": IMPL_VERSION, "steps": self.steps() });
        let digest = Sha256::digest(canonical_json(&doc).as_bytes());
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// JSON text with object keys sorted and no whitespace, independent of serde_json's map order.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let fields: Vec<String> = keys
                .iter()
                .map(|k| format!("{}:{}", Value::String((*k).clone()), canonical_json(&map[*k])))
                .collect();
            format!("{{{}}}", fields.join(","))
        }
        Value::Array(items) => {
            format!("[{}]", items.iter().map(canonical_json).collect::<Vec<_>>().join(","))
        }
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST_STEPS: &str = r#"
        [[steps]]
        op = "threshold_mask"
        resolution = 2.0
        sigma = 4.0
        remove_islands = true
        fill_holes = true

        [[steps]]
        op = "conform"
        min_voxel_size = 1.0
        fov = [192.0, 240.0, 192.0]
        top_margin = 5.0
        fixed_grid = true
        order = 1

        [[steps]]
        op = "quantize"
        low = 0.005
        high = 0.995

        [[steps]]
        op = "normalize"
        method = "max_zscore"
        std_floor = 1e-6
    "#;

    fn tables(text: &str) -> Vec<toml::Table> {
        let doc: toml::Table = text.parse().unwrap();
        doc["steps"].as_array().unwrap().iter().map(|v| v.as_table().unwrap().clone()).collect()
    }

    #[test]
    fn manifest_steps_parse_to_the_default_recipe() {
        let recipe = Recipe::from_steps(&tables(MANIFEST_STEPS)).unwrap();
        assert_eq!(recipe, Recipe::nanobrain_eval());
        assert_eq!(recipe.grid_shape(), [192, 240, 192]);
    }

    #[test]
    fn omitted_keys_take_defaults_and_hash_the_same() {
        let sparse = "[[steps]]\nop='threshold_mask'\n[[steps]]\nop='conform'\n[[steps]]\nop='quantize'\n[[steps]]\nop='normalize'";
        let recipe = Recipe::from_steps(&tables(sparse)).unwrap();
        assert_eq!(recipe.version_hash(), Recipe::nanobrain_eval().version_hash());
        assert_eq!(recipe.version_hash().len(), 64);
    }

    #[test]
    fn integer_values_are_accepted_for_floats() {
        let text = MANIFEST_STEPS.replace("resolution = 2.0", "resolution = 2");
        assert_eq!(Recipe::from_steps(&tables(&text)).unwrap().mask.resolution, 2.0);
    }

    #[test]
    fn canonical_json_is_stable() {
        let doc =
            serde_json::json!({ "impl_version": IMPL_VERSION, "steps": Recipe::nanobrain_eval().steps() });
        let expected = concat!(
            r#"{"impl_version":"1","steps":["#,
            r#"{"fill_holes":true,"op":"threshold_mask","remove_islands":true,"resolution":2.0,"sigma":4.0},"#,
            r#"{"fixed_grid":true,"fov":[192.0,240.0,192.0],"min_voxel_size":1.0,"op":"conform","order":1,"top_margin":5.0},"#,
            r#"{"high":0.995,"low":0.005,"op":"quantize"},"#,
            r#"{"method":"max_zscore","op":"normalize","std_floor":1e-6}]}"#
        );
        assert_eq!(canonical_json(&doc), expected);
    }

    #[test]
    fn changing_a_value_changes_the_hash() {
        let mut other = Recipe::nanobrain_eval();
        other.quantize.high = 0.99;
        assert_ne!(other.version_hash(), Recipe::nanobrain_eval().version_hash());
    }

    #[test]
    fn errors_name_the_problem() {
        let unknown_op = MANIFEST_STEPS.replace("op = \"quantize\"", "op = \"histogram_match\"");
        let err = Recipe::from_steps(&tables(&unknown_op)).unwrap_err().to_string();
        assert!(err.contains("step 3") && err.contains("histogram_match"), "{err}");

        let unknown_key = MANIFEST_STEPS.replace("top_margin = 5.0", "top_margin = 5.0\ncrop = true");
        let err = Recipe::from_steps(&tables(&unknown_key)).unwrap_err().to_string();
        assert!(err.contains("step 2") && err.contains("crop"), "{err}");

        let cubic = MANIFEST_STEPS.replace("order = 1", "order = 3");
        let err = Recipe::from_steps(&tables(&cubic)).unwrap_err();
        assert!(matches!(err, RecipeError::Value { op: "conform", .. }), "{err}");

        let free_grid = MANIFEST_STEPS.replace("fixed_grid = true", "fixed_grid = false");
        assert!(Recipe::from_steps(&tables(&free_grid)).is_err());

        let mut steps = tables(MANIFEST_STEPS);
        steps.swap(2, 3);
        let err = Recipe::from_steps(&steps).unwrap_err().to_string();
        assert!(err.contains("threshold_mask, conform, normalize, quantize"), "{err}");
    }
}
