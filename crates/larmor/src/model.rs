//! Finding a model by `--model` and pairing its manifest with its preprocessing recipe.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use larmor_model::Manifest;
use larmor_prep::Recipe;

use crate::locate;

/// A model as the CLI uses it: the parsed manifest, where it was found, and its recipe.
#[derive(Clone, Debug)]
pub struct Model {
    pub manifest: Manifest,
    /// The `model.toml` file.
    pub path: PathBuf,
    pub recipe: Recipe,
    /// [`Recipe::version_hash`], the preprocessing part of cache keys.
    pub prep_version: String,
}

impl Model {
    /// Pairs a manifest with its `[[preprocessing.steps]]` and checks that the recipe produces the
    /// grid the encoder expects.
    pub fn new(manifest: Manifest, path: PathBuf) -> Result<Model> {
        let recipe = Recipe::from_steps(&manifest.preprocessing.steps)
            .with_context(|| format!("{}: [[preprocessing.steps]]", path.display()))?;
        let grid = recipe.grid_shape();
        if grid != manifest.architecture.grid {
            bail!(
                "{}: preprocessing makes a {grid:?} grid but architecture.grid is {:?}",
                path.display(),
                manifest.architecture.grid
            );
        }
        let prep_version = recipe.version_hash();
        Ok(Model { manifest, path, recipe, prep_version })
    }

    /// Reads and checks a `model.toml`.
    pub fn load(path: &Path) -> Result<Model> {
        let manifest = Manifest::load(path)?;
        Model::new(manifest, path.to_path_buf())
    }

    /// `--model`: a path to a `.toml` file, or a model id looked up in [`locate::model_dirs`].
    pub fn resolve(arg: &str) -> Result<Model> {
        let path = Path::new(arg);
        if arg.ends_with(".toml") || path.is_file() {
            if !path.is_file() {
                bail!("model manifest {arg} does not exist");
            }
            return Model::load(path);
        }
        Model::find(arg)
    }

    /// The first manifest with this id in the model directories.
    pub fn find(id: &str) -> Result<Model> {
        let dirs = locate::model_dirs();
        for dir in &dirs {
            // `<id>.toml` is the usual name; a broken one is reported rather than skipped
            let direct = dir.join(format!("{id}.toml"));
            if direct.is_file() {
                let model = Model::load(&direct)?;
                if model.manifest.id == id {
                    return Ok(model);
                }
            }
            for found in manifests_in(dir) {
                if let Ok(manifest) = found.manifest
                    && manifest.id == id
                {
                    return Model::new(manifest, found.path);
                }
            }
        }
        let available: Vec<String> =
            discover().into_iter().filter_map(|f| f.manifest.ok().map(|m| m.id)).collect();
        bail!(
            "no model {id:?} found (searched {}); available: {}.\n  Pass a path to a model.toml, or add its folder to LARMOR_MODELS.",
            dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "),
            if available.is_empty() { "none".to_string() } else { available.join(", ") }
        )
    }

    pub fn id(&self) -> &str {
        &self.manifest.id
    }

    /// The weights SHA-256 from the manifest (lower case), the model part of cache keys.
    pub fn weights_sha256(&self) -> String {
        self.manifest.weights.sha256.to_ascii_lowercase()
    }

    pub fn patch(&self) -> usize {
        self.manifest.architecture.patch
    }

    pub fn width(&self) -> usize {
        self.manifest.architecture.width
    }
}

/// A `.toml` file found in a model directory, parsed as a manifest or not.
pub struct FoundManifest {
    pub path: PathBuf,
    pub manifest: Result<Manifest, String>,
}

/// Every `.toml` file directly inside `dir`, sorted by name.
pub fn manifests_in(dir: &Path) -> Vec<FoundManifest> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "toml") && p.is_file())
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| {
            let manifest = Manifest::load(&path).map_err(|e| e.to_string());
            FoundManifest { path, manifest }
        })
        .collect()
}

/// Every `.toml` file in every model directory, in search order.
pub fn discover() -> Vec<FoundManifest> {
    locate::model_dirs().iter().flat_map(|dir| manifests_in(dir)).collect()
}
