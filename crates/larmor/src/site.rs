//! `site.toml`: where scans arrive and where `larmor watch` writes their embeddings, like Onset's
//! rig files. `examples/site.toml` is an annotated example.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use larmor_model::{DeviceChoice, Precision};
use serde::Deserialize;

/// A parsed site file, with every path made absolute.
#[derive(Clone, Debug)]
pub struct Site {
    /// The site file.
    pub path: PathBuf,
    pub sources: Vec<Source>,
    /// The log file every action is appended to.
    pub log: PathBuf,
    /// Where the watcher remembers what it encoded (`<output>/.larmor-watch.json`).
    pub ledger: PathBuf,
    /// Model ids, or absolute paths to `model.toml` files.
    pub models: Vec<String>,
    pub device: DeviceChoice,
    pub precision: Precision,
    pub save_preprocessed: bool,
    /// Seconds between two looks at the sources.
    pub poll_seconds: f64,
    /// A scan is encoded once its files have not changed for this long.
    pub settle_seconds: f64,
}

/// A watched folder.
#[derive(Clone, Debug)]
pub struct Source {
    /// The label in the log.
    pub name: String,
    /// The folder, searched recursively like `larmor encode <folder>`.
    pub path: PathBuf,
    /// The derivatives folder its scans' outputs go to (mirroring their folders below `path`).
    pub out: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SiteFile {
    #[serde(default)]
    source: Vec<SourceFile>,
    output: OutputFile,
    encode: EncodeFile,
    #[serde(default)]
    watch: WatchFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceFile {
    name: Option<String>,
    path: String,
    out: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputFile {
    dir: String,
    log: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EncodeFile {
    models: Vec<String>,
    #[serde(default = "auto")]
    device: String,
    #[serde(default = "f32")]
    precision: String,
    #[serde(default)]
    save_preprocessed: bool,
}

fn auto() -> String {
    "auto".into()
}

fn f32() -> String {
    "f32".into()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WatchFile {
    #[serde(default = "poll")]
    poll_seconds: f64,
    #[serde(default = "settle")]
    settle_seconds: f64,
}

impl Default for WatchFile {
    fn default() -> Self {
        WatchFile { poll_seconds: poll(), settle_seconds: settle() }
    }
}

fn poll() -> f64 {
    10.0
}

fn settle() -> f64 {
    60.0
}

/// `~/x` is in the home folder; a relative path is relative to the site file's folder.
fn expand(text: &str, base: &Path) -> Result<PathBuf> {
    let path = if text == "~" || text.starts_with("~/") {
        let home = dirs::home_dir().context("no home folder to expand ~ in")?;
        home.join(text.trim_start_matches('~').trim_start_matches('/'))
    } else {
        PathBuf::from(text)
    };
    let path = if path.is_absolute() { path } else { base.join(path) };
    Ok(std::path::absolute(&path).unwrap_or(path))
}

/// A folder name usable as a log label and an output folder.
fn label(path: &Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "source".into());
    let clean: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect();
    if clean.is_empty() { "source".into() } else { clean }
}

impl Site {
    pub fn read(path: &Path) -> Result<Site> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Site::parse(&text, path)
    }

    pub fn parse(text: &str, path: &Path) -> Result<Site> {
        let file: SiteFile = toml::from_str(text).with_context(|| format!("parsing {}", path.display()))?;
        let base = std::path::absolute(path)?.parent().map(Path::to_path_buf).unwrap_or_default();
        ensure!(!file.source.is_empty(), "{} lists no [[source]] folders to watch", path.display());
        ensure!(!file.encode.models.is_empty(), "{}: [encode] models is empty", path.display());
        let output = expand(&file.output.dir, &base)?;
        let mut sources = Vec::new();
        for source in &file.source {
            let folder = expand(&source.path, &base)?;
            let name = source.name.clone().unwrap_or_else(|| label(&folder));
            ensure!(
                !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
                "{}: source name {name:?} must be letters, digits, - or _",
                path.display()
            );
            if sources.iter().any(|s: &Source| s.name == name) {
                bail!("{}: two sources are named {name:?}; give them different `name`s", path.display());
            }
            let out = match &source.out {
                Some(out) => expand(out, &base)?,
                None => output.join(&name),
            };
            if folder.starts_with(&out) || out.starts_with(&folder) {
                bail!(
                    "{}: source {name} ({}) and its output folder ({}) must not contain each other",
                    path.display(),
                    folder.display(),
                    out.display()
                );
            }
            sources.push(Source { name, path: folder, out });
        }
        let models = file
            .encode
            .models
            .iter()
            .map(|m| {
                // a path to a model.toml is relative to the site file; anything else is a model id
                if m.ends_with(".toml") || m.contains('/') {
                    expand(m, &base).map(|p| p.display().to_string())
                } else {
                    Ok(m.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let device: DeviceChoice = file
            .encode
            .device
            .parse()
            .map_err(|e| anyhow::anyhow!("{}: [encode] device: {e}", path.display()))?;
        let precision: Precision = file
            .encode
            .precision
            .parse()
            .map_err(|e| anyhow::anyhow!("{}: [encode] precision: {e}", path.display()))?;
        let watch = file.watch;
        ensure!(watch.poll_seconds > 0.0, "{}: [watch] poll_seconds must be positive", path.display());
        ensure!(
            watch.settle_seconds >= 0.0,
            "{}: [watch] settle_seconds must not be negative",
            path.display()
        );
        let log = match &file.output.log {
            Some(log) => expand(log, &output)?,
            None => output.join("larmor-watch.log"),
        };
        Ok(Site {
            path: std::path::absolute(path)?,
            sources,
            log,
            ledger: output.join(".larmor-watch.json"),
            models,
            device,
            precision,
            save_preprocessed: file.encode.save_preprocessed,
            poll_seconds: watch.poll_seconds,
            settle_seconds: watch.settle_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_relative_to_the_site_file_and_names_default_to_folders() {
        let site = Site::parse(
            r#"
            [[source]]
            path = "incoming/Box Scans"

            [[source]]
            name = "scanner"
            path = "/data/scanner"
            out = "/data/out/scanner"

            [output]
            dir = "out"

            [encode]
            models = ["nanobrain-vits-p8", "models/other.toml"]
            precision = "bf16"
            "#,
            Path::new("/lab/site.toml"),
        )
        .unwrap();
        assert_eq!(site.sources[0].name, "Box_Scans");
        assert_eq!(site.sources[0].path, PathBuf::from("/lab/incoming/Box Scans"));
        assert_eq!(site.sources[0].out, PathBuf::from("/lab/out/Box_Scans"));
        assert_eq!(site.sources[1].out, PathBuf::from("/data/out/scanner"));
        assert_eq!(site.models, ["nanobrain-vits-p8", "/lab/models/other.toml"]);
        assert_eq!(site.log, PathBuf::from("/lab/out/larmor-watch.log"));
        assert_eq!((site.poll_seconds, site.settle_seconds), (10.0, 60.0));
        assert_eq!(site.precision, Precision::Bf16);
    }

    #[test]
    fn mistakes_are_refused() {
        let parse = |text: &str| Site::parse(text, Path::new("/lab/site.toml")).unwrap_err().to_string();
        let base = "[output]\ndir = \"/out\"\n[encode]\nmodels = [\"m\"]\n";
        assert!(parse(base).contains("no [[source]]"));
        let inside = format!("[[source]]\npath = \"/out/incoming\"\n{base}");
        assert!(parse(&inside).contains("must not contain each other"), "{}", parse(&inside));
        let typo = format!("[[source]]\npath = \"/in\"\nsetle = 3\n{base}");
        assert!(parse(&typo).contains("parsing"), "{}", parse(&typo));
        let twice = format!("[[source]]\npath = \"/a/in\"\n[[source]]\npath = \"/b/in\"\n{base}");
        assert!(parse(&twice).contains("two sources are named \"in\""), "{}", parse(&twice));
    }
}
