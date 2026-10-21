//! `larmor models`: list, verify and download model weights.

use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use larmor_model::Manifest;
use rayon::prelude::*;

use crate::console::{bold, dim, green, red, secs, yellow};
use crate::hashing::sha256_file;
use crate::locate;
use crate::model::{self, Model};

/// The state of a manifest's weights file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WeightsStatus {
    Ok,
    Missing,
    Mismatch(String),
    Unreadable(String),
}

impl WeightsStatus {
    fn of(manifest: &Manifest) -> WeightsStatus {
        let path = manifest.weights_path();
        if !path.is_file() {
            return WeightsStatus::Missing;
        }
        match sha256_file(&path) {
            Err(err) => WeightsStatus::Unreadable(err.to_string()),
            Ok(actual) => {
                if actual.eq_ignore_ascii_case(&manifest.weights.sha256) {
                    WeightsStatus::Ok
                } else {
                    WeightsStatus::Mismatch(actual)
                }
            }
        }
    }

    fn label(&self) -> String {
        match self {
            WeightsStatus::Ok => green("ok"),
            WeightsStatus::Missing => yellow("missing"),
            WeightsStatus::Mismatch(_) => red("sha256 MISMATCH"),
            WeightsStatus::Unreadable(_) => red("unreadable"),
        }
    }
}

/// `larmor models` / `larmor models list`.
pub fn cmd_list() -> Result<()> {
    let found = model::discover();
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    let mut broken = Vec::new();
    for f in found {
        match f.manifest {
            Ok(manifest) => {
                let shadowed = !seen.insert(manifest.id.clone());
                rows.push((manifest, f.path, shadowed));
            }
            Err(err) => broken.push((f.path, err)),
        }
    }
    let statuses: Vec<WeightsStatus> = rows.par_iter().map(|(m, _, _)| WeightsStatus::of(m)).collect();

    println!("{}", bold("Model folders, in search order:"));
    for dir in locate::model_dirs() {
        let state = if dir.is_dir() { String::new() } else { dim(" (does not exist)") };
        println!("  {}{state}", dir.display());
    }
    println!();
    if rows.is_empty() {
        println!("No models found. See `larmor models --help`, or pass --model path/to/model.toml.");
    } else {
        println!(
            "{}",
            dim(&format!(
                "{:<28} {:<7} {:<13} {:>5} {:>6} {:>5}  {:<16} {}",
                "id", "family", "grid", "patch", "width", "depth", "weights", "manifest"
            ))
        );
    }
    for ((manifest, path, shadowed), status) in rows.iter().zip(&statuses) {
        let arch = &manifest.architecture;
        let grid = format!("{}x{}x{}", arch.grid[0], arch.grid[1], arch.grid[2]);
        let mut weights = status.label();
        if *status == WeightsStatus::Missing && !manifest.weights.url.is_empty() {
            weights.push_str(&dim(" (download)"));
        }
        let pad = 16usize.saturating_sub(visible_len(&weights));
        let note = if *shadowed { dim("  (shadowed by an earlier one)") } else { String::new() };
        println!(
            "{:<28} {:<7} {:<13} {:>5} {:>6} {:>5}  {weights}{} {}{note}",
            manifest.id,
            manifest.family,
            grid,
            arch.patch,
            arch.width,
            arch.depth,
            " ".repeat(pad),
            path.display()
        );
    }
    for (path, err) in broken {
        println!("{} {} is not a usable model.toml: {}", yellow("note:"), path.display(), first_line(&err));
    }
    Ok(())
}

/// `larmor models verify [ID]`: recompute the weights SHA-256 of one model or of all.
pub fn cmd_verify(id: Option<&str>) -> Result<()> {
    let models: Vec<(Manifest, std::path::PathBuf)> = match id {
        Some(id) => {
            let model = Model::resolve(id)?;
            vec![(model.manifest, model.path)]
        }
        None => {
            let mut seen = HashSet::new();
            model::discover()
                .into_iter()
                .filter_map(|f| f.manifest.ok().map(|m| (m, f.path)))
                .filter(|(m, _)| seen.insert(m.id.clone()))
                .collect()
        }
    };
    if models.is_empty() {
        bail!("no models found; see `larmor models list`");
    }
    let mut bad = 0;
    for (manifest, path) in &models {
        let start = Instant::now();
        let status = WeightsStatus::of(manifest);
        let detail = match &status {
            WeightsStatus::Ok => {
                dim(&format!("{} in {}", &manifest.weights.sha256[..12], secs(start.elapsed().as_secs_f64())))
            }
            WeightsStatus::Missing => format!("{} not found", manifest.weights_path().display()),
            WeightsStatus::Mismatch(actual) => {
                format!("file has {actual}, {} expects {}", path.display(), manifest.weights.sha256)
            }
            WeightsStatus::Unreadable(err) => err.clone(),
        };
        println!("  {:<28} {}  {detail}", manifest.id, status.label());
        bad += usize::from(status != WeightsStatus::Ok);
    }
    if bad > 0 {
        bail!("{bad} of {} models failed verification", models.len());
    }
    Ok(())
}

/// `larmor models download ID`: fetch `weights.url` to the manifest's weights path and verify it.
pub fn cmd_download(id: &str) -> Result<()> {
    let model = Model::resolve(id)?;
    let manifest = &model.manifest;
    let dest = manifest.weights_path();
    if WeightsStatus::of(manifest) == WeightsStatus::Ok {
        println!("{} {} is already present and verified ({})", green("✓"), manifest.id, dest.display());
        return Ok(());
    }
    let url = manifest.weights.url.trim();
    if url.is_empty() {
        bail!(
            "{} has no weights.url; export the weights yourself (see models/README.md) to {}",
            model.path.display(),
            dest.display()
        );
    }
    println!("{} {} → {}", bold("downloading"), url, dest.display());
    let start = Instant::now();
    let part = dest.with_extension("part");
    let bytes = download(url, &part).inspect_err(|_| {
        let _ = std::fs::remove_file(&part);
    })?;
    let actual = sha256_file(&part)?;
    if !actual.eq_ignore_ascii_case(&manifest.weights.sha256) {
        let _ = std::fs::remove_file(&part);
        bail!(
            "downloaded file has sha256 {actual}, but {} expects {}",
            model.path.display(),
            manifest.weights.sha256
        );
    }
    std::fs::rename(&part, &dest).with_context(|| format!("moving the download to {}", dest.display()))?;
    println!(
        "{} {} verified · {:.1} MB in {}",
        green("✓"),
        dest.display(),
        bytes as f64 / 1e6,
        secs(start.elapsed().as_secs_f64())
    );
    Ok(())
}

/// Streams `url` into `path` with a progress line on stderr; returns the byte count.
///
/// `HF_TOKEN` is sent as a bearer token, but only to Hugging Face hosts so it never leaks to a
/// server named in someone else's manifest.
fn download(url: &str, path: &Path) -> Result<u64> {
    let mut request = ureq::get(url);
    if let Some(token) = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
        && is_hugging_face(url)
    {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    let mut response = request.call().with_context(|| format!("GET {url}"))?;
    let total = response.body().content_length();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut file = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?,
    );
    let mut reader = response.body_mut().as_reader();
    let mut buffer = vec![0u8; 1 << 20];
    let (mut done, mut last_report) = (0u64, Instant::now());
    let live = std::io::stderr().is_terminal();
    loop {
        let n = reader.read(&mut buffer).with_context(|| format!("reading {url}"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])?;
        done += n as u64;
        if live && last_report.elapsed().as_millis() >= 200 {
            last_report = Instant::now();
            eprint!("\r\x1b[2K  {}", progress(done, total));
        }
    }
    file.flush()?;
    if live {
        eprintln!("\r\x1b[2K  {}", progress(done, total));
    }
    if let Some(total) = total
        && done != total
    {
        bail!("download ended after {done} of {total} bytes");
    }
    Ok(done)
}

fn progress(done: u64, total: Option<u64>) -> String {
    match total {
        Some(total) if total > 0 => {
            format!(
                "{:.1} / {:.1} MB ({:.0}%)",
                done as f64 / 1e6,
                total as f64 / 1e6,
                100.0 * done as f64 / total as f64
            )
        }
        _ => format!("{:.1} MB", done as f64 / 1e6),
    }
}

/// True for `huggingface.co`, `hf.co` and their subdomains, over HTTPS.
fn is_hugging_face(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else { return false };
    let host = rest.split(['/', ':', '?', '#']).next().unwrap_or("").to_ascii_lowercase();
    ["huggingface.co", "hf.co"].iter().any(|h| host == *h || host.ends_with(&format!(".{h}")))
}

/// Printed width of text that may hold color escapes.
fn visible_len(text: &str) -> usize {
    let mut len = 0;
    let mut in_escape = false;
    for c in text.chars() {
        match (in_escape, c) {
            (false, '\x1b') => in_escape = true,
            (true, 'm') => in_escape = false,
            (false, _) => len += 1,
            _ => {}
        }
    }
    len
}

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_only_go_to_hugging_face() {
        assert!(is_hugging_face("https://huggingface.co/medarc/nanobrain/resolve/main/x.safetensors"));
        assert!(is_hugging_face("https://cdn-lfs.hf.co/x"));
        assert!(!is_hugging_face("http://huggingface.co/x"));
        assert!(!is_hugging_face("https://huggingface.co.evil.com/x"));
        assert!(!is_hugging_face("https://example.com/huggingface.co"));
    }

    #[test]
    fn escapes_do_not_count_as_width() {
        assert_eq!(visible_len("\x1b[32mok\x1b[0m"), 2);
    }
}
