//! `larmor models`: list, verify and download model weights.

use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use larmor_model::Manifest;
use rayon::prelude::*;
use ureq::http::Uri;

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

/// Options of `larmor models download`.
pub struct DownloadOptions {
    pub id: String,
    /// Replace an existing weights file whose SHA-256 does not match the manifest.
    pub force: bool,
    /// Accept a plain `http://` URL.
    pub allow_http: bool,
}

/// `larmor models download ID`: fetch `weights.url` to the manifest's weights path and verify it.
pub fn cmd_download(options: &DownloadOptions) -> Result<()> {
    let model = Model::resolve(&options.id)?;
    let manifest = &model.manifest;
    let dest = manifest.weights_path();
    match WeightsStatus::of(manifest) {
        WeightsStatus::Ok => {
            println!("{} {} is already present and verified ({})", green("✓"), manifest.id, dest.display());
            return Ok(());
        }
        WeightsStatus::Missing => {}
        WeightsStatus::Mismatch(actual) if !options.force => bail!(
            "{} exists, but its sha256 is {actual}, not the {} that {} names; refusing to replace it (pass \
             --force to replace it with the download)",
            dest.display(),
            manifest.weights.sha256,
            model.path.display()
        ),
        WeightsStatus::Unreadable(err) if !options.force => {
            bail!("{} exists but cannot be read ({err}); pass --force to replace it", dest.display())
        }
        _ => {}
    }
    let url = manifest.weights.url.trim();
    if url.is_empty() {
        bail!(
            "{} has no weights.url; export the weights yourself (see models/README.md) to {}",
            model.path.display(),
            dest.display()
        );
    }
    let url = download_url(url, options.allow_http)
        .with_context(|| format!("{}: weights.url", model.path.display()))?;
    check_inside(&dest, &manifest.dir)?;
    println!("{} {} → {}", bold("downloading"), url, dest.display());
    let start = Instant::now();
    let part = dest.with_extension("part");
    let bytes = download(&url, &part).inspect_err(|_| {
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

/// The URL to fetch: `https://` (or `http://` with `allow_http`), with no user name or password in
/// it. A bare `hf.co` host becomes `huggingface.co`, where `hf.co` redirects: the bearer token is
/// not sent on across a redirect to another host, so gated files would fail through `hf.co`.
fn download_url(url: &str, allow_http: bool) -> Result<String> {
    let uri: Uri = url.parse().with_context(|| format!("{url:?} is not a valid URL"))?;
    let scheme = uri.scheme_str().unwrap_or("").to_ascii_lowercase();
    match scheme.as_str() {
        "https" => {}
        "http" if allow_http => {}
        "http" => bail!("{url} is plain http; use an https:// URL (or pass --allow-http)"),
        _ => bail!("{url}: only https:// URLs are downloaded"),
    }
    let authority = uri.authority().with_context(|| format!("{url} has no host"))?;
    if authority.as_str().contains('@') {
        bail!("{url} has a user name or password before its host; larmor refuses such URLs");
    }
    let host = authority.host().to_ascii_lowercase();
    if host == "hf.co" {
        let rest = uri.path_and_query().map_or("/", |p| p.as_str());
        let port = authority.port().map_or_else(String::new, |p| format!(":{p}"));
        return Ok(format!("{scheme}://huggingface.co{port}{rest}"));
    }
    Ok(url.to_string())
}

/// Refuses a destination outside the manifest's folder (the manifest check already requires a
/// plain file name; this guards the folder itself, through symlinks too).
fn check_inside(dest: &Path, manifest_dir: &Path) -> Result<()> {
    let dir = if manifest_dir.as_os_str().is_empty() { Path::new(".") } else { manifest_dir };
    let canonical_dir = std::fs::canonicalize(dir).with_context(|| format!("resolving {}", dir.display()))?;
    let parent = dest.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let canonical_parent =
        std::fs::canonicalize(parent).with_context(|| format!("resolving {}", parent.display()))?;
    if canonical_parent != canonical_dir {
        bail!("{} is not in the manifest's folder {}", dest.display(), dir.display());
    }
    Ok(())
}

/// The `Authorization` header for `url`: `HF_TOKEN` as a bearer token, and only for Hugging Face
/// hosts over HTTPS, so it never leaks to a server named in someone else's manifest.
fn authorization(url: &str, token: Option<&str>) -> Option<String> {
    let token = token.filter(|t| !t.is_empty())?;
    is_hugging_face(url).then(|| format!("Bearer {token}"))
}

/// Streams `url` into `path` with a progress line on stderr; returns the byte count.
fn download(url: &str, path: &Path) -> Result<u64> {
    let mut request = ureq::get(url);
    if let Some(value) = authorization(url, std::env::var("HF_TOKEN").ok().as_deref()) {
        request = request.header("Authorization", value);
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

/// True for `huggingface.co`, `hf.co` and their subdomains, over HTTPS, as a URL parser reads the
/// host (the part after any `user:password@`, which is refused outright).
fn is_hugging_face(url: &str) -> bool {
    let Ok(uri) = url.parse::<Uri>() else { return false };
    let Some(authority) = uri.authority() else { return false };
    if !uri.scheme_str().is_some_and(|s| s.eq_ignore_ascii_case("https")) || authority.as_str().contains('@')
    {
        return false;
    }
    let host = authority.host().to_ascii_lowercase();
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
        assert!(is_hugging_face("HTTPS://HuggingFace.co/x"));
        assert!(is_hugging_face("https://huggingface.co:443/x"));
        assert!(!is_hugging_face("http://huggingface.co/x"));
        assert!(!is_hugging_face("https://huggingface.co.evil.com/x"));
        assert!(!is_hugging_face("https://example.com/huggingface.co"));
        // a host before '@' is user information; the request goes to the host after it
        for url in [
            "https://huggingface.co:443@evil.example/x",
            "https://huggingface.co:x@evil.example/x",
            "https://hf.co:1@evil.example/x",
            "https://huggingface.co@evil.example/x",
            "https://huggingface.co%40evil.example/x",
            "HTTPS://evil.example/?x=huggingface.co",
            "https://evil.example#huggingface.co",
        ] {
            assert!(!is_hugging_face(url), "{url}");
            assert_eq!(authorization(url, Some("hf_secret")), None, "{url}");
        }
        assert_eq!(authorization("https://huggingface.co/x", Some("t")).as_deref(), Some("Bearer t"));
        assert_eq!(authorization("https://huggingface.co/x", Some("")), None);
        assert_eq!(authorization("https://huggingface.co/x", None), None);
    }

    #[test]
    fn download_urls_must_be_https_without_user_information() {
        assert_eq!(
            download_url("https://example.org/w.safetensors", false).unwrap(),
            "https://example.org/w.safetensors"
        );
        assert!(
            download_url("http://example.org/w", false).unwrap_err().to_string().contains("--allow-http")
        );
        assert!(download_url("http://127.0.0.1:8765/w", true).is_ok());
        assert!(download_url("ftp://example.org/w", true).is_err());
        assert!(
            download_url("https://huggingface.co:443@evil.example/x", false)
                .unwrap_err()
                .to_string()
                .contains("user name")
        );
        assert!(download_url("not a url", false).is_err());
        // hf.co redirects to huggingface.co, which would drop the token: go there directly
        assert_eq!(
            download_url("https://hf.co/org/repo/resolve/main/w.safetensors?download=1", false).unwrap(),
            "https://huggingface.co/org/repo/resolve/main/w.safetensors?download=1"
        );
        assert_eq!(download_url("https://cdn-lfs.hf.co/x", false).unwrap(), "https://cdn-lfs.hf.co/x");
    }

    #[test]
    fn escapes_do_not_count_as_width() {
        assert_eq!(visible_len("\x1b[32mok\x1b[0m"), 2);
    }
}
