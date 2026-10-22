//! `larmor watch --site site.toml`: encode scans as they arrive in watched folders.
//!
//! The sources are listed every `poll_seconds` (polling, not file events: cloud folders such as Box
//! Drive and network shares do not report changes reliably). A file is settled once its size and
//! modification time have not changed for `settle_seconds` of our own observation (Box Drive keeps
//! a file's original modification time, so its age says nothing about when it arrived). A NIfTI
//! file is encoded once it is settled; a DICOM series once every file in its folder is (a series
//! arrives a file at a time). Headers are read only from settled files, once. Every scan goes
//! through the embedding cache, its outputs mirror its folder below the source, and every action is
//! appended to the log. What was encoded (or failed) is remembered in a ledger by the files' sizes
//! and times, so a restart does not read them again, and a scan whose files change is encoded again.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::encode_cmd::{WriteOptions, write_outputs};
use crate::locate::{self, Analysis, Candidate, Skipped};
use crate::model::Model;
use crate::outputs::ScanOutputs;
use crate::pipeline::{Outcome, Pipeline};
use crate::run_info::now;
use crate::scan::Scan;
use crate::site::{Site, Source};

/// Options of `larmor watch`.
pub struct WatchOptions {
    pub site: PathBuf,
    /// Encode what is settled now, then exit.
    pub once: bool,
}

/// The log: every line to the file (with the time) and to stdout.
struct Log {
    file: std::fs::File,
}

impl Log {
    fn open(path: &Path) -> Result<Log> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening the log {}", path.display()))?;
        Ok(Log { file })
    }

    fn line(&mut self, text: &str) {
        let line = format!("{} {text}", now());
        println!("{line}");
        // a full disk must not stop the watcher; the line is on stdout either way
        let _ = writeln!(self.file, "{line}");
        let _ = self.file.flush();
    }
}

/// What the watcher did with a scan, kept across restarts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Ledger {
    /// By scan id: `nifti:<path>` or `dicom:<SeriesInstanceUID>@<folder>`.
    scans: BTreeMap<String, LedgerEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LedgerEntry {
    /// Its files' paths, sizes and modification times, hashed.
    fingerprint: String,
    /// `encoded`, `cached` or `failed`.
    status: String,
    /// The error of a failed scan.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    message: String,
    /// The scan's SHA-256 (as in run.json), when it was read.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    sha256: String,
    time: String,
}

impl Ledger {
    fn read(path: &Path) -> Result<Ledger> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display())),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Ledger::default()),
            Err(err) => Err(err).with_context(|| format!("reading {}", path.display())),
        }
    }

    fn write(&self, path: &Path) -> Result<()> {
        crate::outputs::write_json(path, self)
    }
}

/// A file's size and modification time, and since when they have been what they are.
struct FileState {
    size: u64,
    modified: Option<SystemTime>,
    /// When this poll loop first saw the file with this size and time.
    since: Instant,
    /// Its header, read once it settled.
    analysis: Option<Analysis>,
}

/// What the sources hold, poll after poll.
struct Watcher {
    settle: Duration,
    files: HashMap<PathBuf, FileState>,
    /// Folders with files that have not settled, as last logged.
    busy: HashSet<PathBuf>,
    /// Sources found missing (logged once until they come back).
    missing: HashSet<String>,
    /// Notes already logged (skipped series, unreadable files), so each is logged once.
    noted: HashSet<String>,
    /// Sources listed at least once (the first listing is summarised in one line).
    listed: HashSet<String>,
}

impl Watcher {
    fn new(settle_seconds: f64) -> Watcher {
        Watcher {
            settle: Duration::from_secs_f64(settle_seconds),
            files: HashMap::new(),
            busy: HashSet::new(),
            missing: HashSet::new(),
            noted: HashSet::new(),
            listed: HashSet::new(),
        }
    }

    /// Lists `source`, updates the file states, and returns its scans whose files are settled.
    fn poll(&mut self, source: &Source, skip: &[PathBuf], log: &mut Log) -> Vec<Scan> {
        if !source.path.is_dir() {
            if self.missing.insert(source.name.clone()) {
                log.line(&format!(
                    "source {}: {} is not a folder (yet); waiting for it",
                    source.name,
                    source.path.display()
                ));
            }
            return Vec::new();
        }
        if self.missing.remove(&source.name) {
            log.line(&format!("source {}: {} is back", source.name, source.path.display()));
        }
        let mut skipped = Skipped::default();
        let candidates = match locate::candidates(&source.path, skip, &mut skipped) {
            Ok(candidates) => candidates,
            Err(err) => {
                log.line(&format!("source {}: cannot list {}: {err}", source.name, source.path.display()));
                return Vec::new();
            }
        };
        let now = Instant::now();
        let mut seen = HashSet::new();
        let mut changed_folders: BTreeMap<PathBuf, usize> = BTreeMap::new();
        for candidate in &candidates {
            let path = candidate.path().to_path_buf();
            seen.insert(path.clone());
            // a file that vanished between listing and stat is simply not there this time
            let Ok(meta) = std::fs::metadata(&path) else { continue };
            let (size, modified) = (meta.len(), meta.modified().ok());
            let unchanged = self.files.get(&path).is_some_and(|s| s.size == size && s.modified == modified);
            if !unchanged {
                self.files.insert(path.clone(), FileState { size, modified, since: now, analysis: None });
                *changed_folders.entry(folder_of(&path)).or_default() += 1;
            }
        }
        // files that disappeared from this source
        let root = &source.path;
        let gone: Vec<PathBuf> =
            self.files.keys().filter(|p| p.starts_with(root) && !seen.contains(*p)).cloned().collect();
        for path in &gone {
            self.files.remove(path);
        }
        let shown = |folder: &Path| folder.strip_prefix(root).unwrap_or(folder).display().to_string();
        if self.listed.insert(source.name.clone()) {
            // everything is new to a watcher that just started
            self.busy.extend(changed_folders.keys().cloned());
            log.line(&format!(
                "source {}: {} candidate files in {} folders; each is read once it has not changed for {} s",
                source.name,
                candidates.len(),
                changed_folders.len(),
                self.settle.as_secs_f64()
            ));
        } else {
            for (folder, n) in &changed_folders {
                if self.busy.insert(folder.clone()) {
                    log.line(&format!(
                        "source {}: {n} new or changed file{} in {}/; waiting until they settle",
                        source.name,
                        if *n == 1 { "" } else { "s" },
                        shown(folder)
                    ));
                }
            }
        }

        let settled = |state: &FileState| now.duration_since(state.since) >= self.settle;
        // a folder is settled when every candidate file directly in it is
        let mut unsettled_folders = HashSet::new();
        for candidate in &candidates {
            if self.files.get(candidate.path()).is_some_and(|s| !settled(s)) {
                unsettled_folders.insert(folder_of(candidate.path()));
            }
        }
        self.busy.retain(|folder| unsettled_folders.contains(folder));

        let mut analyzed: Vec<(Candidate, Analysis)> = Vec::new();
        for candidate in candidates {
            let Some(state) = self.files.get_mut(candidate.path()) else { continue };
            let ready = match &candidate {
                Candidate::Nifti(_) => now.duration_since(state.since) >= self.settle,
                Candidate::MaybeDicom(path) => !unsettled_folders.contains(&folder_of(path)),
            };
            if !ready {
                continue;
            }
            let analysis = state.analysis.get_or_insert_with(|| locate::analyze(&candidate)).clone();
            analyzed.push((candidate, analysis));
        }
        let scans = locate::assemble(analyzed, &mut skipped);
        for note in skipped.notes() {
            if self.noted.insert(note.clone()) {
                log.line(&format!("source {}: {note}", source.name));
            }
        }
        scans
    }

    /// The fingerprint of a scan's files as they are now (paths, sizes and modification times) and
    /// of how it is encoded (`how`: the models' encoder keys, the precision...).
    fn fingerprint(&self, scan: &Scan, how: &str) -> String {
        let files = match scan {
            Scan::Nifti(path) => vec![path.clone()],
            Scan::Dicom(dicom) => dicom.series.files(),
        };
        let mut text = format!("{how}\n");
        for path in files {
            let state = self.files.get(&path);
            let modified = state
                .and_then(|s| s.modified)
                .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            text.push_str(&format!("{}\t{}\t{modified}\n", path.display(), state.map_or(0, |s| s.size)));
        }
        crate::hashing::sha256_hex(text.as_bytes())
    }
}

fn folder_of(path: &Path) -> PathBuf {
    path.parent().map(Path::to_path_buf).unwrap_or_default()
}

/// The ledger key of a scan.
fn scan_id(scan: &Scan) -> String {
    match scan {
        Scan::Nifti(path) => format!("nifti:{}", path.display()),
        Scan::Dicom(dicom) => format!("dicom:{}@{}", dicom.series.uid, dicom.folder.display()),
    }
}

pub fn cmd_watch(options: WatchOptions) -> Result<()> {
    let site = Site::read(&options.site)?;
    let models = site.models.iter().map(|m| Model::resolve(m)).collect::<Result<Vec<Model>>>()?;
    let cache_root = locate::cache_root()?;
    let mut pipelines: Vec<Pipeline> = models
        .iter()
        .map(|model| Pipeline::new(model, site.device, site.precision, cache_root.clone()))
        .collect();
    let mut log = Log::open(&site.log)?;
    let mut ledger = Ledger::read(&site.ledger)?;
    let mut watcher = Watcher::new(site.settle_seconds);
    // outputs are never inputs (created now, so their canonical paths match what a listing sees)
    for source in &site.sources {
        std::fs::create_dir_all(&source.out).with_context(|| format!("creating {}", source.out.display()))?;
    }
    let skip: Vec<PathBuf> = site
        .sources
        .iter()
        .map(|s| s.out.clone())
        .chain(site.log.parent().map(Path::to_path_buf))
        .filter_map(|dir| std::fs::canonicalize(&dir).ok().or(Some(dir)))
        .collect();
    let model_names: Vec<String> =
        models.iter().map(|m| format!("{} ({})", m.id(), &m.encoder_key[..8])).collect();
    // a scan is done for this site when its files and this are what they were
    let how = format!(
        "{} {} {} {}",
        models.iter().map(|m| m.encoder_key.as_str()).collect::<Vec<_>>().join(","),
        models.iter().map(|m| m.prep_version.as_str()).collect::<Vec<_>>().join(","),
        pipelines[0].precision(),
        site.save_preprocessed
    );
    log.line(&format!(
        "watching {} with {} · {} · poll {} s, settle {} s{}",
        site.sources
            .iter()
            .map(|s| format!("{} ({})", s.name, s.path.display()))
            .collect::<Vec<_>>()
            .join(", "),
        model_names.join(", "),
        pipelines[0].precision(),
        site.poll_seconds,
        site.settle_seconds,
        if options.once { " · once" } else { "" }
    ));
    log.line(&format!(
        "site {} · cache {} · log {}",
        site.path.display(),
        cache_root.display(),
        site.log.display()
    ));

    if options.once && site.settle_seconds > 0.0 {
        // the first look starts every file's settle clock; the second one encodes what stayed put
        for source in &site.sources {
            watcher.poll(source, &skip, &mut log);
        }
        std::thread::sleep(Duration::from_secs_f64(site.settle_seconds));
    }
    loop {
        for source in &site.sources {
            for scan in watcher.poll(source, &skip, &mut log) {
                let id = scan_id(&scan);
                let fingerprint = watcher.fingerprint(&scan, &how);
                let outputs = ScanOutputs::new(&source.out, &scan, Some(&source.path));
                let written = models.iter().all(|m| outputs.run_json(m.id()).is_file());
                let done = ledger.scans.get(&id).is_some_and(|e| e.fingerprint == fingerprint);
                // a failed scan is tried again only when its files change; a done one also when its
                // outputs were removed
                if done && (written || ledger.scans[&id].status == "failed") {
                    continue;
                }
                let entry = encode(&site, source, &scan, &mut pipelines, fingerprint, &mut log);
                ledger.scans.insert(id, entry);
                if let Err(err) = ledger.write(&site.ledger) {
                    log.line(&format!("cannot write the ledger {}: {err:#}", site.ledger.display()));
                }
            }
        }
        if options.once {
            log.line("done (--once)");
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs_f64(site.poll_seconds));
    }
}

/// Encodes one settled scan with every model, writes its outputs and logs what happened.
fn encode(
    site: &Site,
    source: &Source,
    scan: &Scan,
    pipelines: &mut [Pipeline],
    fingerprint: String,
    log: &mut Log,
) -> LedgerEntry {
    let name = scan.path().strip_prefix(&source.path).unwrap_or(scan.path()).display().to_string();
    let what = match scan {
        Scan::Nifti(_) => String::new(),
        Scan::Dicom(dicom) => {
            format!(" (DICOM {}, {} files)", dicom.series.label(), dicom.series.headers.len())
        }
    };
    log.line(&format!("source {}: {name}{what} settled; encoding", source.name));
    let outputs = ScanOutputs::new(&source.out, scan, Some(&source.path));
    let write = WriteOptions { save_preprocessed: site.save_preprocessed, force: false };
    let (mut status, mut message, mut sha256) = ("cached".to_string(), String::new(), String::new());
    for pipeline in pipelines.iter_mut() {
        let start = Instant::now();
        let model = pipeline.model();
        let result = pipeline.run(scan, None, false).and_then(|result| {
            write_outputs(model, scan, None, &source.out, &outputs, &result, &write)?;
            Ok(result)
        });
        match result {
            Ok(result) => {
                let outcome = match result.outcome {
                    Outcome::Encoded => {
                        status = "encoded".into();
                        "encoded"
                    }
                    Outcome::Cached => "cached",
                };
                sha256 = result.entry.info.scan.sha256.clone();
                log.line(&format!(
                    "source {}: {outcome} {name} with {} · {} tokens · {:.2} s → {}",
                    source.name,
                    model.id(),
                    result.entry.info.tokens,
                    start.elapsed().as_secs_f64(),
                    outputs.embedding(model.id()).display()
                ));
            }
            Err(err) => {
                status = "failed".into();
                message = format!("{err:#}");
                log.line(&format!("source {}: failed {name} with {}: {message}", source.name, model.id()));
            }
        }
    }
    LedgerEntry { fingerprint, status, message, sha256, time: now() }
}
