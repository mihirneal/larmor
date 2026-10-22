//! One scan through the cache: hash it, and on a miss read, preprocess, encode and cache it.
//!
//! The work is split in two so `larmor encode` can preprocess the next scan on another thread
//! while the GPU encodes the current one: [`Stager::stage`] does everything up to the patch tokens,
//! [`Pipeline::finish`] encodes and writes the cache entry.

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use larmor_dicom::SeriesInfo;
use larmor_model::{DeviceChoice, Encoder, EncoderError, ManifestError, Precision};
use larmor_prep::{Prepared, Tokens};
use larmor_volume::Nifti;

use crate::cache::{self, CacheEntry, EntryKey};
use crate::console::{dim, secs, yellow};
use crate::model::Model;
use crate::run_info::{Encoding, ReferenceInfo, RunInfo, StageSeconds, absolute, now};
use crate::scan::{ReadScan, Scan};

/// Whether a scan's embedding came from the cache or was computed now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Cached,
    Encoded,
}

/// The embedding of one scan, cached or computed.
pub struct ScanResult {
    pub entry: CacheEntry,
    pub cache_path: PathBuf,
    pub outcome: Outcome,
    /// The preprocessed scan, when this run computed it.
    pub prepared: Option<Prepared>,
    /// The parsed scan, when this run read it.
    pub nifti: Option<Nifti>,
    /// Seconds spent reading and hashing the file on this run (the cache entry's timings are from
    /// the run that encoded it).
    pub read_hash_seconds: f64,
}

/// A scan after [`Stager::stage`]: found in the cache, or preprocessed and waiting for the encoder.
pub enum Staged {
    Cached(Box<ScanResult>),
    Ready(Box<Ready>),
}

/// A preprocessed scan whose embedding is not cached yet.
pub struct Ready {
    scan: Scan,
    sha: String,
    reference: Option<ReferenceInfo>,
    cache_path: PathBuf,
    nifti: Nifti,
    dicom: Option<SeriesInfo>,
    prepared: Prepared,
    tokens: Tokens,
    seconds: StageSeconds,
    started: String,
}

/// The reference scan's SHA-256 as the cache key needs it: `None` when there is no reference or it
/// is the scan itself (whose own head mask gives the same result).
pub fn reference_key<'a>(scan_sha256: &str, reference_sha256: Option<&'a str>) -> Option<&'a str> {
    reference_sha256.filter(|r| *r != scan_sha256)
}

/// Everything before the encoder: read, hash, look up the cache, preprocess, cut tokens. It holds
/// no encoder, so it can run on a worker thread.
#[derive(Clone)]
pub struct Stager<'a> {
    pub model: &'a Model,
    pub cache_root: PathBuf,
    /// Preprocess again even when the cache has the scan.
    pub force: bool,
    /// The precision the encoder computes in, part of the cache key.
    pub precision: Precision,
    /// Set once the encoder failed to load; uncached scans are then not preprocessed for nothing.
    encoder_failed: Arc<AtomicBool>,
}

impl Stager<'_> {
    /// This stager keyed by another precision (to look for entries made in it).
    pub fn with_precision(&self, precision: Precision) -> Self {
        Stager { precision, ..self.clone() }
    }

    /// The cache file for a scan (and reference scan) with these SHA-256s.
    pub fn entry_path(&self, scan_sha256: &str, reference_sha256: Option<&str>) -> PathBuf {
        cache::entry_path(&self.cache_root, self.model, &self.key(scan_sha256, reference_sha256))
    }

    fn key<'k>(&self, scan_sha256: &'k str, reference_sha256: Option<&'k str>) -> EntryKey<'k> {
        EntryKey {
            scan_sha256,
            reference_sha256: reference_key(scan_sha256, reference_sha256),
            precision: self.precision,
        }
    }

    /// The cached embedding of a scan, `Ok(None)` when there is none, or the reason an existing
    /// entry cannot be used (unreadable, truncated, another key in its metadata).
    pub fn lookup(
        &self,
        scan_sha256: &str,
        reference_sha256: Option<&str>,
        tokens: bool,
    ) -> Result<Option<CacheEntry>> {
        let key = self.key(scan_sha256, reference_sha256);
        let path = cache::entry_path(&self.cache_root, self.model, &key);
        if !path.is_file() {
            return Ok(None);
        }
        CacheEntry::read(&path, self.model, &key, tokens).map(Some)
    }

    /// [`Stager::stage`], with a panic turned into this scan's error, so a worker thread keeps
    /// going with the next scan.
    pub fn stage_catching(&self, scan: &Scan, reference: Option<&Scan>, tokens: bool) -> Result<Staged> {
        panic::catch_unwind(AssertUnwindSafe(|| self.stage(scan, reference, tokens))).unwrap_or_else(
            |payload| {
                Err(anyhow!(
                    "internal error preprocessing {}: {}",
                    scan.display(),
                    panic_message(payload.as_ref())
                ))
            },
        )
    }

    /// Reads `scan` (and `reference`, the scan whose head mask and grid to use) and either finds its
    /// embedding in the cache (with the patch tokens when `tokens`) or preprocesses it for the
    /// encoder.
    pub fn stage(&self, scan: &Scan, reference: Option<&Scan>, tokens: bool) -> Result<Staged> {
        let started = now();
        let start = Instant::now();
        let content = scan.read_content()?;
        let reference_content = reference
            .map(|r| r.read_content().with_context(|| format!("reading the reference {}", r.display())))
            .transpose()?;
        let read = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let target = ReadScan::new(scan.clone(), content);
        let reference =
            reference.zip(reference_content).map(|(r, content)| ReadScan::new(r.clone(), content));
        // a scan that is its own reference is simply the scan
        let reference = reference.filter(|r| r.sha256 != target.sha256);
        let hash = start.elapsed().as_secs_f64();
        let reference_sha = reference.as_ref().map(|r| r.sha256.as_str());
        let cache_path = self.entry_path(&target.sha256, reference_sha);

        if !self.force {
            match self.lookup(&target.sha256, reference_sha, tokens) {
                Ok(Some(entry)) => {
                    return Ok(Staged::Cached(Box::new(ScanResult {
                        entry,
                        cache_path,
                        outcome: Outcome::Cached,
                        prepared: None,
                        nifti: None,
                        read_hash_seconds: read + hash,
                    })));
                }
                Ok(None) => {}
                Err(err) => eprintln!("{} re-encoding: {err:#}", yellow("warning:")),
            }
        }
        if self.encoder_failed.load(Ordering::Relaxed) {
            bail!("not encoded: the encoder could not be loaded (see the first error)");
        }

        let start = Instant::now();
        let parsed = target.parse()?;
        for note in &parsed.notes {
            eprintln!("{} {}: {note}", dim("note:"), scan.display());
        }
        let reference_nifti = reference.as_ref().map(|r| r.parse().map(|p| p.nifti)).transpose()?;
        let read = read + start.elapsed().as_secs_f64();
        let nifti = parsed.nifti;
        let prepared = match &reference_nifti {
            None => self.model.recipe.run(&nifti),
            Some(reference_nifti) => self.model.recipe.run_with_reference(&nifti, reference_nifti),
        };
        let prepared = prepared.with_context(|| match &reference {
            None => format!("preprocessing {}", scan.display()),
            Some(r) => format!("preprocessing {} on the head mask of {}", scan.display(), r.scan.display()),
        })?;
        let start = Instant::now();
        let patch_tokens = prepared.tokens(self.model.patch())?;
        let t = prepared.timings;
        let seconds = StageSeconds {
            read,
            hash,
            mask: t.mask.as_secs_f64(),
            conform: t.conform.as_secs_f64(),
            quantize: t.quantize.as_secs_f64(),
            normalize: t.normalize.as_secs_f64(),
            tokens: start.elapsed().as_secs_f64(),
            ..StageSeconds::default()
        };
        Ok(Staged::Ready(Box::new(Ready {
            scan: scan.clone(),
            sha: target.sha256,
            reference: reference.map(|r| ReferenceInfo { path: absolute(r.scan.path()), sha256: r.sha256 }),
            cache_path,
            nifti,
            dicom: parsed.dicom,
            prepared,
            tokens: patch_tokens,
            seconds,
            started,
        })))
    }
}

/// Encodes scans with one model, loading the encoder only when a scan is not cached.
pub struct Pipeline<'a> {
    pub stager: Stager<'a>,
    pub device: DeviceChoice,
    encoder: Option<Encoder>,
}

impl<'a> Pipeline<'a> {
    /// A pipeline that encodes in `precision`, or in f32 when `device` is `auto`, half precision is
    /// asked for and this machine has no GPU (with a warning): the precision the encoder will
    /// actually run in, which is part of the cache key.
    pub fn new(model: &'a Model, device: DeviceChoice, precision: Precision, cache_root: PathBuf) -> Self {
        let precision = if precision != Precision::F32
            && device == DeviceChoice::Auto
            && device.resolve() == DeviceChoice::Cpu
        {
            eprintln!("{} no GPU found; encoding in f32 on the CPU", yellow("warning:"));
            Precision::F32
        } else {
            precision
        };
        let stager = Stager {
            model,
            cache_root,
            force: false,
            precision,
            encoder_failed: Arc::new(AtomicBool::new(false)),
        };
        Pipeline { stager, device, encoder: None }
    }

    pub fn model(&self) -> &'a Model {
        self.stager.model
    }

    /// The precision embeddings are computed in and cached under.
    pub fn precision(&self) -> Precision {
        self.stager.precision
    }

    /// The cached or freshly computed embedding of one scan, preprocessed on its own or on the head
    /// mask and grid of `reference`. With `tokens`, the entry carries the patch tokens as well as
    /// the pooled embedding.
    pub fn run(&mut self, scan: &Scan, reference: Option<&Scan>, tokens: bool) -> Result<ScanResult> {
        let staged = self.stager.stage_catching(scan, reference, tokens)?;
        self.finish(staged, tokens)
    }

    /// Encodes a staged scan and writes its cache entry; a cached scan passes through.
    pub fn finish(&mut self, staged: Staged, tokens: bool) -> Result<ScanResult> {
        let ready = match staged {
            Staged::Cached(result) => return Ok(*result),
            Staged::Ready(ready) => *ready,
        };
        let Ready {
            scan,
            sha,
            reference,
            cache_path,
            nifti,
            dicom,
            prepared,
            tokens: patch_tokens,
            mut seconds,
            started,
        } = ready;
        let encoder = self.encoder()?;
        let start = Instant::now();
        let encoded = encoder
            .encode(&patch_tokens.patches, &patch_tokens.coords)
            .with_context(|| format!("encoding {}", scan.display()))?;
        seconds.encode = start.elapsed().as_secs_f64();
        // what a cache hit hands out, so probes see the same features either way
        let stored_tokens = cache::as_stored(&encoded.tokens);
        if !encoded.pooled.iter().chain(&stored_tokens).all(|v| v.is_finite()) {
            bail!(
                "encoding {}: the encoder produced NaN or infinite values (or tokens beyond float16's \
                 ±65504, the cache's storage type); nothing was cached",
                scan.display()
            );
        }

        let encoding = Encoding::new(encoder, seconds, started);
        let model = self.model();
        let reference_sha = reference.as_ref().map(|r| r.sha256.clone());
        let mut info = RunInfo::new(
            model,
            encoding,
            scan.path(),
            &sha,
            &nifti,
            dicom.as_ref(),
            &prepared,
            encoded.n_tokens,
            reference,
        )?;
        let key = EntryKey {
            scan_sha256: &sha,
            reference_sha256: reference_sha.as_deref(),
            precision: self.precision(),
        };
        let start = Instant::now();
        CacheEntry::write(&cache_path, &key, &encoded.tokens, &patch_tokens.coords, &encoded.pooled, &info)?;
        let s = &mut info.seconds;
        s.cache_write = start.elapsed().as_secs_f64();
        s.total = s.read
            + s.hash
            + s.mask
            + s.conform
            + s.quantize
            + s.normalize
            + s.tokens
            + s.encode
            + s.cache_write;
        let read_hash_seconds = s.read + s.hash;
        let entry = CacheEntry {
            tokens: if tokens { stored_tokens } else { Vec::new() },
            coords: patch_tokens.coords,
            pooled: encoded.pooled,
            info,
        };
        Ok(ScanResult {
            entry,
            cache_path,
            outcome: Outcome::Encoded,
            prepared: Some(prepared),
            nifti: Some(nifti),
            read_hash_seconds,
        })
    }

    /// The encoder, loaded on first use. A load failure is a configuration error, the same for
    /// every scan: it is reported once, and later scans fail at once without loading again.
    fn encoder(&mut self) -> Result<&Encoder> {
        if self.stager.encoder_failed.load(Ordering::Relaxed) {
            bail!("not encoded: the encoder could not be loaded (see the first error)");
        }
        if self.encoder.is_none() {
            let start = Instant::now();
            let encoder = match load_encoder(self.model(), self.device, self.precision()) {
                Ok(encoder) => encoder,
                Err(err) => {
                    self.stager.encoder_failed.store(true, Ordering::Relaxed);
                    return Err(err);
                }
            };
            let seconds = secs(start.elapsed().as_secs_f64());
            let options = encoder.options();
            let line = format!(
                "loaded {} on {} ({}) in {seconds}",
                self.model().id(),
                options.device,
                options.precision
            );
            eprintln!("{}", dim(&line));
            self.encoder = Some(encoder);
        }
        Ok(self.encoder.as_ref().expect("loaded above"))
    }
}

/// Loads a model's encoder on the requested device; [`Encoder::options`] says which device and
/// precision it got. `auto` picks CUDA, then Metal, then the CPU. Half precision needs a GPU, so
/// when `auto` finds none it falls back to f32 on the CPU, with a warning.
pub fn load_encoder(model: &Model, device: DeviceChoice, precision: Precision) -> Result<Encoder> {
    let manifest = &model.manifest;
    let loaded = match Encoder::load(manifest, device, precision) {
        // under `auto` only the half-precision-on-the-CPU check reports a device error
        Err(EncoderError::Device(reason)) if device == DeviceChoice::Auto && precision != Precision::F32 => {
            eprintln!("{} no GPU found ({reason}); encoding in f32 on the CPU", yellow("warning:"));
            Encoder::load(manifest, DeviceChoice::Cpu, Precision::F32)
        }
        other => other,
    };
    loaded.map_err(|err| {
        let hint = match &err {
            EncoderError::Device(_) => {
                " (choose another --device, such as metal or cpu, or for CUDA build larmor with --features cuda)"
                    .to_string()
            }
            EncoderError::Manifest(ManifestError::Checksum { .. } | ManifestError::Read { .. }) => format!(
                " (run `larmor models verify {id}`, or `larmor models download {id}` if the manifest has a \
                 weights url)",
                id = model.id()
            ),
            _ => String::new(),
        };
        anyhow!("{err}{hint}").context(format!("loading {}", model.path.display()))
    })
}

/// The text of a panic payload.
pub fn panic_message(payload: &(dyn Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic")
        .to_string()
}
