//! One scan through the cache: hash it, and on a miss read, preprocess, encode and cache it.
//!
//! The work is split in two so `larmor encode` can preprocess the next scan on another thread
//! while the GPU encodes the current one: [`Stager::stage`] does everything up to the patch tokens,
//! [`Pipeline::finish`] encodes and writes the cache entry.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use larmor_model::{DeviceChoice, Encoder};
use larmor_prep::{Prepared, Tokens};
use larmor_volume::Nifti;

use crate::cache::{self, CacheEntry};
use crate::console::{dim, secs, yellow};
use crate::device::{DeviceArg, PrecisionArg};
use crate::hashing::sha256_hex;
use crate::model::Model;
use crate::run_info::{Encoding, RunInfo, StageSeconds, now};

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
    scan: PathBuf,
    sha: String,
    cache_path: PathBuf,
    nifti: Nifti,
    prepared: Prepared,
    tokens: Tokens,
    seconds: StageSeconds,
    started: String,
}

/// Everything before the encoder: read, hash, look up the cache, preprocess, cut tokens. It holds
/// no encoder, so it can run on a worker thread.
#[derive(Clone)]
pub struct Stager<'a> {
    pub model: &'a Model,
    pub cache_root: PathBuf,
    /// Preprocess again even when the cache has the scan.
    pub force: bool,
}

impl Stager<'_> {
    /// Reads `scan` and either finds its embedding in the cache (with the patch tokens when
    /// `tokens`) or preprocesses it for the encoder.
    pub fn stage(&self, scan: &Path, tokens: bool) -> Result<Staged> {
        let started = now();
        let start = Instant::now();
        let bytes = std::fs::read(scan).with_context(|| format!("reading {}", scan.display()))?;
        let read = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let sha = sha256_hex(&bytes);
        let hash = start.elapsed().as_secs_f64();
        let cache_path = cache::entry_path(&self.cache_root, self.model, &sha);

        if !self.force && cache_path.is_file() {
            match CacheEntry::read(&cache_path, tokens) {
                Ok(entry) => {
                    return Ok(Staged::Cached(Box::new(ScanResult {
                        entry,
                        cache_path,
                        outcome: Outcome::Cached,
                        prepared: None,
                        nifti: None,
                        read_hash_seconds: read + hash,
                    })));
                }
                Err(err) => eprintln!("{} re-encoding: {err:#}", yellow("warning:")),
            }
        }

        let start = Instant::now();
        let nifti = Nifti::from_bytes(&bytes).with_context(|| format!("parsing {}", scan.display()))?;
        let read = read + start.elapsed().as_secs_f64();
        let prepared =
            self.model.recipe.run(&nifti).with_context(|| format!("preprocessing {}", scan.display()))?;
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
            scan: scan.to_path_buf(),
            sha,
            cache_path,
            nifti,
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
    pub device: DeviceArg,
    pub precision: PrecisionArg,
    encoder: Option<(Encoder, DeviceArg)>,
}

impl<'a> Pipeline<'a> {
    pub fn new(model: &'a Model, device: DeviceArg, precision: PrecisionArg, cache_root: PathBuf) -> Self {
        let stager = Stager { model, cache_root, force: false };
        Pipeline { stager, device, precision, encoder: None }
    }

    pub fn model(&self) -> &'a Model {
        self.stager.model
    }

    /// The cached or freshly computed embedding of one scan. With `tokens`, the entry carries the
    /// patch tokens as well as the pooled embedding.
    pub fn run(&mut self, scan: &Path, tokens: bool) -> Result<ScanResult> {
        let staged = self.stager.stage(scan, tokens)?;
        self.finish(staged, tokens)
    }

    /// Encodes a staged scan and writes its cache entry; a cached scan passes through.
    pub fn finish(&mut self, staged: Staged, tokens: bool) -> Result<ScanResult> {
        let ready = match staged {
            Staged::Cached(result) => return Ok(*result),
            Staged::Ready(ready) => *ready,
        };
        let Ready { scan, sha, cache_path, nifti, prepared, tokens: patch_tokens, mut seconds, started } =
            ready;
        let (encoder, device) = self.encoder()?;
        let start = Instant::now();
        let encoded = encoder
            .encode(&patch_tokens.patches, &patch_tokens.coords)
            .with_context(|| format!("encoding {}", scan.display()))?;
        seconds.encode = start.elapsed().as_secs_f64();

        let encoding = Encoding { device, precision: self.precision, seconds, started };
        let model = self.model();
        let mut info = RunInfo::new(model, encoding, &scan, &sha, &nifti, &prepared, encoded.n_tokens)?;
        let start = Instant::now();
        CacheEntry::write(&cache_path, &encoded.tokens, &patch_tokens.coords, &encoded.pooled, &info)?;
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
            tokens: if tokens { encoded.tokens } else { Vec::new() },
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

    /// The encoder, loaded on first use.
    fn encoder(&mut self) -> Result<(&Encoder, DeviceArg)> {
        if self.encoder.is_none() {
            let start = Instant::now();
            let (encoder, device) = load_encoder(self.model(), self.device, self.precision)?;
            let seconds = secs(start.elapsed().as_secs_f64());
            let line = format!("loaded {} on {device} ({}) in {seconds}", self.model().id(), self.precision);
            eprintln!("{}", dim(&line));
            self.encoder = Some((encoder, device));
        }
        let (encoder, device) = self.encoder.as_ref().expect("loaded above");
        Ok((encoder, *device))
    }
}

/// Loads a model's encoder on the requested device and says which device it got. `auto` resolves
/// to the GPU and falls back to the CPU, with a warning, when the GPU cannot be used.
pub fn load_encoder(
    model: &Model,
    requested: DeviceArg,
    precision: PrecisionArg,
) -> Result<(Encoder, DeviceArg)> {
    let device = requested.resolve();
    let manifest = &model.manifest;
    match Encoder::load(manifest, device.0, precision.0) {
        Ok(encoder) => Ok((encoder, device)),
        Err(err) if requested.0 == DeviceChoice::Auto && device.0 != DeviceChoice::Cpu => {
            eprintln!("{} {device} unavailable ({err}); using the CPU", yellow("warning:"));
            let cpu = DeviceArg(DeviceChoice::Cpu);
            let encoder = Encoder::load(manifest, cpu.0, precision.0)
                .with_context(|| format!("loading {}", model.path.display()))?;
            Ok((encoder, cpu))
        }
        Err(err) => Err(err).with_context(|| format!("loading {}", model.path.display())),
    }
}
