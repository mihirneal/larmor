//! One scan through the cache: hash it, and on a miss read, preprocess, encode and cache it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use larmor_model::{DeviceChoice, Encoder};
use larmor_prep::Prepared;
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

/// The result of [`Pipeline::run`] for one scan.
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

/// Encodes scans with one model, loading the encoder only when a scan is not cached.
pub struct Pipeline<'a> {
    pub model: &'a Model,
    pub device: DeviceArg,
    pub precision: PrecisionArg,
    pub cache_root: PathBuf,
    /// Re-encode even when the cache has the scan.
    pub force: bool,
    encoder: Option<(Encoder, DeviceArg)>,
}

impl<'a> Pipeline<'a> {
    pub fn new(model: &'a Model, device: DeviceArg, precision: PrecisionArg, cache_root: PathBuf) -> Self {
        Pipeline { model, device, precision, cache_root, force: false, encoder: None }
    }

    /// The cached or freshly computed embedding of one scan. With `tokens`, the entry carries the
    /// patch tokens as well as the pooled embedding.
    pub fn run(&mut self, scan: &Path, tokens: bool) -> Result<ScanResult> {
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
                    return Ok(ScanResult {
                        entry,
                        cache_path,
                        outcome: Outcome::Cached,
                        prepared: None,
                        nifti: None,
                        read_hash_seconds: read + hash,
                    });
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
        let tokenize = start.elapsed().as_secs_f64();
        let (encoder, device) = self.encoder()?;
        let start = Instant::now();
        let encoded = encoder
            .encode(&patch_tokens.patches, &patch_tokens.coords)
            .with_context(|| format!("encoding {}", scan.display()))?;
        let encode = start.elapsed().as_secs_f64();

        let t = prepared.timings;
        let seconds = StageSeconds {
            read,
            hash,
            mask: t.mask.as_secs_f64(),
            conform: t.conform.as_secs_f64(),
            quantize: t.quantize.as_secs_f64(),
            normalize: t.normalize.as_secs_f64(),
            tokens: tokenize,
            encode,
            cache_write: 0.0,
            total: 0.0,
        };
        let mut info = RunInfo::new(
            self.model,
            Encoding { device, precision: self.precision, seconds, started },
            scan,
            &sha,
            &nifti,
            &prepared,
            encoded.n_tokens,
        )?;
        let start = Instant::now();
        CacheEntry::write(&cache_path, &encoded.tokens, &patch_tokens.coords, &encoded.pooled, &info)?;
        info.seconds.cache_write = start.elapsed().as_secs_f64();
        info.seconds.total = info.seconds.read
            + info.seconds.hash
            + t.total().as_secs_f64()
            + info.seconds.tokens
            + info.seconds.encode
            + info.seconds.cache_write;
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
            read_hash_seconds: read + hash,
        })
    }

    /// The encoder, loaded on first use.
    fn encoder(&mut self) -> Result<(&Encoder, DeviceArg)> {
        if self.encoder.is_none() {
            let start = Instant::now();
            let (encoder, device) = load_encoder(self.model, self.device, self.precision)?;
            let seconds = secs(start.elapsed().as_secs_f64());
            let line = format!("loaded {} on {device} ({}) in {seconds}", self.model.id(), self.precision);
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
