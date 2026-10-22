# Larmor

One Rust program for Macs that turns an MRI scan into foundation-model embeddings and fits and applies
linear probes on them, in the style of Onset (`~/Developer/onset`). Every number is checked against the
PyTorch code it mirrors: MedARC-AI/nanobrain's eval framework at commit `a1bfdd1`.

**macOS only.** Larmor runs on Apple silicon Macs with Metal. Don't add Windows or Linux code, docs, or CI.

## How we work

Claude builds and fixes Larmor directly. The owner is a Rust beginner who knows Python best and knows the
MRI side well (NIfTI, BIDS, preprocessing, foundation models, probes):

- When a change uses a Rust idea worth knowing, explain it briefly by comparing it to Python. Don't explain
  the MRI side.
- Larmor should stay model-agnostic: a new model is a new `model.toml`, not new code, as long as its
  architecture and preprocessing use supported values. nanobrain's ViT is the only model today.
- Work happens on more than one Mac, so keep this file current when decisions change.

## Layout

| Path | What it is |
|---|---|
| `docs/DESIGN.md` | The contract: recipe, manifest, fixture layouts, cache, probes, gates, runtime choice |
| `docs/CLI.md` | Every command, flag and output file |
| `crates/larmor-volume` | NIfTI I/O, affines, orientation, resampling, Gaussian filter, morphology, quantiles (mirrors nibabel, scipy.ndimage, numpy) |
| `crates/larmor-dicom` | DICOM series (dicom-rs) to the NIfTI image dcm2niix v1.0.20260724 writes, bit for bit; synthetic series for tests |
| `crates/larmor-prep` | The `nanobrain-eval` preprocessing recipe, tokens, mapping back to native space |
| `crates/larmor-model` | `model.toml` manifests, safetensors weights, the 3D ViT encoder on Candle |
| `crates/larmor-probe` | Ridge, logistic and segmentation probes, sklearn-identical folds, numpy RNG ports, metrics, probe files |
| `crates/larmor` | The CLI: `encode`, `probe fit`/`apply`, `watch`, `models`, `check-parity`, `bench` |
| `examples/` | `labels.tsv` and `site.toml` (for `larmor watch`), annotated; never real lab paths |
| `tools/` | Python reference tools (pinned to nanobrain's `uv.lock`, plus dcm2niix): fixtures, PyTorch references, checkpoint export |
| `fixtures/` | Not committed. 36 scans, 15 DICOM sets and their references, rebuilt by `tools/setup_fixtures.sh`. In a worktree, symlink it: `ln -s /Users/mihir/Developer/larmor/fixtures fixtures` |

## Checks

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
  CI on GitHub runs these on macOS. Its runners may not expose Metal, so tests run on the CPU and the ones
  that need Metal are `#[ignore]`d.
- Parity (needs `fixtures/`): `cargo test --workspace --release -- --ignored` and
  `target/release/larmor check-parity fixtures` (both reference models and the DICOM sets; exits nonzero if
  any gate fails). Run these after any change to preprocessing, the encoder, the probes or the DICOM reader.
- Timings: run `larmor bench` on a quiet machine (no other builds or GPU jobs) and say so next to the numbers.

## Design decisions

- **Parity target:** the eval framework's path (`eval/prepare.py::prepare_images` then
  `ViTMAE3D.dense_embed`), not pretraining's `data.py`. The mask is nanobrain's threshold head mask, so no
  skull stripper is needed.
- **Bit-exact preprocessing on this Mac:** matching nibabel/scipy/numpy to the bit needed their exact
  operation order (Accelerate's 4x4 matmul and LAPACK inverse, clang's fused multiply-adds in scipy, float32
  vs float64 promotion, half-to-even rounding). Against a Linux-built Python stack expect last-bit
  differences. The normalized input differs by one float32 step on 4 of 25 scans because torch's float32
  mean depends on its thread count; Larmor keeps the exactly rounded mean.
- **Runtime: Candle, not Burn.** Both passed parity; Burn was 2–3× slower on Metal (see DESIGN.md). Devices:
  Metal (the default) and the CPU (fallback and tests). Fused `sdpa` on Metal, chunked attention elsewhere.
  `f16`/`bf16` are autocast-like and GPU-only.
- **Cache:** key = scan bytes SHA-256 (+ the reference scan's for multi-modal input) + recipe `version_hash`
  (includes `larmor_prep::IMPL_VERSION`) + `Manifest::encoder_key` (weights SHA-256, every architecture and
  readout value, `larmor_model::ENCODER_IMPL_VERSION`) + precision. Bump `IMPL_VERSION` whenever a change
  could move any preprocessing output, and `ENCODER_IMPL_VERSION` for any encoder output.
- **DICOM:** the target is dcm2niix v1.0.20260724's NIfTI (default options) as nibabel loads it, because
  that is how the lab's data reach nanobrain; `larmor-dicom` ports `saveDcm2NiiCore` for one 3-D classic
  volume (instance-number sort, position re-sort, `nifti_dicom2mat`, `nii_flipZ`, `nii_setOrtho` for 3-D
  acquisitions, `nii_flipY` otherwise, Philips precise scaling) in float32 with clang's fused multiply-adds
  (`-ffp-contract=on`), so affines are bit-identical to the macOS/PyPI build. What dcm2niix would not write
  as one volume (localizers, 4-D, several echoes, varying spacing, mosaics) is refused with the reason.
  Enhanced multi-frame and JPEG-LS are refused for now. Bump `larmor_dicom::IMPL_VERSION` for any change
  that could move a converted image: it is part of a series' SHA-256 and so of its cache key.
- **Gate data for DICOM:** CC0 DICOM of MR heads is scarce (OpenNeuro keeps NIfTI), so the fixtures use
  dcm2niix's own validation sets (`neurolabusc/dcm_qa_*`, BSD-2-Clause), pinned by commit and git blob SHA-1.
- **Watching:** `larmor watch` polls (cloud folders such as Box Drive do not deliver file events reliably)
  and waits until files have not changed for `settle_seconds` of its own observation; a DICOM series waits
  for its whole folder. A ledger keyed by file sizes and times plus the models' encoder keys makes restarts
  cheap and re-encodes series that grow.
- **Multi-modal input:** `Recipe::run_with_reference` / `--reference` mirrors `prepare_images(...,
  reference=)` (FOMO tasks 1 and 2: ADC and DWI on the FLAIR's mask). `fomo_task5_t1w_pmg`'s
  `strip_and_crop_ap` (SynthSeg brain mask and AP crop) is not supported.
- **Fail where the eval framework fails:** inputs that make sklearn or torch raise (too few of a class per
  training fold, NaN features, empty subsampled segmentation sets) are errors, not numbers.
- **Probes match the eval framework:** sklearn's folds and numpy's bootstrap resamples are reproduced exactly
  (MT19937 and PCG64 ports). The segmentation probe converges to the exact optimum; torch's L-BFGS stops a
  little short of it, so out-of-fold probabilities differ from torch's by up to about 1.5e-2 while the chosen
  alpha and threshold match.
- **Weights:** trained nanobrain checkpoints live on the cluster as `.pth`; export them with
  `tools/make_models.py export`. Reference models in `fixtures/models` are perturbed random inits.

## Status and open questions

- Phases 0–3 are done: `larmor encode` (NIfTI and DICOM), the cache, probes, `larmor watch` and every parity
  gate pass on all 36 fixture scans, the multi-modal pair and the 15 DICOM sets (`[[pair]]` and `[[dicom]]`
  in `fixtures/manifest.toml`).
- Not yet: Phase 4 (the egui viewer); enhanced multi-frame DICOM (Philips and Siemens XA export it; the
  dcm_qa_philips_enh and dcm_qa_enh repositories have sets with dcm2niix references) and JPEG-LS (needs
  CharLS, C++).
- Open: which trained checkpoint ships first (rerun `check-parity` with it); check-parity against references
  made on the cluster; a `watch` trial on the lab's real Box Drive folder (only local folders are tested).

## Setup on a new Mac

- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- If links fail with a macOS SDK error, pin builds to an installed older SDK with `SDKROOT` under `[env]` in
  `~/.cargo/config.toml` (as for Onset). Keep that file out of the repo.
- Fixtures and the Python reference env: `tools/setup_fixtures.sh` (needs `uv`; downloads about 120 MB of
  public scans and templates).
