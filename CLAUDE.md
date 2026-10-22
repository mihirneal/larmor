# Larmor

One Rust program for Macs that turns an MRI scan into foundation-model embeddings and fits and applies
linear probes on them, in the style of Onset (`~/Developer/onset`). Every number is checked against the
PyTorch code it mirrors: MedARC-AI/nanobrain's eval framework at commit `a1bfdd1`.

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
| `crates/larmor-prep` | The `nanobrain-eval` preprocessing recipe, tokens, mapping back to native space |
| `crates/larmor-model` | `model.toml` manifests, safetensors weights, the 3D ViT encoder on Candle |
| `crates/larmor-probe` | Ridge, logistic and segmentation probes, sklearn-identical folds, numpy RNG ports, metrics, probe files |
| `crates/larmor` | The CLI: `encode`, `probe fit`/`apply`, `models`, `check-parity`, `bench` |
| `tools/` | Python reference tools (pinned to nanobrain's `uv.lock`): fixtures, PyTorch references, checkpoint export |
| `fixtures/` | Not committed. 25 scans plus their references, rebuilt by `tools/setup_fixtures.sh` |

## Checks

- `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`.
  CI on GitHub runs these on macOS, and builds and tests on Linux.
- Parity (needs `fixtures/`): `cargo test --workspace --release -- --ignored` and
  `target/release/larmor check-parity fixtures` (both reference models; exits nonzero if any gate fails).
  Run these after any change to preprocessing, the encoder or the probes.
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
- **Runtime: Candle, not Burn.** Both passed parity; Burn was 2–3× slower on Metal (see DESIGN.md). Fused
  `sdpa` on Metal, chunked attention elsewhere. `f16`/`bf16` are autocast-like and GPU-only.
- **Cache:** key = scan bytes SHA-256 (+ the reference scan's for multi-modal input) + recipe `version_hash`
  (includes `larmor_prep::IMPL_VERSION`) + `Manifest::encoder_key` (weights SHA-256, every architecture and
  readout value, `larmor_model::ENCODER_IMPL_VERSION`) + precision. Bump `IMPL_VERSION` whenever a change
  could move any preprocessing output, and `ENCODER_IMPL_VERSION` for any encoder output.
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

- Phases 0–2 are done: `larmor encode`, the cache, probes and every parity gate pass on all 25 fixtures and
  the multi-modal pair (`[[pair]]` in `fixtures/manifest.toml`).
- Not yet: Phase 3 (DICOM through dicom-rs, `larmor watch` on a Box Drive folder via `site.toml`) and
  Phase 4 (the egui viewer).
- Open: which trained checkpoint ships first (rerun `check-parity` with it); the CUDA build (`--features cuda`)
  is untested because it needs the CUDA toolkit; check-parity against references made on the cluster.

## Setup on a new Mac

- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- If links fail with a macOS SDK error, pin builds to an installed older SDK with `SDKROOT` under `[env]` in
  `~/.cargo/config.toml` (as for Onset). Keep that file out of the repo.
- Fixtures and the Python reference env: `tools/setup_fixtures.sh` (needs `uv`; downloads about 120 MB of
  public scans and templates).
