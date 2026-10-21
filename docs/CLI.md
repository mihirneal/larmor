# The `larmor` command line

`larmor` turns MRI scans into foundation-model embeddings and fits and applies linear probes on them.
This page lists every command, its flags and what it writes. `docs/DESIGN.md` is the contract behind it
(recipe, cache, formats, gates).

```sh
cargo build --release -p larmor                  # macOS: Metal; Linux + NVIDIA: add --features cuda
target/release/larmor --help
```

Errors print as `error: …` and exit with status 1. Colors turn off when stdout is not a terminal or
`NO_COLOR` is set.

## Common flags

| Flag | Meaning |
|---|---|
| `--model ID\|PATH` | A model id (looked up as below) or a path to a `model.toml` |
| `--device auto\|cpu\|metal\|cuda[:N]` | Where the encoder runs. `auto` (default): Metal on a Mac, CUDA in a `--features cuda` build, else the CPU; if that GPU cannot be used, `auto` falls back to the CPU with a warning |
| `--precision f32\|f16\|bf16` | Encoder matmul precision. `f32` (default) is the parity target; the half types mirror PyTorch autocast |

## Environment

| Variable | Default | What |
|---|---|---|
| `LARMOR_MODELS` | | Colon-separated folders searched first for model ids |
| `LARMOR_CACHE` | `~/Library/Caches/larmor` (macOS), `$XDG_CACHE_HOME/larmor` or `~/.cache/larmor` (Linux) | Embedding cache |
| `HF_TOKEN` | | Bearer token for `models download` from Hugging Face hosts only |
| `LARMOR_FIXTURES` | `./fixtures` | Fixtures folder `bench` takes its default scan and model from |
| `NO_COLOR` | | Plain output |

Model ids are looked up in `$LARMOR_MODELS`, then the user data folder
(`~/Library/Application Support/larmor/models` on macOS, `~/.local/share/larmor/models` on Linux), then
`./models`; the first manifest with that `id` wins. See `models/README.md`.

## `larmor encode <scan or folder>... --model ID`

Preprocesses and encodes scans, caches their embeddings and writes BIDS derivatives.

| Flag | |
|---|---|
| `--out DIR` | Derivatives folder. Default: `<BIDS root>/derivatives/larmor` when the scan is inside a BIDS dataset (a folder up the tree has `dataset_description.json`), else `./derivatives/larmor` |
| `--device`, `--precision` | As above |
| `--force` | Encode again even when the cache has the embedding |
| `--save-preprocessed` | Also write the quantized model-grid image and its head mask |

Inputs are `.nii` / `.nii.gz` files or folders. Folders are searched recursively, skipping hidden
folders, `derivatives/`, and label images (BIDS suffixes `_mask`, `_dseg`, `_probseg`, `_roi`,
`_defacemask`); a file named on the command line is always used.

For each scan: SHA-256 of the file bytes, then the cache key (scan SHA-256, preprocessing version, model
weights SHA-256). On a cache hit nothing is computed; on a miss the scan is read, preprocessed with the
model's recipe, cut into in-mask patch tokens, encoded and cached. The encoder is loaded only when some
scan is not cached, and while it encodes one scan the next is read and preprocessed on another thread.
One status line per scan, then a summary:

```text
larmor encode  2 scans · model nanobrain-vits-p8 (98562bb2) · cache ~/Library/Caches/larmor
loaded nanobrain-vits-p8 on metal (f32) in 0.188 s
  encoded sub-01_T1w  8150 tokens · read 0.051 s · prep 0.059 s · encode 0.150 s · total 0.270 s
  cached  sub-02_T1w  8516 tokens · read 0.006 s
1 encoded, 1 cached, 0 failed in 0.6 s · outputs in /data/bids/derivatives/larmor
```

A scan that fails (unreadable, empty head mask…) is reported and the others go on; the exit status is 1
if any failed.

### Derivatives layout

```text
<out>/
├── dataset_description.json                     BIDS derivative description (written once)
├── sub-01/[ses-1/]anat/                         mirrors sub-XX/[ses-YY/]<datatype>/ of the input
│   ├── sub-01_T1w_model-<id>_embedding.npy      pooled embedding, float32 [D] (np.load)
│   ├── sub-01_T1w_model-<id>_run.json           how it was made (below)
│   ├── sub-01_T1w_model-<id>_desc-quantized.nii.gz   --save-preprocessed: uint16 grid image
│   ├── sub-01_T1w_model-<id>_desc-head_mask.nii.gz   --save-preprocessed: uint8 grid mask
│   ├── sub-01_T1w_desc-<probe>_mask.nii.gz      probe apply (segmentation): uint8, scan space
│   └── sub-01_T1w_desc-<probe>_probseg.nii.gz   probe apply (segmentation): float32, model grid
├── brain/brain_model-<id>_embedding.npy         a scan outside sub-XX/ folders: <out>/<file stem>/
└── <probe>_predictions.tsv                      probe apply (regression, classification)
```

Model ids are used as is in `model-<id>`, so an id with hyphens is not a strict BIDS label. Two inputs
that would write the same file (same stem, same output folder) stop the run before anything is encoded.

### `run.json`

| Field | |
|---|---|
| `larmor_version` | |
| `model` | `id`, weights `sha256`, `manifest` path, `family`, `patch`, `width`, `depth` |
| `preprocessing` | `version` (the cache-key hash), `impl_version`, every `steps` value |
| `scan` | absolute `path`, `sha256`, `shape`, `zooms`, `orientation` (axis codes), stored `dtype`, `affine` |
| `grid` | `shape`, `fit_affine` (maps native labels onto the grid), `centroid`, anti-alias `fwhm`, `patch` |
| `vmin`, `vmax` | quantization window (in-mask 0.5% and 99.5% quantiles) |
| `head_volume_ml` | native-space head mask volume |
| `tokens`, `width` | in-mask patch tokens, embedding width |
| `device`, `precision` | what encoded it |
| `seconds` | `read`, `hash`, `mask`, `conform`, `quantize`, `normalize`, `tokens`, `encode`, `cache_write`, `total` |
| `started`, `finished` | RFC 3339 local times of the encode |
| `cache` | `path` of the cache entry, `hit`: whether this run found it there |

On a cache hit the fields describe the run that encoded the scan, with `cache.hit = true`.

### Cache entries

```text
<cache>/<model_id>-<sha8>/<prep_version8>/<scan_sha256>.safetensors
    tokens f16 [N, D], coords u16 [N, 3] (gx, gy, gz), pooled f32 [D]
    metadata: format, model_id, model_sha256, preprocessing_version, scan_sha256, run (the run.json JSON)
```

`pooled` is the mean of the float32 tokens, before they are stored as float16. Delete the folder to clear
the cache; nothing else refers to it.

## `larmor models [list | verify [ID] | download ID]`

| Command | |
|---|---|
| `models` / `models list` | The model folders in search order, then one row per manifest: id, family, grid, patch, width, depth, weights status (`ok`, `missing`, `missing (download)` when it has a URL, `sha256 MISMATCH`), manifest path. A later manifest with an id already listed is marked shadowed; `.toml` files that are not manifests are listed as notes |
| `models verify [ID]` | Recomputes the weights SHA-256 of one model or of every model found; exit 1 if any is missing or differs |
| `models download ID` | Fetches `[weights] url` to the manifest's weights path (via a `.part` file, with progress), checks the SHA-256 and only then moves it into place. `HF_TOKEN` is sent as a bearer token to `huggingface.co` / `hf.co` URLs only |

## `larmor probe fit <labels.tsv> --model ID --kind KIND --label COLUMN`

Fits a probe on cached embeddings with the eval framework's cross-validation, then a deployable probe on
all scans.

| Flag | |
|---|---|
| `--kind regression\|classification\|segmentation` | What to fit (DESIGN.md, "Probes") |
| `--label COLUMN` | The label column |
| `--out FILE` | Probe file, default `probe.toml`; the weights go next to it (`probe.safetensors`) |
| `--encode` | Encode scans that are not cached yet instead of stopping |
| `--max-negative-ratio R` | Segmentation: keep at most `R` all-negative patches per positive patch when fitting (nanobrain's `max_negative_ratio`) |
| `--device`, `--precision` | For `--encode` |

`labels.tsv` (annotated example: `examples/labels.tsv`): tab-separated, one row per scan; `#` lines are
comments. `scan` is a path relative to the TSV. Other columns are labels; empty and `n/a` cells are
missing and their rows are left out. An optional `fold` column (0-based) fixes the outer folds; without
it the folds are the eval framework's (KFold(5) or StratifiedKFold(5), shuffled with seed 0).

* **regression**: numeric labels; `StandardScaler` + `RidgeCV` (exact leave-one-out) on pooled
  embeddings. Prints MAE, Pearson r and R² with 95% bootstrap intervals.
* **classification**: exactly two values; numbers sort numerically, text alphabetically, the second is
  the positive class. `StandardScaler` + `LogisticRegressionCV`. Prints AUROC and balanced accuracy.
* **segmentation**: a binary NIfTI per scan in the scan's own space. It is resampled onto the cached fit
  grid with nearest neighbour (`resample_from_to(seg, grid, order=0)`, rounded, as `eval/prepare.py`
  does) and cut into the voxels of each cached in-mask patch. Per-voxel logistic heads on patch tokens;
  prints Dice, voxel average precision and voxel AUROC.

Scans not in the cache stop the fit with a list of them and the `larmor encode` command to run, unless
`--encode` is given. The probe file records the model id and weights SHA-256, the preprocessing version,
the layer and pooling, the label names, the SHA-256 of `labels.tsv`, the outer-CV metrics and per-fold
choices, and the chosen hyperparameters.

Known gap: larmor-probe's segmentation solver takes its backend's own device and offers no way to build
one from `--device`, so segmentation fits run on that backend's default device (the GPU when one is
compiled in) whatever `--device` says.

## `larmor probe apply <probe.toml> <scan or folder>...`

| Flag | |
|---|---|
| `--out DIR` | Output folder (default: the derivatives folder, as for `encode`) |
| `--model ID\|PATH` | Model to embed with (default: the probe's `model_id`, looked up as usual) |
| `--device`, `--precision` | For scans that are not cached |

Refuses a probe whose model weights SHA-256, preprocessing version or embedding width differs from the
model's. Scans are encoded through the cache when needed.

* **regression / classification**: writes `<out>/<probe>_predictions.tsv` (`scan`, `prediction`, and
  `probability` of the positive class for classification; the predicted class is the positive one when
  the probability is above 0.5) and prints the predictions.
* **segmentation**: per scan, voxel probabilities on the model grid, thresholded at the probe's
  threshold (`p >= t`), mapped back to the scan's voxels with nearest neighbour (the recipe is re-run,
  about 0.1 s) and written as `<stem>_desc-<probe>_mask.nii.gz` (uint8, the scan's header geometry),
  plus the probabilities on the grid as `<stem>_desc-<probe>_probseg.nii.gz` (float32, fit affine).
  Prints the mask volume per scan.

`<probe>` is the probe file's stem (letters and digits only), or the label column when the file is
`probe.toml`.

## `larmor check-parity <fixtures> [--models ID,...]`

Runs the recipe on every scan in `<fixtures>/manifest.toml` and compares each stage with
`reference/prep/<scan>/`, then encodes with each model that has references in
`reference/encode/<model>/` (manifests in `<fixtures>/models/`) and compares tokens, class tokens and
the pooled embedding with PyTorch's fp32 output. `--models` restricts the models; `--device` and
`--precision` choose how to encode. Prints one row per scan and model, the gates with their worst
values, and exits 1 if any gate fails.

| Column / gate | Limit |
|---|---|
| head mask Δ, fit mask Δ: voxels that disagree | ≤ 0.01% of voxels |
| conform: max abs error / reference value range | ≤ 1e-5 |
| quant: in-mask voxels with the exact uint16 value | ≥ 99.99% |
| vmin, vmax, centroid, fit affine, anti-alias FWHM | exactly equal |
| tokens: in-mask patch grid equal to the reference | all scans |
| 1-cos min / avg: per-token cosine distance | min token cosine ≥ 0.999 |
| 1-ctr min: cosine distance after centering each side's tokens | reported |
| 1-pooled: pooled embedding cosine distance | pooled cosine ≥ 0.999 |
| max abs Δ of the tokens | reported |

Also printed: PyTorch's own bf16-vs-fp32 cosines from the reference metadata (the noise floor of the
eval framework's bf16 autocast) and the mean encode time.

## `larmor bench`

Per-stage timing on this machine, like `onset timing-test`.

| Flag | |
|---|---|
| `--model ID\|PATH` | Default: `<fixtures>/models/nanobrain-vits-p8-ref.toml`, else the first installed model with verified weights, else random weights in the shape of nanobrain ViT-S (patch 8) |
| `--scan FILE` | Default: the first scan of `<fixtures>/manifest.toml`, else a built-in synthetic head (176 x 208 x 176 at 1 mm) |
| `--runs N` | Default 5; the first encode is reported separately as the cold one |
| `--device`, `--precision` | As above |
| `--json` | The report as JSON |

Prints the chip, cores (performance and efficiency), GPU cores, memory and OS; the encoder load time and
first (cold) encode; then median, min and max seconds for read, hash, mask, conform, quantize, normalize,
tokens, warm encode, cache write and the warm total; and the load average and GPU utilization just
before the timing started, so timings taken on a busy machine can be recognized.
