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
| `--device auto\|cpu\|metal\|cuda[:N]` | Where the encoder and the segmentation probe run. `auto` (default): CUDA in a `--features cuda` build, else Metal on a Mac, else the CPU |
| `--precision f32\|f16\|bf16` | Encoder precision. `f32` (default) is the parity target. `f16` and `bf16` work like PyTorch autocast (linears, attention and GELU in the half type, the residual stream and norms in f32) and need a GPU; with `--device auto` and no GPU, larmor encodes in f32 on the CPU with a warning. The precision is part of the cache key, so an f32 request is never served a half-precision embedding (or the reverse) |

## Environment

| Variable | Default | What |
|---|---|---|
| `LARMOR_MODELS` | | Colon-separated folders searched first for model ids |
| `LARMOR_CACHE` | `~/Library/Caches/larmor` (macOS), `$XDG_CACHE_HOME/larmor` or `~/.cache/larmor` (Linux) | Embedding cache |
| `HF_TOKEN` | | Bearer token for `models download` from Hugging Face hosts only (`https://huggingface.co`, `hf.co` and their subdomains, as a URL parser reads the host) |
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
| `--reference SCAN` | Prepare every scan on this scan's head mask and grid (below) |
| `--device`, `--precision` | As above |
| `--force` | Encode again even when the cache has the embedding, and replace outputs that another scan wrote (below) |
| `--save-preprocessed` | Also write the quantized model-grid image and its head mask |

Inputs are `.nii` / `.nii.gz` files (single-file NIfTI-1 or NIfTI-2 with integer or real float data) or
folders. Two-file images (`.hdr`/`.img`), Analyze 7.5, compressed variants other than gzip (`.nii.bz2`,
`.nii.zst`), FreeSurfer `.mgz` and complex or RGB data are not read: convert them to `.nii.gz` (complex
data to a real magnitude image first). Folders are searched recursively, skipping:

- hidden folders, `derivatives/`, the run's `--out` folder and any folder whose `dataset_description.json`
  says `"DatasetType": "derivative"`;
- inside a BIDS dataset, the non-anatomical datatype folders `func/`, `dwi/`, `fmap/`, `perf/`, `pet/`,
  `meg/`, `eeg/`, `ieeg/`, `beh/`, `micr/`, `nirs/`, `motion/` and `mrs/` (name such a folder, or a file in
  it, to include it);
- label images (BIDS suffixes `_mask`, `_dseg`, `_probseg`, `_roi`, `_defacemask`) and larmor's own grid
  images (`_model-<id>_desc-quantized`);
- images whose header (read alone, without the voxels) is not 3-D, such as 4-D fMRI or DWI series.

A note on stderr lists what was skipped (and files in the formats above). A file named on the command line
is always used; a 4-D one fails with an error that says so, before its voxels are read.

For each scan: SHA-256 of the file bytes, then the cache key (scan SHA-256, the reference scan's SHA-256 with
`--reference`, preprocessing version, encoder key, precision; see "Cache entries"). On a cache hit nothing is
computed; on a miss the scan is read, preprocessed with the model's recipe, cut into in-mask patch tokens,
encoded and cached. The encoder is loaded only when some scan is not cached, and while it encodes one scan
the next is read and preprocessed on another thread. One status line per scan, then a summary:

```text
larmor encode  2 scans · model nanobrain-vits-p8 (3f0c51a7) · f32 · cache ~/Library/Caches/larmor
loaded nanobrain-vits-p8 on metal (f32) in 0.188 s
  encoded sub-01_T1w  8150 tokens · read 0.051 s · prep 0.059 s · encode 0.150 s · total 0.270 s
  cached  sub-02_T1w  8516 tokens · read 0.006 s
1 encoded, 1 cached, 0 failed in 0.6 s · outputs in /data/bids/derivatives/larmor
```

The hex after the model id is the start of its encoder key. A scan that fails (unreadable, a corrupt or
4-D header, an empty head mask…) is reported and the others go on; the exit status is 1 if any failed.
If the encoder cannot be loaded (a weights SHA-256 mismatch, a device this build lacks), the error is
printed once, with what to do, and the remaining uncached scans fail without loading it again.

### A second modality: `--reference`

nanobrain's eval framework prepares some modalities on another one's head mask: `prepare_images(images,
segs, reference="flair")` computes one head mask from the FLAIR and conforms and quantizes ADC and DWI with
that mask and the FLAIR's grid (FOMO tasks 1 and 2). `--reference SCAN` does the same for every scan of the
run: the head mask comes from `SCAN`, each scan is conformed with it, its grid must agree with the
reference's to 1e-3 mm (the eval framework's assertion; a scan on another voxel grid fails), and it is
quantized and normalised inside the reference's grid mask. The scan's embedding then differs from the one
it gets on its own mask, and is cached under its own key. Encode one subject at a time this way
(`larmor encode sub-01/dwi/ --reference sub-01/anat/sub-01_FLAIR.nii.gz`), or give `probe fit` a
`reference` column. A scan that is its own reference is simply encoded on its own mask.

Which of nanobrain's `DEFAULT_TASKS` Larmor reproduces: `fomo_task3_t1w_age`,
`fomo_task3_t1w_hippocampus`, `fomo_task4_t2w_nerve`, `fomo_task4_t2w_vessel` and
`fomo_task5_t1w_cortex` (single scans), and `fomo_task1_adc_lesion` and `fomo_task2_dwi_tumor` with the
FLAIR as `--reference`. Not `fomo_task5_t1w_pmg`: its `strip_and_crop_ap` transform multiplies the image
and the mask by a SynthSeg brain mask and a 133 mm anterior-posterior window before `dense_embed`, which
Larmor does not do.

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
├── subj01/T1_model-<id>_embedding.npy          outside sub-XX/: the scan's folder below the inputs' folder
├── brain/brain_model-<id>_embedding.npy         a scan directly in that folder: <out>/<file stem>/
└── <probe>_predictions.tsv                      probe apply (regression, classification)
```

A scan outside `sub-XX/` folders mirrors its folder below the inputs' common folder (the folders named
on the command line, and the folders of the files named), so `larmor encode /data/study` with
`subj01/T1.nii.gz` and `subj02/T1.nii.gz` writes `<out>/subj01/` and `<out>/subj02/`, and
`larmor encode a b` writes `<out>/a/` and `<out>/b/`. A scan directly in that folder (or named alone) gets
`<out>/<file stem>/`.

Model ids are used as is in `model-<id>`, so an id with hyphens is not a strict BIDS label. Two inputs
that would write the same file stop the run before anything is encoded; names that differ only in case
count as the same file, as they are on macOS's default file system. An output whose existing `run.json`
belongs to another scan (another path and other bytes: a second run with the same file name and the same
`--out`) is not replaced unless `--force` is given.

### `run.json`

| Field | |
|---|---|
| `larmor_version` | |
| `model` | `id`, weights `sha256`, `encoder_key` (the encoder part of the cache key), `manifest` path, `family`, `patch`, `width`, `depth` |
| `preprocessing` | `version` (the cache-key hash), `impl_version`, every `steps` value |
| `scan` | absolute `path`, `sha256`, `shape`, `zooms`, `orientation` (axis codes), stored `dtype`, `affine` |
| `grid` | `shape`, `fit_affine` (maps native labels onto the grid), `centroid`, anti-alias `fwhm`, `patch` |
| `vmin`, `vmax` | quantization window (in-mask 0.5% and 99.5% quantiles) |
| `head_volume_ml` | native-space head mask volume |
| `tokens`, `width` | in-mask patch tokens, embedding width |
| `device`, `precision` | what encoded it |
| `seconds` | `read`, `hash`, `mask`, `conform`, `quantize`, `normalize`, `tokens`, `encode`, `cache_write`, `total` |
| `started`, `finished` | RFC 3339 local times of the encode |
| `reference` | with `--reference`: the reference scan's absolute `path` and `sha256` |
| `cache` | `path` of the cache entry, `hit`: whether this run found it there |
| `encoded_from` | on a cache hit: the `scan_path`, `manifest`, `started` and `finished` of the run that computed the embedding |

On a cache hit `scan.path` and `model.manifest` are this run's scan and manifest, and the other fields
describe the run that encoded the embedding (which may have read another file with the same bytes, or
used another manifest with the same encoder key), with `cache.hit = true`.

### Cache entries

```text
<cache>/<model_id>-<encoder_key8>/<prep_version8>/<precision>/<scan_sha256>[_ref-<reference_sha256>].safetensors
    tokens f16 [N, D], coords u16 [N, 3] (gx, gy, gz), pooled f32 [D]
    metadata: format, model_id, model_sha256, encoder_key, precision, preprocessing_version, scan_sha256,
              reference_sha256 (with --reference), run (the run.json JSON)
```

The encoder key is the SHA-256 of the weights' SHA-256 and tensor naming, `family`, every
`[architecture]` and `[readout]` value and larmor's encoder implementation version: values such as
`heads` or `norm_eps` change the embeddings without changing any weight, so correcting them in a
manifest gives new entries instead of the old embeddings. A read checks the key in the metadata as well;
an entry that is unreadable, truncated or holds another key is encoded again (with a warning).

`pooled` is the mean of the float32 tokens, before they are stored as float16. A freshly encoded scan hands
the probes its tokens rounded to float16 too, exactly what a cache hit returns, so a segmentation probe
gives the same output on the first and the second run. Entries are written atomically (a synced
temporary file renamed into place). Delete the folder to clear the cache; nothing else refers to it.

## `larmor models [list | verify [ID] | download ID]`

| Command | |
|---|---|
| `models` / `models list` | The model folders in search order, then one row per manifest: id, family, grid, patch, width, depth, weights status (`ok`, `missing`, `missing (download)` when it has a URL, `sha256 MISMATCH`), manifest path. A later manifest with an id already listed is marked shadowed; `.toml` files that are not manifests are listed as notes |
| `models verify [ID]` | Recomputes the weights SHA-256 of one model or of every model found; exit 1 if any is missing or differs |
| `models download ID [--force] [--allow-http]` | Fetches `[weights] url` to the manifest's weights path (via a `.part` file, with progress), checks the SHA-256 and only then moves it into place. The URL must be `https://` (`--allow-http` accepts `http://`) without a user name or password in it. An existing weights file whose SHA-256 differs is not replaced unless `--force` is given. `HF_TOKEN` is sent as a bearer token only to `huggingface.co`, `hf.co` and their subdomains over HTTPS; an `https://hf.co/...` URL is fetched from `huggingface.co` directly, since the token is not passed on across `hf.co`'s redirect |

A manifest's `id` is 1 to 64 letters, digits, `.`, `_` or `-`, starting with a letter or digit (it names
cache folders and output files; write `org--name` for a Hugging Face style `org/name`), and
`[weights] file` is a plain file name in the manifest's folder, so neither a download nor a cache entry
lands outside the folders meant for them.

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
| `--device` | Where the segmentation probe fits, and the encoder with `--encode` |
| `--precision` | For `--encode` |

`labels.tsv` (annotated example: `examples/labels.tsv`): tab-separated, one row per scan; `#` lines are
comments, and a UTF-8 byte order mark (spreadsheet exports) is ignored. `scan` is a path relative to the
TSV. Other columns are labels; empty and `n/a` cells are missing and their rows are left out. An optional
`fold` column (0-based) fixes the outer folds; without it the folds are the eval framework's (KFold(5) or
StratifiedKFold(5), shuffled with seed 0). An optional `reference` column names, per row, the scan whose
head mask and grid that row's scan is prepared on (as `encode --reference`).

A scan may appear only once: two rows naming the same file, or files with identical bytes, stop the fit
with their line numbers, because copies in different folds leak between training and test and inflate the
metrics. Copies that a `fold` column keeps in one fold are accepted, with a note.

* **regression**: numeric labels; `StandardScaler` + `RidgeCV` (exact leave-one-out) on pooled
  embeddings. Prints MAE, Pearson r and R² with 95% bootstrap intervals.
* **classification**: exactly two values; numbers sort numerically when every value is a number, else
  everything sorts as text; the second is the positive class. `StandardScaler` + `LogisticRegressionCV`.
  Prints AUROC and balanced accuracy. Every outer training fold, and the full set for the deployable
  probe, needs at least 5 scans of each class: below that the inner `StratifiedKFold(5)` leaves a
  validation fold with one class and scikit-learn raises, so larmor stops with the fold and its counts
  instead of reporting numbers the eval framework cannot produce (with the default folds, at least 7 of
  the smaller class).
* **segmentation**: a binary NIfTI per scan in the scan's own space. It is resampled onto the cached fit
  grid with nearest neighbour (`resample_from_to(seg, grid, order=0)`, rounded, as `eval/prepare.py`
  does) and cut into the voxels of each cached in-mask patch. Per-voxel logistic heads on patch tokens;
  prints Dice, voxel average precision and voxel AUROC. Labelled voxels outside the in-mask patches are
  not used (as in the eval framework; a note counts them), a label set with no labelled voxel inside them
  stops the fit, and so does a training set that `--max-negative-ratio` leaves empty (a training fold
  without positive patches: nanobrain raises there). Every outer training fold needs at least 5 scans
  for the inner KFold(5).

The outer 5-fold cross-validation needs at least 5 labelled scans. Embeddings with NaN or infinite values
are refused (scikit-learn raises on them). Scans not in the cache, or whose cache entry cannot be read,
stop the fit with a list of them and the `larmor encode` command to run (with `--force` for unreadable
entries), unless `--encode` is given, which encodes them (again). The cache is looked up in the
`--precision` asked for (f32 by default), and a hint says when the missing scans are cached in another
precision. The probe file records the model id, weights SHA-256 and encoder key, the embeddings'
precision, the preprocessing version, the layer and pooling, the label names, the SHA-256 of
`labels.tsv`, the outer-CV metrics and per-fold choices, and the chosen hyperparameters.

## `larmor probe apply <probe.toml> <scan or folder>...`

| Flag | |
|---|---|
| `--out DIR` | Output folder (default: the derivatives folder, as for `encode`) |
| `--model ID\|PATH` | Model to embed with (default: the probe's `model_id`, looked up as usual) |
| `--reference SCAN` | Prepare every scan on this scan's head mask and grid (as `encode --reference`) |
| `--device` | Where the segmentation probe predicts, and the encoder for scans that are not cached |
| `--precision` | The embeddings' precision (a warning says when it is not the one the probe was fit on) |

Refuses a probe whose model weights SHA-256, encoder key (probes written before larmor recorded it skip
this check), preprocessing version or embedding width differs from the model's. Scans are encoded through
the cache when needed. A scan that fails is reported and the others go on; the exit status is 1 if any
failed.

* **regression / classification**: writes `<out>/<probe>_predictions.tsv` (`scan`, `prediction`, and
  `probability` of the positive class for classification; the predicted class is the positive one when
  the probability is above 0.5) and prints the predictions.
* **segmentation**: per scan, voxel probabilities on the model grid, thresholded at the probe's
  threshold (`p >= t`), mapped back to the scan's voxels with nearest neighbour (the recipe is re-run,
  about 0.1 s) and written as `<stem>_desc-<probe>_mask.nii.gz` (uint8, the scan's header geometry),
  plus the probabilities on the grid as `<stem>_desc-<probe>_probseg.nii.gz` (float32, fit affine).
  Prints the mask volume per scan. Outputs go where `encode` puts them; two scans that would write the
  same mask stop the run before anything is written.

`<probe>` is the probe file's stem (letters and digits only), or the label column when the file is
`probe.toml`.

## `larmor check-parity <fixtures> [--models ID,...]`

Runs the recipe on every scan in `<fixtures>/manifest.toml` and compares each stage with
`reference/prep/<scan>/`, then encodes with each model that has references in
`reference/encode/<model>/` (manifests in `<fixtures>/models/`) and compares tokens, class tokens and
the pooled embedding with PyTorch's fp32 output. Each `[[pair]]` of the manifest (a target scan prepared
on a reference scan's head mask, `prepare_images` with `reference=`) is checked the same way, and its fit
mask, quantized volume, grid and image affines and token grid must be bit-exact. `--models` restricts the models; `--device` and
`--precision` choose how to encode. Prints one row per scan and model, then each gate with its worst
value and the scan it came from, and exits 1 if any gate fails.

| Column / gate | Limit |
|---|---|
| head mask Δ, fit mask Δ: voxels that disagree | ≤ 0.01% of voxels |
| conform: max abs error / reference value range | ≤ 1e-5 |
| quant: in-mask voxels with the exact uint16 value | ≥ 99.99% |
| vmin, vmax, centroid, fit affine, anti-alias FWHM | exactly equal |
| tokens: in-mask patch grid equal to the reference | all scans |
| 1-cos min / avg: per-token cosine distance | min token cosine ≥ 0.999 |
| 1-ctr min: cosine distance after subtracting the reference's mean token from both sides | reported |
| 1-pooled: pooled embedding cosine distance | pooled cosine ≥ 0.999 |
| max abs Δ of the tokens | reported |

Also printed: the worst centered cosine and max abs Δ with their scans, the mean encode time, and
PyTorch's own bf16-vs-fp32 cosines from the reference metadata (the noise floor of the eval framework's
bf16 autocast).

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
