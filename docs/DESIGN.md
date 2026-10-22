# Larmor design

Larmor is one Rust program for Macs that turns an MRI scan into foundation-model embeddings and fits and
applies linear probes on them. The forward pass is the easy part; matching the training preprocessing
exactly is the hard part, so every phase is gated on embedding parity with PyTorch.

The living design doc is <https://claude.ai/artifact/G8gTiPP8S9zjQsZtN2R7bV>. This file is the
implementation contract: formats, layouts and gates that the crates and tools agree on.

## Pipeline

```
scan.nii.gz ─────────────────┐
DICOM series ─▶ dcm2niix's ──┴▶ head mask ─▶ conform ─▶ quantize ─▶ normalize ─▶ tokens ─▶ encoder ─▶ cache ─▶ probes
                NIfTI image     (native)     (192×240×192, 1 mm)                 (in-mask 8³ patches)   (tokens + pooled)
```

A new probe never re-runs the encoder: it trains on the cache. Only new scans are preprocessed and encoded.
`larmor watch` runs the same pipeline on scans as they arrive in watched folders.

## Workspace

| Crate | What it is | Mirrors |
|---|---|---|
| `larmor-volume` | NIfTI I/O, affines, orientation, resampling, Gaussian filter, morphology, quantiles | nibabel, scipy.ndimage, numpy |
| `larmor-dicom` | DICOM series (via dicom-rs) to the NIfTI image dcm2niix writes for them | dcm2niix v1.0.20260724 |
| `larmor-prep` | Preprocessing recipes built from those ops; patchify and tokens | `nanobrain.eval.preprocessing`, `eval/prepare.py::prepare_images`, `ViTMAE3D.dense_embed` |
| `larmor-model` | `model.toml` manifests, safetensors weights, the 3D ViT encoder on Candle | `nanobrain.model.ViTMAE3D`, `nanobrain.modules` |
| `larmor-probe` | Whole-scan and dense linear probes, CV folds, metrics, probe files | `nanobrain.eval.probe`, sklearn |
| `larmor` | The CLI: encode, cache, outputs, probe fit/apply, watch, check-parity, bench, models ([CLI.md](CLI.md)) | |

Larmor is macOS-only: it runs on Apple silicon Macs, on the GPU through Metal. The CPU is the fallback
where Metal is missing (CI runners may not expose it) and runs the tests.

## Runtime: Candle, not Burn

The Phase 0 spike built the encoder on both. Both passed parity (f32 min token cosine 1 − 1e-12), but on an
M3 Ultra Burn was 2–3× slower on Metal, which is the doc's own rule for falling back to Candle:

| ViT-S, Metal, one scan | Burn | Candle | PyTorch MPS |
|---|---|---|---|
| patch 8 (8,150 tokens), f32 / f16 | 0.30 s / 0.20 s | 0.16 s / 0.14 s | 0.135 s / 0.11 s |
| patch 4 (59,359 tokens), f32 / f16 | 13–20 s / 13 s | 5.4 s / 4.1 s | 6.3 s / 5.5 s |

Burn's other reasons no longer applied: the mask is classical image processing (no Conv3d needed) and the
probes use analytic gradients (no autodiff). Candle runs fused attention (`candle_nn::ops::sdpa`) on Metal
and query-chunked attention on the CPU. `F16` and `Bf16` are autocast-like: linears, attention and GELU in
half precision, the residual stream and LayerNorms in f32. Half precision is GPU-only, because Candle's CPU
f16 matmul accumulates in f16.

## Preprocessing: the `nanobrain-eval` recipe

Target: exactly what nanobrain's eval framework feeds the model (`eval/prepare.py::prepare_images` then
`ViTMAE3D.dense_embed`), at commit `a1bfdd1` of MedARC-AI/nanobrain. Pretraining's `data.py` path differs
slightly and is not the target.

1. **Head mask** (`utils/preprocessing.py::threshold_mask` on the raw image): `resample_to_output` to 2 mm
   (order 1, keeps the input dtype, so integer images are rounded), `get_fdata(float32)`, scipy
   `gaussian_filter` with σ = 4 mm / 2 mm = 2 voxels (`mode="reflect"`, truncate 4), threshold
   `data > data.mean()`, keep the largest 6-connected component (`ndi.label`, ties to the lowest label),
   `binary_fill_holes` (6-connected), then `resample_from_to(mask, original, order=0)`.
2. **Reorient**: `nib.as_closest_canonical` (RAS+), image cast to float32 (`eval` version of
   `conform_image`), mask reoriented too.
3. **Anti-alias**: if any axis is finer than 1 mm, `smooth_image` with FWHM = √max(1² − zoom², 0) mm per
   axis (`mode="nearest"`); axes at or above 1 mm get σ = 0 and are skipped.
4. **Conform** (`conform_image(..., max_fov=(192, 240, 192), fixed_grid=True)`): x/y center on the mask
   bounding-box midpoint, z anchored at `top = mask_z_max + 5 mm / zoom_z` (capped at the image height),
   center z = `round(top − 96 mm / zoom_z)`. Python `round` is half-to-even. `rescale_affine` builds the
   1 mm output affine; `resample_from_to(..., order=1)` (scipy `affine_transform`, `mode="constant"`, 0
   outside `[0, n−1]` on any axis) fills the 192×240×192 grid.
5. **Mask on the grid**: `resample_from_to(native_mask, conformed, order=0)` (nearest, `floor(x + 0.5)`).
6. **Quantize** (`truncate_image`): in-mask values, `np.quantile(values, [0.005, 0.995])` (linear method),
   `clip((v − vmin) / (vmax − vmin), 0, 1) * 65535` cast to uint16 (truncation), zeros outside the mask.
7. **Normalize** (`dense_embed`): in-mask values `/ max`, then `(v − mean) / max(std, 1e-6)` with the
   population std; zeros outside the mask.
8. **Tokens**: `patchify3d` = einops `b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz)`. A patch is in
   the mask if any of its voxels is. Coordinates are `(gx, gy, gz)` in `ij` meshgrid order.
9. **Readout**: encoder output after the final LayerNorm, class tokens dropped. Pooled = mean of the in-mask
   tokens.

**Inputs.** Single-file NIfTI-1 and NIfTI-2 (`.nii`, `.nii.gz`), read with nibabel's header fixes; integer
and real float data. A NIfTI-2 scan goes through the recipe exactly as nibabel takes it: its float64 zooms
are rounded to float32 where nanobrain's resampling turns it into a `Nifti1Image`. Two-file images,
Analyze headers and complex or RGB data are refused (convert them; for complex data, to a magnitude
image). A file is read header first, and only the voxel bytes the header describes are decompressed. A
header whose sizes overflow, or geometry so corrupt that a smoothing kernel or the 2 mm mask grid would be
absurd (voxel sizes like 1e-30 or 1e30 mm), is an error for that scan (where nibabel and scipy fail too).

### DICOM input: dcm2niix's NIfTI

A DICOM series goes through the recipe as the NIfTI file dcm2niix v1.0.20260724 (default options) writes for
it, loaded by nibabel: `larmor-dicom` builds that image in memory, with the same voxel array and dtype, sform
(`img.affine`), `pixdim` (`get_zooms()`) and `scl_slope`/`scl_inter`, so the rest of the pipeline cannot
tell the two apart (`check-parity` gates both). It mirrors `saveDcm2NiiCore` for one 3-D volume of classic
DICOM, read with dicom-rs:

1. **Series**: files grouped by SeriesInstanceUID; copies (same SOPInstanceUID) count once. Every slice must
   have one frame of grey levels, ImagePositionPatient, ImageOrientationPatient and PixelSpacing, and the
   same size, sample format, PixelSpacing, orientation (within 1e-4, as dcm2niix's localizer test), echo and
   MRAcquisitionType; no gantry tilt.
2. **Order**: by `(SeriesNumber << 32) + InstanceNumber`, then AcquisitionTime in ms (`fillTDCMsort`).
   If the signed steps along the slice normal (`cross(row, column)`) between consecutive slices are not all
   the first step within 0.2 mm, re-sorted by signed distance from the first slice
   (`ensureSequentialSlicePositions`).
3. **Spacing**: `dx` = the distance between the first two positions (float32, `intersliceDistance`). Any
   step differing from `dx` by more than 0.2 mm (missing slices), or a repeated position (4-D), is refused:
   NIfTI cannot hold either, and dcm2niix writes them with a wrong geometry or as 4-D.
4. **Header zooms**: `pixdim` = (column spacing, row spacing, SpacingBetweenSlices, else SliceThickness,
   else 1), the third replaced by `dx` when they differ by more than 0.0001 mm. The sform always uses `dx`
   (so a 2-D series can carry 4.799995 in the affine and 4.8 in `pixdim`, as dcm2niix's file does).
5. **sform** (`nifti_dicom2mat`, `verify_slice_dir`): rows of the orientation normalised, their cross
   product, times the voxel sizes, the first slice's position as the origin; the slice column negated when
   the second slice lies against it; x and y negated (LPS to RAS). If the column was negated, the stack is
   reversed and the last slice becomes the origin (`nii_flipZ`).
6. **Reorientation**: a 3-D acquisition (MRAcquisitionType `3D`, or Philips' private copy) that is not
   echo-planar is permuted and flipped to its nearest orthogonal orientation with the most left, posterior,
   inferior corner as voxel 0 (`nii_setOrtho`); anything else has its rows reversed (`nii_flipY`).
7. **Samples and scaling**: int16 for signed or fewer-than-16-bit samples (12-bit ones masked as
   `nii_mask12bit`), uint16, uint8 or int32 as dcm2niix stores them; RescaleSlope and RescaleIntercept as
   the header's scale factors, replaced for Philips by the precise values `(R·RS + RI) / (RS·SS)` when
   (2005,100E) ScaleSlope is present (`PhilipsPrecise`, dcm2niix's default `-p y`). When slope or intercept
   differ between slices, every slice is scaled into float32 voxels (`nii_iVaries`; refused for Philips,
   whose precise-value rescaling of such series Larmor does not reproduce).

**Bit-exact in float32.** dcm2niix keeps its matrices in `float` and the macOS and PyPI builds are compiled
by clang, which fuses a multiply and an add written in one C expression into one fused multiply-add
(`-ffp-contract=on`): `a*b + c*d` becomes `fma(a, b, c*d)`, a `+=` of a product fuses into the sum. Larmor
ports each of those expressions with `mul_add` in exactly those places (`larmor-dicom/src/geometry.rs`),
and parses decimal strings as dcm2niix does (`atof` to double, then float), so the sforms are bit-identical
on every gated series. A dcm2niix built by another compiler (GCC on x86-64 does not fuse) can differ from
it, and so from Larmor, in the last bit of an oblique affine.

**Identity.** A series' SHA-256, which stands in for a file's in `run.json` and the cache key, is the
SHA-256 of `larmor-dicom series v<IMPL_VERSION>` and its files' SHA-256s, sorted: renaming or moving the
files keeps the cached embedding, and bumping `larmor_dicom::IMPL_VERSION` (any change that could move a
converted image) gives every DICOM series new cache entries.

**Refused**, with the reason: enhanced multi-frame files (not read yet), Siemens mosaics, colour images,
JPEG-LS (no pure-Rust decoder; dcm2niix needs CharLS), localizers and other series with mixed
orientations or sizes, 4-D series, several echoes in one series (dcm2niix writes `_e1`, `_e2`...), varying
slice spacing, gantry tilt, single slices. A folder search skips such series with a note; inside a BIDS
dataset it skips `sourcedata/`, where a dataset's DICOM usually lives next to its converted NIfTI.

### Multi-modal: a reference scan's head mask

`prepare_images(images, segs, reference=...)` prepares every modality of a subject on one head mask, the
reference's: `threshold_mask` runs on the reference, each modality is conformed with that native mask
(steps 2–4 on the modality's own data, so its grid placement uses the reference's mask), the grids must
agree (`np.allclose(fit_img.affine, affine, atol=1e-3)`), and steps 5–6 use the reference's fit-grid mask
(`resample_from_to(mask, reference_img, order=0)`); `dense_embed` then normalises inside that mask. The
fit affine of the mask and of segmentation labels is the reference's. FOMO tasks 1 and 2 prepare ADC and
DWI this way on the FLAIR. `Recipe::run_with_reference(target, reference)` mirrors it (`larmor encode
--reference`, a `reference` column in `labels.tsv`) and refuses a target on another voxel grid (shape,
orientation, or a grid more than 1e-3 off); `run_with_reference(x, x)` equals `run(x)`.

Which of nanobrain's `DEFAULT_TASKS` Larmor reproduces:

| Task | Input | Larmor |
|---|---|---|
| `fomo_task1_adc_lesion` | ADC on the FLAIR's mask | yes, with the FLAIR as reference |
| `fomo_task2_dwi_tumor` | DWI b1000 on the FLAIR's mask | yes, with the FLAIR as reference |
| `fomo_task3_t1w_age`, `fomo_task3_t1w_hippocampus` | T1w | yes |
| `fomo_task4_t2w_nerve`, `fomo_task4_t2w_vessel` | T2w | yes |
| `fomo_task5_t1w_cortex` | T1w (labels from SynthSeg, `select_labels`) | yes |
| `fomo_task5_t1w_pmg` | T1w and mask multiplied by the SynthSeg brain mask and a 133 mm AP window (`strip_and_crop_ap`) before `dense_embed` | no: needs SynthSeg and that transform |

## Model manifest (`model.toml`)

```toml
id = "nanobrain-vits-p8"            # what --model takes
family = "vit3d"                     # architecture family implemented in larmor-model
description = "nanobrain ViT-S MAE encoder, 8 mm patches"

[architecture]
grid = [192, 240, 192]               # input voxels, 1 mm
patch = 8
width = 384
depth = 12
heads = 6
mlp_ratio = 4.0
class_tokens = 1
pos_embed = "separable"              # SeparablePosEmbed3D: per-axis learned tables, summed
qkv = "separate"                     # separate q/k/v linears (nanobrain); "fused" = timm-style qkv
qkv_bias = true
proj_bias = true                     # attention proj and MLP biases (nanobrain ties them)
norm_eps = 1e-6
activation = "gelu"                  # exact (erf) GELU

[readout]
tokens = "last"                      # last block, after the final norm
pool = "mean"                        # over in-mask tokens

[weights]
file = "nanobrain-vits-p8.safetensors"   # relative to this manifest
sha256 = "<hex>"
url = ""                             # optional download location
naming = "nanobrain"                 # tensor names, see below

[[preprocessing.steps]]
op = "threshold_mask"
resolution = 2.0
sigma = 4.0
remove_islands = true
fill_holes = true

[[preprocessing.steps]]
op = "conform"
min_voxel_size = 1.0
fov = [192.0, 240.0, 192.0]
top_margin = 5.0
fixed_grid = true
order = 1

[[preprocessing.steps]]
op = "quantize"
low = 0.005
high = 0.995

[[preprocessing.steps]]
op = "normalize"
method = "max_zscore"
std_floor = 1e-6

[source]                             # provenance, free-form
model_class = "nanobrain.model:ViTMAE3D"
checkpoint = "vits_run2_epoch40.pth"
git_sha = "fe34e0d"
```

`naming = "nanobrain"` tensor names (encoder only; the MAE decoder is not exported):
`patch_embed.{weight,bias}`, `pos_embed.weight_{x,y,z}`, `cls_token` `[1, C, D]`,
`blocks.{i}.norm1.{weight,bias}`, `blocks.{i}.attn.{q,k,v,proj}.{weight,bias}`,
`blocks.{i}.norm2.{weight,bias}`, `blocks.{i}.mlp.fc1.{weight,bias}`, `blocks.{i}.mlp.fc2.{weight,bias}`,
`norm.{weight,bias}`. Linear weights are PyTorch layout `[out, in]`; dtype f32.

A new model family is a new manifest, not new code, as long as its architecture and preprocessing use
supported values. The **preprocessing version** in cache keys is the SHA-256 of the canonical JSON of
`preprocessing.steps` plus `larmor_prep::IMPL_VERSION` (bumped whenever a numeric change could move
outputs). The **encoder key** (`Manifest::encoder_key`) is the SHA-256 of the canonical JSON of `family`,
`[architecture]`, `[readout]`, the weights' `sha256` and `naming`, and `larmor_model::ENCODER_IMPL_VERSION`
(bumped whenever a numeric change to the encoder could move outputs): `heads` and `norm_eps` change the
embeddings without changing any weight shape, so the weights' checksum alone does not identify them.

`id` is 1 to 64 letters, digits, `.`, `_` or `-`, starting with a letter or digit, without `..` (it
becomes part of cache folders and output names); `weights.file` is a plain file name in the manifest's
folder.

## Reference fixtures (`fixtures/`, never committed)

Produced by `tools/` with the PyTorch/nibabel/scipy stack pinned to nanobrain's `uv.lock`
(`tools/pyproject.toml`, `uv sync` in `tools/`).

```
fixtures/
├── nanobrain/                          pinned checkout of MedARC-AI/nanobrain @ a1bfdd1
├── manifest.toml                       [[scan]] name, file, source, sha256, notes; [[pair]]; [[dicom]]
├── scans/<scan>.nii.gz                 inputs (public data, synthetic variants, dcm2niix's NIfTI of DICOM series)
├── dicom/<name>/                       DICOM series (dcm2niix's validation sets, pinned by commit)
├── models/<model_id>.toml + .safetensors
└── reference/
    ├── dicom/<name>/dcm2niix.json      dcm2niix's BIDS sidecar of each converted series
    ├── prep/<scan>/
    │   ├── head_mask.nii.gz            uint8, native space (threshold_mask)
    │   ├── conform.nii.gz              float32, 192×240×192 (conform_image, before quantize)
    │   ├── fit_mask.nii.gz             uint8, 192×240×192
    │   ├── quantized.nii.gz            uint16, 192×240×192 (truncate_image)
    │   └── info.json                   vmin, vmax, centroid, affines, zooms, fwhm, versions
    └── encode/<model_id>/<scan>.safetensors
```

Encode reference tensors (`__metadata__` holds model_id, model_sha256, scan, torch version):

| Tensor | dtype, shape | What |
|---|---|---|
| `image` | f32 `[X, Y, Z]` | normalized model input (step 7), zeros outside the mask |
| `mask` | u8 `[X, Y, Z]` | fit-grid mask |
| `coords` | i32 `[N, 3]` | in-mask patch grid coordinates |
| `patches` | f32 `[N, P]` | patchified `image` for in-mask patches |
| `tokens` | f32 `[N, D]` | fp32 encoder output, class tokens dropped |
| `cls` | f32 `[C, D]` | fp32 class-token output |
| `pooled` | f32 `[D]` | mean of `tokens` |
| `tokens_bf16`, `pooled_bf16` | f32 | the same under bf16 autocast (the eval framework's default) |

A `[[dicom]]` entry is a DICOM set: `kind = "series"` must convert bit for bit to the `[[scan]]` named
`scan` (made by the pinned dcm2niix, `dcm2niix==1.0.20260724` in `tools/`) and then pass that scan's prep
and encode references from the DICOM files; `"refuse"` must be refused with a message containing `refuse`;
`"pixels"` must decode to the same samples in every transfer syntax except those in `unsupported`. The sets
come from dcm2niix's own validation repositories (`neurolabusc/dcm_qa_*`, BSD-2-Clause), since CC0 DICOM
of MR heads is scarce (OpenNeuro keeps converted NIfTI, not DICOM).

Reference models `nanobrain-vits-p8-ref` / `nanobrain-vits-p4-ref` are nanobrain's ViT-S config with
`torch.manual_seed(0)` init, then every parameter perturbed so biases and norms are non-trivial. Real
checkpoints export with `tools/export_checkpoint.py` where they live (the cluster).

## Embedding cache

Key = `sha256(scan file bytes)` (for a DICOM series, its series SHA-256 above) + the reference scan's sha256
(multi-modal) + preprocessing version + encoder key + precision (`f32`, `f16`, `bf16`). Location `$LARMOR_CACHE` or `~/Library/Caches/larmor`:

```
<cache>/<model_id>-<encoder_key8>/<prep_version8>/<precision>/<scan_sha256>[_ref-<reference_sha256>].safetensors
    tokens f16 [N, D], coords u16 [N, 3], pooled f32 [D]; metadata: the key and the run.json fields
```

Changing any part invalidates the entry (a different path); a read also checks the key recorded in the
metadata. A fresh encode hands the probes its tokens rounded to f16, as a cache hit does, so results never
depend on whether a scan was cached.

## CLI

| Command | What it does |
|---|---|
| `larmor encode <scan or folder> --model <id>` | Preprocess, encode, cache tokens and the pooled embedding, write `run.json`; scans are NIfTI files or DICOM series |
| `larmor probe fit <labels.tsv> --model <id> --kind <kind>` | Train a regression, classification or segmentation probe from cached embeddings with CV; write `probe.toml` and weights |
| `larmor probe apply <probe.toml> <scan or folder>` | Predict with a saved probe; segmentation writes a NIfTI mask |
| `larmor models` | List, download and verify model weights |
| `larmor watch --site site.toml` | Encode scans as they arrive in watched folders (Box Drive, network shares), logging every action |
| `larmor check-parity <fixtures>` | Compare DICOM conversions with dcm2niix and preprocessing stages and embeddings with the PyTorch references |
| `larmor bench` | Per-scan timing on this machine, like `onset timing-test` |

Outputs go to a BIDS-derivatives folder (`derivatives/larmor/sub-XX/[ses-YY/]anat/…`), with a `run.json`
per encode: model id and sha256, preprocessing version, larmor version, mask volume, token count, timings.

`labels.tsv`: one row per scan, like BIDS `participants.tsv`. Column `scan` (path, relative to the TSV),
label columns, optional `fold` (0-based) to force CV folds. Segmentation labels are a column of NIfTI paths
in the scan's native space.

## Watching folders (`site.toml`)

`larmor watch` is the scanner-side service: a site file (like Onset's rig files; `examples/site.toml`)
lists source folders, an output folder, the models and two timings, and the watcher polls the sources,
waits until a scan's files stop changing (`settle_seconds`; a DICOM series' whole folder must be quiet,
since cloud folders sync a series a file at a time), encodes it through the cache with every model and
appends every action to a log. Polling, not file events: Box Drive (a macOS File Provider under
`~/Library/CloudStorage/`) and network shares do not deliver events reliably, listing does not download
files, and settling is timed by the watcher's own observation because synced files keep their original
modification times. Headers are read once per settled file. A ledger next to the log keys each scan by
its files' paths, sizes and times and the models' encoder keys, so restarts skip finished scans and a scan
whose files change (late slices) is encoded again. No lab paths are built in.

## Probes (must equal the eval framework's numbers)

| Kind | Eval framework | Larmor |
|---|---|---|
| regression | `StandardScaler` + `RidgeCV(alphas=logspace(-2, 6, 9))` (efficient LOO), outer `KFold(5, shuffle, seed 0)` | closed-form ridge with exact LOO, same folds |
| classification | `StandardScaler` + `LogisticRegressionCV(Cs=logspace(-4, 4, 9), l1_ratios=(0,), neg_log_loss, class_weight="balanced")`, outer `StratifiedKFold(5, shuffle, seed 0)` | Newton/L-BFGS logistic, same objective, same inner and outer folds |
| segmentation | per-patch features standardized, L2 logistic (L-BFGS) predicting the 8³ voxels of each patch, α ∈ {1e1…1e5} and threshold tuned by inner `KFold(5, shuffle, seed 0)` on mean Dice | same objective on the GPU, same folds, same thresholds |

Folds and bootstrap resamples reproduce numpy's generators (legacy `RandomState` for sklearn folds,
`default_rng` for bootstraps), so the numbers match without a `fold` column.

Probe file: `probe.toml` (model id and sha256, encoder key, embedding precision, preprocessing version,
layer, pooling, kind, label names, CV metrics, sha256 of the training label table) plus `probe.safetensors`,
each written atomically. `probe apply` refuses a probe whose model sha256, encoder key or preprocessing
version doesn't match.

Where the eval framework fails, `probe fit` fails too instead of reporting numbers it cannot produce: a
class with fewer than 5 samples in an outer training fold (sklearn's `LogisticRegressionCV` raises), NaN
or infinite embeddings (sklearn raises), a segmentation training set that `max_negative_ratio` leaves
empty (torch raises), fewer than 5 samples for the outer 5-fold CV. Solvers treat a NaN gradient as an
error, never as convergence. A `labels.tsv` naming one scan twice is refused, since copies in different
folds leak between training and test.

## Gates

- **Phase 0, runtime spike:** token and pooled cosine ≥ 0.999 vs PyTorch fp32; patch 8 and 4 timed on a Mac.
- **Phase 1, `larmor encode`:** parity holds end to end from raw NIfTI on ~20 fixture scans. Each stage is
  also compared on its own (mask agreement, conform/quantize max error) so a mismatch points to its step.
- **Phase 2, probes:** reproduce the eval framework on one scan-level and one segmentation task. Here that
  is checked by running nanobrain's own probe functions on shared features and comparing every output.
- **Phase 3, ingest:** DICOM series convert bit for bit to dcm2niix's NIfTI (voxels, dtype, affine, zooms,
  scaling) on Siemens (VE11, XA30), GE, Philips and UIH series, 2-D and 3-D, four patient positions,
  uncompressed, fragmented JPEG lossless and JPEG 2000, and the whole recipe and encode then match
  nanobrain's prepare on dcm2niix's NIfTI; series dcm2niix makes something other than one volume of are
  refused with the reason; `watch` encodes a local folder end to end (CLI tests).
