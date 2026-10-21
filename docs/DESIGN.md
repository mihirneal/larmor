# Larmor design

Larmor is one Rust program for Macs that turns an MRI scan into foundation-model embeddings and fits and
applies linear probes on them. The forward pass is the easy part; matching the training preprocessing
exactly is the hard part, so every phase is gated on embedding parity with PyTorch.

The living design doc is <https://claude.ai/artifact/G8gTiPP8S9zjQsZtN2R7bV>. This file is the
implementation contract: formats, layouts and gates that the crates and tools agree on.

## Pipeline

```
scan.nii.gz ─▶ head mask ─▶ conform ─▶ quantize ─▶ normalize ─▶ tokens ─▶ encoder ─▶ cache ─▶ probes
              (native)     (192×240×192, 1 mm)                 (in-mask 8³ patches)   (tokens + pooled)
```

A new probe never re-runs the encoder: it trains on the cache. Only new scans are preprocessed and encoded.

## Workspace

| Crate | What it is | Mirrors |
|---|---|---|
| `larmor-volume` | NIfTI I/O, affines, orientation, resampling, Gaussian filter, morphology, quantiles | nibabel, scipy.ndimage, numpy |
| `larmor-prep` | Preprocessing recipes built from those ops; patchify and tokens | `nanobrain.eval.preprocessing`, `eval/prepare.py::prepare_images`, `ViTMAE3D.dense_embed` |
| `larmor-model` | `model.toml` manifests, safetensors weights, the 3D ViT encoder on Burn | `nanobrain.model.ViTMAE3D`, `nanobrain.modules` |
| `larmor-probe` | Whole-scan and dense linear probes, CV folds, metrics, probe files | `nanobrain.eval.probe`, sklearn |
| `larmor` | The CLI: encode, cache, outputs, probe fit/apply, check-parity, bench, models | |

The core crates build on macOS (Metal) and Linux (CUDA). Only the future viewer is macOS-only.

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
outputs).

## Reference fixtures (`fixtures/`, never committed)

Produced by `tools/` with the PyTorch/nibabel/scipy stack pinned to nanobrain's `uv.lock`
(`tools/pyproject.toml`, `uv sync` in `tools/`).

```
fixtures/
├── nanobrain/                          pinned checkout of MedARC-AI/nanobrain @ a1bfdd1
├── manifest.toml                       [[scan]] name, file, source, sha256, notes
├── scans/<scan>.nii.gz                 inputs (public data and synthetic variants)
├── models/<model_id>.toml + .safetensors
└── reference/
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

Reference models `nanobrain-vits-p8-ref` / `nanobrain-vits-p4-ref` are nanobrain's ViT-S config with
`torch.manual_seed(0)` init, then every parameter perturbed so biases and norms are non-trivial. Real
checkpoints export with `tools/export_checkpoint.py` where they live (the cluster).

## Embedding cache

Key = `sha256(scan file bytes)` + preprocessing version + model weights sha256. Location
`$LARMOR_CACHE` or `~/Library/Caches/larmor` (`$XDG_CACHE_HOME/larmor` on Linux):

```
<cache>/<model_id>-<model_sha8>/<prep_version8>/<scan_sha256>.safetensors
    tokens f16 [N, D], coords u16 [N, 3], pooled f32 [D]; metadata: run.json fields
```

Changing any of the three invalidates the entry (a different path).

## CLI

| Command | What it does |
|---|---|
| `larmor encode <scan or folder> --model <id>` | Preprocess, encode, cache tokens and the pooled embedding, write `run.json` |
| `larmor probe fit <labels.tsv> --model <id> --kind <kind>` | Train a regression, classification or segmentation probe from cached embeddings with CV; write `probe.toml` and weights |
| `larmor probe apply <probe.toml> <scan or folder>` | Predict with a saved probe; segmentation writes a NIfTI mask |
| `larmor models` | List, download and verify model weights |
| `larmor check-parity <fixtures>` | Compare preprocessing stages and embeddings with the PyTorch references |
| `larmor bench` | Per-scan timing on this machine, like `onset timing-test` |

Outputs go to a BIDS-derivatives folder (`derivatives/larmor/sub-XX/[ses-YY/]anat/…`), with a `run.json`
per encode: model id and sha256, preprocessing version, larmor version, mask volume, token count, timings.

`labels.tsv`: one row per scan, like BIDS `participants.tsv`. Column `scan` (path, relative to the TSV),
label columns, optional `fold` (0-based) to force CV folds. Segmentation labels are a column of NIfTI paths
in the scan's native space.

## Probes (must equal the eval framework's numbers)

| Kind | Eval framework | Larmor |
|---|---|---|
| regression | `StandardScaler` + `RidgeCV(alphas=logspace(-2, 6, 9))` (efficient LOO), outer `KFold(5, shuffle, seed 0)` | closed-form ridge with exact LOO, same folds |
| classification | `StandardScaler` + `LogisticRegressionCV(Cs=logspace(-4, 4, 9), l1_ratios=(0,), neg_log_loss, class_weight="balanced")`, outer `StratifiedKFold(5, shuffle, seed 0)` | Newton/L-BFGS logistic, same objective, same inner and outer folds |
| segmentation | per-patch features standardized, L2 logistic (L-BFGS) predicting the 8³ voxels of each patch, α ∈ {1e1…1e5} and threshold tuned by inner `KFold(5, shuffle, seed 0)` on mean Dice | same objective on the GPU, same folds, same thresholds |

Folds and bootstrap resamples reproduce numpy's generators (legacy `RandomState` for sklearn folds,
`default_rng` for bootstraps), so the numbers match without a `fold` column.

Probe file: `probe.toml` (model id and sha256, preprocessing version, layer, pooling, kind, label names,
CV metrics, sha256 of the training label table) plus `probe.safetensors`. `probe apply` refuses a probe
whose model sha256 or preprocessing version doesn't match.

## Gates

- **Phase 0, runtime spike:** token and pooled cosine ≥ 0.999 vs PyTorch fp32; patch 8 and 4 timed on a Mac.
- **Phase 1, `larmor encode`:** parity holds end to end from raw NIfTI on ~20 fixture scans. Each stage is
  also compared on its own (mask agreement, conform/quantize max error) so a mismatch points to its step.
- **Phase 2, probes:** reproduce the eval framework on one scan-level and one segmentation task. Here that
  is checked by running nanobrain's own probe functions on shared features and comparing every output.
