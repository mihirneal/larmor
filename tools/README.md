# tools: PyTorch references for Larmor

Larmor is checked against the PyTorch code it mirrors: MedARC-AI/nanobrain at commit `a1bfdd1`, run with
the library versions pinned in `pyproject.toml` / `uv.lock` (the same as nanobrain's own lock). Everything
here writes into `fixtures/` (git-ignored; override with `LARMOR_FIXTURES`). The formats and layouts are the
contract in `docs/DESIGN.md`, "Reference fixtures".

## Setup

```sh
tools/setup_fixtures.sh            # from scratch; safe to re-run, it only fills in what is missing
tools/setup_fixtures.sh --no-test  # same, without the pytest sanity checks at the end
```

Needs `git`, [`uv`](https://docs.astral.sh/uv/) and network access. It

1. clones nanobrain into `fixtures/nanobrain` at `a1bfdd1` if it is missing (and stops if an existing
   checkout is at another commit),
2. runs `uv sync --frozen --inexact` in `tools/` (Python 3.12, CPU/MPS torch, and dcm2niix
   v1.0.20260724 from PyPI, whose wheel has the JPEG 2000 and JPEG-LS decoders Homebrew's build lacks),
3. runs `make_models.py reference` only if `fixtures/models/` lacks the reference models,
4. runs `fetch_fixtures.py` (about 120 MB of NIfTI downloads and 260 MB of DICOM from GitHub; the DICOM
   file lists come from the GitHub API, 9 requests, cached in `fixtures/downloads/github/`; set
   `GITHUB_TOKEN` if the anonymous rate limit runs out),
5. runs `make_reference.py all` (about 45 minutes on CPU, most of it the four patch-4 bf16 passes),
6. runs `pytest test_reference.py`.

Disk: about 0.4 GB of scans, 0.3 GB of DICOM, 0.6 GB of prep references and 4 GB of encode references
(patch 8: 86 MB per scan; patch 4: 240 MB per scan).

Run any script by hand from `tools/` with `.venv/bin/python`. Each imports `larmor_ref` first, which puts
nanobrain's `src/` on `sys.path` (`NANOBRAIN_SRC` overrides it) and stubs `SynthSeg_pytorch`, which
`eval/prepare.py` imports but the preprocessing never calls.

## Scripts and what they write

| Script | Writes |
|---|---|
| `larmor_ref.py` | nothing; shared helpers (paths, sha256, a minimal TOML writer, library versions) |
| `make_models.py reference` | `fixtures/models/nanobrain-vits-p{8,4}-ref.{toml,safetensors}`: nanobrain's ViT-S with `manual_seed(0)` init and every bias, norm, position table and class token perturbed |
| `make_models.py export CKPT --id ID` | `ID.toml` + `ID.safetensors` from a real nanobrain checkpoint (below) |
| `fetch_fixtures.py` | `fixtures/scans/*.nii.gz`, `fixtures/manifest.toml`, `fixtures/downloads/` (the 4D series the DWI b0 is cut from, GitHub tree listings), `fixtures/templateflow/` (TemplateFlow cache), `fixtures/dicom/<name>/` (DICOM series) and `fixtures/reference/dicom/<name>/dcm2niix.json` (dcm2niix's sidecar of each converted series) |
| `make_reference.py prep SCAN --name NAME` | `fixtures/reference/prep/NAME/`: `head_mask`, `conform`, `fit_mask`, `quantized` (.nii.gz) and `info.json` |
| `make_reference.py prep SCAN --name NAME --reference REF` | the same for SCAN prepared on REF's head mask and grid (`prepare_images` with `reference=`); `head_mask` is REF's, `info.json` adds `reference`, `reference_sha256` and `image_affine` |
| `make_reference.py encode NAME --model M.toml` | `fixtures/reference/encode/<model id>/NAME.safetensors` |
| `make_reference.py all` | both of the above for every manifest scan and `[[pair]]`, skipping outputs that are current |
| `test_reference.py` | nothing; pytest consistency checks over the scans and every reference |
| `reference/probe_*.py` | probe references for Phase 2 (owned by the probe work, see their docstrings); `probe_reference.py minority` records where scikit-learn raises on a small minority class |

`fetch_fixtures.py` pins every download by sha256 (DICOM files by their git blob SHA-1 and each set by
`files_sha256`) and keeps existing synthetic and dcm2niix files (`--force` rebuilds them, `--check` only
verifies the manifest). `make_reference.py all` regenerates a prep when the scan's
sha256 differs from the one in `info.json`, and an encode when the quantized volume it was made from has
changed; `--force` regenerates everything, and scan names restrict it to those scans.

### The fixture set (`fixtures/manifest.toml`)

One `[[scan]]` per file: `name`, `file`, `kind`, `modality`, `source` (exact URL, or
`synthetic from <name>: <transform>`), `license`, `sha256`, a header summary (`shape`, `zooms`,
`orientation`, `dtype`, raw `scl_slope`/`scl_inter`, `qform_code`, `sform_code`), `encode` (the models
with an encode reference) and `notes`.

* **openneuro**: real CC0 scans downloaded byte for byte: sagittal (PIL, PSR), axial (LPS, LAS) and
  coronal (LIA) storage; 0.43 to 2 mm, isotropic and 3 to 5 mm slices; int16, uint16, uint8 and float32;
  scl_slope unset, 0, 1 and real scaling; plus a DWI b0 cut from a 4D series.
* **templateflow**: MNI152NLin6Asym T1w (`mni152-t1w`) and MNI152NLin2009cAsym T1w and T2w, 1 mm.
* **pairs** (`[[pair]]`: `name`, `target`, `reference`, `encode`, `notes`): a target scan prepared on a
  reference scan's head mask and grid, the multi-modal path of `prepare_images` (FOMO tasks 1 and 2 prepare
  ADC and DWI on the FLAIR this way). `mni152-2009c-t2w-ref-t1w` is the 2009c T2w on the T1w's mask;
  `check-parity` gates its fit mask, quantized volume, affines and token grid bit-exact.
* **dcm2niix**: dcm2niix's NIfTI of each converted `[[dicom]]` series (below), so the recipe and encode
  references of the DICOM path are made from exactly what dcm2niix writes.
* **dicom** (`[[dicom]]`: `name`, `folder`, `kind`, `source`, `license`, `manufacturer`, `modality`,
  `files`, `files_sha256`, then `scan`, `refuse` or `unsupported`, `notes`): DICOM series from dcm2niix's
  validation repositories (`neurolabusc/dcm_qa_*`, BSD-2-Clause; CC0 DICOM of MR heads is scarce, since
  OpenNeuro keeps converted NIfTI). `kind = "series"`: Larmor must convert it bit for bit to the `[[scan]]`
  named `scan` and pass that scan's references: Siemens VE11 and XA30, GE, Philips (precise scaling) and
  UIH 3-D T1s (reoriented like dcm2niix's `nii_setOrtho`), 2-D sagittal T1s in four patient positions
  (rows flipped), an EPI volume, and XA30 T1s recompressed as fragmented JPEG lossless and JPEG 2000.
  `kind = "refuse"`: a dual-echo series and a three-orientation localizer, refused with the reason.
  `kind = "pixels"`: one image in nine transfer syntaxes that must decode alike (JPEG-LS refused).
* **synthetic**: variants of `mni152-t1w` (and one of `ds001246-t1w`), each aimed at a branch:
  (0.5, 1, 3) mm resample as in nanobrain's `test_preprocessing.py`, 0.8 mm (anti-alias smoothing), LPS /
  PIR / ASL storage, a 15 degree oblique affine, qform-only and sform-only headers (with stale fields
  whose code is 0, and a sheared sform), int16 with `scl_slope`/`scl_inter`, the top of the head cut off
  (top capped at the image height), an FOV smaller than the grid, a large FOV with neck and shoulders,
  and a 2 mm scan (the z center lands on a .5 rounding tie).

`info.json` records, besides the fields DESIGN.md lists, a `branches` object: the dtype threshold_mask
resamples in, the 2 mm grid, connected components, filled holes, the reorientation, conform's float32
zooms, whether and how much it smooths, the mask bounding box, rounding ties, `top` and whether it was
capped. Use it to see which code path a fixture takes, and where a mismatch starts.

## Exporting a real checkpoint (on the cluster)

Checkpoints stay where they are trained; only the encoder is exported. On the cluster, in a nanobrain
environment that has `safetensors`:

```sh
cd larmor/tools
NANOBRAIN_SRC=/path/to/nanobrain/src python make_models.py export \
    /path/to/checkpoints/vits_run2_epoch40.pth \
    --id nanobrain-vits-p8 --out /path/to/export \
    --description "nanobrain ViT-S MAE encoder, 8 mm patches, run 2 epoch 40"
```

The checkpoint must be nanobrain's own format (`model_class`, `args`, `model`, optionally `git_sha`). The
export writes `nanobrain-vits-p8.safetensors` (encoder tensors only, f32, nanobrain names) and
`nanobrain-vits-p8.toml` (architecture from `args`, the weights' sha256, the nanobrain-eval preprocessing
steps, and the checkpoint name and git sha under `[source]`). Copy both files to the Mac and keep them
side by side (for example in `models/`); the manifest refers to the weights by file name.
