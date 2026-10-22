"""Sanity checks over every fixture scan and reference output (docs/DESIGN.md, "Reference fixtures").

    .venv/bin/python -m pytest test_reference.py -q

These check that the references are internally consistent (shapes, dtypes, masks, token bookkeeping), not
that they match nanobrain: make_reference.py already asserts that while it writes them.
"""

import json
import tomllib

import nibabel as nib
import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import load_file

from larmor_ref import FIXTURES, sha256_file

GRID = (192, 240, 192)
REFERENCE = FIXTURES / "reference"
MANIFEST = tomllib.loads((FIXTURES / "manifest.toml").read_text())
SCANS = MANIFEST["scan"]
PAIRS = MANIFEST.get("pair", [])
ENCODES = [(scan, model_id) for scan in SCANS + PAIRS for model_id in scan["encode"]]


def scan_id(scan):
    return scan["name"]


def encode_id(item):
    return f"{item[0]['name']}-{item[1].split('-')[2]}"


def model_manifest(model_id: str) -> dict:
    return tomllib.loads((FIXTURES / "models" / f"{model_id}.toml").read_text())


def in_mask_patches(mask: np.ndarray, patch: int) -> np.ndarray:
    """patchify3d(mask).any(-1) on the (gx, gy, gz) patch grid."""
    gx, gy, gz = (n // patch for n in mask.shape)
    return mask.reshape(gx, patch, gy, patch, gz, patch).any(axis=(1, 3, 5))


def patchify(image: np.ndarray, patch: int) -> np.ndarray:
    """einops b (gx px) (gy py) (gz pz) -> b (gx gy gz) (px py pz), without the batch axis."""
    gx, gy, gz = (n // patch for n in image.shape)
    blocks = image.reshape(gx, patch, gy, patch, gz, patch).transpose(0, 2, 4, 1, 3, 5)
    return blocks.reshape(gx * gy * gz, patch**3)


def test_manifest_lists_about_forty_unique_scans():
    names = [scan["name"] for scan in SCANS]
    assert len(names) == len(set(names))
    assert 30 <= len(names) <= 45
    for scan in SCANS:
        assert {"name", "file", "source", "license", "sha256", "modality", "notes", "encode"} <= scan.keys()
        assert "nanobrain-vits-p8-ref" in scan["encode"]


@pytest.mark.parametrize("scan", SCANS, ids=scan_id)
def test_scan_matches_manifest(scan):
    path = FIXTURES / scan["file"]
    assert sha256_file(path) == scan["sha256"]
    img = nib.load(path)
    assert type(img) is nib.Nifti1Image and img.ndim == 3
    assert list(img.shape) == scan["shape"]
    assert str(img.get_data_dtype()) == scan["dtype"]
    assert "".join(nib.aff2axcodes(img.affine)) == scan["orientation"]


@pytest.mark.parametrize("scan", SCANS, ids=scan_id)
def test_prep_reference(scan):
    prep_dir = REFERENCE / "prep" / scan["name"]
    info = json.loads((prep_dir / "info.json").read_text())
    assert info["scan_sha256"] == scan["sha256"]
    design_keys = {"vmin", "vmax", "centroid", "affine", "canonical_affine", "fit_affine", "zooms", "fwhm", "versions"}
    assert design_keys <= info.keys()
    scan_img = nib.load(FIXTURES / scan["file"])

    head_mask = nib.load(prep_dir / "head_mask.nii.gz")
    assert head_mask.shape == scan_img.shape
    assert np.allclose(head_mask.affine, scan_img.affine)
    head = np.asanyarray(head_mask.dataobj)
    assert head.dtype == np.uint8 and set(np.unique(head)) <= {0, 1}
    assert int(head.sum()) == info["head_mask_voxels"] > 0

    conform = nib.load(prep_dir / "conform.nii.gz")
    fit_mask = nib.load(prep_dir / "fit_mask.nii.gz")
    quantized = nib.load(prep_dir / "quantized.nii.gz")
    for img in (conform, fit_mask, quantized):
        assert img.shape == GRID
        assert np.allclose(img.affine, info["fit_affine"])
        assert np.allclose(nib.affines.voxel_sizes(img.affine), 1.0)
    assert conform.get_data_dtype() == np.float32
    mask = np.asanyarray(fit_mask.dataobj)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 1}
    assert int(mask.sum()) == info["fit_mask_voxels"] > 0
    q = np.asanyarray(quantized.dataobj)
    assert q.dtype == np.uint16
    assert not q[mask == 0].any(), "quantized is nonzero outside the fit mask"
    assert q[mask > 0].max() == 65535 and info["vmax"] > info["vmin"]


@pytest.mark.parametrize("pair", PAIRS, ids=scan_id)
def test_pair_reference(pair):
    scans = {scan["name"]: scan for scan in SCANS}
    target, reference = scans[pair["target"]], scans[pair["reference"]]
    prep_dir = REFERENCE / "prep" / pair["name"]
    info = json.loads((prep_dir / "info.json").read_text())
    assert info["scan_sha256"] == target["sha256"] and info["reference_sha256"] == reference["sha256"]
    # the head mask is the reference's own
    reference_mask = REFERENCE / "prep" / reference["name"] / "head_mask.nii.gz"
    assert np.array_equal(
        np.asanyarray(nib.load(prep_dir / "head_mask.nii.gz").dataobj), np.asanyarray(nib.load(reference_mask).dataobj)
    )
    assert np.allclose(info["image_affine"], info["fit_affine"], atol=1e-3)
    mask = np.asanyarray(nib.load(prep_dir / "fit_mask.nii.gz").dataobj)
    assert int(mask.sum()) == info["fit_mask_voxels"] > 0
    q = np.asanyarray(nib.load(prep_dir / "quantized.nii.gz").dataobj)
    assert q.dtype == np.uint16 and not q[mask == 0].any() and q[mask > 0].max() == 65535


@pytest.mark.parametrize("item", ENCODES, ids=encode_id)
def test_encode_reference(item):
    scan, model_id = item
    model = model_manifest(model_id)
    patch, width = model["architecture"]["patch"], model["architecture"]["width"]
    classes = model["architecture"]["class_tokens"]
    path = REFERENCE / "encode" / model_id / f"{scan['name']}.safetensors"
    with safe_open(path, "np") as f:
        meta = f.metadata()
    assert meta["model_id"] == model_id and meta["scan"] == scan["name"]
    assert meta["model_sha256"] == model["weights"]["sha256"]
    prep_dir = REFERENCE / "prep" / scan["name"]
    if "quantized_sha256" in meta:
        assert meta["quantized_sha256"] == sha256_file(prep_dir / "quantized.nii.gz")

    t = load_file(path)
    fit_mask = np.asanyarray(nib.load(prep_dir / "fit_mask.nii.gz").dataobj)
    grid_in_mask = in_mask_patches(fit_mask > 0, patch)
    n = int(grid_in_mask.sum())
    expected = {
        "image": (np.float32, GRID),
        "mask": (np.uint8, GRID),
        "coords": (np.int32, (n, 3)),
        "patches": (np.float32, (n, patch**3)),
        "tokens": (np.float32, (n, width)),
        "cls": (np.float32, (classes, width)),
        "pooled": (np.float32, (width,)),
        "tokens_bf16": (np.float32, (n, width)),
        "pooled_bf16": (np.float32, (width,)),
    }
    assert set(t) == set(expected)
    for key, (dtype, shape) in expected.items():
        assert t[key].dtype == dtype and t[key].shape == shape, key
    assert int(meta["n_tokens"]) == n

    assert np.array_equal(t["mask"], fit_mask)
    image = t["image"]
    assert not image[t["mask"] == 0].any()
    values = image[t["mask"] > 0].astype(np.float64)
    assert abs(values.mean()) < 1e-4 and abs(values.std() - 1) < 1e-4

    # coords: (gx, gy, gz) of in-mask patches, ij meshgrid order, inside the patch grid
    coords = t["coords"]
    assert np.array_equal(coords, np.argwhere(grid_in_mask))
    assert (coords >= 0).all() and (coords < np.array(GRID) // patch).all()
    assert np.array_equal(t["patches"], patchify(image, patch)[grid_in_mask.reshape(-1)])

    assert np.isfinite(t["tokens"]).all() and np.isfinite(t["tokens_bf16"]).all()
    assert np.allclose(t["pooled"], t["tokens"].mean(axis=0, dtype=np.float64), atol=1e-5)
    assert np.allclose(t["pooled_bf16"], t["tokens_bf16"].mean(axis=0, dtype=np.float64), atol=1e-5)


DICOM = MANIFEST.get("dicom", [])


@pytest.mark.parametrize("entry", DICOM, ids=lambda e: e["name"])
def test_dicom_set_matches_manifest(entry):
    from fetch_fixtures import files_sha256

    folder = FIXTURES / entry["folder"]
    assert files_sha256(folder) == entry["files_sha256"]
    assert sum(1 for p in folder.rglob("*") if p.is_file()) == entry["files"]
    assert entry["kind"] in ("series", "refuse", "pixels")
    if entry["kind"] == "series":
        assert entry["scan"] in {scan["name"] for scan in SCANS}
        sidecar = REFERENCE / "dicom" / entry["name"] / "dcm2niix.json"
        assert json.loads(sidecar.read_text())["ConversionSoftwareVersion"] == "v1.0.20260724"
    if entry["kind"] == "refuse":
        assert entry["refuse"]
