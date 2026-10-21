"""PyTorch/nibabel reference outputs that Larmor is checked against.

    python make_reference.py prep SCAN --name NAME          # preprocessing stages -> reference/prep/NAME/
    python make_reference.py encode NAME --model MODEL.toml  # encoder outputs -> reference/encode/ID/NAME.safetensors
    python make_reference.py all [--force] [NAME ...]        # prep and encode for every scan in fixtures/manifest.toml

Layouts are specified in docs/DESIGN.md ("Reference fixtures"). Every stage is produced by nanobrain's own
functions; `prep` also checks that the captured stages equal `prepare_images` byte for byte.

`all` encodes each scan with the models its manifest entry lists under `encode` (patch 8 for every scan,
patch 4 for a few) and skips outputs that are already current: a prep whose info.json records the scan's
manifest sha256, an encode whose metadata records the sha256 of the quantized volume it was made from.
Outputs are written to a temporary name and renamed, so a reader never sees a half-written file.
"""

import argparse
import json
import math
import os
import time
import tomllib
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from nibabel.orientations import io_orientation
from nibabel.processing import fwhm2sigma, resample_from_to
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from larmor_ref import FIXTURES, sha256_file, versions

import nanobrain.eval.preprocessing as preproc  # noqa: E402
import nanobrain.eval.prepare as prepare  # noqa: E402
import nanobrain.model as nbm  # noqa: E402

REFERENCE = FIXTURES / "reference"
MODELS = FIXTURES / "models"
MANIFEST = FIXTURES / "manifest.toml"
PREP_FILES = ("head_mask.nii.gz", "conform.nii.gz", "fit_mask.nii.gz", "quantized.nii.gz", "info.json")
INFO_KEYS = ("scan_sha256", "fwhm", "branches")  # an info.json without these predates `all`


def save_nifti(img: nib.Nifti1Image, path: Path) -> None:
    tmp = path.with_name(f".{path.name}")  # keeps the .nii.gz suffix nibabel picks the format from
    nib.save(img, tmp)
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}")
    tmp.write_text(text)
    os.replace(tmp, path)


def display_path(path: Path) -> str:
    """Scan paths inside fixtures/ are recorded relative to it, so info.json does not name this machine."""
    try:
        return Path(path).resolve().relative_to(FIXTURES.resolve()).as_posix()
    except ValueError:
        return str(path)


def conform_branches(img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> dict:
    """The decisions eval/preprocessing.py::conform_image takes for this scan, with its own expressions.

    Kept literal (float32 header zooms, numpy scalar promotion) so the recorded values are the ones the
    reference used; run_prep checks the result against the centroid it captured.
    """
    fov = prepare.GRID_FOV
    canonical = nib.as_closest_canonical(img)
    # conform_image rebuilds the image from its affine, so zooms are float32 column norms of the affine
    zooms = nib.Nifti1Image(np.zeros((1, 1, 1), np.float32), canonical.affine).header.get_zooms()
    new_zooms = 3 * [1.0]
    smoothed = any(sz_ > sz for sz_, sz in zip(new_zooms, zooms))
    fwhm = [math.sqrt(max(sz_**2 - sz**2, 0.0)) for sz_, sz in zip(new_zooms, zooms)] if smoothed else None
    # smooth_image converts FWHM to voxel sigmas with float64 column norms, not the float32 zooms
    vox = np.sqrt(np.sum(canonical.affine[:, :3] ** 2, 0))
    sigma = fwhm2sigma(np.asarray(fwhm) / vox).tolist() if smoothed else None

    mask = np.asarray(nib.as_closest_canonical(mask_img).dataobj) > 0
    ids = [np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
    bbox = [[int(i.min()), int(i.max())] for i in ids]
    top = ids[2].max() + 5.0 / zooms[2]
    capped = bool(top > canonical.shape[2])
    top = min(top, canonical.shape[2])
    half_height = fov[2] / zooms[2] / 2
    z_center = top - half_height
    ornt = io_orientation(img.affine).astype(int)
    return {
        "canonical_ornt": ornt.tolist(),
        "reoriented": not np.array_equal(ornt, [[0, 1], [1, 1], [2, 1]]),
        "conform_zooms": [float(z) for z in zooms],
        "smoothed": smoothed,
        "smooth_sigma_vox": sigma,
        "mask_bbox": bbox,
        # x/y centers round((min + max) / 2) half to even when min + max is odd
        "centroid_ties": [ax for ax in (0, 1) if (bbox[ax][0] + bbox[ax][1]) % 2 == 1],
        "top": float(top),
        "top_capped": capped,
        "half_height": float(half_height),
        "z_center": float(z_center),
        "z_center_tie": bool(float(z_center) % 1 == 0.5),
        "_fwhm": fwhm,
    }


def run_prep(scan: Path, name: str, out_root: Path = REFERENCE / "prep") -> dict:
    """prepare_images for one scan as its own reference, keeping every intermediate."""
    img = nib.load(scan)

    captured = {}
    rescale_affine = preproc.rescale_affine
    largest_component = preproc.largest_component

    def capture_rescale(affine, shape, zooms, new_shape=None, centroid=None):
        captured.update(centroid=np.asarray(centroid).tolist(), new_shape=list(new_shape))
        return rescale_affine(affine, shape, zooms, new_shape=new_shape, centroid=centroid)

    def capture_components(mask):
        _, count = ndi.label(mask)
        kept = largest_component(mask)
        captured.update(
            mask_grid=list(mask.shape),
            components=int(count),
            largest_voxels=int(kept.sum()),
            holes_filled=int(ndi.binary_fill_holes(kept).sum() - kept.sum()),
        )
        return kept

    t0 = time.perf_counter()
    preproc.largest_component = capture_components
    try:
        mask_img = preproc.threshold_mask(img)
    finally:
        preproc.largest_component = largest_component
    t1 = time.perf_counter()
    preproc.rescale_affine = capture_rescale
    try:
        conform_img = preproc.conform_image(img, mask_img, max_fov=prepare.GRID_FOV, fixed_grid=True)
    finally:
        preproc.rescale_affine = rescale_affine
    t2 = time.perf_counter()
    affine = conform_img.affine
    fit_mask = np.asarray(resample_from_to(mask_img, conform_img, order=0).dataobj) > 0
    fit_mask_img = nib.Nifti1Image(fit_mask.astype(np.uint8), affine)
    quant_img = prepare.truncate_image(conform_img, fit_mask_img, (prepare.VMIN_QUANTILE, prepare.VMAX_QUANTILE))
    t3 = time.perf_counter()

    # the stages above must be exactly what the eval framework stores
    encoded = prepare.prepare_images({"image": img}, {}, reference="image")
    ref_mask = prepare.decode_nifti_zst(encoded["mask"])
    ref_image = prepare.decode_nifti_zst(encoded["image"])
    assert np.array_equal(np.asarray(ref_mask.dataobj), fit_mask.astype(np.uint8)), "fit mask differs"
    assert np.array_equal(np.asarray(ref_image.dataobj), np.asarray(quant_img.dataobj)), "quantized differs"
    assert np.allclose(ref_image.affine, affine), "affine differs"

    branches = conform_branches(img, mask_img)
    fwhm = branches.pop("_fwhm")
    assert round(branches["z_center"]) == captured["centroid"][2], "z center recomputed differently"
    assert all(round(sum(branches["mask_bbox"][ax]) / 2) == captured["centroid"][ax] for ax in (0, 1)), "x/y center"
    branches = {
        # threshold_mask resamples whatever dtype the scaled dataobj has (ints are rounded, scaled ints are float)
        "mask_resample_dtype": str(np.asanyarray(img.dataobj).dtype),
        "mask_grid": captured["mask_grid"],
        "components": captured["components"],
        "largest_voxels": captured["largest_voxels"],
        "holes_filled": captured["holes_filled"],
        **branches,
    }

    values = conform_img.get_fdata(dtype=np.float32)[fit_mask]
    vmin, vmax = np.quantile(values, (prepare.VMIN_QUANTILE, prepare.VMAX_QUANTILE))
    canonical = nib.as_closest_canonical(img)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_nifti(mask_img, out_dir / "head_mask.nii.gz")
    save_nifti(conform_img, out_dir / "conform.nii.gz")
    save_nifti(fit_mask_img, out_dir / "fit_mask.nii.gz")
    save_nifti(quant_img, out_dir / "quantized.nii.gz")
    info = {
        "scan": display_path(scan),
        "scan_sha256": sha256_file(scan),
        "shape": list(img.shape),
        "dtype": str(img.get_data_dtype()),
        "zooms": [float(z) for z in img.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "affine": img.affine.tolist(),
        "canonical_affine": canonical.affine.tolist(),
        "canonical_zooms": [float(z) for z in nib.affines.voxel_sizes(canonical.affine)],
        "head_mask_voxels": int((np.asarray(mask_img.dataobj) > 0).sum()),
        "centroid": captured["centroid"],
        "fit_affine": affine.tolist(),
        "fit_mask_voxels": int(fit_mask.sum()),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "fwhm": fwhm,
        "branches": branches,
        "seconds": {"mask": t1 - t0, "conform": t2 - t1, "quantize": t3 - t2},
        "versions": versions(),
    }
    write_text(out_dir / "info.json", json.dumps(info, indent=2))
    print(f"prep {name}: mask {info['head_mask_voxels']} vox, fit mask {info['fit_mask_voxels']} vox, "
          f"{sum(info['seconds'].values()):.1f}s")
    return info


def load_model(manifest_path: Path) -> tuple[dict, nbm.ViTMAE3D]:
    manifest = tomllib.loads(manifest_path.read_text())
    arch = manifest["architecture"]
    args = {
        "grid_size": arch["grid"],
        "patch_size": arch["patch"],
        "model_kwargs": {
            "depth": arch["depth"],
            "embed_dim": arch["width"],
            "num_heads": arch["heads"],
            "mlp_ratio": arch["mlp_ratio"],
            "class_tokens": arch["class_tokens"],
        },
    }
    model = nbm.ViTMAE3D.from_config(args)
    weights_path = manifest_path.parent / manifest["weights"]["file"]
    assert sha256_file(weights_path) == manifest["weights"]["sha256"], "weights sha256 mismatch"
    missing, unexpected = model.load_state_dict(load_file(weights_path), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("decoder") or k == "mask_token" for k in missing), missing
    return manifest, model.eval()


@torch.inference_mode()
def run_encode(name: str, manifest_path: Path, out_root: Path = REFERENCE / "encode") -> dict:
    manifest, model = load_model(manifest_path)
    prep_dir = REFERENCE / "prep" / name
    images = torch.from_numpy(nib.load(prep_dir / "quantized.nii.gz").get_fdata(dtype=np.float32))[None]
    mask = torch.from_numpy(np.asarray(nib.load(prep_dir / "fit_mask.nii.gz").dataobj).astype(np.float32))[None]
    p = model.patch_size
    c = model.class_tokens

    # dense_embed, step for step, keeping what it throws away (patches, coords, class tokens)
    image_mask = mask[0] > 0
    values = images[0][image_mask]
    values = values / values.max()
    values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
    image = torch.zeros_like(images[0])
    image[image_mask] = values
    patches = nbm.patchify3d(image[None], p)
    in_mask = nbm.patchify3d(image_mask[None], p).any(dim=2)[0]
    coords = model.coord_grid[0, in_mask]

    t0 = time.perf_counter()
    x = model.forward_encoder(patches[:, in_mask], model.coord_grid[:, in_mask])[0]
    seconds_fp32 = time.perf_counter() - t0
    tokens, cls = x[c:], x[:c]
    pooled = model.global_embed(images, mask)[0]
    assert torch.allclose(pooled, tokens.mean(dim=0), atol=1e-5), "pooled != mean of tokens"

    t0 = time.perf_counter()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dense_bf16 = model.dense_embed(images, mask).float().flatten(1, 3)[0]
    seconds_bf16 = time.perf_counter() - t0
    tokens_bf16 = dense_bf16[in_mask]
    pooled_bf16 = tokens_bf16.mean(dim=0)

    out_dir = out_root / manifest["id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "image": image.contiguous(),
        "mask": image_mask.to(torch.uint8).contiguous(),
        "coords": coords.to(torch.int32).contiguous(),
        "patches": patches[0, in_mask].contiguous(),
        "tokens": tokens.contiguous(),
        "cls": cls.contiguous(),
        "pooled": pooled.contiguous(),
        "tokens_bf16": tokens_bf16.contiguous(),
        "pooled_bf16": pooled_bf16.contiguous(),
    }
    cos = torch.nn.functional.cosine_similarity(tokens, tokens_bf16, dim=1)
    info = json.loads((prep_dir / "info.json").read_text())
    meta = {
        "model_id": manifest["id"],
        "model_sha256": manifest["weights"]["sha256"],
        "scan": name,
        "scan_sha256": info["scan_sha256"],
        "quantized_sha256": sha256_file(prep_dir / "quantized.nii.gz"),
        "n_tokens": str(len(tokens)),
        "seconds_fp32": f"{seconds_fp32:.3f}",
        "seconds_bf16": f"{seconds_bf16:.3f}",
        "bf16_min_token_cosine": f"{cos.min().item():.6f}",
        "bf16_pooled_cosine": f"{torch.nn.functional.cosine_similarity(pooled, pooled_bf16, dim=0).item():.6f}",
        **{f"version_{k}": v for k, v in versions().items()},
    }
    path = out_dir / f"{name}.safetensors"
    tmp = path.with_name(f".{path.name}")
    save_file(tensors, tmp, metadata=meta)
    os.replace(tmp, path)
    print(f"encode {name} with {manifest['id']}: {len(tokens)} tokens, fp32 {seconds_fp32:.1f}s, "
          f"bf16 {seconds_bf16:.1f}s, bf16 min token cosine {meta['bf16_min_token_cosine']}")
    return meta


# ---------------------------------------------------------------------------------------------------------
# all: every manifest scan


def load_manifest() -> list[dict]:
    return tomllib.loads(MANIFEST.read_text())["scan"]


def prep_is_current(scan: dict) -> bool:
    prep_dir = REFERENCE / "prep" / scan["name"]
    if not all((prep_dir / f).exists() for f in PREP_FILES):
        return False
    info = json.loads((prep_dir / "info.json").read_text())
    return all(k in info for k in INFO_KEYS) and info["scan_sha256"] == scan["sha256"]


def encode_metadata(name: str, model_id: str) -> dict | None:
    path = REFERENCE / "encode" / model_id / f"{name}.safetensors"
    if not path.exists():
        return None
    with safe_open(path, "pt") as f:
        return f.metadata()


def encode_is_current(name: str, model_id: str) -> bool:
    meta = encode_metadata(name, model_id)
    if meta is None:
        return False
    # encodes written before `all` existed carry no quantized_sha256; they are trusted as they are
    recorded = meta.get("quantized_sha256")
    return recorded is None or recorded == sha256_file(REFERENCE / "prep" / name / "quantized.nii.gz")


def run_all(names: list[str], force: bool) -> None:
    scans = load_manifest()
    unknown = set(names) - {s["name"] for s in scans}
    assert not unknown, f"not in {MANIFEST}: {sorted(unknown)}"
    scans = [s for s in scans if not names or s["name"] in names]
    t_start = time.perf_counter()
    for scan in scans:
        path = FIXTURES / scan["file"]
        assert sha256_file(path) == scan["sha256"], f"{path}: sha256 differs from the manifest"
        if force or not prep_is_current(scan):
            run_prep(path, scan["name"])
    model_ids = sorted({m for s in scans for m in s["encode"]}, key=lambda m: m.endswith("p4-ref"))
    for model_id in model_ids:
        for scan in scans:
            if model_id in scan["encode"] and (force or not encode_is_current(scan["name"], model_id)):
                run_encode(scan["name"], MODELS / f"{model_id}.toml")
    print_summary(scans, model_ids)
    print(f"all: {time.perf_counter() - t_start:.0f}s")


def print_summary(scans: list[dict], model_ids: list[str]) -> None:
    header = f"{'scan':26} {'prep s':>7} " + " ".join(f"{m.split('-')[2]:>6} tok {'fp32 s':>7} {'bf16 s':>7}"
                                                     for m in model_ids)
    print(header)
    for scan in scans:
        info = json.loads((REFERENCE / "prep" / scan["name"] / "info.json").read_text())
        row = f"{scan['name']:26} {sum(info['seconds'].values()):7.1f} "
        for model_id in model_ids:
            meta = encode_metadata(scan["name"], model_id)
            if meta is None:
                row += f"{'':>10} {'':>7} {'':>7} "
            else:
                row += f"{meta['n_tokens']:>10} {float(meta['seconds_fp32']):7.1f} {float(meta['seconds_bf16']):7.1f} "
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("scan", type=Path)
    prep.add_argument("--name", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("name")
    enc.add_argument("--model", type=Path, required=True)
    everything = sub.add_parser("all")
    everything.add_argument("names", nargs="*", help="only these manifest scans")
    everything.add_argument("--force", action="store_true", help="regenerate outputs that are current")
    args = parser.parse_args()
    if args.cmd == "prep":
        run_prep(args.scan, args.name)
    elif args.cmd == "encode":
        run_encode(args.name, args.model)
    else:
        run_all(args.names, args.force)
