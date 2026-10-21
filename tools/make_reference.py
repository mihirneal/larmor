"""PyTorch/nibabel reference outputs that Larmor is checked against.

    python make_reference.py prep SCAN --name NAME          # preprocessing stages -> reference/prep/NAME/
    python make_reference.py encode NAME --model MODEL.toml  # encoder outputs -> reference/encode/ID/NAME.safetensors

Layouts are specified in docs/DESIGN.md ("Reference fixtures"). Every stage is produced by nanobrain's own
functions; `prep` also checks that the captured stages equal `prepare_images` byte for byte.
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to
from safetensors.torch import load_file, save_file

from larmor_ref import FIXTURES, sha256_file, versions

import nanobrain.eval.preprocessing as preproc  # noqa: E402
import nanobrain.eval.prepare as prepare  # noqa: E402
import nanobrain.model as nbm  # noqa: E402

REFERENCE = FIXTURES / "reference"


def run_prep(scan: Path, name: str, out_root: Path = REFERENCE / "prep") -> dict:
    """prepare_images for one scan as its own reference, keeping every intermediate."""
    img = nib.load(scan)

    captured = {}
    rescale_affine = preproc.rescale_affine

    def capture(affine, shape, zooms, new_shape=None, centroid=None):
        captured.update(centroid=np.asarray(centroid).tolist(), new_shape=list(new_shape))
        return rescale_affine(affine, shape, zooms, new_shape=new_shape, centroid=centroid)

    t0 = time.perf_counter()
    mask_img = preproc.threshold_mask(img)
    t1 = time.perf_counter()
    preproc.rescale_affine = capture
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

    values = conform_img.get_fdata(dtype=np.float32)[fit_mask]
    vmin, vmax = np.quantile(values, (prepare.VMIN_QUANTILE, prepare.VMAX_QUANTILE))
    canonical = nib.as_closest_canonical(img)
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(mask_img, out_dir / "head_mask.nii.gz")
    nib.save(conform_img, out_dir / "conform.nii.gz")
    nib.save(fit_mask_img, out_dir / "fit_mask.nii.gz")
    nib.save(quant_img, out_dir / "quantized.nii.gz")
    info = {
        "scan": str(scan),
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
        "seconds": {"mask": t1 - t0, "conform": t2 - t1, "quantize": t3 - t2},
        "versions": versions(),
    }
    (out_dir / "info.json").write_text(json.dumps(info, indent=2))
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
    meta = {
        "model_id": manifest["id"],
        "model_sha256": manifest["weights"]["sha256"],
        "scan": name,
        "n_tokens": str(len(tokens)),
        "seconds_fp32": f"{seconds_fp32:.3f}",
        "seconds_bf16": f"{seconds_bf16:.3f}",
        "bf16_min_token_cosine": f"{cos.min().item():.6f}",
        "bf16_pooled_cosine": f"{torch.nn.functional.cosine_similarity(pooled, pooled_bf16, dim=0).item():.6f}",
        **{f"version_{k}": v for k, v in versions().items()},
    }
    save_file(tensors, out_dir / f"{name}.safetensors", metadata=meta)
    print(f"encode {name} with {manifest['id']}: {len(tokens)} tokens, fp32 {seconds_fp32:.1f}s, "
          f"bf16 min token cosine {meta['bf16_min_token_cosine']}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prep")
    prep.add_argument("scan", type=Path)
    prep.add_argument("--name", required=True)
    enc = sub.add_parser("encode")
    enc.add_argument("name")
    enc.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "prep":
        run_prep(args.scan, args.name)
    else:
        run_encode(args.name, args.model)
