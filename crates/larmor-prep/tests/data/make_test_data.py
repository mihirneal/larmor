"""Small end-to-end references for larmor-prep, from nanobrain's own preprocessing functions.

Run with the tools venv:

    tools/.venv/bin/python crates/larmor-prep/tests/data/make_test_data.py

Each case is a synthetic head phantom (a NIfTI with an awkward geometry) run through
`threshold_mask`, `conform_image` (fixed grid, a small field of view so the outputs stay small),
the fit-grid mask of `prepare_images`, `truncate_image`, the normalisation of
`ViTMAE3D.dense_embed`, the patch coordinates, and a nearest-neighbour map of a label volume back
to native space. Everything lands in `pipeline.safetensors` plus `pipeline.json`.
"""

import gzip
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[3] / "tools"))

import larmor_ref  # noqa: E402,F401  (puts nanobrain on sys.path)
import torch  # noqa: E402
from nibabel.processing import resample_from_to  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

import nanobrain.eval.prepare as prepare  # noqa: E402
import nanobrain.eval.preprocessing as preproc  # noqa: E402
import nanobrain.model as nbm  # noqa: E402

FOV = (24.0, 32.0, 24.0)
PATCH = 4
rng = np.random.default_rng(7)


def rotation(axis: str, degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def phantom(shape, affine) -> np.ndarray:
    """A head-like ellipsoid with a brain, a dark hole, an island and noise, in world mm."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    xyz = nib.affines.apply_affine(affine, ijk).reshape(*shape, 3)
    centre = np.array([1.0, -2.0, 0.5])
    r = ((xyz - centre) / np.array([10.0, 13.0, 10.0])) ** 2
    head = r.sum(-1) < 1
    brain = (((xyz - centre) / np.array([7.0, 9.5, 7.0])) ** 2).sum(-1) < 1
    hole = (((xyz - centre - [0, 2, 1]) / np.array([2.0, 2.5, 2.0])) ** 2).sum(-1) < 1
    island = ((xyz - centre - [0, 0, 13.5]) ** 2).sum(-1) < 4
    data = 900.0 * head + 400.0 * brain - 1100.0 * hole + 700.0 * island
    data = np.clip(data, 0, None) + rng.normal(0, 40, shape)
    return data


def nifti_bytes(data, affine, dtype, slope=None, inter=None) -> bytes:
    """A NIfTI with sform and qform (code 1), optionally storing integers with scale factors."""
    stored = data if slope is None else np.round((data - inter) / slope)
    img = nib.Nifti1Image(stored.astype(dtype), affine)
    img.header.set_qform(affine, code=1)
    img.header.set_sform(affine, code=1)
    if slope is not None:
        img.header.set_slope_inter(slope, inter)
    return img.to_bytes()


def nifti2_bytes(data, affine, dtype, zooms) -> bytes:
    """A NIfTI-2 with an sform (code 1) and float64 header zooms that differ from the affine's."""
    img = nib.Nifti2Image(data.astype(dtype), affine)
    img.header.set_sform(affine, code=1)
    img.header.set_qform(affine, code=0)
    img.header["pixdim"][1:4] = zooms
    return img.to_bytes()


def run_case(name, t, cases, raw: bytes):
    nifti2 = int.from_bytes(raw[:4], "little") == 540
    img = (nib.Nifti2Image if nifti2 else nib.Nifti1Image).from_bytes(raw)
    captured = {}
    rescale = preproc.rescale_affine

    def capture(affine, shape, zooms, new_shape=None, centroid=None):
        captured["centroid"] = np.asarray(centroid).tolist()
        return rescale(affine, shape, zooms, new_shape=new_shape, centroid=centroid)

    mask_img = preproc.threshold_mask(img)
    preproc.rescale_affine = capture
    try:
        conform_img = preproc.conform_image(img, mask_img, max_fov=FOV, fixed_grid=True)
    finally:
        preproc.rescale_affine = rescale
    fit_mask = np.asarray(resample_from_to(mask_img, conform_img, order=0).dataobj) > 0
    fit_mask_img = nib.Nifti1Image(fit_mask.astype(np.uint8), conform_img.affine)
    quant_img = prepare.truncate_image(conform_img, fit_mask_img, (prepare.VMIN_QUANTILE, prepare.VMAX_QUANTILE))
    values = conform_img.get_fdata(dtype=np.float32)[fit_mask]
    vmin, vmax = np.quantile(values, (prepare.VMIN_QUANTILE, prepare.VMAX_QUANTILE))

    # ViTMAE3D.dense_embed, step for step
    images = torch.from_numpy(np.asarray(quant_img.dataobj).astype(np.float32))[None]
    image_mask = torch.from_numpy(fit_mask)
    vals = images[0][image_mask]
    vals = vals / vals.max()
    vals = (vals - vals.mean()) / vals.std(correction=0).clamp_min(1e-6)
    image = torch.zeros_like(images[0])
    image[image_mask] = vals
    in_mask = nbm.patchify3d(image_mask[None], PATCH).any(dim=2)[0]
    coords = nbm.make_coord_grid(conform_img.shape, PATCH)[0, in_mask]

    # a label volume on the grid, mapped back to the scan's voxels
    grid = np.indices(conform_img.shape).sum(0) // 5 % 4
    labels = (grid * fit_mask).astype(np.uint8)
    native_labels = resample_from_to(nib.Nifti1Image(labels, conform_img.affine), img, order=0)

    t[f"{name}/scan"] = np.frombuffer(gzip.compress(raw, mtime=0), dtype=np.uint8)
    t[f"{name}/head_mask"] = np.asarray(mask_img.dataobj)
    t[f"{name}/conform"] = conform_img.get_fdata(dtype=np.float32)
    t[f"{name}/fit_mask"] = fit_mask
    t[f"{name}/quantized"] = np.asarray(quant_img.dataobj)
    t[f"{name}/image"] = image.numpy()
    t[f"{name}/window"] = np.array([vmin, vmax])
    t[f"{name}/centroid"] = np.array(captured["centroid"], dtype=np.int64)
    t[f"{name}/fit_affine"] = conform_img.affine
    t[f"{name}/coords"] = coords.numpy().astype(np.int64)
    t[f"{name}/labels"] = labels
    t[f"{name}/native_labels"] = np.asarray(native_labels.dataobj)
    cases[name] = {
        "fov": list(FOV),
        "patch": PATCH,
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "zooms": [float(z) for z in img.header.get_zooms()],
        "dtype": img.get_data_dtype().name,
        "tokens": int(in_mask.sum()),
    }
    print(f"{name}: {cases[name]['orientation']} {img.shape} {cases[name]['dtype']}, "
          f"{int(fit_mask.sum())} grid voxels in mask, {cases[name]['tokens']} tokens")


def main():
    t, cases = {}, {}
    # oblique LPS-ish, anisotropic with one axis finer than 1 mm (anti-aliased), int16 with scaling
    rzs = rotation("z", 12.0) @ rotation("x", -7.0) @ np.diag([-1.1, -0.9, 1.3])
    affine = nib.affines.from_matvec(rzs, [17.0, 15.5, -18.0])
    shape = (30, 36, 24)
    run_case("oblique_scaled_int16", t, cases,
             nifti_bytes(phantom(shape, affine), affine, np.int16, slope=0.37, inter=12.0))

    # PIL permuted axes, 1.5 mm, float32
    rzs = np.array([[0.0, 0.0, -1.5], [-1.5, 0.0, 0.0], [0.0, -1.5, 0.0]])
    affine = nib.affines.from_matvec(rzs, [17.0, 20.0, 16.0])
    shape = (24, 22, 22)
    run_case("pil_float32", t, cases, nifti_bytes(phantom(shape, affine), affine, np.float32))

    # RAS 0.8 mm uint8 (anti-aliased on every axis)
    affine = nib.affines.from_matvec(np.diag([0.8, 0.8, 0.8]), [-14.0, -18.0, -14.0])
    shape = (36, 44, 36)
    data = phantom(shape, affine)
    run_case("ras_fine_uint8", t, cases, nifti_bytes(np.clip(data / 6, 0, 255), affine, np.uint8))

    # NIfTI-2, 2 mm RAS: the 2 mm mask grid is the scan's own, so threshold_mask smooths with the
    # header's float64 zooms, which resample_to_output (a Nifti1Image) rounds to float32
    affine = nib.affines.from_matvec(np.diag([2.0, 2.0, 2.0]), [-14.0, -18.0, -14.0])
    shape = (15, 19, 15)
    run_case("nifti2_ras_2mm", t, cases,
             nifti2_bytes(phantom(shape, affine), affine, np.float32, (1.9999999, 2.0000003, 1.99999)))

    tensors = {k: np.ascontiguousarray(v) for k, v in t.items()}
    save_file(tensors, OUT / "pipeline.safetensors")
    (OUT / "pipeline.json").write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")
    print(f"pipeline.safetensors: {(OUT / 'pipeline.safetensors').stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
