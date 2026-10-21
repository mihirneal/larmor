"""Build Larmor's fixture scans (fixtures/scans/) and write fixtures/manifest.toml.

    python fetch_fixtures.py            # download and build whatever is missing, then write the manifest
    python fetch_fixtures.py --force    # also rebuild synthetic variants that already exist
    python fetch_fixtures.py --check    # only verify the files listed in the manifest

Three kinds of scan, all 3D NIfTI-1 .nii.gz:

* openneuro: real public scans over anonymous HTTPS, byte-identical to the source and pinned by sha256.
  The DWI b0 is the one derived file: volume 0 cut out of a pinned 4D series kept in fixtures/downloads/.
* templateflow: MNI templates fetched by the templateflow client (TEMPLATEFLOW_HOME=fixtures/templateflow).
* synthetic: deterministic nibabel variants of mni152-t1w (and one of ds001246-t1w), chosen so that every
  branch of nanobrain's threshold_mask and conform_image runs on at least one fixture.

Synthetic files are written field by field (so a header is exactly what the fixture tests, including stale
qform/sform matrices whose code is 0) with a fixed gzip header, so a rebuild with the same zlib gives the
same bytes. Existing synthetic files are kept unless --force, so their sha256 values stay stable.
"""

import argparse
import gzip
import math
import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from larmor_ref import FIXTURES, sha256_file, toml_value

os.environ.setdefault("TEMPLATEFLOW_HOME", str(FIXTURES / "templateflow"))

import nibabel as nib  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402
from nibabel.affines import apply_affine, from_matvec, voxel_sizes  # noqa: E402
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform  # noqa: E402
from nibabel.processing import resample_to_output, smooth_image  # noqa: E402

SCANS = FIXTURES / "scans"
DOWNLOADS = FIXTURES / "downloads"
MANIFEST = FIXTURES / "manifest.toml"
OPENNEURO = "https://s3.amazonaws.com/openneuro.org"
TEMPLATEFLOW_S3 = "https://templateflow.s3.amazonaws.com"

# Scans that also get a patch-4 encode reference (about 60k tokens each; CPU torch takes ~8 min per scan):
# mni152-t1w (the Phase 0 reference) plus a real sagittal scan, a thick-slice scan and an oblique affine.
P4_SCANS = ("mni152-t1w", "ds001246-t1w", "ds000221-flair-lowres", "mni152-oblique")
MODELS = ("nanobrain-vits-p8-ref", "nanobrain-vits-p4-ref")

CC0 = "CC0"
MNI6_LICENSE = "FSL MNI152 (ICBM 152 6th generation) terms; TemplateFlow tpl-MNI152NLin6Asym"
MNI2009_LICENSE = "MNI ICBM 152 2009 terms: free use with copyright notice; TemplateFlow tpl-MNI152NLin2009cAsym"


@dataclass(frozen=True)
class Fixture:
    name: str
    kind: str  # openneuro | templateflow | synthetic
    modality: str
    source: str
    license: str
    notes: str
    build: Callable[[Path], None]

    @property
    def path(self) -> Path:
        return SCANS / f"{self.name}.nii.gz"


# ---------------------------------------------------------------------------------------------------------
# writing NIfTI files


def nifti_header(
    dtype,
    affine: np.ndarray,
    *,
    qform_code: int = 1,
    sform_code: int = 1,
    qform: np.ndarray | None = None,
    sform: np.ndarray | None = None,
    slope: float = np.nan,
    inter: float = np.nan,
    descrip: str = "larmor fixture",
) -> nib.Nifti1Header:
    """A fresh NIfTI-1 header; `qform` / `sform` override what is stored in a field whose code may be 0."""
    hdr = nib.Nifti1Header()
    hdr.set_data_dtype(dtype)
    hdr.set_xyzt_units("mm", "sec")
    hdr.set_qform(affine if qform is None else qform, code=qform_code)
    hdr.set_sform(affine if sform is None else sform, code=sform_code)
    hdr["scl_slope"] = slope
    hdr["scl_inter"] = inter
    hdr["descrip"] = descrip.encode()[:79]
    return hdr


def write_nifti(path: Path, raw: np.ndarray, hdr: nib.Nifti1Header) -> None:
    """Write a single-file NIfTI-1 (.nii.gz) with exactly these header fields and these raw values.

    nibabel's own writer recomputes scl_slope/scl_inter, so the bytes are written by hand: the 348-byte
    header, 4 zero bytes (no extensions), then the array in Fortran order at vox_offset 352.
    """
    hdr = hdr.copy()
    assert raw.ndim == 3, raw.shape
    assert raw.dtype == hdr.get_data_dtype(), (raw.dtype, hdr.get_data_dtype())
    hdr.set_data_shape(raw.shape)
    hdr["vox_offset"] = 352
    part = path.with_name(path.name + ".part")
    with part.open("wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
        gz.write(hdr.binaryblock)
        gz.write(b"\0" * 4)
        gz.write(np.asarray(raw, dtype=hdr.get_data_dtype()).tobytes(order="F"))
    part.replace(path)

    img = nib.load(path)
    assert img.shape == raw.shape and img.get_data_dtype() == raw.dtype
    assert np.array_equal(np.asanyarray(img.dataobj.get_unscaled()), raw)


def assert_3d_nifti1(path: Path) -> None:
    img = nib.load(path)
    assert type(img) is nib.Nifti1Image, f"{path}: {type(img).__name__}, not NIfTI-1"
    assert img.ndim == 3, f"{path}: {img.ndim}D"


# ---------------------------------------------------------------------------------------------------------
# downloads


def download(url: str, dest: Path, sha256: str, attempts: int = 4) -> None:
    """Fetch url into dest atomically, unless dest already holds the pinned bytes."""
    if dest.exists() and sha256_file(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            break
        except OSError as err:
            if attempt == attempts:
                raise
            print(f"  retry {attempt}: {err}")
            time.sleep(2 * attempt)
    got = sha256_file(part)
    if got != sha256:
        part.unlink()
        raise RuntimeError(f"{url}: sha256 {got} is not the pinned {sha256}")
    part.replace(dest)


def openneuro(name, modality, dataset, key, sha256, notes) -> Fixture:
    url = f"{OPENNEURO}/{dataset}/{key}"

    def build(dest: Path):
        download(url, dest, sha256)
        assert_3d_nifti1(dest)

    return Fixture(name, "openneuro", modality, url, f"{CC0} (OpenNeuro {dataset})", notes, build)


def openneuro_volume(name, modality, dataset, key, sha256, volume, notes) -> Fixture:
    """One volume of a 4D OpenNeuro series, header copied from the source with the 4th dim dropped."""
    url = f"{OPENNEURO}/{dataset}/{key}"
    source_path = DOWNLOADS / f"{dataset}_{Path(key).name}"

    def build(dest: Path):
        download(url, source_path, sha256)
        img = nib.load(source_path)
        hdr = raw_header(source_path)
        hdr.set_data_shape(img.shape[:3])
        write_nifti(dest, np.asanyarray(img.dataobj.get_unscaled())[..., volume], hdr)

    source = f"extracted from {url}: volume {volume}"
    return Fixture(name, "openneuro", modality, source, f"{CC0} (OpenNeuro {dataset})", notes, build)


def templateflow(name, modality, template, suffix, sha256, notes, license) -> Fixture:
    filename = f"tpl-{template}_res-01_{suffix}.nii.gz"

    def build(dest: Path):
        from templateflow.client import TemplateFlowClient

        path = TemplateFlowClient().get(template, resolution=1, suffix=suffix, desc=None, extension="nii.gz")
        assert sha256_file(path) == sha256, f"{path}: not the pinned templateflow file"
        if not dest.exists() or sha256_file(dest) != sha256:
            shutil.copyfile(path, dest)
        assert_3d_nifti1(dest)

    url = f"{TEMPLATEFLOW_S3}/tpl-{template}/{filename}"
    return Fixture(name, "templateflow", modality, url, license, notes, build)


# ---------------------------------------------------------------------------------------------------------
# synthetic variants


def synthetic(name, modality, base, transform, notes, make, license=MNI6_LICENSE) -> Fixture:
    """`make(base_img)` returns the raw on-disk array and the header to write it with."""

    def build(dest: Path):
        raw, hdr = make(nib.load(SCANS / f"{base}.nii.gz"))
        write_nifti(dest, raw, hdr)

    return Fixture(name, "synthetic", modality, f"synthetic from {base}: {transform}", license, notes, build)


def float32_image(img: nib.Nifti1Image) -> nib.Nifti1Image:
    return nib.Nifti1Image(img.get_fdata(dtype=np.float32), img.affine)


def rotation(axis, degrees: float) -> np.ndarray:
    """Rodrigues rotation matrix about `axis`."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    theta = math.radians(degrees)
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def rotated_about_center(affine: np.ndarray, shape, rot: np.ndarray) -> np.ndarray:
    """Rotate the physical image about its own center: the same voxels, a rotated affine."""
    center = apply_affine(affine, (np.asarray(shape) - 1) / 2)
    world = from_matvec(rot, center - rot @ center)
    return world @ affine


def voxel_shift(affine: np.ndarray, offset) -> np.ndarray:
    """Affine of a crop or pad whose voxel (0, 0, 0) sits at voxel `offset` of the original."""
    return affine @ from_matvec(np.eye(3), np.asarray(offset, dtype=np.float64))


def make_resampled(zooms, dtype=None):
    def make(base):
        img = base if dtype is None else float32_image(base)
        out = resample_to_output(img, voxel_sizes=zooms, order=1)
        raw = np.asanyarray(out.dataobj)
        return raw, nifti_header(raw.dtype, out.affine, qform_code=4, sform_code=4)

    return make


def make_reoriented(axcodes: str):
    def make(base):
        ornt = ornt_transform(io_orientation(base.affine), axcodes2ornt(tuple(axcodes)))
        out = base.as_reoriented(ornt)
        assert "".join(nib.aff2axcodes(out.affine)) == axcodes
        raw = np.asanyarray(out.dataobj)
        return raw, nifti_header(raw.dtype, out.affine, qform_code=4, sform_code=4)

    return make


def make_oblique(base):
    rot = rotation((1.0, 0.4, 0.25), 15.0)
    affine = rotated_about_center(base.affine, base.shape, rot)
    return np.asanyarray(base.dataobj), nifti_header(np.int16, affine, qform_code=1, sform_code=1)


def make_qform_only(base):
    # stored left-handed (x flipped, qfac = -1) and yawed 10 degrees; the stale sform is the original
    # unflipped, unrotated affine, so a reader that ignores sform_code = 0 gets a mirrored head
    raw = np.asanyarray(base.dataobj)[::-1]
    flipped = voxel_shift(base.affine, (base.shape[0] - 1, 0, 0)) @ np.diag([-1.0, 1, 1, 1])
    qform = rotated_about_center(flipped, base.shape, rotation((0, 0, 1), 10.0))
    hdr = nifti_header(np.int16, qform, qform_code=1, sform_code=0, sform=base.affine)
    return np.ascontiguousarray(raw), hdr


def make_sform_only(base):
    # a sheared sform (only sform can store shear); the qform is a stale identity with qform_code = 0,
    # so pixdim says 1 mm while the sform's columns are not unit length
    shear = np.eye(4)
    shear[0, 1] = 0.05
    shear[2, 0] = -0.03
    sform = base.affine @ shear
    hdr = nifti_header(np.int16, sform, qform_code=0, sform_code=2, qform=np.eye(4))
    return np.asanyarray(base.dataobj), hdr


def make_int16_scaled(base):
    slope, inter = np.float32(0.37), np.float32(12.5)
    values = base.get_fdata(dtype=np.float64)
    raw = np.round((values - float(inter)) / float(slope)).astype(np.int16)
    return raw, nifti_header(np.int16, base.affine, qform_code=2, sform_code=2, slope=slope, inter=inter)


def make_crop(start, stop):
    def make(base):
        sl = tuple(slice(a, b) for a, b in zip(start, stop))
        raw = np.ascontiguousarray(np.asanyarray(base.dataobj)[sl])
        return raw, nifti_header(raw.dtype, voxel_shift(base.affine, start), qform_code=4, sform_code=4)

    return make


def make_2mm(base):
    # a low-resolution acquisition: blur to a 2 mm PSF, resample to 2 mm, store as float64
    smoothed = smooth_image(float32_image(base), fwhm=math.sqrt(2.0**2 - 1.0**2))
    out = resample_to_output(smoothed, voxel_sizes=2.0, order=1)
    raw = np.asanyarray(out.dataobj).astype(np.float64)
    return raw, nifti_header(np.float64, out.affine, qform_code=4, sform_code=4)


def make_neck_shoulders(base):
    """ds001246-t1w (sagittal, FOV ends at the chin) in RAS with a synthetic neck and shoulders below.

    The real voxels from the first full tissue slice up are kept bit for bit, in raw int16 units with the
    source's slope and intercept. Below them the cross-section of that slice narrows into a neck over 30 mm,
    then widens over 60 mm into shoulders that run off both sides of the 340 mm FOV. Tissue texture and
    background noise come from a seeded generator, so the file is deterministic.
    """
    rng = np.random.default_rng(20261021)
    canonical = nib.as_closest_canonical(nib.Nifti1Image(base.dataobj.get_unscaled(), base.affine))
    real = np.asanyarray(canonical.dataobj)
    slope, inter = float(base.dataobj.slope), float(base.dataobj.inter)
    values = real * slope + inter
    nx, ny, nz = real.shape
    # the source's lowest ~22 slices are empty; start at the second slice with a full cross-section
    first = int(np.nonzero((values > 20).sum(axis=(0, 1)) > 10_000)[0][1])
    pad_x, pad_z = 74, 90
    shape = (nx + 2 * pad_x, ny, nz + pad_z)
    keep_from = pad_z + first

    out = rng.rayleigh(1.5, size=shape)
    bottom = np.zeros(shape[:2])
    bottom[pad_x : pad_x + nx] = values[:, :, first]
    section = ndi.binary_fill_holes(ndi.gaussian_filter(bottom, 2.0) > 40.0)
    tissue = float(np.median(bottom[section]))
    noise = ndi.gaussian_filter(rng.normal(size=(*shape[:2], keep_from)), 3.0)
    texture = tissue * (1.0 + 0.15 * noise / noise.std())
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    neck = section
    for depth in range(1, keep_from + 1):  # mm below the kept real voxels
        z = keep_from - depth
        if depth <= 30:
            if depth % 2 == 0:
                neck = ndi.binary_erosion(neck)
            body = neck
        else:
            if depth == 31:
                xs, ys = np.nonzero(neck.any(axis=1))[0], np.nonzero(neck.any(axis=0))[0]
                cx, cy = xs.mean(), ys.mean()
                neck_x, neck_y = (xs.max() - xs.min()) / 2, (ys.max() - ys.min()) / 2
            t = min(1.0, (depth - 30) / 60)
            widen = t * t * (3 - 2 * t)  # smoothstep from the neck to the shoulders
            half_x = neck_x + (200 - neck_x) * widen
            half_y = neck_y * (1 + 0.5 * widen)
            body = neck | (((xx - cx) / half_x) ** 2 + ((yy - cy) / half_y) ** 2 <= 1.0)
        out[:, :, z] = np.where(body, texture[:, :, z], out[:, :, z])
    out[pad_x : pad_x + nx, :, keep_from:] = values[:, :, first:]

    raw = np.clip(np.round((out - inter) / slope), -32768, 32767).astype(np.int16)
    raw[pad_x : pad_x + nx, :, keep_from:] = real[:, :, first:]
    affine = voxel_shift(canonical.affine, (-pad_x, 0, -pad_z))
    hdr = nifti_header(np.int16, affine, qform_code=1, sform_code=1, slope=slope, inter=inter)
    return raw, hdr


# ---------------------------------------------------------------------------------------------------------
# the fixture set

FIXTURE_SET: list[Fixture] = [
    # real scans (OpenNeuro, all CC0)
    openneuro(
        "ds001246-t1w", "T1w", "ds001246", "sub-01/ses-anatomy/anat/sub-01_ses-anatomy_T1w.nii.gz",
        "9fc7e09e8487b343473d448d2631a590109e1c7f44a6fe4e36272cd3501f9ea7",
        "sagittal MPRAGE stored PIL, 1 mm, slightly oblique affine, int16 with scl_slope/scl_inter; "
        "FOV reaches the upper neck, lowest 22 slices empty",
    ),
    openneuro(
        "ds002345-t1w", "T1w", "ds002345", "sub-001/anat/sub-001_T1w.nii.gz",
        "b6894c4b257e33d9ebdfdd3716cce3ff3bf41f8426a81bfdaa47c9619df0b3da",
        "sagittal stored PSR, 0.86 x 0.86 x 0.9 mm (anti-alias smoothing on all axes), int16 scaled",
    ),
    openneuro(
        "ds002424-t1w", "T1w", "ds002424", "sub-01/ses-T1/anat/sub-01_ses-T1_T1w.nii.gz",
        "105f185473b796910d133d9f574689f51042dc0793520ee15eb87d0140f3f439",
        "stored LPS, uint16 with scl_slope, qform_code 1 / sform_code 2; top capped; float32 zooms just under "
        "1 mm switch on a near-zero anti-alias smoothing (FWHM 0.0003 / 0 / 0.0055 mm)",
    ),
    openneuro(
        "ds003020-t1w", "T1w", "ds003020", "sub-UTS01/ses-1/anat/sub-UTS01_ses-1_T1w.nii.gz",
        "65cdecf72a4ea161ff42d9ba1e70ed3cc0ac86eee4c4e1be57549542236839d9",
        "coronal LIA (FreeSurfer conformed 256^3), uint8 with scl_slope 0.918",
    ),
    openneuro(
        "ds000221-t2w", "T2w", "ds000221", "sub-010002/ses-01/anat/sub-010002_ses-01_T2w.nii.gz",
        "5c98cd48d0be7ebf6196340e2d1f966d8fcca5cfe04085787c09d523a6bcde0f",
        "RAS 1 mm, int16 with a small scl_slope (0.0061) and large scl_inter (201.5)",
    ),
    openneuro(
        "ds001246-t2w-inplane", "T2w", "ds001246",
        "sub-01/ses-perceptionTest01/anat/sub-01_ses-perceptionTest01_inplaneT2.nii.gz",
        "a7b182147dee50faec085c0f76e3657f6bd394ea84b410717f2a44f5b1638c97",
        "axial LPS, 0.75 x 0.75 x 3 mm partial coverage, int16 with scl_slope 0 (means unscaled); "
        "threshold finds 2 components; top capped",
    ),
    openneuro(
        "ds000221-flair-lowres", "FLAIR", "ds000221",
        "sub-010002/ses-01/anat/sub-010002_ses-01_acq-lowres_FLAIR.nii.gz",
        "3856b5033c02be322d8c03a6104c6c68aa14f6f04d58ba0cbac90f4fab85d2a7",
        "axial LAS, 0.43 x 0.43 x 5 mm (smoothing in-plane, none through-plane), int16; top capped",
    ),
    openneuro(
        "ds004199-flair", "FLAIR", "ds004199", "sub-00001/anat/sub-00001_acq-T2sel_FLAIR.nii.gz",
        "e82ca267867e1ba3f0201c04cba2f5e2614eb3a26d07607dac20cc4c72573050",
        "3D FLAIR, RAS 1 mm, int16 scaled; threshold finds 2 components",
    ),
    openneuro_volume(
        "ds002785-dwi-b0", "DWI", "ds002785", "sub-0001/dwi/sub-0001_dwi.nii.gz",
        "e6b31afe1938de9e50e9fae8ffcece0a608f219f99aeafea2b95639953b430b0", 0,
        "b=0 volume (bval 0) of a 33-volume DWI series, LAS 2 mm, float32; header copied from the 4D file",
    ),
    # templates
    templateflow(
        "mni152-t1w", "T1w", "MNI152NLin6Asym", "T1w",
        "f675946c0a077e0682885c3d6085f10acb0dd22b2274a6cbbf04969c08a77258",
        "MNI152NLin6Asym 1 mm, RAS, int16, qform/sform code 4; base of most synthetic variants",
        MNI6_LICENSE,
    ),
    templateflow(
        "mni152-2009c-t1w", "T1w", "MNI152NLin2009cAsym", "T1w",
        "1f27aabea9f7183dc0c69dafa71e1787b921ca778d655d9c1bf302b273d5627a",
        "MNI152NLin2009cAsym 1 mm T1w template", MNI2009_LICENSE,
    ),
    templateflow(
        "mni152-2009c-t2w", "T2w", "MNI152NLin2009cAsym", "T2w",
        "6e39951e4478ff041f1290d31935dc05aa5425f9fd3f20ed57d4646f0d5be0ff",
        "MNI152NLin2009cAsym 1 mm T2w template", MNI2009_LICENSE,
    ),
    # synthetic variants
    synthetic(
        "mni152-aniso", "T1w", "mni152-t1w",
        "resample_to_output(img, (0.5, 1.0, 3.0), order=1), as in nanobrain utils/test_preprocessing.py",
        "anisotropic 0.5 x 1 x 3 mm int16 (resampling rounds); smoothing on x only",
        make_resampled((0.5, 1.0, 3.0)),
    ),
    synthetic(
        "mni152-iso08", "T1w", "mni152-t1w", "cast to float32, resample_to_output(img, 0.8, order=1)",
        "0.8 mm isotropic float32: anti-alias smoothing with FWHM 0.6 mm on every axis",
        make_resampled(0.8, dtype=np.float32),
    ),
    synthetic(
        "mni152-lps", "T1w", "mni152-t1w", "as_reoriented to LPS (flip x and y, affine updated)",
        "same physical image stored LPS", make_reoriented("LPS"),
    ),
    synthetic(
        "mni152-pir", "T1w", "mni152-t1w", "as_reoriented to PIR (permute and flip, affine updated)",
        "same physical image stored PIR (sagittal-style axis order)", make_reoriented("PIR"),
    ),
    synthetic(
        "mni152-asl", "T1w", "mni152-t1w", "as_reoriented to ASL (permute and flip, affine updated)",
        "same physical image stored ASL", make_reoriented("ASL"),
    ),
    synthetic(
        "mni152-oblique", "T1w", "mni152-t1w",
        "affine rotated 15 degrees about axis (1, 0.4, 0.25) through the volume center",
        "oblique affine: non-diagonal canonical affine and fit grid; qform and sform both code 1",
        make_oblique,
    ),
    synthetic(
        "mni152-qform-only", "T1w", "mni152-t1w",
        "data flipped in x, qform = flipped affine yawed 10 degrees (qfac -1), sform_code 0",
        "qform-only header: sform_code 0 with a stale unflipped sform that must be ignored",
        make_qform_only,
    ),
    synthetic(
        "mni152-sform-only", "T1w", "mni152-t1w",
        "sform = affine with shears 0.05 (x by j) and -0.03 (z by i), sform_code 2, qform_code 0",
        "sform-only header with a sheared affine; pixdim (1 mm) disagrees with the sform column norms",
        make_sform_only,
    ),
    synthetic(
        "mni152-int16-scaled", "T1w", "mni152-t1w",
        "raw = round((v - 12.5) / 0.37) as int16 with scl_slope 0.37, scl_inter 12.5",
        "int16 with scl_slope and scl_inter: scaled values are not integers", make_int16_scaled,
    ),
    synthetic(
        "mni152-crop-top", "T1w", "mni152-t1w", "crop z to [0, 141): top of the head cut off",
        "head mask reaches the top slice, so top = min(mask top + 5 mm, image height) is capped",
        make_crop((0, 0, 0), (182, 218, 141)),
    ),
    synthetic(
        "mni152-small-fov", "T1w", "mni152-t1w", "crop to x [20, 162), y [20, 202), z [10, 160)",
        "142 x 182 x 150 mm FOV, well inside the 192 x 240 x 192 grid; scalp cut on five faces",
        make_crop((20, 20, 10), (162, 202, 160)),
    ),
    synthetic(
        "ds001246-neck-shoulders", "T1w", "ds001246-t1w",
        "reoriented to RAS, padded to 340 x 256 x 346 mm with a seeded synthetic neck and shoulders below "
        "the real FOV and seeded background noise",
        "large FOV: mask spans neck and shoulders that run off both x borders; z anchored at the head top",
        make_neck_shoulders, license=f"{CC0} (OpenNeuro ds001246)",
    ),
    synthetic(
        "mni152-2mm", "T1w", "mni152-t1w",
        "cast to float32, smooth_image(fwhm=sqrt(3)), resample_to_output(img, 2.0, order=1), stored float64",
        "2 mm low resolution float64: the 2 mm mask resample is an identity; z center lands on a .5 tie",
        make_2mm,
    ),
]


# ---------------------------------------------------------------------------------------------------------
# manifest


def raw_header(path: Path) -> nib.Nifti1Header:
    """The header fields exactly as stored (a loaded image's header has scl_slope/scl_inter reset)."""
    with gzip.open(path, "rb") as f:
        return nib.Nifti1Header.from_fileobj(f)


def header_summary(path: Path) -> dict:
    img = nib.load(path)
    hdr = raw_header(path)
    slope, inter = (float(hdr[k]) for k in ("scl_slope", "scl_inter"))
    return {
        "shape": [int(n) for n in img.shape],
        "zooms": [round(float(z), 4) for z in voxel_sizes(img.affine)],
        "orientation": "".join(nib.aff2axcodes(img.affine)),
        "dtype": str(img.get_data_dtype()),
        "scl_slope": slope,
        "scl_inter": inter,
        "qform_code": int(hdr["qform_code"]),
        "sform_code": int(hdr["sform_code"]),
    }


def manifest_entry(fixture: Fixture) -> list[str]:
    encode = [m for m in MODELS if m.endswith("p8-ref") or fixture.name in P4_SCANS]
    fields = {
        "name": fixture.name,
        "file": fixture.path.relative_to(FIXTURES).as_posix(),
        "kind": fixture.kind,
        "modality": fixture.modality,
        "source": fixture.source,
        "license": fixture.license,
        "sha256": sha256_file(fixture.path),
        **header_summary(fixture.path),
        "encode": encode,
        "notes": fixture.notes,
    }
    return ["[[scan]]", *(f"{k} = {toml_value(v)}" for k, v in fields.items()), ""]


def write_manifest(fixtures: list[Fixture]) -> None:
    lines = [
        "# Larmor fixture scans: written by tools/fetch_fixtures.py, do not edit by hand.",
        "# file is relative to this directory. zooms are the affine's column norms (what nibabel resamples",
        "# with); scl_slope = nan or 0 means unscaled. encode lists the models with an encode reference.",
        "",
    ]
    for fixture in fixtures:
        lines += manifest_entry(fixture)
    MANIFEST.write_text("\n".join(lines))
    print(f"wrote {MANIFEST} ({len(fixtures)} scans)")


def check_manifest() -> None:
    import tomllib

    scans = tomllib.loads(MANIFEST.read_text())["scan"]
    for scan in scans:
        path = FIXTURES / scan["file"]
        assert sha256_file(path) == scan["sha256"], f"{scan['name']}: sha256 mismatch"
        assert_3d_nifti1(path)
    print(f"{len(scans)} scans match {MANIFEST}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="only build these fixtures (the manifest still lists all)")
    parser.add_argument("--force", action="store_true", help="rebuild synthetic variants that exist")
    parser.add_argument("--check", action="store_true", help="only verify files against the manifest")
    args = parser.parse_args()
    if args.check:
        check_manifest()
        return

    SCANS.mkdir(parents=True, exist_ok=True)
    unknown = set(args.names) - {f.name for f in FIXTURE_SET}
    assert not unknown, f"unknown fixtures {sorted(unknown)}"
    for fixture in FIXTURE_SET:
        if args.names and fixture.name not in args.names:
            continue
        if fixture.kind == "synthetic" and fixture.path.exists() and not (args.force or args.names):
            continue
        t0 = time.perf_counter()
        fixture.build(fixture.path)
        print(f"{fixture.name}: {fixture.path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s")
    missing = [f.name for f in FIXTURE_SET if not f.path.exists()]
    assert not missing, f"missing fixtures {missing}"
    write_manifest(FIXTURE_SET)


if __name__ == "__main__":
    main()
