"""Build Larmor's fixture scans (fixtures/scans/) and write fixtures/manifest.toml.

    python fetch_fixtures.py            # download and build whatever is missing, then write the manifest
    python fetch_fixtures.py --force    # also rebuild synthetic and dcm2niix scans that already exist
    python fetch_fixtures.py --check    # only verify the files listed in the manifest

Four kinds of scan, all 3D NIfTI-1 .nii.gz (plus `[[pair]]` and `[[dicom]]` entries, below):

* openneuro: real public scans over anonymous HTTPS, byte-identical to the source and pinned by sha256.
  The DWI b0 is the one derived file: volume 0 cut out of a pinned 4D series kept in fixtures/downloads/.
* templateflow: MNI templates fetched by the templateflow client (TEMPLATEFLOW_HOME=fixtures/templateflow).
* synthetic: deterministic nibabel variants of mni152-t1w (and one of ds001246-t1w), chosen so that every
  branch of nanobrain's threshold_mask and conform_image runs on at least one fixture.
* dcm2niix: the NIfTI the pinned dcm2niix (tools/.venv/bin/dcm2niix, v1.0.20260724) writes for a DICOM
  series of `[[dicom]]`, gzipped with a fixed header.

A `[[dicom]]` is a DICOM series from dcm2niix's own validation repositories (github.com/neurolabusc/dcm_qa_*,
BSD-2-Clause), pinned by commit, each file checked against its git blob SHA-1 and the set by `files_sha256`,
unpacked into fixtures/dicom/<name>/. `kind = "series"` must convert to the `[[scan]]` named `scan`
(dcm2niix's NIfTI of it) bit for bit and then pass that scan's recipe and encode references; `"refuse"` must
be refused with a message containing `refuse`; `"pixels"` holds one image in several transfer syntaxes that
must decode alike. CC0 DICOM of MR heads is scarce (OpenNeuro keeps converted NIfTI, not DICOM), which is why
these BSD-licensed sets are used; nothing from them is committed.

A `[[pair]]` names two scans of one subject on the same voxel grid, a target prepared on its reference's head
mask and grid as `eval/prepare.py::prepare_images(images, segs, reference=...)` prepares FOMO tasks 1 and 2 (ADC and
DWI on the FLAIR's mask); make_reference.py writes their references under reference/prep/<pair name>/.

Synthetic files are written field by field (so a header is exactly what the fixture tests, including stale
qform/sform matrices whose code is 0) with a fixed gzip header, so a rebuild with the same zlib gives the
same bytes. Existing synthetic and dcm2niix files are kept unless --force, so their sha256 values stay stable.
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
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
# DICOM series and dcm2niix's NIfTI of them

DICOM = FIXTURES / "dicom"
GITHUB_RAW = "https://raw.githubusercontent.com"
GITHUB_API = "https://api.github.com"
DCM2NIIX_VERSION = "v1.0.20260724"
DCM2NIIX = Path(sys.executable).parent / "dcm2niix"


@dataclass(frozen=True)
class DicomFixture:
    """A DICOM series from one of dcm2niix's validation repositories (github.com/neurolabusc/dcm_qa_*),
    pinned by commit; every downloaded file is checked against its git blob SHA-1 in that commit's tree.

    `path` selects the files: a folder, a single .zip file, or a file-name prefix. `unpack` is "files"
    (copy), "gz" (gunzip each `.gz`) or "zip" (extract). `kind` is "series" (Larmor converts it bit-exact
    to dcm2niix's NIfTI, the [[scan]] named `scan`, and runs the recipe on it), "refuse" (Larmor refuses it
    with a message containing `refuse`) or "pixels" (every file decodes to the same samples, except the
    transfer syntaxes in `unsupported`, which Larmor refuses).
    """

    name: str
    repo: str
    commit: str
    path: str
    unpack: str
    files_sha256: str
    manufacturer: str
    modality: str
    kind: str
    notes: str
    scan: str | None = None
    refuse: str | None = None
    unsupported: tuple[str, ...] = ()

    @property
    def folder(self) -> Path:
        return DICOM / self.name

    @property
    def source(self) -> str:
        return f"https://github.com/{self.repo}/tree/{self.commit}/{self.path}"

    @property
    def license(self) -> str:
        return f"BSD-2-Clause ({self.repo})"


def files_sha256(folder: Path) -> str:
    """The SHA-256 of a folder's files' SHA-256s, sorted (names and order do not matter)."""
    digests = sorted(sha256_file(p) for p in folder.rglob("*") if p.is_file())
    return hashlib.sha256("".join(d + "\n" for d in digests).encode()).hexdigest()


def git_tree(repo: str, commit: str) -> list[dict]:
    """The blobs of a commit's tree, from the GitHub API (cached in fixtures/downloads/github/)."""
    cache = DOWNLOADS / "github" / f"{repo.replace('/', '--')}@{commit}.json"
    if not cache.exists():
        request = urllib.request.Request(f"{GITHUB_API}/repos/{repo}/git/trees/{commit}?recursive=1")
        if os.environ.get("GITHUB_TOKEN"):
            request.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
        with urllib.request.urlopen(request, timeout=60) as response:
            tree = json.load(response)
        assert not tree.get("truncated"), f"{repo}@{commit}: the tree listing is truncated"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps([t for t in tree["tree"] if t["type"] == "blob"]))
    return json.loads(cache.read_text())


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fetch_dicom(fixture: DicomFixture) -> None:
    """Download and unpack the series into fixtures/dicom/<name>/, unless it already holds the pinned files."""
    if fixture.folder.exists() and files_sha256(fixture.folder) == fixture.files_sha256:
        return
    tree = git_tree(fixture.repo, fixture.commit)
    # a single file (a .zip), a folder's files, or the files whose names start with a prefix
    if any(b["path"] == fixture.path for b in tree):
        blobs, base = [b for b in tree if b["path"] == fixture.path], str(Path(fixture.path).parent)
    elif any(b["path"].startswith(fixture.path + "/") for b in tree):
        blobs, base = [b for b in tree if b["path"].startswith(fixture.path + "/")], fixture.path
    else:
        blobs, base = [b for b in tree if b["path"].startswith(fixture.path)], str(Path(fixture.path).parent)
    blobs = [b for b in blobs if not Path(b["path"]).name.startswith(".")]
    assert blobs, f"{fixture.name}: nothing at {fixture.path} in {fixture.repo}@{fixture.commit}"
    part = fixture.folder.with_name(fixture.folder.name + ".part")
    shutil.rmtree(part, ignore_errors=True)
    part.mkdir(parents=True)
    print(f"  downloading {len(blobs)} files of {fixture.source}")
    for blob in blobs:
        url = f"{GITHUB_RAW}/{fixture.repo}/{fixture.commit}/{urllib.parse.quote(blob['path'])}"
        for attempt in range(1, 5):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    data = response.read()
                break
            except OSError as err:
                if attempt == 4:
                    raise
                print(f"  retry {attempt}: {err}")
                time.sleep(2 * attempt)
        assert git_blob_sha1(data) == blob["sha"], f"{url}: not the pinned blob"
        relative = Path(blob["path"]).relative_to(base)
        if fixture.unpack == "zip":
            # the archive's files, flattened into the series folder
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    name = Path(member.filename)
                    if member.is_dir() or "__MACOSX" in name.parts or name.name.startswith("."):
                        continue
                    target = part / name.name
                    assert not target.exists(), f"{fixture.name}: two members named {name.name}"
                    target.write_bytes(archive.read(member))
        elif fixture.unpack == "gz":
            assert relative.suffix == ".gz", blob["path"]
            target = part / relative.with_suffix("")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(gzip.decompress(data))
        else:
            (part / relative).parent.mkdir(parents=True, exist_ok=True)
            (part / relative).write_bytes(data)
    got = files_sha256(part)
    if os.environ.get("LARMOR_PRINT_DICOM_HASHES"):
        print(f"  {fixture.name}: files_sha256 {got}")
    else:
        assert got == fixture.files_sha256, f"{fixture.name}: files_sha256 {got} is not the pinned one"
    shutil.rmtree(fixture.folder, ignore_errors=True)
    part.replace(fixture.folder)


def dcm2niix_nifti(fixture: DicomFixture, out_dir: Path) -> Path:
    """dcm2niix's NIfTI-1 (uncompressed) and JSON sidecar of the series; it must make exactly one image."""
    assert DCM2NIIX.exists(), f"{DCM2NIIX} is missing: run `uv sync` in tools/"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*"):
        old.unlink()
    result = subprocess.run(
        [str(DCM2NIIX), "-z", "n", "-b", "y", "-f", "%s", "-o", str(out_dir), str(fixture.folder)],
        capture_output=True, text=True, check=True,
    )
    assert DCM2NIIX_VERSION in result.stdout, result.stdout[:200]
    images = sorted(out_dir.glob("*.nii"))
    assert len(images) == 1, f"{fixture.name}: dcm2niix wrote {[p.name for p in images]}\n{result.stdout}"
    return images[0]


def same_image(a: Path, b: Path) -> bool:
    """The parts of two NIfTI files that nibabel's loaded image exposes: voxels, dtype, affine, zooms, scaling."""
    ia, ib = nib.load(a), nib.load(b)
    ha, hb = raw_header(a), raw_header(b)
    return (
        ia.shape == ib.shape
        and ia.get_data_dtype() == ib.get_data_dtype()
        and np.array_equal(ia.affine, ib.affine)
        and ia.header.get_zooms() == ib.header.get_zooms()
        and all(np.array_equal(ha[k], hb[k], equal_nan=True) for k in ("scl_slope", "scl_inter"))
        and np.array_equal(np.asanyarray(ia.dataobj.get_unscaled()), np.asanyarray(ib.dataobj.get_unscaled()))
    )


def dcm2niix_fixture(fixture: DicomFixture) -> Fixture:
    """The [[scan]] holding dcm2niix's NIfTI of a series, gzipped with a fixed header (stable bytes)."""

    def build(dest: Path):
        fetch_dicom(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            nii = dcm2niix_nifti(fixture, Path(tmp))
            sidecar = REFERENCE_DICOM / fixture.name / "dcm2niix.json"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(nii.with_suffix(".json"), sidecar)
            # nibabel picks the format from the name, so the temporary file keeps the .nii.gz ending
            part = dest.with_name(f".part-{dest.name}")
            with part.open("wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
                gz.write(nii.read_bytes())
            assert same_image(nii, part)
            part.replace(dest)
        assert_3d_nifti1(dest)

    source = f"dcm2niix {DCM2NIIX_VERSION} (-z n) of dicom/{fixture.name}, from {fixture.source}"
    return Fixture(fixture.name, "dcm2niix", fixture.modality, source, fixture.license, fixture.notes, build)


REFERENCE_DICOM = FIXTURES / "reference" / "dicom"
NEUROLABUSC = "neurolabusc"

QA = "neurolabusc/dcm_qa"
MPRAGE = (f"{QA}_mprage", "01f876ffb29461c2e65702856cfe10c8c8534cd9")
GE = (f"{QA}_ge", "e92133ae414e2b23e8c4d07a4030ab4dbb41518e")
PHILIPS = (f"{QA}_philips_enh", "31426f16861bab6af7b23d15638a190749041a3e")
UIH = (f"{QA}_uih", "f3bd991a98eaf49a6f928f276a41f666d658a64b")
DECUBITUS = (f"{QA}_decubitus", "92cd635dc0a8f971d0e1d4e4ce69efff70fdc04e")
XA30 = (f"{QA}_xa30i", "699b8c5f6a10efa5e563435d2ce515a511a1e965")
FRAG = (f"{QA}_frag", "8d102ee4545e248f40d0db81024297cefac1e9c9")
PDT2 = (f"{QA}_pdt2", "ae6013ba4d8613fb28050055344e53d3325447cf")
TS = (f"{QA}_ts", "5ea2cf0c48e04a943b9473ced7db701bc4ae57d8")


def dicom(name, at, path, unpack, sha, manufacturer, modality, kind, notes, **kw) -> DicomFixture:
    return DicomFixture(name, at[0], at[1], path, unpack, sha, manufacturer, modality, kind, notes, **kw)


DICOM_SET: list[DicomFixture] = [
    dicom(
        "dcm-siemens-mprage-sag", MPRAGE, "In/6_T1_mprage_ns_sag_p2.zip", "zip",
        "324870c3ae6d715feb908362fbce897137cd719adab458e0fb10a7c198c34ccd",
        "Siemens", "T1w", "series",
        "Siemens Prisma VE11 3-D sagittal MPRAGE, 1 mm, 192 slices, slightly oblique: reoriented like any 3-D "
        "acquisition (nii_setOrtho)", scan="dcm-siemens-mprage-sag",
    ),
    dicom(
        "dcm-ge-mprage-ax", GE, "In/05_3D_Ax_T1_MP_RAGE_R22", "files",
        "f737ca09be1b08b2df54d60b4bdb4aab00c66deef55e17e9f268e5549bd555fe",
        "GE", "T1w", "series",
        "GE 3-D axial MPRAGE, 0.94 x 0.94 x 1.2 mm, oblique in two axes; file names out of instance order",
        scan="dcm-ge-mprage-ax",
    ),
    dicom(
        "dcm-philips-mprage-sag", PHILIPS, "In/cl/1301_", "gz",
        "4f7a8bdefb044974178924928aa58819724c6f438df9167c6c140f4447bb6546",
        "Philips", "T1w", "series",
        "Philips classic 3-D sagittal MPRAGE (ADNI), 170 slices; Philips precise-value scaling (ScaleSlope)",
        scan="dcm-philips-mprage-sag",
    ),
    dicom(
        "dcm-uih-t1-sag", UIH, "In/DTI_132225/t1_gre_fsp_3d_sag__132750", "files",
        "0f33b3d8fc6c179ea1baa55bb2377635521636ab7cee833cdbb484d5e8e83e0a",
        "UIH", "T1w", "series",
        "United Imaging 3-D sagittal T1, 1 x 0.5 x 0.5 mm (anti-alias smoothing in-plane), uint16",
        scan="dcm-uih-t1-sag",
    ),
    *[
        dicom(
            f"dcm-siemens-t1-2d-{position.lower()}", DECUBITUS, f"In/{position}/{series}", "files", sha,
            "Siemens", "T1w", "series",
            f"Siemens 2-D sagittal T1 FLASH, 35 slices of 4 mm every 4.8 mm, patient position {position}: rows "
            "flipped (nii_flipY), the sform's measured 4.799995 mm spacing next to pixdim's 4.8",
            scan=f"dcm-siemens-t1-2d-{position.lower()}",
        )
        for position, series, sha in [
            ("HFS", "3_t1_fl2d_sag", "a462a9d43d11b9f2a7a152205046d522671d965e2669d6d6c15388aa9e738a18"),
            ("HFP", "2_t1_fl2d_sag", "57d1802c2ea54fe954041d6efd196b7375ea5dafc4dc2cd1c03e657c5e5c039d"),
            ("HFDL", "2_t1_fl2d_sag", "205e45f4dde3df19911367fc5e609e3bddc8dd9c64174790a70930da52e77526"),
            ("HFDR", "2_t1_fl2d_sag", "145b8e559509a58525340b0e134fa19930fc18c0c31eaf9f97349a7affbcc388"),
        ]
    ],
    dicom(
        "dcm-siemens-xa30-t1w", XA30, "In/5001_anat-T1w_acq-tfl", "gz",
        "17bd356dfea6e465f102a91ff72deaf3ffee2cf624f090893df8606b4350ae6a",
        "Siemens", "T1w", "series",
        "Siemens XA30 classic (interoperability) 3-D T1w, 128 x 128 x 96", scan="dcm-siemens-xa30-t1w",
    ),
    dicom(
        "dcm-siemens-xa30-jpegll", FRAG, "In/5001_anat-T1w_acq-tfl", "files",
        "a40a4d404db2a147cd4813356d984e07549fd91ceb075ec22b9eb9d69844f62d",
        "Siemens", "T1w", "series",
        "the dcm-siemens-xa30-t1w series recompressed as JPEG lossless, each slice split over several fragments; "
        "converts to the same image", scan="dcm-siemens-xa30-t1w",
    ),
    dicom(
        "dcm-siemens-xa30-t1w-j2k", FRAG, "In/6001_anat-T1w_acq-tfl", "files",
        "0aa571d5a9216c292f8e5c0ddb749f5be3ff784f846990e727fe852e4d82163c",
        "Siemens", "T1w", "series",
        "Siemens XA30 3-D T1w recompressed as JPEG 2000 lossless in several fragments per slice (uint16)",
        scan="dcm-siemens-xa30-t1w-j2k",
    ),
    dicom(
        "dcm-siemens-xa30-bold", XA30, "In/7001_func-bold_task-fa_run-1", "gz",
        "32f257f2e0931e39c7053f70527bf90b37330c8f07c494f04739d451ee0189f1",
        "Siemens", "BOLD", "series",
        "one 2-D EPI volume (64 x 64 x 33, 3.5 mm) saved as its own series: the echo-planar path (rows flipped)",
        scan="dcm-siemens-xa30-bold",
    ),
    dicom(
        "dcm-siemens-pdt2", PDT2, "In/Siemens/VE11", "files",
        "c8c3c6ae1855775917a48a696da6cfbac8112073178cd3e229db9bae3f8750b3",
        "Siemens", "PD/T2", "refuse",
        "dual-echo PD/T2 in one series (dcm2niix writes _e1 and _e2): refused", refuse="more than one echo",
    ),
    dicom(
        "dcm-siemens-xa30-scout", XA30, "In/2001_AAHead_Scout", "gz",
        "6ed6340110c9f83dd51938c8fcc2fbb3973ec90ceb9cdf686664aeb14a96f9f0",
        "Siemens", "localizer", "refuse",
        "MPR localizer slices in three orientations in one series: refused", refuse="different orientations",
    ),
    dicom(
        "dcm-transfer-syntaxes", TS, "In", "files",
        "43ea40ef2c2c3afedf210a491c167bd9c629cbf0ebb7d0c0467279542df7523e",
        "Siemens", "BOLD", "pixels",
        "one Siemens mosaic saved in nine transfer syntaxes (explicit and implicit VR little endian, big endian, "
        "RLE, JPEG lossless, JPEG-LS, JPEG 2000): every decoder gives the same samples; JPEG-LS is refused",
        unsupported=("1.2.840.10008.1.2.4.80", "1.2.840.10008.1.2.4.81"),
    ),
]

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
    # dcm2niix's NIfTI of each DICOM series (one per [[dicom]] that names itself as its scan)
    *[dcm2niix_fixture(d) for d in DICOM_SET if d.kind == "series" and d.scan == d.name],
]


# ---------------------------------------------------------------------------------------------------------
# manifest


@dataclass(frozen=True)
class Pair:
    name: str
    target: str  # a FIXTURE_SET name
    reference: str  # a FIXTURE_SET name on the same voxel grid
    notes: str


PAIRS: list[Pair] = [
    Pair(
        "mni152-2009c-t2w-ref-t1w",
        "mni152-2009c-t2w",
        "mni152-2009c-t1w",
        "T2w prepared on the T1w's head mask and grid (prepare_images with reference), the multi-modal path",
    ),
]


def pair_entry(pair: Pair) -> list[str]:
    fields = {
        "name": pair.name,
        "target": pair.target,
        "reference": pair.reference,
        "encode": [m for m in MODELS if m.endswith("p8-ref")],
        "notes": pair.notes,
    }
    return ["[[pair]]", *(f"{k} = {toml_value(v)}" for k, v in fields.items()), ""]


def dicom_entry(fixture: DicomFixture) -> list[str]:
    fields = {
        "name": fixture.name,
        "folder": fixture.folder.relative_to(FIXTURES).as_posix(),
        "kind": fixture.kind,
        "source": fixture.source,
        "license": fixture.license,
        "manufacturer": fixture.manufacturer,
        "modality": fixture.modality,
        "files": sum(1 for p in fixture.folder.rglob("*") if p.is_file()),
        "files_sha256": fixture.files_sha256,
    }
    if fixture.scan:
        fields["scan"] = fixture.scan
    if fixture.refuse:
        fields["refuse"] = fixture.refuse
    if fixture.unsupported:
        fields["unsupported"] = list(fixture.unsupported)
    fields["notes"] = fixture.notes
    return ["[[dicom]]", *(f"{k} = {toml_value(v)}" for k, v in fields.items()), ""]


def check_same_conversions() -> None:
    """A series whose [[scan]] is another series' dcm2niix NIfTI must convert to that same image."""
    for fixture in DICOM_SET:
        if fixture.kind != "series" or fixture.scan == fixture.name:
            continue
        fetch_dicom(fixture)
        with tempfile.TemporaryDirectory() as tmp:
            nii = dcm2niix_nifti(fixture, Path(tmp))
            assert same_image(nii, SCANS / f"{fixture.scan}.nii.gz"), f"{fixture.name}: not the image of {fixture.scan}"
            sidecar = REFERENCE_DICOM / fixture.name / "dcm2niix.json"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(nii.with_suffix(".json"), sidecar)


def raw_header(path: Path) -> nib.Nifti1Header:
    """The header fields exactly as stored (a loaded image's header has scl_slope/scl_inter reset)."""
    with path.open("rb") as raw:
        gzipped = raw.read(2) == b"\x1f\x8b"
    with (gzip.open(path, "rb") if gzipped else path.open("rb")) as f:
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
        "# A [[pair]] is a target scan prepared on a reference scan's head mask and grid (prepare_images).",
        "# A [[dicom]] is a DICOM series under folder: kind series converts to the [[scan]] named scan (dcm2niix's",
        "# NIfTI of it), refuse is refused with a message containing refuse, pixels decodes to the same samples.",
        "",
    ]
    for fixture in fixtures:
        lines += manifest_entry(fixture)
    names = {f.name for f in fixtures}
    for pair in PAIRS:
        assert {pair.target, pair.reference} <= names, pair
        lines += pair_entry(pair)
    for fixture in DICOM_SET:
        assert fixture.scan is None or fixture.scan in names, fixture
        lines += dicom_entry(fixture)
    MANIFEST.write_text("\n".join(lines))
    print(f"wrote {MANIFEST} ({len(fixtures)} scans, {len(PAIRS)} pairs, {len(DICOM_SET)} DICOM sets)")


def check_manifest() -> None:
    import tomllib

    manifest = tomllib.loads(MANIFEST.read_text())
    scans = manifest["scan"]
    for scan in scans:
        path = FIXTURES / scan["file"]
        assert sha256_file(path) == scan["sha256"], f"{scan['name']}: sha256 mismatch"
        assert_3d_nifti1(path)
    for entry in manifest.get("dicom", []):
        assert files_sha256(FIXTURES / entry["folder"]) == entry["files_sha256"], f"{entry['name']}: files differ"
    print(f"{len(scans)} scans and {len(manifest.get('dicom', []))} DICOM sets match {MANIFEST}")


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
    unknown = set(args.names) - {f.name for f in FIXTURE_SET} - {d.name for d in DICOM_SET}
    assert not unknown, f"unknown fixtures {sorted(unknown)}"
    for fixture in DICOM_SET:
        if not args.names or fixture.name in args.names:
            fetch_dicom(fixture)
    for fixture in FIXTURE_SET:
        if args.names and fixture.name not in args.names:
            continue
        # built files are kept (their sha256 stays stable) unless rebuilt on purpose
        if fixture.kind in ("synthetic", "dcm2niix") and fixture.path.exists() and not (args.force or args.names):
            continue
        t0 = time.perf_counter()
        fixture.build(fixture.path)
        print(f"{fixture.name}: {fixture.path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - t0:.1f}s")
    missing = [f.name for f in FIXTURE_SET if not f.path.exists()]
    assert not missing, f"missing fixtures {missing}"
    check_same_conversions()
    write_manifest(FIXTURE_SET)


if __name__ == "__main__":
    main()
