"""Reference outputs for larmor-volume's op-level tests, from numpy, scipy and nibabel.

Run with the tools venv (the versions nanobrain pins), from anywhere:

    tools/.venv/bin/python crates/larmor-volume/tests/data/make_test_data.py

Writes `<module>.safetensors` (every number, so nothing is lost to JSON float parsing) and
`<module>.json` (case names and non-numeric parameters) next to this script. The outputs are small
and committed, so `cargo test` needs no Python.
"""

import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from nibabel.processing import resample_from_to
from nibabel.spaces import vox2out_vox
from safetensors.numpy import save_file

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(20261021)


def save(module: str, tensors: dict, cases: dict):
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    save_file(tensors, OUT / f"{module}.safetensors")
    (OUT / f"{module}.json").write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")
    size = (OUT / f"{module}.safetensors").stat().st_size
    print(f"{module}: {len(cases)} cases, {len(tensors)} tensors, {size / 1024:.0f} KiB")


def formula_values(n: int) -> np.ndarray:
    """Deterministic float32 values that Rust can regenerate exactly (integer ops, one division)."""
    i = np.arange(n, dtype=np.int64)
    return (((i * 7919) % 10007) / 37.0 - 100.0).astype(np.float32)


def rotation(axis: str, degrees: float) -> np.ndarray:
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def affine_from(rzs, trans) -> np.ndarray:
    return nib.affines.from_matvec(np.asarray(rzs, dtype=np.float64), np.asarray(trans, dtype=np.float64))


# --------------------------------------------------------------------------------------- affine
def make_affine():
    t, cases = {}, {}
    for i in range(40):
        a = rng.standard_normal((4, 4)) * rng.choice([1e-2, 1.0, 100.0], size=(4, 4))
        b = rng.standard_normal((4, 4))
        if i % 4:
            a[3] = b[3] = [0, 0, 0, 1]
        pts = rng.integers(-300, 300, (8, 3)).astype(np.float64)
        t[f"a{i}/a"] = a
        t[f"a{i}/b"] = b
        t[f"a{i}/inv"] = np.linalg.inv(a)
        t[f"a{i}/dot"] = a.dot(b)
        t[f"a{i}/points"] = pts
        t[f"a{i}/apply_many"] = nib.affines.apply_affine(a, pts)
        t[f"a{i}/apply_one"] = nib.affines.apply_affine(a, pts[0])
        t[f"a{i}/voxel_sizes"] = nib.affines.voxel_sizes(a)
        cases[f"a{i}"] = {"kind": "affine"}
    save("affine", t, cases)


# ---------------------------------------------------------------------------------------- stats
def make_stats():
    t, cases = {}, {}
    for n in [1, 2, 7, 8, 9, 15, 16, 17, 31, 127, 128, 129, 136, 255, 256, 1000, 4103]:
        x = (rng.standard_normal(n) * 100 + 30).astype(np.float32)
        t[f"sum{n}/f32"] = x
        t[f"sum{n}/f32_expected"] = np.array([np.add.reduce(x)], dtype=np.float32)
        x64 = rng.standard_normal(n) * 1e3
        t[f"sum{n}/f64"] = x64
        t[f"sum{n}/f64_expected"] = np.array([np.add.reduce(x64)])
        cases[f"sum{n}"] = {"kind": "sum", "n": n}
    for n in [8193, 70001, 300007, 1000003]:
        t[f"formula{n}/f32_expected"] = np.array([np.add.reduce(formula_values(n))], dtype=np.float32)
        t[f"formula{n}/mean_expected"] = np.array([formula_values(n).mean()], dtype=np.float32)
        cases[f"formula{n}"] = {"kind": "formula", "n": n}
    vol = (rng.standard_normal((23, 31, 19)) * 50 + 400).astype(np.float32)
    t["mean3d/x"] = vol
    t["mean3d/expected"] = np.array([vol.mean()], dtype=np.float32)
    cases["mean3d"] = {"kind": "mean"}

    qs_default = np.array([0.005, 0.995])
    quantile_cases = {
        "rand1000": ((rng.standard_normal(1000) * 300 + 2000).astype(np.float32), [0.005, 0.995, 0.5, 0.25, 0.0, 1.0]),
        "ties": (rng.integers(0, 20, 503).astype(np.float32), [0.005, 0.995, 0.5, 0.1]),
        "n1": (np.array([7.25], dtype=np.float32), [0.005, 0.995]),
        "n2": (np.array([7.0, 3.0], dtype=np.float32), [0.005, 0.995, 0.5]),
        "n3": (np.array([5.0, -1.5, 2.0], dtype=np.float32), [0.25, 0.75]),
        "half_gamma": ((rng.standard_normal(201) * 10).astype(np.float32), [0.0025, 0.9975, 0.5]),
        "uint16_like": (rng.integers(0, 65536, 4099).astype(np.float32), list(qs_default)),
    }
    for name, (values, qs) in quantile_cases.items():
        qs = np.asarray(qs, dtype=np.float64)
        t[f"q_{name}/values"] = values
        t[f"q_{name}/qs"] = qs
        t[f"q_{name}/expected"] = np.asarray(np.quantile(values, tuple(qs)), dtype=np.float64)
        cases[f"q_{name}"] = {"kind": "quantile"}
    save("stats", t, cases)


# --------------------------------------------------------------------------------------- filter
def make_filter():
    t, cases = {}, {}
    kernels = {
        "f32_2": np.float32(2.0),
        "f32_8over3": np.float32(4.0) / np.float32(1.5),
        "f64_0p6164": np.float64(0.6164),
        "f64_1p3": np.float64(1.3),
        "f64_tiny": np.float64(1e-3),
    }
    from scipy.ndimage._filters import _gaussian_kernel1d

    for name, sigma in kernels.items():
        radius = int(4.0 * float(sigma) + 0.5)
        t[f"k_{name}/sigma"] = np.array([sigma])
        t[f"k_{name}/expected"] = _gaussian_kernel1d(sigma, 0, radius)[::-1].copy()
        cases[f"k_{name}"] = {"kind": "kernel", "dtype": str(np.asarray(sigma).dtype), "radius": radius}

    filters = {
        "threshold": ((21, 17, 23), np.full(3, 2.0, dtype=np.float32), "reflect"),
        "short_lines": ((5, 3, 7), np.full(3, 2.0, dtype=np.float32), "reflect"),
        "nearest_mixed": ((9, 8, 7), np.array([0.6, 0.0, 1.3]), "nearest"),
        "tiny_sigma": ((4, 20, 3), np.array([1e-16, 0.37, 2.5]), "nearest"),
        "single_voxel_axis": ((1, 6, 20), np.array([2.0, 1.0, 0.5], dtype=np.float32), "reflect"),
        "smooth_like": ((12, 10, 9), np.array([0.3127, 0.0, 0.4416]), "nearest"),
    }
    for name, (shape, sigmas, mode) in filters.items():
        x = (rng.standard_normal(shape) * 100 + 500).astype(np.float32)
        t[f"g_{name}/x"] = x
        t[f"g_{name}/sigmas"] = sigmas
        t[f"g_{name}/expected"] = ndi.gaussian_filter(x, sigma=sigmas, mode=mode)
        cases[f"g_{name}"] = {"kind": "gaussian", "mode": mode, "sigma_dtype": str(sigmas.dtype)}
    save("filter", t, cases)


# ------------------------------------------------------------------------------------- resample
def make_resample():
    t, cases = {}, {}
    oblique = rotation("z", 17.0) @ rotation("x", -9.0) @ np.diag([0.93, 1.07, 1.21])

    def case(name, x, matrix, offset, out_shape, order):
        matrix = np.asarray(matrix, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        t[f"at_{name}/x"] = x
        t[f"at_{name}/matrix"] = matrix
        t[f"at_{name}/offset"] = offset
        t[f"at_{name}/expected"] = ndi.affine_transform(
            x, matrix, offset, output_shape=out_shape, order=order, mode="constant", cval=0.0
        )
        cases[f"at_{name}"] = {"kind": "affine_transform", "order": order, "out_shape": list(out_shape)}

    f32 = (rng.standard_normal((11, 13, 9)) * 100).astype(np.float32)
    case("shift_int", f32, np.eye(3), [-2, 3, 1], (12, 10, 11), 1)
    case("downsample", (rng.standard_normal((17, 15, 13)) * 100).astype(np.float32), np.diag([2.0, 2, 2]),
         [0.0, 0.5, -0.5], (9, 8, 7), 1)
    f64 = rng.standard_normal((9, 11, 7))
    case("oblique_linear_f64", f64, oblique, [0.37, -0.41, 0.53], (10, 12, 8), 1)
    case("oblique_nearest_u8", rng.integers(0, 256, (9, 11, 7)).astype(np.uint8), oblique, [0.37, -0.41, 0.53],
         (10, 12, 8), 0)
    case("oblique_round_i16", rng.integers(-500, 500, (9, 11, 7)).astype(np.int16), oblique, [0.2, 0.1, -0.3],
         (10, 12, 8), 1)
    case("round_u8", rng.integers(0, 256, (9, 11, 7)).astype(np.uint8), oblique, [0.2, 0.1, -0.3], (10, 12, 8), 1)
    case("round_u16", rng.integers(0, 65536, (6, 7, 5)).astype(np.uint16), np.diag([0.7, 0.9, 1.1]),
         [0.1, 0.0, 0.0], (9, 8, 6), 1)
    edges = (rng.standard_normal((5, 4, 3)) * 10).astype(np.float32)
    case("edge_exact", edges, np.diag([4 / 6, 3 / 5, 2 / 4]), [0, 0, 0], (7, 6, 5), 1)
    case("edge_beyond", edges, np.eye(3), [1e-13, -1e-13, 0.0], (5, 4, 3), 1)
    case("edge_nearest", edges, np.eye(3), [1e-13, -1e-13, 0.5], (5, 4, 3), 0)
    case("half_ties", (rng.standard_normal((6, 6, 6)) * 10).astype(np.float32), np.diag([0.5, 0.5, 0.5]),
         [0.0, 0.25, 0.75], (11, 11, 11), 0)
    case("length_one_axis", (rng.standard_normal((1, 5, 4)) * 10).astype(np.float32), np.eye(3), [0, 0.5, 0.25],
         (1, 5, 4), 1)
    case("int32_order0", rng.integers(-(2**31), 2**31 - 1, (4, 5, 6)).astype(np.int32), np.eye(3)[[1, 0, 2]],
         [0.4, -0.4, 0.0], (5, 4, 6), 0)

    # nibabel wrappers: vox2out_vox and resample_from_to
    geoms = {
        "ras_1mm": (affine_from(np.eye(3), [-91, -126, -72]), (182, 218, 182)),
        "lps_oblique": (affine_from(np.diag([-0.9375, -0.9375, 1.2]) @ rotation("y", 8.0), [120.5, 110.25, -60.0]),
                        (256, 256, 150)),
        "psr_aniso": (affine_from([[0, 0, 1.1], [-0.8, 0, 0], [0, 1.3, 0]], [-80, 100, -40]), (160, 200, 140)),
    }
    for name, (aff, shape) in geoms.items():
        out_shape, out_aff = vox2out_vox((shape, aff), (2.0, 2.0, 2.0))
        t[f"v2o_{name}/affine"] = aff
        t[f"v2o_{name}/shape"] = np.array(shape, dtype=np.int64)
        t[f"v2o_{name}/out_shape"] = np.array(out_shape, dtype=np.int64)
        t[f"v2o_{name}/out_affine"] = out_aff
        cases[f"v2o_{name}"] = {"kind": "vox2out_vox"}

    def rft(name, x, from_aff, to_shape, to_aff, order):
        img = nib.Nifti1Image(x, from_aff)
        out = resample_from_to(img, (to_shape, to_aff), order=order)
        t[f"rft_{name}/x"] = x
        t[f"rft_{name}/from_affine"] = from_aff
        t[f"rft_{name}/to_affine"] = to_aff
        t[f"rft_{name}/expected"] = np.asarray(out.dataobj)
        t[f"rft_{name}/zooms"] = np.array(out.header.get_zooms()[:3], dtype=np.float32)
        cases[f"rft_{name}"] = {"kind": "resample_from_to", "order": order, "to_shape": list(to_shape)}

    small = (rng.standard_normal((10, 12, 9)) * 100 + 300).astype(np.float32)
    rft("oblique_to_1mm", small, affine_from(oblique, [3.0, -4.0, 5.5]), (12, 13, 11),
        affine_from(np.eye(3), [-1.0, -6.0, 2.0]), 1)
    rft("mask_nearest", (small > 300).astype(np.uint8), affine_from(np.diag([2.0, 2, 2]), [-10.0, -12.0, -8.0]),
        (19, 23, 17), affine_from(np.eye(3), [-10.0, -12.0, -8.0]), 0)
    rft("same_grid_keeps_zooms", small, affine_from(np.eye(3) * 1.0000001, [0.0, 0, 0]), (10, 12, 9),
        affine_from(np.eye(3), [0.0, 0, 0]), 1)
    save("resample", t, cases)


# ---------------------------------------------------------------------------------- orientation
def make_orientation():
    t, cases = {}, {}
    affines = {
        "ras": affine_from(np.eye(3), [0, 0, 0]),
        "las": affine_from(np.diag([-1.0, 1, 1]), [90, -126, -72]),
        "lps": affine_from(np.diag([-0.9, -0.9, 1.2]), [100, 110, -60]),
        "pil": affine_from([[0, 0, -1.2], [-0.8, 0, 0], [0, -1.1, 0]], [10, 20, 30]),
        "asl": affine_from([[0, 0, -1.0], [1.0, 0, 0], [0, 1.0, 0]], [0, 0, 0]),
        "oblique10": affine_from(rotation("x", 10) @ rotation("z", -12) @ np.diag([0.9, 1.0, 1.3]), [5, 6, 7]),
        "oblique30_flipped": affine_from(rotation("y", 30) @ np.diag([-1.0, 1.0, -2.0]), [1, 2, 3]),
        "near45": affine_from(rotation("z", 44.9) @ np.diag([1.0, 1.0, 1.0]), [0, 0, 0]),
        "exact45": affine_from([[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]], [0, 0, 0]),
        "tie_columns": affine_from([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 2.0]], [0, 0, 0]),
        "sheared": affine_from([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 2.0]], [0, 0, 0]),
    }
    shape = (2, 3, 4)
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    for name, aff in affines.items():
        ornt = nib.orientations.io_orientation(aff)
        img = nib.Nifti1Image(data, aff)
        canon = nib.as_closest_canonical(img)
        t[f"o_{name}/affine"] = aff
        t[f"o_{name}/ornt"] = ornt.astype(np.int64)
        t[f"o_{name}/canonical_affine"] = canon.affine
        t[f"o_{name}/canonical_data"] = np.asarray(canon.dataobj)
        t[f"o_{name}/canonical_zooms"] = np.array(canon.header.get_zooms()[:3], dtype=np.float32)
        cases[f"o_{name}"] = {"kind": "orientation", "axcodes": "".join(nib.aff2axcodes(aff))}
    t["data"] = data
    save("orientation", t, cases)


# ----------------------------------------------------------------------------------- morphology
def make_morphology():
    import larmor_ref  # noqa: F401  (puts nanobrain on sys.path)
    from nanobrain.utils.preprocessing import largest_component

    t, cases = {}, {}

    def case(name, mask):
        labels, count = ndi.label(mask)
        t[f"m_{name}/mask"] = mask
        t[f"m_{name}/labels"] = labels.astype(np.uint32)
        t[f"m_{name}/count"] = np.array([count], dtype=np.int64)
        t[f"m_{name}/largest"] = np.asarray(largest_component(mask), dtype=bool)
        t[f"m_{name}/filled"] = ndi.binary_fill_holes(mask)
        cases[f"m_{name}"] = {"kind": "morphology"}

    case("sparse", rng.random((15, 17, 13)) < 0.3)
    case("dense", rng.random((14, 11, 16)) < 0.62)
    tie = np.zeros((6, 9, 7), dtype=bool)
    tie[4:6, 6:8, 4:6] = True  # 8 voxels, labelled first? no: (4,6,4) comes after (1,1,1)
    tie[1:3, 1:3, 1:3] = True  # 8 voxels, first in C order
    tie[0, 8, 0] = True
    case("size_tie", tie)
    x, y, z = np.mgrid[:13, :13, :13]
    r = np.sqrt((x - 6) ** 2 + (y - 6) ** 2 + (z - 6) ** 2)
    case("hollow_sphere", (r < 6) & (r > 3.5))
    shell = (r < 6) & (r > 3.5)
    shell[6, 6, 9:] = False  # a tunnel to the outside: the hole is no longer enclosed
    case("tunnelled_sphere", shell)
    border = np.zeros((5, 5, 5), dtype=bool)
    border[0, :, :] = True
    border[:, 0, :] = True
    case("touching_border", border)
    save("morphology", t, cases)


# ---------------------------------------------------------------------------------------- nifti
def make_nifti():
    t, cases = {}, {}
    oblique = rotation("z", 17.0) @ rotation("x", -9.0)

    def build(name, data, *, endian="<", sform=None, sform_code=0, qform=None, qform_code=0, pixdim=None,
              quatern=None, qoffset=(0.0, 0.0, 0.0), qfac=1.0, slope=1.0, inter=0.0, notes=""):
        hdr = nib.Nifti1Header(endianness=endian)
        hdr.set_data_shape(data.shape)
        hdr.set_data_dtype(data.dtype)
        if qform is not None:
            hdr.set_qform(qform, code=qform_code)
        if sform is not None:
            hdr.set_sform(sform, code=sform_code)
        hdr["sform_code"] = sform_code
        hdr["qform_code"] = qform_code
        if quatern is not None:
            hdr["quatern_b"], hdr["quatern_c"], hdr["quatern_d"] = quatern
            hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"] = qoffset
        if pixdim is not None:
            hdr["pixdim"][1 : 1 + len(pixdim)] = pixdim
        hdr["pixdim"][0] = qfac
        hdr["scl_slope"] = slope
        hdr["scl_inter"] = inter
        hdr["vox_offset"] = 352
        raw = hdr.binaryblock + b"\0" * 4 + data.astype(data.dtype.newbyteorder(endian)).tobytes(order="F")
        img = nib.Nifti1Image.from_bytes(raw)
        t[f"n_{name}/bytes"] = np.frombuffer(raw, dtype=np.uint8)
        t[f"n_{name}/gz"] = np.frombuffer(gzip.compress(raw, mtime=0), dtype=np.uint8)
        t[f"n_{name}/affine"] = img.affine
        t[f"n_{name}/zooms"] = np.array(img.header.get_zooms(), dtype=np.float32)
        t[f"n_{name}/fdata"] = img.get_fdata(dtype=np.float32)
        dataobj = np.asanyarray(img.dataobj)
        t[f"n_{name}/dataobj"] = dataobj
        cases[f"n_{name}"] = {
            "kind": "nifti",
            "shape": list(img.shape),
            "dtype": img.get_data_dtype().name,
            "dataobj_dtype": dataobj.dtype.name,
            "notes": notes,
        }

    shape = (4, 5, 3)
    i16 = rng.integers(-1000, 1000, shape).astype(np.int16)
    build("sform_wins", i16, sform=affine_from(np.diag([1.0, 1, 1]), [-91, -126, -72]), sform_code=4,
          qform=affine_from(oblique, [1, 2, 3]), qform_code=1, notes="both codes set, sform takes precedence")
    build("qform_qfac_neg", rng.standard_normal(shape).astype(np.float32), qform_code=2,
          quatern=(0.1, -0.2, 0.3), qoffset=(10.5, -20.25, 30.0), pixdim=(0.9, 1.1, 1.3), qfac=-1.0,
          notes="qform only, qfac -1")
    build("base_affine", rng.integers(0, 255, shape).astype(np.uint8), pixdim=(1.2, 0.9, 3.0),
          notes="no codes: centred base affine with x flip")
    build("scaled_i16", i16, sform=affine_from(np.eye(3) * 2, [0, 0, 0]), sform_code=2, slope=2.5, inter=-3.0,
          notes="int16 with slope and intercept")
    build("scaled_f32", rng.standard_normal(shape).astype(np.float32) * 100, sform=affine_from(np.eye(3), [0, 0, 0]),
          sform_code=1, slope=0.5, inter=10.0, notes="float32 with slope and intercept")
    build("big_endian_f64", rng.standard_normal(shape), endian=">", sform=affine_from(oblique * 1.5, [5, 6, 7]),
          sform_code=1, notes="big-endian header and data")
    build("slope_zero_u16", rng.integers(0, 65535, shape).astype(np.uint16), sform=affine_from(np.eye(3), [0, 0, 0]),
          sform_code=1, slope=0.0, inter=5.0, notes="slope 0 means unscaled")
    build("fixups", i16, qform_code=7, quatern=(0.0, 0.0, 0.0), pixdim=(0.0, -1.5, 2.0), qfac=0.0,
          notes="qfac 0 -> 1, pixdim 0 -> 1 and negative -> abs, invalid qform code -> 0")
    build("quat_w_zero", i16, qform_code=1, quatern=(1.0, 0.0, 0.0), pixdim=(1.0, 2.0, 3.0),
          notes="180 degree rotation, w recovered as 0")
    build("quat_w_threshold", i16, qform_code=1, quatern=(0.6, 0.8, 1e-4), pixdim=(1.0, 1.0, 1.0),
          notes="1 - |bcd|^2 slightly negative but within the threshold")
    for dt in [np.int8, np.int32, np.uint32, np.int64, np.uint64]:
        info = np.iinfo(dt)
        lo, hi = max(info.min, -(2**40)), min(info.max, 2**40)
        build(f"dtype_{np.dtype(dt).name}", rng.integers(lo, hi, shape, dtype=np.int64).astype(dt),
              sform=affine_from(np.eye(3), [0, 0, 0]), sform_code=1)
    build("four_d", rng.integers(0, 100, (3, 4, 5, 2)).astype(np.int16), sform=affine_from(np.eye(3), [0, 0, 0]),
          sform_code=1, pixdim=(1.0, 1.0, 1.0, 2.5), notes="4-D image")

    # NIfTI-2: 540-byte header, int64 dims, float64 pixdim, quaternion, offsets and sform
    def build2(name, data, *, endian="<", sform=None, sform_code=0, qform_code=0, pixdim=None, quatern=None,
               qoffset=(0.0, 0.0, 0.0), qfac=1.0, slope=1.0, inter=0.0, notes=""):
        hdr = nib.Nifti2Header(endianness=endian)
        hdr.set_data_shape(data.shape)
        hdr.set_data_dtype(data.dtype)
        if sform is not None:
            hdr.set_sform(sform, code=sform_code)
        hdr["sform_code"] = sform_code
        hdr["qform_code"] = qform_code
        if quatern is not None:
            hdr["quatern_b"], hdr["quatern_c"], hdr["quatern_d"] = quatern
            hdr["qoffset_x"], hdr["qoffset_y"], hdr["qoffset_z"] = qoffset
        if pixdim is not None:
            hdr["pixdim"][1 : 1 + len(pixdim)] = pixdim
        hdr["pixdim"][0] = qfac
        hdr["scl_slope"] = slope
        hdr["scl_inter"] = inter
        hdr["vox_offset"] = 544
        raw = hdr.binaryblock + b"\0" * 4 + data.astype(data.dtype.newbyteorder(endian)).tobytes(order="F")
        img = nib.Nifti2Image.from_bytes(raw)
        t[f"n2_{name}/bytes"] = np.frombuffer(raw, dtype=np.uint8)
        t[f"n2_{name}/gz"] = np.frombuffer(gzip.compress(raw, mtime=0), dtype=np.uint8)
        t[f"n2_{name}/affine"] = img.affine
        t[f"n2_{name}/zooms"] = np.array(img.header.get_zooms(), dtype=np.float64)
        t[f"n2_{name}/fdata"] = img.get_fdata(dtype=np.float32)
        dataobj = np.asanyarray(img.dataobj)
        t[f"n2_{name}/dataobj"] = dataobj
        cases[f"n2_{name}"] = {
            "kind": "nifti2",
            "shape": list(img.shape),
            "dtype": img.get_data_dtype().name,
            "dataobj_dtype": dataobj.dtype.name,
            "notes": notes,
        }

    # float64 values that float32 cannot hold, so a NIfTI-1 reading would round them
    fine = affine_from(oblique * np.array([0.9000000001, 1.1000000003, 1.3]), [-91.123456789012, 2.5, 3.0000001])
    build2("sform", i16, sform=fine, sform_code=1, pixdim=(0.9000000001, 1.1000000003, 1.3),
           notes="float64 sform and pixdim")
    build2("qform_qfac_neg", rng.standard_normal(shape).astype(np.float32), qform_code=2,
           quatern=(0.1000000001, -0.2, 0.3), qoffset=(10.5000000001, -20.25, 30.0), pixdim=(0.9, 1.1, 1.3),
           qfac=-1.0, notes="qform only in float64, qfac -1")
    build2("quat_w_threshold", i16, qform_code=1, quatern=(0.6, 0.8, 1e-9), pixdim=(1.0, 1.0, 1.0),
           notes="1 - |bcd|^2 within float64's threshold (float32's is wider)")
    build2("base_affine", rng.integers(0, 255, shape).astype(np.uint8), pixdim=(1.2000000001, 0.9, 3.0),
           notes="no codes: centred base affine from float64 zooms")
    build2("big_endian_scaled", rng.integers(-1000, 1000, shape).astype(np.int16), endian=">", sform=fine,
           sform_code=4, slope=2.5000000001, inter=-3.0, notes="big-endian, float64 slope")
    save("nifti", t, cases)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tools"))
    make_affine()
    make_stats()
    make_filter()
    make_resample()
    make_orientation()
    make_morphology()
    make_nifti()
