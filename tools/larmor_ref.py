"""Shared helpers for Larmor's reference tools.

Imports nanobrain from a pinned checkout (NANOBRAIN_SRC, default fixtures/nanobrain/src) so the reference
outputs come from the exact code the cluster runs. SynthSeg is stubbed: the eval prep module imports it,
but the preprocessing Larmor mirrors never calls it.
"""

import hashlib
import json
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(os.environ.get("LARMOR_FIXTURES", REPO / "fixtures"))
NANOBRAIN_SRC = Path(os.environ.get("NANOBRAIN_SRC", FIXTURES / "nanobrain" / "src"))
NANOBRAIN_COMMIT = "a1bfdd1"

if str(NANOBRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(NANOBRAIN_SRC))
sys.modules.setdefault("SynthSeg_pytorch", types.SimpleNamespace(SynthSegPredictor=None))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def toml_value(value) -> str:
    """Minimal TOML encoder for the flat values a manifest needs."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"unsupported TOML value {value!r}")


def versions() -> dict[str, str]:
    import nibabel
    import numpy
    import scipy
    import sklearn
    import torch

    return {
        "nanobrain": NANOBRAIN_COMMIT,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "nibabel": nibabel.__version__,
        "sklearn": sklearn.__version__,
    }
