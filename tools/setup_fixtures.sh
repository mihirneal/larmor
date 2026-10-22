#!/usr/bin/env bash
# Build Larmor's fixtures from scratch. Idempotent: every step skips work that is already done, so a
# re-run only fills in what is missing (and re-checks pinned hashes).
#
#   tools/setup_fixtures.sh            # everything, then the sanity tests
#   tools/setup_fixtures.sh --no-test  # skip the pytest sanity checks at the end
#
# LARMOR_FIXTURES overrides the fixtures directory (default: <repo>/fixtures).
set -euo pipefail

NANOBRAIN_URL="https://github.com/MedARC-AI/nanobrain.git"
NANOBRAIN_COMMIT="a1bfdd1"

TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$TOOLS")"
export LARMOR_FIXTURES="${LARMOR_FIXTURES:-$REPO/fixtures}"
FIXTURES="$LARMOR_FIXTURES"
PY="$TOOLS/.venv/bin/python"
RUN_TESTS=1
[[ "${1:-}" == "--no-test" ]] && RUN_TESTS=0

step() { printf '\n== %s\n' "$*"; }

step "nanobrain @ $NANOBRAIN_COMMIT"
if [[ ! -d "$FIXTURES/nanobrain/.git" ]]; then
  mkdir -p "$FIXTURES"
  git clone --quiet "$NANOBRAIN_URL" "$FIXTURES/nanobrain"
  git -C "$FIXTURES/nanobrain" checkout --quiet --detach "$NANOBRAIN_COMMIT"
fi
head="$(git -C "$FIXTURES/nanobrain" rev-parse HEAD)"
if [[ "$head" != "$NANOBRAIN_COMMIT"* ]]; then
  echo "fixtures/nanobrain is at $head, not $NANOBRAIN_COMMIT; check it out by hand" >&2
  exit 1
fi
echo "ok: $head"

step "python env (uv sync in tools/)"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
(cd "$TOOLS" && uv sync --frozen --inexact --quiet)
"$PY" -c "import torch, nibabel, scipy; print('ok: torch', torch.__version__)"
# dcm2niix --version exits with status 3
echo "ok: dcm2niix $("$TOOLS/.venv/bin/dcm2niix" --version 2>&1 | tail -1 || true)"

step "reference models"
models_present=1
for patch in 8 4; do
  for ext in toml safetensors; do
    [[ -f "$FIXTURES/models/nanobrain-vits-p$patch-ref.$ext" ]] || models_present=0
  done
done
if [[ $models_present -eq 1 ]]; then
  echo "ok: fixtures/models already has the reference models (never regenerated in place)"
else
  (cd "$TOOLS" && "$PY" make_models.py reference)
fi

step "fixture scans, DICOM series (dcm2niix's NIfTI of each) and manifest"
(cd "$TOOLS" && "$PY" fetch_fixtures.py)

step "PyTorch references (prep for every scan, patch-8 encode for every scan, patch-4 for a few)"
(cd "$TOOLS" && "$PY" -u make_reference.py all)

if [[ $RUN_TESTS -eq 1 ]]; then
  step "sanity checks"
  (cd "$TOOLS" && "$PY" -m pytest test_reference.py -q)
fi
