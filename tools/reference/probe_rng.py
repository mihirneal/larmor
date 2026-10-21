"""Reference draws for larmor-probe's numpy/torch RNG ports and sklearn folds.

    tools/.venv/bin/python tools/reference/probe_rng.py   # -> crates/larmor-probe/tests/data/rng_folds.json

Everything here comes from the libraries themselves: legacy RandomState(seed).shuffle (what sklearn's
KFold/StratifiedKFold call through check_random_state), default_rng(seed).integers (the bootstrap), torch.randperm
with a seeded Generator (subsample_negatives), and the sklearn splitters.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import larmor_ref  # noqa: E402,F401  (pins nanobrain on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

OUT = larmor_ref.REPO / "crates" / "larmor-probe" / "tests" / "data" / "rng_folds.json"


def uint32_words(n: int) -> list[int]:
    """An integer's 32-bit words, lowest first, as SeedSequence splits its entropy."""
    words = [n & 0xFFFFFFFF]
    while n >> 32:
        n >>= 32
        words.append(n & 0xFFFFFFFF)
    return words


def shuffle_cases():
    cases = []
    for seed in (0, 1, 7, 42, 12345, 2**31 - 1, 2**32 - 1):
        for n in (0, 1, 2, 3, 5, 16, 17, 100, 623, 624, 625, 1000, 5000):
            values = np.arange(n)
            np.random.RandomState(seed).shuffle(values)
            cases.append({"seed": seed, "n": n, "permutation": values.tolist()})
    return cases


def mt19937_raw():
    # the first 2000 32-bit outputs, crossing the 624-word regeneration twice
    return {
        str(seed): np.random.RandomState(seed)._bit_generator.random_raw(2000).tolist()
        for seed in (0, 5489, 2**32 - 1)
    }


def integers_cases():
    cases = []
    for seed in (0, 1, 2, 99, 2**32 + 5, 2**63 + 11, 2**128 + 3):
        for n in (1, 2, 3, 7, 20, 96, 100, 120, 1000, 2**31 + 1, 2**32 - 1, 2**32, 2**32 + 1, 2**40 + 7):
            rng = np.random.default_rng(seed)
            # several calls so the buffered 32-bit half carries across calls
            draws = [rng.integers(0, n, size).tolist() for size in (n if n < 2000 else 3, 1, 5, 2)]
            cases.append(
                {
                    "seed": str(seed),
                    "seed_words": uint32_words(seed),
                    "high": str(n),
                    "draws": [[str(v) for v in d] for d in draws],
                }
            )
    return cases


def seed_sequence_cases():
    return [
        {
            "seed": str(seed),
            "seed_words": uint32_words(seed),
            "state": [str(v) for v in np.random.SeedSequence(seed).generate_state(4, np.uint64)],
        }
        for seed in (0, 1, 2**32 - 1, 2**32, 2**64 + 1, 2**160 + 2**70 + 3)
    ]


def pcg64_raw():
    return {
        str(seed): [str(v) for v in np.random.default_rng(seed).bit_generator.random_raw(50)]
        for seed in (0, 1, 2**70 + 1)
    }


def randperm_cases():
    cases = []
    for seed in (0, 1, 3):
        for n in (0, 1, 2, 10, 623, 624, 625, 5000, 54321):
            generator = torch.Generator().manual_seed(seed)
            cases.append({"seed": seed, "n": n, "permutation": torch.randperm(n, generator=generator).tolist()})
    return cases


def kfold_cases():
    cases = []
    for n in (5, 6, 7, 16, 20, 23, 96, 100, 120, 301):
        for n_splits in (2, 3, 5, 10):
            if n_splits > n:
                continue
            for shuffle, seed in ((False, None), (True, 0), (True, 1), (True, 2**32 - 1)):
                folds = KFold(n_splits, shuffle=shuffle, random_state=seed)
                tests = [test.tolist() for _, test in folds.split(np.zeros(n))]
                cases.append({"n": n, "n_splits": n_splits, "shuffle": shuffle, "seed": seed, "tests": tests})
    return cases


def stratified_cases():
    rng = np.random.default_rng(1234)
    cases = []
    label_sets = [
        [0, 1] * 10,
        [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        (rng.random(100) < 0.3).astype(int).tolist(),
        (rng.random(80) < 0.2).astype(int).tolist(),
        (rng.random(97) < 0.5).astype(int).tolist(),
        [2, 0, 1] * 9 + [2, 2, 2],
        rng.integers(0, 4, 61).tolist(),
        [1] * 6 + [0] * 25,
    ]
    for labels in label_sets:
        for n_splits in (2, 3, 5):
            for shuffle, seed in ((False, None), (True, 0), (True, 3)):
                folds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=seed)
                y = np.asarray(labels)
                tests = [test.tolist() for _, test in folds.split(np.zeros(len(y)), y)]
                cases.append(
                    {"labels": labels, "n_splits": n_splits, "shuffle": shuffle, "seed": seed, "tests": tests}
                )
    return cases


def main():
    data = {
        "versions": larmor_ref.versions(),
        "shuffle": shuffle_cases(),
        "mt19937_raw": mt19937_raw(),
        "seed_sequence": seed_sequence_cases(),
        "pcg64_raw": pcg64_raw(),
        "integers": integers_cases(),
        "randperm": randperm_cases(),
        "kfold": kfold_cases(),
        "stratified_kfold": stratified_cases(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
