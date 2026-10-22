"""Reference outputs of nanobrain's own probe functions on synthetic features, for larmor-probe.

    tools/.venv/bin/python tools/reference/probe_reference.py regression       # -> crates/larmor-probe/tests/data/
    tools/.venv/bin/python tools/reference/probe_reference.py classification   # -> crates/larmor-probe/tests/data/
    tools/.venv/bin/python tools/reference/probe_reference.py segmentation --ratio 10    # -> fixtures/reference/probes/
    tools/.venv/bin/python tools/reference/probe_reference.py segmentation --ratio none
    tools/.venv/bin/python tools/reference/probe_reference.py fit              # -> crates/larmor-probe/tests/data/
    tools/.venv/bin/python tools/reference/probe_reference.py minority         # -> crates/larmor-probe/tests/data/

A fake Encoder returns features we choose: the sample index is written into voxel (0, 0, 0) of each 192x240x192
image and the encoder looks its features up. probe_regression, probe_binary_classification and
probe_binary_segmentation then run unmodified (num_workers=0, device="cpu", amp=False). Their internals are
recorded by wrapping the module-level helpers they call (make_pipeline, fit_predict_segmentation, fit_logistic,
subsample_negatives), which changes nothing they compute.

Besides the eval numbers, each run stores what a deployable probe needs (the same pipeline refit on all samples)
and, for the iterative solvers, a tightly converged solution of the same objective, so the Rust solvers can be
checked against the optimum as well as against sklearn's and torch's stopping points.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import larmor_ref  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV, RidgeCV  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

import nanobrain.eval.probe as probe  # noqa: E402

TEST_DATA = larmor_ref.REPO / "crates" / "larmor-probe" / "tests" / "data"
FIXTURE_DIR = larmor_ref.FIXTURES / "reference" / "probes"
GRID = probe.GRID_SHAPE
PATCH_GRID = probe.PATCH_GRID_SHAPE
N_PATCHES = int(np.prod(PATCH_GRID))
RUN_ARGS = {"num_workers": 0, "device": "cpu", "amp": False}


class TableEncoder(nn.Module):
    """An Encoder whose embeddings are looked up by the sample index stored at voxel (0, 0, 0)."""

    def __init__(self, global_table: np.ndarray | None = None, dense_tables: list | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))
        self.global_table = None if global_table is None else torch.from_numpy(global_table)
        self.dense_tables = dense_tables

    @staticmethod
    def sample_ids(images):
        return images[:, 0, 0, 0].round().long().tolist()

    def global_embed(self, images, mask):
        return self.global_table[self.sample_ids(images)]

    def dense_embed(self, images, mask):
        embed_dim = self.dense_tables[0][1].shape[1]
        out = torch.zeros(len(images), N_PATCHES, embed_dim)
        for row, sample in enumerate(self.sample_ids(images)):
            patch_ids, features = self.dense_tables[sample]
            out[row, torch.from_numpy(patch_ids)] = torch.from_numpy(features)
        return out.reshape(len(images), *PATCH_GRID, embed_dim)


def index_image(index: int) -> np.ndarray:
    image = np.zeros(GRID, dtype=np.float32)
    image[0, 0, 0] = index
    return image


class GlobalDataset(Dataset):
    def __init__(self, targets: np.ndarray, key: str):
        self.targets = targets
        self.key = key

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"image": index_image(index), "mask": np.zeros(1, dtype=np.uint8), self.key: self.targets[index]}


def latent_features(rng, n: int, dim: int, n_factors: int, shift: np.ndarray | None = None):
    """Embedding-like features: correlated through shared factors, uneven scales, nonzero means, one dead dim."""
    factors = rng.normal(size=(n, n_factors))
    if shift is not None:
        factors = factors + shift
    loadings = rng.normal(size=(n_factors, dim)) / np.sqrt(n_factors)
    scales = np.exp(rng.normal(0.0, 0.5, dim))
    offsets = rng.normal(0.0, 0.5, dim)
    noise = rng.normal(size=(n, dim))
    features = offsets + scales * (1.2 * factors @ loadings + 0.8 * noise)
    features[:, 17] = 0.25  # a constant feature exercises StandardScaler's zero-variance rule
    return features.astype(np.float32), factors


def record_pipelines():
    """Wrap probe.make_pipeline so the fitted pipelines can be read after the probe returns."""
    pipelines = []

    def wrapped(*steps):
        pipeline = make_pipeline(*steps)
        pipelines.append(pipeline)
        return pipeline

    probe.make_pipeline = wrapped
    return pipelines


def bootstrap_draws(n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([rng.integers(0, n_samples, n_samples) for _ in range(probe.N_BOOTSTRAP)])


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def write(out_dir: Path, name: str, arrays: dict, info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    save_file(arrays, str(out_dir / f"{name}.safetensors"), metadata={"versions": json.dumps(larmor_ref.versions())})
    info = {"versions": larmor_ref.versions(), **info}
    (out_dir / f"{name}.json").write_text(json.dumps(jsonable(info), indent=1))
    size = sum((out_dir / f"{name}{ext}").stat().st_size for ext in (".safetensors", ".json"))
    print(f"wrote {out_dir / name}.{{safetensors,json}} ({size / 1e6:.2f} MB)")


def run_regression(out_dir: Path, dim: int = 384, name: str = "probe_regression"):
    """probe_regression on 120 samples. With dim < 96 the inner RidgeCV runs in "svd" mode, else "eigen"."""
    rng = np.random.default_rng(20261021)
    n = 120
    features, factors = latent_features(rng, n, dim, n_factors=12)
    age = 55 + 9 * factors[:, 0] - 5 * factors[:, 1] + 3 * factors[:, 2] + rng.normal(0, 4.5, n)
    targets = age.astype(np.float64)

    pipelines = record_pipelines()
    result, state = probe.probe_regression(
        TableEncoder(features), GlobalDataset(targets, "age"), image_key="image", target_key="age", **RUN_ARGS
    )
    assert np.array_equal(state["features"], features)

    folds = list(KFold(5, shuffle=True, random_state=probe.CV_SEED).split(features))
    fold_of = np.zeros(n, dtype=np.int64)
    for fold, (_, test) in enumerate(folds):
        fold_of[test] = fold

    arrays = {"features": features, "targets": targets, "fold": fold_of}
    per_fold = []
    for fold, ((train, _), pipeline) in enumerate(zip(folds, pipelines)):
        scaler, ridge = pipeline
        # the LOO errors the alpha was chosen on: same fit with the per-sample errors kept
        loo = RidgeCV(alphas=probe.RIDGE_ALPHAS, store_cv_results=True).fit(
            scaler.transform(features[train]), targets[train]
        )
        assert loo.alpha_ == ridge.alpha_
        arrays[f"fold{fold}.scaler_mean"] = scaler.mean_
        arrays[f"fold{fold}.scaler_scale"] = scaler.scale_
        arrays[f"fold{fold}.coef"] = ridge.coef_
        arrays[f"fold{fold}.loo_mse"] = loo.cv_results_.mean(axis=0)
        per_fold.append({"alpha": ridge.alpha_, "intercept": ridge.intercept_, "best_score": ridge.best_score_})

    # deployable probe: the same pipeline on every sample
    final = make_pipeline(StandardScaler(), RidgeCV(alphas=probe.RIDGE_ALPHAS)).fit(features, targets)
    arrays["final.scaler_mean"] = final[0].mean_
    arrays["final.scaler_scale"] = final[0].scale_
    arrays["final.coef"] = final[1].coef_
    arrays["final.predictions"] = final.predict(features)

    predictions = np.asarray(result["predictions"])
    draws = bootstrap_draws(n)
    arrays["bootstrap_ids"] = draws.astype(np.int32)
    arrays["bootstrap_mae"] = np.array([mean_absolute_error(targets[i], predictions[i]) for i in draws])
    arrays["bootstrap_r"] = np.array([np.corrcoef(targets[i], predictions[i])[0, 1] for i in draws])
    arrays["bootstrap_r2"] = np.array([r2_score(targets[i], predictions[i]) for i in draws])

    info = {
        "result": {k: v for k, v in result.items() if k not in ("targets",)},
        "folds": per_fold,
        "final": {"alpha": final[1].alpha_, "intercept": final[1].intercept_},
        "gcv_mode": "eigen" if len(folds[0][0]) <= dim else "svd",
    }
    write(out_dir, name, arrays, info)


def run_classification(out_dir: Path):
    rng = np.random.default_rng(20261022)
    n, dim = 100, 384
    labels = (rng.random(n) < 0.32).astype(np.int64)
    shift = np.zeros((n, 10))
    shift[:, 0] = 0.9 * labels
    shift[:, 3] = -0.5 * labels
    features, _ = latent_features(rng, n, dim, n_factors=10, shift=shift)

    pipelines = record_pipelines()
    result, state = probe.probe_binary_classification(
        TableEncoder(features), GlobalDataset(labels, "label"), image_key="image", label_key="label", **RUN_ARGS
    )
    assert np.array_equal(state["features"], features)

    folds = list(StratifiedKFold(5, shuffle=True, random_state=probe.CV_SEED).split(features, labels))
    fold_of = np.zeros(n, dtype=np.int64)
    for fold, (_, test) in enumerate(folds):
        fold_of[test] = fold

    arrays = {"features": features, "labels": labels, "fold": fold_of}
    exact_probabilities = np.zeros(n)
    per_fold = []
    for fold, ((train, test), pipeline) in enumerate(zip(folds, pipelines)):
        scaler, logistic = pipeline
        arrays[f"fold{fold}.scaler_mean"] = scaler.mean_
        arrays[f"fold{fold}.scaler_scale"] = scaler.scale_
        arrays[f"fold{fold}.coef"] = logistic.coef_[0]
        arrays[f"fold{fold}.scores"] = logistic.scores_[:, 0, :]  # (inner fold, C)
        arrays[f"fold{fold}.coefs_paths"] = logistic.coefs_paths_[:, 0, :, 0, :]  # (inner fold, C, dof)
        # the same objectives solved to the optimum (Newton with a tight tolerance)
        exact = exact_pipeline().fit(features[train], labels[train])
        arrays[f"fold{fold}.exact_scores"] = exact[1].scores_[:, 0, :]
        arrays[f"fold{fold}.exact_coefs_paths"] = exact[1].coefs_paths_[:, 0, :, 0, :]
        arrays[f"fold{fold}.exact_coef"] = exact[1].coef_[0]
        exact_probabilities[test] = exact.predict_proba(features[test])[:, 1]
        per_fold.append(
            {
                "C": logistic.C_,
                "intercept": float(logistic.intercept_[0]),
                "exact_C": exact[1].C_,
                "exact_intercept": float(exact[1].intercept_[0]),
                "n_iter": logistic.n_iter_[:, 0, :],
            }
        )
    arrays["exact_probabilities"] = exact_probabilities

    final = make_pipeline(StandardScaler(), logistic_cv()).fit(features, labels)
    exact_final = exact_pipeline().fit(features, labels)
    arrays["final.scaler_mean"] = final[0].mean_
    arrays["final.scaler_scale"] = final[0].scale_
    arrays["final.coef"] = final[1].coef_[0]
    arrays["final.exact_coef"] = exact_final[1].coef_[0]
    arrays["final.probabilities"] = final.predict_proba(features)[:, 1]

    probabilities = np.asarray(result["probabilities"])
    predictions = probabilities > 0.5
    draws = bootstrap_draws(n)
    kept = np.array([len(np.unique(labels[i])) == 2 for i in draws])
    arrays["bootstrap_ids"] = draws.astype(np.int32)
    arrays["bootstrap_kept"] = kept.astype(np.uint8)
    arrays["bootstrap_auroc"] = np.array([roc_auc_score(labels[i], probabilities[i]) for i in draws[kept]])
    arrays["bootstrap_balanced_accuracy"] = np.array(
        [balanced_accuracy_score(labels[i], predictions[i]) for i in draws[kept]]
    )

    info = {
        "result": {k: v for k, v in result.items() if k not in ("labels",)},
        "folds": per_fold,
        "final": {
            "C": final[1].C_,
            "intercept": float(final[1].intercept_[0]),
            "exact_C": exact_final[1].C_,
            "exact_intercept": float(exact_final[1].intercept_[0]),
        },
        "exact": {
            "auroc": roc_auc_score(labels, exact_probabilities),
            "balanced_accuracy": balanced_accuracy_score(labels, exact_probabilities > 0.5),
        },
    }
    write(out_dir, "probe_classification", arrays, info)


def run_minority(out_dir: Path):
    """probe_binary_classification with 30 negatives and 1, 6 or 7 positives.

    With fewer than 5 positives in an outer training fold (6 or fewer overall under the outer
    StratifiedKFold(5)), LogisticRegressionCV's unshuffled inner StratifiedKFold(5) leaves a validation fold
    without positives and sklearn raises; with 7 it runs. Records the error or the numbers.
    """
    arrays, info = {}, {"cases": {}}
    for n_pos in (1, 6, 7):
        rng = np.random.default_rng(0)
        n_neg, dim = 30, 20
        labels = np.r_[np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)]
        shift = np.zeros((len(labels), 4))
        shift[:, 0] = 1.2 * labels
        features, _ = latent_features(rng, len(labels), dim, n_factors=4, shift=shift)
        name = f"pos{n_pos}"
        arrays[f"{name}.features"] = features
        arrays[f"{name}.labels"] = labels
        try:
            result, _ = probe.probe_binary_classification(
                TableEncoder(features), GlobalDataset(labels, "label"), image_key="image", label_key="label",
                **RUN_ARGS,
            )
        except ValueError as err:
            info["cases"][name] = {"raises": str(err)}
            print(name, "raises:", str(err)[:90])
            continue
        arrays[f"{name}.probabilities"] = np.asarray(result["probabilities"])
        info["cases"][name] = {k: v for k, v in result.items() if k not in ("labels", "probabilities")}
        print(name, "auroc", result["auroc"], "fold_C", result["fold_C"])
    write(out_dir, "probe_minority", arrays, info)


def logistic_cv(**overrides):
    """LogisticRegressionCV exactly as probe_binary_classification builds it."""
    kwargs = dict(
        Cs=probe.LOGISTIC_CS,
        l1_ratios=(0.0,),
        scoring="neg_log_loss",
        class_weight="balanced",
        max_iter=1000,
        use_legacy_attributes=False,
    )
    return LogisticRegressionCV(**{**kwargs, **overrides})


def exact_pipeline():
    return make_pipeline(StandardScaler(), logistic_cv(solver="newton-cholesky", tol=1e-11, max_iter=200))


# --- segmentation ---------------------------------------------------------------------------------------------


def ellipsoid(center, radii, grid=None):
    grid = grid if grid is not None else np.ogrid[: GRID[0], : GRID[1], : GRID[2]]
    return sum(((axis - c) / r) ** 2 for axis, c, r in zip(grid, center, radii)) < 1


def make_subjects(n_subjects: int, embed_dim: int, seed: int):
    """Head-like masks with lesion blobs, and f16-exact patch features carrying the blobs' layout plus noise."""
    rng = np.random.default_rng(seed)
    grid = np.ogrid[: GRID[0], : GRID[1], : GRID[2]]
    center = np.array(GRID) / 2
    octant_weights = rng.normal(size=(8, embed_dim))
    fraction_weights = rng.normal(size=embed_dim)
    position_weights = rng.normal(size=(3, embed_dim))
    mixing = rng.normal(size=(embed_dim, embed_dim)) / np.sqrt(embed_dim)
    offsets = rng.normal(0, 0.3, embed_dim)
    subjects = []
    for index in range(n_subjects):
        radii = np.array([36.0, 46.0, 34.0]) + rng.uniform(-3, 3, 3)
        mask = ellipsoid(center + rng.uniform(-4, 4, 3), radii, grid)
        label = np.zeros(GRID, dtype=bool)
        n_lesions = 0 if index == 13 else int(rng.integers(1, 7))
        inside = np.argwhere(mask)
        for _ in range(n_lesions):
            lesion_center = inside[rng.integers(len(inside))]
            label |= ellipsoid(lesion_center, rng.uniform(2.5, 9.0, 3), grid)
        label &= mask

        mask_patches = probe.patchify3d(torch.from_numpy(mask[None].astype(np.uint8)))[0]
        label_patches = probe.patchify3d(torch.from_numpy(label[None].astype(np.uint8)))[0]
        patch_ids = mask_patches.any(dim=1).nonzero()[:, 0].numpy()
        patch_labels = label_patches[patch_ids].numpy()

        octants = patch_labels.reshape(-1, 2, 4, 2, 4, 2, 4).mean(axis=(2, 4, 6)).reshape(-1, 8)
        fraction = patch_labels.mean(axis=1)
        position = np.stack(np.unravel_index(patch_ids, PATCH_GRID), axis=1) / np.array(PATCH_GRID) - 0.5
        noise = rng.normal(size=(len(patch_ids), embed_dim)) @ mixing
        features = (
            offsets
            + 2.0 * octants @ octant_weights
            + 1.5 * np.sqrt(fraction)[:, None] * fraction_weights
            + 0.8 * position @ position_weights
            + 0.9 * noise
        )
        # the cache stores tokens as f16; the probe sees them as f32
        features = features.astype(np.float16).astype(np.float32)
        subjects.append({"mask": mask, "label": label, "patch_ids": patch_ids, "features": features})
    return subjects


class SegmentationDataset(Dataset):
    def __init__(self, subjects):
        self.subjects = subjects

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        subject = self.subjects[index]
        return {
            "image": index_image(index),
            "mask": subject["mask"].astype(np.uint8),
            "label": subject["label"].astype(np.uint8),
        }


class SegmentationRecorder:
    """Wraps fit_predict_segmentation, fit_logistic and subsample_negatives to keep every fit's inputs and output.

    With `solver`, fit_logistic itself is replaced (same arguments, same return types)."""

    def __init__(self, solver=None):
        self.calls = []
        self._current = {}
        self._fit_predict = probe.fit_predict_segmentation
        self._fit_logistic = probe.fit_logistic
        self._subsample = probe.subsample_negatives
        self._solver = solver or probe.fit_logistic

    def install(self):
        probe.fit_predict_segmentation = self.fit_predict_segmentation
        probe.fit_logistic = self.fit_logistic
        probe.subsample_negatives = self.subsample_negatives

    def uninstall(self):
        probe.fit_predict_segmentation = self._fit_predict
        probe.fit_logistic = self._fit_logistic
        probe.subsample_negatives = self._subsample

    def subsample_negatives(self, labels, max_negative_ratio):
        keep = self._subsample(labels, max_negative_ratio)
        self._current["keep_ids"] = keep.numpy().astype(np.int32)
        return keep

    def fit_logistic(self, features, targets, alpha, max_iter=1000):
        start = time.perf_counter()
        coef, intercept = self._solver(features, targets, alpha, max_iter)
        self._current.update(
            coef=coef.numpy().copy(),
            intercept=intercept.numpy().copy(),
            seconds=time.perf_counter() - start,
            n_rows=len(features),
            standardized=features,
            targets=targets,
        )
        return coef, intercept

    def fit_predict_segmentation(self, patch_features, patch_labels, train_ids, test_ids, alpha, max_negative_ratio=None):
        self._current = {}
        probabilities = self._fit_predict(patch_features, patch_labels, train_ids, test_ids, alpha, max_negative_ratio)
        fit = self._current
        fit.update(
            train_ids=np.asarray(train_ids),
            test_ids=np.asarray(test_ids),
            alpha=alpha,
            dice=np.stack([probe.dice_by_threshold(p, patch_labels[i]).numpy() for i, p in zip(test_ids, probabilities)]),
        )
        self.calls.append(fit)
        return probabilities


def objective_terms(coef, intercept, features, targets, alpha):
    """fit_logistic's loss and the largest gradient component, in float64, at a given solution."""
    x = features.double()
    y = targets.double()
    w = torch.as_tensor(coef, dtype=torch.float64)
    b = torch.as_tensor(intercept, dtype=torch.float64)
    n, n_outputs = y.shape
    logits = x @ w + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + alpha * w.square().sum() / (n * n_outputs)
    residual = (torch.sigmoid(logits) - y) / (n * n_outputs)
    grad_w = x.T @ residual + 2 * alpha * w / (n * n_outputs)
    grad_b = residual.sum(dim=0)
    return float(loss), float(torch.cat([grad_w.flatten(), grad_b]).abs().max())


def exact_fit_logistic(features, targets, alpha, max_iter=1000, tol_grad=1e-14, history=10):
    """fit_logistic's objective solved to the optimum: float64 L-BFGS, preconditioned and with a line search on
    the slope only.

    torch's L-BFGS stops at a max gradient of 1e-7 on a loss averaged over n x 512 entries, which leaves the
    solution up to ~1e-2 from the optimum in probability (and it changes that much with the row order). This
    solver goes to the optimum so the probe's numbers there are well defined. The preconditioner is each output's
    Hessian with p(1 - p) replaced by its value at the mean label: c_j X'X / n + 2 alpha / n, intercept apart.
    """
    x = features.double()
    y = targets.double()
    n, n_outputs = y.shape
    d = x.shape[1]
    scale = n * n_outputs
    eigenvalues, vectors = torch.linalg.eigh(x.T @ x / n)
    curvature = (y.mean(dim=0) * (1 - y.mean(dim=0))).clamp_min(1e-4)
    denominator = (curvature[None, :] * eigenvalues[:, None] + 2 * alpha / n) / n_outputs

    def precondition(r):
        out = torch.empty_like(r)
        out[:-1] = vectors @ ((vectors.T @ r[:-1]) / denominator)
        out[-1] = r[-1] / (curvature / n_outputs)
        return out

    def gradient(theta):
        logits = x @ theta[:-1] + theta[-1]
        residual = (torch.sigmoid(logits) - y) / scale
        g = torch.empty_like(theta)
        g[:-1] = x.T @ residual + 2 * alpha * theta[:-1] / scale
        g[-1] = residual.sum(dim=0)
        return g

    def slope_search(theta, direction, slope0, c2=0.9, max_evals=30):
        low, high, slope_low, slope_high, step = 0.0, None, slope0, None, 1.0
        for _ in range(max_evals):
            g = gradient(theta + step * direction)
            slope = float((g * direction).sum())
            if abs(slope) <= c2 * -slope0:
                break
            if slope < 0:
                low, slope_low = step, slope
                if high is None:
                    step *= 4
                    continue
            else:
                high, slope_high = step, slope
            secant = low - slope_low * (high - low) / (slope_high - slope_low)
            width = high - low
            step = min(max(secant, low + 0.1 * width), high - 0.1 * width)
        return step, g

    theta = torch.zeros(d + 1, n_outputs, dtype=torch.float64)
    theta[-1] = torch.logit(y.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    g = gradient(theta)
    pairs, gamma = [], 1.0
    best, stalled = float("inf"), 0
    for _ in range(max_iter):
        q = -g.clone()
        alphas = []
        for s, yv, rho in reversed(pairs):
            a = rho * float((s * q).sum())
            alphas.append(a)
            q -= a * yv
        direction = gamma * precondition(q)
        for (s, yv, rho), a in zip(pairs, reversed(alphas)):
            direction += (a - rho * float((yv * direction).sum())) * s
        slope0 = float((g * direction).sum())
        if slope0 >= 0:
            pairs, gamma = [], 1.0
            direction = precondition(-g)
            slope0 = float((g * direction).sum())
        step, g_new = slope_search(theta, direction, slope0)
        s = step * direction
        yv = g_new - g
        theta, g = theta + s, g_new
        g_max = float(g.abs().max())
        if g_max <= tol_grad:
            break
        # stop once the gradient no longer shrinks: rounding has been reached
        stalled = stalled + 1 if g_max > 0.9 * best else 0
        best = min(best, g_max)
        if stalled >= 30:
            break
        sy = float((s * yv).sum())
        if sy > 0:
            pairs = pairs[-(history - 1) :] + [(s, yv, 1.0 / sy)]
            gamma = sy / float((yv * precondition(yv)).sum())
    return theta[:-1].float(), theta[-1].float()


def newton_optimum(features, targets, alpha, iterations=60):
    """The exact optimum of fit_logistic's objective by damped Newton per output column, float64 (slow; a check
    on exact_fit_logistic)."""
    x = torch.cat([features.double(), torch.ones(len(features), 1, dtype=torch.float64)], dim=1)
    y = targets.double()
    n, n_outputs = y.shape
    dim = x.shape[1]
    penalty = torch.full((dim,), 2 * alpha / n, dtype=torch.float64)
    penalty[-1] = 0

    def objective(theta):
        logits = x @ theta
        bce = torch.nn.functional.softplus(logits) - y * logits
        return bce.mean(dim=0) + 0.5 * (penalty[:, None] * theta.square()).sum(dim=0)

    theta = torch.zeros(dim, n_outputs, dtype=torch.float64)
    theta[-1] = torch.logit(y.mean(dim=0).clamp(1e-6, 1 - 1e-6))
    value = objective(theta)
    for _ in range(iterations):
        p = torch.sigmoid(x @ theta)
        grad = x.T @ (p - y) / n + penalty[:, None] * theta
        if float(grad.abs().max()) < 1e-13:
            break
        weights = p * (1 - p) / n
        step = torch.empty_like(theta)
        for start in range(0, n_outputs, 64):
            block = slice(start, start + 64)
            hessian = torch.einsum("ni,nk,nj->kij", x, weights[:, block], x) + torch.diag(penalty)
            hessian += 1e-14 * torch.eye(dim, dtype=torch.float64)
            step[:, block] = torch.linalg.solve(hessian, grad[:, block].T.unsqueeze(-1)).squeeze(-1).T
        # halve each column's step until its objective does not increase
        scale = torch.ones(n_outputs, dtype=torch.float64)
        pending = torch.ones(n_outputs, dtype=torch.bool)
        for _ in range(40):
            candidate = theta - scale * step
            candidate_value = objective(candidate)
            accept = pending & (candidate_value <= value)
            theta[:, accept] = candidate[:, accept]
            value[accept] = candidate_value[accept]
            pending &= ~accept
            if not pending.any():
                break
            scale[pending] /= 2
    return theta[:-1], theta[-1]


def segmentation_final(patch_features, patch_labels, max_negative_ratio, n_inner_folds=5):
    """The deployable segmentation probe: alpha and threshold by the probe's inner CV over all subjects, then a fit
    on all subjects. Uses nanobrain's own fit_predict_segmentation and dice_by_threshold."""
    n_subjects = len(patch_features)
    all_ids = np.arange(n_subjects)
    thresholds = probe.SEGMENTATION_THRESHOLDS
    inner_dice = torch.zeros(len(probe.SEGMENTATION_ALPHAS), n_subjects, len(thresholds))
    for inner_train, inner_val in KFold(n_inner_folds, shuffle=True, random_state=probe.CV_SEED).split(all_ids):
        for alpha_id, alpha in enumerate(probe.SEGMENTATION_ALPHAS):
            probabilities = probe.fit_predict_segmentation(
                patch_features, patch_labels, all_ids[inner_train], all_ids[inner_val], alpha, max_negative_ratio
            )
            for ii, p in zip(all_ids[inner_val], probabilities):
                inner_dice[alpha_id, ii] = probe.dice_by_threshold(p, patch_labels[ii])
    mean_inner_dice = inner_dice.mean(dim=1)
    alpha_id, threshold_id = np.unravel_index(mean_inner_dice.argmax().item(), mean_inner_dice.shape)
    alpha = probe.SEGMENTATION_ALPHAS[alpha_id]
    probe.fit_predict_segmentation(patch_features, patch_labels, all_ids, all_ids[:1], alpha, max_negative_ratio)
    return {
        "alpha": alpha,
        "threshold": thresholds[threshold_id].item(),
        "threshold_id": int(threshold_id),
        "mean_inner_dice": mean_inner_dice.numpy(),
    }


def run_segmentation_probe(subjects, ratio, solver=None):
    """probe_binary_segmentation (and the deployable refit) with every fit recorded."""
    encoder = TableEncoder(dense_tables=[(s["patch_ids"], s["features"]) for s in subjects])
    recorder = SegmentationRecorder(solver)
    recorder.install()
    start = time.perf_counter()
    try:
        result, state = probe.probe_binary_segmentation(
            encoder,
            SegmentationDataset(subjects),
            image_key="image",
            label_key="label",
            max_negative_ratio=ratio,
            **RUN_ARGS,
        )
        cv_seconds = time.perf_counter() - start
        n_cv_calls = len(recorder.calls)
        patch_features = [torch.from_numpy(s["features"]) for s in subjects]
        patch_labels = [torch.from_numpy(s["label"][None].astype(np.uint8)) for s in subjects]
        patch_labels = [probe.patchify3d(l)[0][torch.from_numpy(s["patch_ids"])] for l, s in zip(patch_labels, subjects)]
        final = segmentation_final(patch_features, patch_labels, ratio)
    finally:
        recorder.uninstall()
    for check, subject in zip(state["in_mask"], subjects):
        assert np.array_equal(check.nonzero()[:, 0].numpy(), subject["patch_ids"])
    return {
        "result": result,
        "state": state,
        "calls": recorder.calls,
        "n_cv_calls": n_cv_calls,
        "final": final,
        "cv_seconds": cv_seconds,
        "patch_labels": patch_labels,
    }


def describe_fits(run, n_subjects, tag, arrays, newton_check):
    """Per-fit records (arrays under `tag.fitNNN.*`) and the mean inner Dice grids rebuilt from the calls."""
    n_alphas = len(probe.SEGMENTATION_ALPHAS)
    calls_per_fold = 5 * n_alphas + 1
    calls, n_cv_calls = run["calls"], run["n_cv_calls"]
    # the call sequence is fixed: per outer fold, (inner split x alpha) fits then the outer fit
    assert n_cv_calls == 5 * calls_per_fold
    fits = []
    for index, call in enumerate(calls):
        if index < n_cv_calls:
            fold, position = divmod(index, calls_per_fold)
            stage = "outer" if position == calls_per_fold - 1 else "inner"
            inner, alpha_id = divmod(position, n_alphas) if stage == "inner" else (None, None)
        else:
            fold, position = None, index - n_cv_calls
            stage = "final" if position == 5 * n_alphas else "final_inner"
            inner, alpha_id = divmod(position, n_alphas) if stage == "final_inner" else (None, None)
        name = f"{tag}.fit{index:03d}"
        arrays[f"{name}.coef"] = call["coef"]
        arrays[f"{name}.intercept"] = call["intercept"]
        arrays[f"{name}.dice"] = call["dice"]
        if "keep_ids" in call and tag == "torch":
            arrays[f"fit{index:03d}.keep_ids"] = call["keep_ids"]
        loss, grad_max = objective_terms(call["coef"], call["intercept"], call["standardized"], call["targets"], call["alpha"])
        fit_info = {
            "index": index,
            "stage": stage,
            "fold": fold,
            "inner": inner,
            "alpha_id": alpha_id,
            "alpha": call["alpha"],
            "train_ids": call["train_ids"],
            "test_ids": call["test_ids"],
            "n_rows": call["n_rows"],
            "seconds": call["seconds"],
            "loss": loss,
            "grad_max": grad_max,
        }
        if newton_check and stage in ("outer", "final"):
            coef, intercept = newton_optimum(call["standardized"], call["targets"], call["alpha"])
            x = call["standardized"].double()
            p_newton = torch.sigmoid(x @ coef + intercept)
            p_fit = torch.sigmoid(x @ torch.from_numpy(call["coef"]).double() + torch.from_numpy(call["intercept"]).double())
            fit_info["newton_prob_diff"] = float((p_newton - p_fit).abs().max())
        fits.append(fit_info)
    grids = []
    for fold in range(5):
        train_ids = calls[fold * calls_per_fold + calls_per_fold - 1]["train_ids"]
        inner_dice = torch.zeros(n_alphas, n_subjects, len(probe.SEGMENTATION_THRESHOLDS))
        for position in range(calls_per_fold - 1):
            call = calls[fold * calls_per_fold + position]
            for ii, dice in zip(call["test_ids"], call["dice"]):
                inner_dice[position % n_alphas, ii] = torch.from_numpy(dice)
        grids.append(inner_dice[:, train_ids].mean(dim=1).numpy())
    arrays[f"{tag}.mean_inner_dice"] = np.stack(grids)
    arrays[f"{tag}.final.mean_inner_dice"] = run["final"]["mean_inner_dice"]
    arrays[f"{tag}.subject_fold"] = run["state"]["fold"]
    arrays[f"{tag}.oof_probabilities"] = np.concatenate([p.numpy() for p in run["state"]["probabilities"]])
    final = {k: v for k, v in run["final"].items() if k != "mean_inner_dice"}
    return {"result": run["result"], "fits": fits, "final": final, "cv_seconds": run["cv_seconds"]}


def run_segmentation(out_dir: Path, ratio: float | None):
    """probe_binary_segmentation twice on the same subjects: as is (torch L-BFGS) and with fit_logistic solved to
    the optimum. Both are nanobrain's code; only the second is reproducible to better than ~1e-3."""
    n_subjects, embed_dim = 20, 48
    subjects = make_subjects(n_subjects, embed_dim, seed=20261023)
    arrays = {
        "patch_counts": np.array([len(s["patch_ids"]) for s in subjects]),
        "patch_ids": np.concatenate([s["patch_ids"] for s in subjects]).astype(np.int32),
        "features": np.concatenate([s["features"] for s in subjects]).astype(np.float16),
    }
    info = {"max_negative_ratio": ratio, "n_subjects": n_subjects, "embed_dim": embed_dim,
            "torch_threads": torch.get_num_threads()}
    for tag, solver in (("torch", None), ("exact", exact_fit_logistic)):
        run = run_segmentation_probe(subjects, ratio, solver)
        if "labels" not in arrays:
            arrays["labels"] = np.concatenate([p.numpy() for p in run["patch_labels"]]).astype(np.uint8)
        info[tag] = describe_fits(run, n_subjects, tag, arrays, newton_check=tag == "exact")
        result = run["result"]
        print(f"{tag}: dice {result['dice']:.5f} ap {result['average_precision']:.5f} "
              f"auroc {result['voxel_auroc']:.5f} alphas {result['fold_alpha']} thresholds "
              f"{[round(t, 5) for t in result['fold_threshold']]} final {info[tag]['final']} {run['cv_seconds']:.1f}s")
    write(out_dir, f"segmentation_ratio{'none' if ratio is None else int(ratio)}", arrays, info)


def run_fit(out_dir: Path):
    """A few single fits of fit_logistic on three subjects of the segmentation data, small enough for the
    always-on tests: torch's own result, the Newton optimum and fit_predict_segmentation's standardization."""
    subjects = make_subjects(20, 48, seed=20261023)
    patch_labels = [
        probe.patchify3d(torch.from_numpy(s["label"][None].astype(np.uint8)))[0][torch.from_numpy(s["patch_ids"])]
        for s in subjects
    ]
    train_ids, test_id = [0, 1, 2], 3
    features = torch.cat([torch.from_numpy(subjects[i]["features"]) for i in train_ids])
    labels = torch.cat([patch_labels[i] for i in train_ids]).float()
    arrays = {
        "features": features.numpy().astype(np.float16),
        "labels": labels.numpy().astype(np.uint8),
        "test_features": subjects[test_id]["features"].astype(np.float16),
        "test_labels": patch_labels[test_id].numpy().astype(np.uint8),
    }
    info = {"train_ids": train_ids, "test_id": test_id, "fits": []}
    for ratio in (None, 10.0):
        f, l = features, labels
        tag = "all" if ratio is None else "ratio10"
        if ratio is not None:
            keep = probe.subsample_negatives(l, ratio)
            arrays[f"{tag}.keep_ids"] = keep.numpy().astype(np.int32)
            f, l = f[keep], l[keep]
        mean = f.mean(dim=0)
        std = f.std(dim=0, correction=0).clamp_min(1e-6)
        arrays[f"{tag}.mean"] = mean.numpy()
        arrays[f"{tag}.std"] = std.numpy()
        standardized = (f - mean) / std
        for alpha in (1e1, 1e3, 1e5):
            name = f"{tag}.alpha{int(alpha)}"
            start = time.perf_counter()
            coef, intercept = probe.fit_logistic(standardized, l, alpha)
            torch_seconds = time.perf_counter() - start
            exact_coef, exact_intercept = newton_optimum(standardized, l, alpha)
            arrays[f"{name}.torch_coef"] = coef.numpy()
            arrays[f"{name}.torch_intercept"] = intercept.numpy()
            arrays[f"{name}.exact_coef"] = exact_coef.float().numpy()
            arrays[f"{name}.exact_intercept"] = exact_intercept.float().numpy()
            test = (torch.from_numpy(subjects[test_id]["features"]) - mean) / std
            exact_p = torch.sigmoid(test.double() @ exact_coef + exact_intercept)
            torch_p = torch.sigmoid(test @ coef + intercept)
            info["fits"].append(
                {
                    "name": name,
                    "alpha": alpha,
                    "max_negative_ratio": ratio,
                    "n_rows": len(f),
                    "torch_seconds": torch_seconds,
                    "torch_vs_exact_test_prob": float((torch_p.double() - exact_p).abs().max()),
                    "exact_test_dice": probe.dice_by_threshold(exact_p.float(), patch_labels[test_id]).tolist(),
                }
            )
            print(name, len(f), f"{torch_seconds:.2f}s", info["fits"][-1]["torch_vs_exact_test_prob"])
    write(out_dir, "segmentation_fit", arrays, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=["regression", "classification", "segmentation", "fit", "minority"])
    parser.add_argument("--ratio", default="10", help="max_negative_ratio for segmentation, or 'none'")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.kind == "regression":
        run_regression(args.out or TEST_DATA)
        run_regression(args.out or TEST_DATA, dim=40, name="probe_regression_svd")
    elif args.kind == "classification":
        run_classification(args.out or TEST_DATA)
    elif args.kind == "fit":
        run_fit(args.out or TEST_DATA)
    elif args.kind == "minority":
        run_minority(args.out or TEST_DATA)
    else:
        ratio = None if args.ratio.lower() == "none" else float(args.ratio)
        run_segmentation(args.out or FIXTURE_DIR, ratio)


if __name__ == "__main__":
    main()
