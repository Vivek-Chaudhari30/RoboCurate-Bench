"""Weighting-track artifacts (D-020) in LeRobot's RA-BC progress-parquet schema.

LeRobot 0.6.1 exposes per-sample loss weighting only through `--sample_weighting.type` in {rabc, uniform}
(utils/sample_weighting.py:125-132; no plugin hook for other types). RA-BC reads per-frame progress from a
parquet (columns index, episode_index, progress_<head_mode>; rewards/sarm/rabc.py:101-126) and turns progress
deltas over chunk_size frames into per-frame weights (rabc.py:262-295, SARM paper Eq. 8-9). A weighting entrant
therefore supplies per-frame PROGRESS; LeRobot's own RABCWeights computes the weights during training.

Placebo (D-025, amends D-020): permute whole per-episode WEIGHT curves across episodes (`permute_weight_curves`).
For permutation seed s, episode e receives the weight curve of episode pi_s(e), resampled onto len(e) by normalized
time (linear interpolation); then every weight is rescaled so the dataset-wide mean frame weight equals the
original. The entrant's weight curves are the RA-BC frame weights of its progress (`rabc_frame_weights`, which
equals LeRobot's RABCWeights before batch normalization). `permute_progress` (the D-020 interpretation: permute
progress curves, let RA-BC recompute weights) is kept only for reference; D-025 rejects it.

LeRobot 0.6.1 normalizes each batch's weights to sum to the batch size (rabc.py:232-233) and divides the loss by
the weight sum (lerobot_train.py:162), so a global rescale has no effect on LeRobot training; the rescale keeps the
artifact's mean equal to the entrant's as D-025 specifies.

Global frame indices: the training dataset is the materialized full-data copy (episodes re-indexed 0..N-1 in
sorted original-index order, frames contiguous), so index = cumulative length over sorted episodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _resample(curve: np.ndarray, n: int) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) == n:
        return curve.copy()
    if len(curve) == 1:
        return np.full(n, curve[0])
    return np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(curve)), curve)


def permute_weight_curves(weights: dict[int, np.ndarray], seed: int | None = None,
                          perm: np.ndarray | None = None) -> dict[int, np.ndarray]:
    """D-025 weighting-track placebo. weights: per-episode frame-weight curves of the entrant.

    Episode e (in sorted order) receives the curve of episode sorted[perm[i]], resampled onto len(e) by normalized
    time (linear interpolation); then all weights are multiplied by one factor so the dataset-wide mean frame weight
    equals the original. perm defaults to np.random.default_rng(seed).permutation(n_episodes).
    """
    eps = sorted(weights)
    if perm is None:
        if seed is None:
            raise ValueError("give a permutation seed or an explicit permutation")
        perm = np.random.default_rng(seed).permutation(len(eps))
    perm = np.asarray(perm)
    if sorted(perm.tolist()) != list(range(len(eps))):
        raise ValueError("perm must be a permutation of range(n_episodes)")
    out = {e: _resample(weights[eps[j]], len(weights[e])) for e, j in zip(eps, perm, strict=True)}
    orig = np.concatenate([np.asarray(weights[e], dtype=np.float64) for e in eps])
    new = np.concatenate([out[e] for e in eps])
    if np.any(orig < 0) or not np.all(np.isfinite(orig)):
        raise ValueError("weights must be finite and >= 0")
    if new.mean() <= 0:
        if orig.mean() == 0:
            return out
        raise ValueError("permuted weights have zero mean; cannot preserve the original mean")
    factor = orig.mean() / new.mean()
    return {e: out[e] * factor for e in eps}


def write_weights_parquet(weights: dict[int, np.ndarray], path: str | Path, meta: dict[str, str] | None = None
                          ) -> Path:
    """Per-frame weights for a materialized dataset whose episodes are sorted(weights) (same index layout as the
    RA-BC progress parquet). Columns: index, episode_index, frame_index, weight."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    idx, ep, fr, w = [], [], [], []
    start = 0
    for new_ep, e in enumerate(sorted(weights)):
        x = np.asarray(weights[e], dtype=np.float64)
        n = len(x)
        idx.append(np.arange(start, start + n, dtype=np.int64))
        ep.append(np.full(n, new_ep, dtype=np.int64))
        fr.append(np.arange(n, dtype=np.int64))
        w.append(x)
        start += n
    t = pa.table({"index": np.concatenate(idx), "episode_index": np.concatenate(ep),
                  "frame_index": np.concatenate(fr), "weight": np.concatenate(w)})
    if meta:
        t = t.replace_schema_metadata({k.encode(): str(v).encode() for k, v in meta.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path)
    return path


def policy_chunk_size(policy_type: str) -> int:
    """RA-BC's delta horizon is the policy config's chunk_size (utils/sample_weighting.py:154-155); the default
    for the task's policy type comes from LeRobot's registered config class (policies/factory.py:102)."""
    from lerobot.policies.factory import make_policy_config

    cs = getattr(make_policy_config(policy_type), "chunk_size", None)
    if cs is None:
        raise ValueError(f"policy type {policy_type!r} has no chunk_size; RA-BC weighting needs one")
    return int(cs)


def permute_progress(progress: dict[int, np.ndarray], seed: int) -> dict[int, np.ndarray]:
    eps = sorted(progress)
    perm = np.random.default_rng(seed).permutation(len(eps))
    out = {}
    for e, j in zip(eps, perm, strict=True):
        src = np.asarray(progress[eps[j]], dtype=np.float64)
        n = len(progress[e])
        out[e] = src.copy() if len(src) == n else np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(src)), src)
    return out


def write_progress_parquet(progress: dict[int, np.ndarray], path: str | Path, meta: dict[str, str] | None = None
                           ) -> Path:
    """Write the RA-BC progress parquet for a materialized dataset whose episodes are sorted(progress)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    idx, ep, fr, pr = [], [], [], []
    start = 0
    for new_ep, e in enumerate(sorted(progress)):
        p = np.asarray(progress[e], dtype=np.float32)
        n = len(p)
        idx.append(np.arange(start, start + n, dtype=np.int64))
        ep.append(np.full(n, new_ep, dtype=np.int64))
        fr.append(np.arange(n, dtype=np.int64))
        pr.append(p)
        start += n
    t = pa.table({"index": np.concatenate(idx), "episode_index": np.concatenate(ep),
                  "frame_index": np.concatenate(fr), "progress_sparse": np.concatenate(pr)})
    if meta:
        t = t.replace_schema_metadata({k.encode(): str(v).encode() for k, v in meta.items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, path)
    return path


def rabc_frame_weights(progress: dict[int, np.ndarray], chunk_size: int, kappa: float = 0.01,
                       epsilon: float = 1e-6) -> dict[int, np.ndarray]:
    """Per-frame RA-BC weights exactly as lerobot's RABCWeights computes them (rabc.py:143-173 global stats,
    :237-260 delta, :262-295 Eq. 8-9), WITHOUT batch normalization. For reporting and toy tests only; real
    training uses LeRobot's class."""
    deltas = {}
    for e, p in progress.items():
        p = np.asarray(p, dtype=np.float64)
        fut = np.minimum(np.arange(len(p)) + chunk_size, len(p) - 1)
        deltas[e] = p[fut] - p
    alld = np.concatenate(list(deltas.values())) if deltas else np.zeros(0)
    mu = max(float(alld.mean()), 0.0) if alld.size else 0.0
    sd = max(float(alld.std()), epsilon) if alld.size else epsilon
    out = {}
    for e, d in deltas.items():
        soft = np.clip((d - (mu - 2 * sd)) / (4 * sd + epsilon), 0.0, 1.0)
        w = np.zeros_like(d)
        w[d > kappa] = 1.0
        mid = (d >= 0) & (d <= kappa)
        w[mid] = soft[mid]
        out[e] = w
    return out
