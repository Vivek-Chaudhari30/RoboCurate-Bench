"""Weighting-track artifacts (D-020) in LeRobot's RA-BC progress-parquet schema.

LeRobot 0.6.1 exposes per-sample loss weighting only through `--sample_weighting.type` in {rabc, uniform}
(utils/sample_weighting.py:125-132; no plugin hook for other types). RA-BC reads per-frame progress from a
parquet (columns index, episode_index, progress_<head_mode>; rewards/sarm/rabc.py:101-126) and turns progress
deltas over chunk_size frames into per-frame weights (rabc.py:262-295, SARM paper Eq. 8-9). A weighting entrant
therefore supplies per-frame PROGRESS; LeRobot's own RABCWeights computes the weights during training.

Placebo (D-020, "the same weight vector randomly permuted across episodes"): RA-BC weights are per frame, so the
unit we permute is the episode's progress CURVE. For permutation seed s, episode e receives the curve of episode
pi_s(e), resampled to len(e) by normalized time (np.interp). The pool of curves (hence the weight distribution)
is preserved; its alignment with episode content is destroyed. [Interpretation recorded in PROGRESS.md for a
protocol decision.]

Global frame indices: the training dataset is the materialized full-data copy (episodes re-indexed 0..N-1 in
sorted original-index order, frames contiguous), so index = cumulative length over sorted episodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


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
