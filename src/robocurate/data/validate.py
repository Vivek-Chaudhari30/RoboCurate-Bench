"""`robocurate validate-defects`: defect statistics of LeRobot datasets from actions, state and timestamps ONLY
(no video download). Supports LeRobot v2.1 (meta/tasks.jsonl, one parquet per episode) and v3.0 (meta/tasks.parquet,
chunked parquet files). Informs the real mixture recipe; never edits a config.

Definitions (fixed before looking at any dataset; same pause rule as rcb-score v1):
  pause frame      ||s[t+1] - s[t]||_2 < tau, tau = 0.05 * median over ALL transitions of the dataset
  pause segment    >= PAUSE_MIN_S seconds (0.5 s) of consecutive pause frames
  pause rate       fraction of episodes with >= 1 pause segment; pause lengths in seconds per segment.
                   Also split into BOUNDARY segments (touching the first or last transition: idle before/after
                   teleoperation) and INTERIOR segments (all others; our injected pauses are interior). The split
                   was added after the first run showed headline rates of 0.85-1.0; both are reported.
  jitter           per episode: share of action power above fps/4 (FFT of each mean-removed action dim, power
                   summed over dims); reported as a distribution and as the share of episodes above 0.10
  near-duplicate   episode pairs whose action trajectories, z-scored per dim with dataset stats and resampled to
                   64 points, have RMS difference < 0.05 and lengths within 5%; exact duplicates (identical
                   action+state bytes) counted separately
  length           episode duration in seconds (frames / fps); timestamp problems (non-monotone, gaps > 2.5/fps)
  task variety     number of distinct task strings, Shannon entropy (bits) of the episode task distribution
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PAUSE_TAU_FRAC, PAUSE_MIN_S, JITTER_SHARE_FLAG, DUP_RMS, DUP_LEN_TOL, RESAMPLE = 0.05, 0.5, 0.10, 0.05, 0.05, 64


@dataclass
class Episode:
    index: int
    task: str
    timestamps: np.ndarray
    state: np.ndarray | None
    actions: np.ndarray


def _stack(col) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float64) for v in col]) if len(col) else np.zeros((0, 0))


def load_parquet_dataset(root: str | Path) -> tuple[dict, list[Episode]]:
    import pandas as pd

    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    if (root / "meta/tasks.parquet").exists():
        t = pd.read_parquet(root / "meta/tasks.parquet")
        tasks = ({int(i): str(k) for k, i in zip(t.index, t["task_index"], strict=True)} if "task_index" in t
                 else {i: str(k) for i, k in enumerate(t.index)})
    else:
        tasks = {}
        for line in (root / "meta/tasks.jsonl").read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tasks[int(d["task_index"])] = str(d["task"])
    files = sorted((root / "data").rglob("*.parquet"))
    cols = ["episode_index", "frame_index", "timestamp", "action", "observation.state", "task_index"]
    df = pd.concat([pd.read_parquet(f, columns=[c for c in cols if c in pd.read_parquet(f).columns])
                    for f in files], ignore_index=True)
    eps = []
    for ep, g in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        eps.append(Episode(int(ep), tasks.get(int(g["task_index"].iloc[0]), "?") if "task_index" in g else "?",
                           g["timestamp"].to_numpy(dtype=np.float64), _stack(g["observation.state"].tolist())
                           if "observation.state" in g else None, _stack(g["action"].tolist())))
    return info, eps


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, length) of runs of True."""
    out, run, start = [], 0, 0
    for t, m in enumerate(mask):
        if m:
            if run == 0:
                start = t
            run += 1
        elif run:
            out.append((start, run))
            run = 0
    if run:
        out.append((start, run))
    return out


def hf_share(actions: np.ndarray, fps: float) -> float:
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 8:
        return 0.0
    a = a - a.mean(0)
    p = np.abs(np.fft.rfft(a, axis=0)) ** 2
    f = np.fft.rfftfreq(len(a), d=1.0 / fps)
    tot = p[1:].sum()
    return float(p[f > fps / 4].sum() / tot) if tot > 0 else 0.0


def _pct(x, qs=(5, 25, 50, 75, 95)) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if x.size else {}


def analyze(info: dict, eps: list[Episode]) -> dict[str, Any]:
    fps = float(info.get("fps") or 30)
    n = len(eps)
    sig = [e.state if e.state is not None and e.state.size else e.actions for e in eps]
    steps = [np.linalg.norm(np.diff(s, axis=0), axis=1) for s in sig]
    allst = np.concatenate([s for s in steps if s.size]) if n else np.zeros(0)
    tau = PAUSE_TAU_FRAC * float(np.median(allst)) if allst.size else 0.0
    min_len = max(1, int(round(PAUSE_MIN_S * fps)))
    seg_lens, int_lens, with_pause, with_interior, pause_frac = [], [], 0, 0, []
    for st in steps:
        segs = [(a, n_) for a, n_ in _segments(st < tau) if n_ >= min_len]
        interior = [(a, n_) for a, n_ in segs if a > 0 and a + n_ < len(st)]
        seg_lens += [n_ / fps for _, n_ in segs]
        int_lens += [n_ / fps for _, n_ in interior]
        with_pause += bool(segs)
        with_interior += bool(interior)
        pause_frac.append(float(np.mean(st < tau)) if st.size else 0.0)
    jit = [hf_share(e.actions, fps) for e in eps]
    lens = [len(e.actions) / fps for e in eps]
    ts_bad = sum(bool(len(e.timestamps) > 1 and (np.any(np.diff(e.timestamps) <= 0) or
                                                 np.any(np.diff(e.timestamps) > 2.5 / fps))) for e in eps)
    # duplicates
    hashes = [hashlib.sha256(e.actions.tobytes() + (e.state.tobytes() if e.state is not None else b"")).hexdigest()
              for e in eps]
    exact = n - len(set(hashes))
    A = np.concatenate([e.actions for e in eps]) if n else np.zeros((0, 1))
    mu, sd = A.mean(0), A.std(0) + 1e-8
    sigs = [np.stack([np.interp(np.linspace(0, len(e.actions) - 1, RESAMPLE), np.arange(len(e.actions)),
                                ((e.actions - mu) / sd)[:, d]) for d in range(A.shape[1])], 1) for e in eps]
    near = []
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] == hashes[j]:
                continue
            li, lj = len(eps[i].actions), len(eps[j].actions)
            if abs(li - lj) > DUP_LEN_TOL * max(li, lj):
                continue
            if float(np.sqrt(np.mean((sigs[i] - sigs[j]) ** 2))) < DUP_RMS:
                near.append((eps[i].index, eps[j].index))
    tasks = [e.task for e in eps]
    _, counts = np.unique(tasks, return_counts=True) if n else ([], np.zeros(0))
    pr = counts / counts.sum() if n else counts
    return {
        "n_episodes": n, "fps": fps, "codebase_version": info.get("codebase_version"), "robot_type":
        info.get("robot_type"),
        "pause": {"tau": tau, "min_segment_s": PAUSE_MIN_S, "episode_rate": with_pause / n if n else 0.0,
                  "segment_seconds": _pct(seg_lens), "n_segments": len(seg_lens),
                  "interior_episode_rate": with_interior / n if n else 0.0, "interior_segment_seconds": _pct(int_lens),
                  "n_interior_segments": len(int_lens),
                  "pause_frame_fraction": _pct(pause_frac)},
        "jitter": {"hf_power_share": _pct(jit), "episodes_above_0.10": float(np.mean(np.array(jit) > JITTER_SHARE_FLAG))
                   if n else 0.0},
        "duplicates": {"exact_duplicate_episodes": exact, "near_duplicate_pairs": len(near),
                       "near_duplicate_examples": near[:5]},
        "length_seconds": _pct(lens), "timestamp_problem_episodes": ts_bad,
        "tasks": {"distinct": int(len(counts)), "entropy_bits": float(-(pr * np.log2(pr)).sum()) if n else 0.0},
    }


def fetch_non_video(repo_id: str, revision: str) -> Path:
    """Download meta/ and data/ only (no videos) into the HF cache under $HF_HOME."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id, repo_type="dataset", revision=revision,
                                  allow_patterns=["meta/**", "data/**"]))
