"""Per-frame task-progress models for cleaners (Robometer via lerobot.rewards; deterministic mocks for tests).

Interface: `ProgressModel.episode_progress(view) -> {episode_index: np.ndarray (T,) in [0, 1]}`, plus `revision`
(cache key component: dataset hash + model revision, D-022).

RobometerProgress wraps `lerobot.rewards.robometer.compute_rabc_weights.compute_robometer_progress`
(lerobot 0.6.1, rewards/robometer/compute_rabc_weights.py:103). That function loads the dataset by repo_id
(`LeRobotDataset(dataset_repo_id, download_videos=True)`, :133), i.e. from HF_LEROBOT_HOME/<repo_id>; it has no
root argument. It writes a parquet with columns index, episode_index, frame_index, progress_sparse (:193-201),
the schema RABCWeights reads (rewards/sarm/rabc.py:101-126). Progress is clamped to [0, 1]
(rewards/robometer/modeling_robometer.py:279-283). Requires a GPU host and the Robometer weights (not
downloaded in this milestone).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocurate.data.episodes import DatasetView


class MockProgressModel:
    """Deterministic stand-in for Robometer (tests / fake backend ONLY; never used for results).

    progress_t = cumulative state path length up to t / total path length (monotone in [0, 1]); a held pose
    (pause) adds no progress, so paused frames get zero progress delta, as a reward model would ideally predict.
    `fn(episode) -> progress` overrides the rule (toy-testbed sanity tests use a goal-distance progress).
    """

    revision = "mock-progress-v1"

    def __init__(self, fn: Any = None):
        self.fn = fn

    def episode_progress(self, view: DatasetView) -> dict[int, np.ndarray]:
        out = {}
        for e in view:
            if self.fn is not None:
                p = np.asarray(self.fn(e), dtype=np.float64)
            else:
                step = np.linalg.norm(np.diff(np.asarray(e.state, dtype=np.float64), axis=0), axis=1)
                c = np.concatenate([[0.0], np.cumsum(step)])
                p = c / c[-1] if c[-1] > 0 else np.zeros_like(c)
            out[e.episode_index] = np.clip(p, 0.0, 1.0)
        return out


class RobometerProgress:  # pragma: no cover - needs GPU + model weights
    def __init__(self, dataset_repo_id: str, reward_model_path: str, revision: str, output_path: str | Path,
                 device: str = "cuda", batch_size: int = 32):
        self.repo_id, self.model_path, self.output_path = dataset_repo_id, reward_model_path, Path(output_path)
        self.revision = f"{reward_model_path}@{revision}"
        self.device, self.batch_size = device, batch_size

    def episode_progress(self, view: DatasetView) -> dict[int, np.ndarray]:
        import pandas as pd
        from lerobot.rewards.robometer.compute_rabc_weights import compute_robometer_progress

        out = compute_robometer_progress(dataset_repo_id=self.repo_id, reward_model_path=self.model_path,
                                         output_path=str(self.output_path), device=self.device,
                                         batch_size=self.batch_size)
        df = pd.read_parquet(out).sort_values(["episode_index", "frame_index"])
        prog = {int(ep): g["progress_sparse"].to_numpy(dtype=np.float64) for ep, g in df.groupby("episode_index")}
        missing = set(view.episode_indices) - set(prog)
        if missing:
            raise RuntimeError(f"Robometer parquet lacks episodes {sorted(missing)[:5]}...")
        return {i: prog[i] for i in view.episode_indices}


class CachedProgress:
    """Score cache keyed by dataset hash + model revision (m1-core item 5b): one progress pass is shared by
    every entrant that uses the same model on the same data (robometer_filter, rabc, rcb_score)."""

    def __init__(self, model: Any, cache_dir: str | Path):
        self.model, self.cache_dir = model, Path(cache_dir)
        self.revision = model.revision
        self.hits = 0

    def path(self, view: DatasetView) -> Path:
        import hashlib

        h = hashlib.sha256(f"{view.fingerprint()}|{self.revision}".encode()).hexdigest()[:24]
        return self.cache_dir / "_progress_cache" / f"{h}.npz"

    def episode_progress(self, view: DatasetView) -> dict[int, np.ndarray]:
        p = self.path(view)
        if p.exists():
            self.hits += 1
            z = np.load(p)
            return {int(k): z[k] for k in z.files}
        out = self.model.episode_progress(view)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npz")
        np.savez(tmp, **{str(k): v for k, v in out.items()})
        tmp.replace(p)
        return out
