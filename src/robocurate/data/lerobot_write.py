"""Write a label-free DatasetView as a LeRobot dataset directory (for checker adapters, which read a dataset dir).

Kept apart from data/mixture.py on purpose: this module imports nothing that can reach ground-truth labels
(CLAUDE.md rule 7; the conformance no-leak check watches robocurate.data.labels).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def write_view_dataset(view: Any, root: str | Path, repo_id: str = "robocurate/view") -> Path:
    """Write a label-free DatasetView as a LeRobot dataset (no video, episodes in view order; same LeRobot calls as
    write_policy_dataset: LeRobotDataset.create / add_frame / save_episode / finalize). For checker adapters that
    read a dataset directory. Writes NO labels sidecar (CLAUDE.md rule 7)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    eps = list(view)
    if [e.episode_index for e in eps] != list(range(len(eps))):
        raise ValueError("view episodes must be indexed 0..N-1 (dataset episode index = view index)")
    sd, ad = eps[0].state.shape[1], eps[0].actions.shape[1]
    features = {"observation.state": {"dtype": "float32", "shape": (sd,), "names": None},
                "action": {"dtype": "float32", "shape": (ad,), "names": None}}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=view.fps, features=features, root=root, use_videos=False)
    try:
        for e in eps:
            for s, a in zip(e.state, e.actions, strict=True):
                ds.add_frame({"task": e.task, "observation.state": np.asarray(s, np.float32),
                              "action": np.asarray(a, np.float32)})
            ds.save_episode()
    finally:
        ds.finalize()
    return root
