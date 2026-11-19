"""Dataset access through LeRobot's own dataset API (lerobot 0.6.1).

LeRobot touchpoints:
- LeRobotDatasetMetadata(repo_id, root, revision)          datasets/dataset_metadata.py:63,71
- LeRobotDataset(repo_id, root, episodes, download_videos)  datasets/lerobot_dataset.py:47
- LeRobotDataset.hf_dataset (parquet rows, no video decode) datasets/lerobot_dataset.py:397
- LeRobotDataset.absolute_to_relative_idx                    datasets/lerobot_dataset.py:405
- meta.episodes columns dataset_from_index / dataset_to_index / length / tasks (v3.0 episodes parquet)
- Cache root: HF_LEROBOT_HOME, read at import (utils/constants.py:69) -> require_storage_env() first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robocurate.data.episodes import DatasetView, EpisodeData
from robocurate.paths import require_storage_env

STATE_KEY = "observation.state"
ACTION_KEY = "action"


def _ensure_env() -> None:
    require_storage_env()


def load_metadata(repo_id: str, root: str | Path | None = None, revision: str | None = None):
    _ensure_env()
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    return LeRobotDatasetMetadata(repo_id, root=root, revision=revision)


def load_dataset(repo_id: str, root: str | Path | None = None, episodes: list[int] | None = None,
                 download_videos: bool = True, **kw: Any):
    _ensure_env()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root, episodes=episodes, download_videos=download_videos, **kw)


def episode_lengths(meta) -> list[int]:
    return [int(x) for x in meta.episodes["length"]]


def episode_frame_range(meta, episode_index: int) -> tuple[int, int]:
    """Absolute [from, to) frame indices of an episode (v3.0 episodes metadata)."""
    return (
        int(meta.episodes["dataset_from_index"][episode_index]),
        int(meta.episodes["dataset_to_index"][episode_index]),
    )


def load_view(repo_id: str, root: str | Path | None = None) -> DatasetView:
    """Label-free DatasetView from parquet rows only (no video download or decode)."""
    ds = load_dataset(repo_id, root=root, download_videos=False)
    cols = (
        ds.hf_dataset.select_columns(["episode_index", ACTION_KEY, STATE_KEY, "timestamp", "task_index"])
        .with_format("numpy")[:]
    )
    task_by_index = {int(v): str(k) for k, v in ds.meta.tasks["task_index"].items()}
    ep = cols["episode_index"].reshape(-1)
    episodes = []
    for e in np.unique(ep):
        m = ep == e
        tidx = np.unique(cols["task_index"].reshape(-1)[m])
        if len(tidx) != 1:
            raise ValueError(f"episode {e} has {len(tidx)} task strings")
        a = cols[ACTION_KEY][m]
        s = cols[STATE_KEY][m]
        ts = cols["timestamp"].reshape(-1)[m]
        for arr in (a, s, ts):
            arr.setflags(write=False)
        episodes.append(EpisodeData(int(e), task_by_index[int(tidx[0])], a, s, ts))
    return DatasetView(episodes, ds.meta.fps, name=repo_id)
