"""Training on an episode subset with subset-only normalization stats (D-013).

Finding (lerobot 0.6.1):
- `lerobot-train --dataset.episodes=[...]` exists (DatasetConfig.episodes, configs/default.py:38) but the
  dataset metadata loads the FULL dataset's meta/stats.json (datasets/dataset_metadata.py:219) and
  lerobot-train hands `dataset.meta.stats` to the policy pre/post-processors (scripts/lerobot_train.py:351,
  376-377). Stats would include removed episodes -> violates D-013.
- `lerobot.datasets.dataset_tools.split_dataset(dataset, {"train": episodes}, output_dir)`
  (datasets/dataset_tools.py:175) materializes the subset as its own dataset and writes stats aggregated
  from ONLY the kept episodes' per-episode stats (`_copy_and_reindex_episodes_metadata`,
  dataset_tools.py:852-948, aggregation at :941-948). This satisfies D-013 and is what we use.
- Image stats are then overridden by ImageNet stats when `dataset.use_imagenet_stats=true` (default,
  configs/default.py:41; datasets/factory.py:137-142) -- identically for every condition, so no leak.
"""

from __future__ import annotations

from pathlib import Path

from robocurate.data.hub import load_dataset

SPLIT_NAME = "train"


def materialize_subset(repo_id: str, root: str | Path | None, episodes: list[int], output_dir: str | Path) -> Path:
    """Write `episodes` of a dataset as a standalone LeRobot dataset with recomputed stats.

    Returns the subset dataset root (output_dir/train). The subset's episodes are re-indexed 0..k-1 in
    ascending order of the original indices (split_dataset sorts them, dataset_tools.py:236).
    """
    from lerobot.datasets.dataset_tools import split_dataset

    if len(set(episodes)) != len(episodes):
        raise ValueError("duplicate episode ids")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty")
    src = load_dataset(repo_id, root=root)
    split_dataset(src, {SPLIT_NAME: sorted(int(e) for e in episodes)}, output_dir=output_dir)
    return output_dir / SPLIT_NAME
