"""Ground-truth label sidecar (D-008). Read ONLY by robocurate.report and robocurate.runs.oracle.

The sidecar lives BESIDE the dataset directory, never inside it: `<dataset_root>.labels.json`. Anything that
hands a dataset directory to a cleaner therefore cannot leak labels by accident. An import-graph test
asserts nothing under robocurate.cleaners can reach this module.

Per episode: source (scripted | human | policy), source dataset + episode, defect types, quality tier.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1
SOURCES = ("scripted", "human", "policy")
DEFECTS = ("pause", "duplicate", "wrong_task", "jitter", "gripper_drop", "hesitation")
TIERS = ("clean", "degraded", "bad")  # best -> worst


@dataclass(frozen=True)
class EpisodeLabel:
    episode_index: int
    source: str
    source_dataset: str
    source_episode: int
    defects: tuple[str, ...] = ()
    quality_tier: str = "clean"

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        bad = set(self.defects) - set(DEFECTS)
        if bad:
            raise ValueError(f"unknown defects {sorted(bad)}")
        if self.quality_tier not in TIERS:
            raise ValueError(f"unknown quality tier {self.quality_tier!r}")


@dataclass
class LabelsSidecar:
    dataset: str
    recipe_sha256: str
    seed: int
    episodes: list[EpisodeLabel] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        for e in d["episodes"]:
            e["defects"] = list(e["defects"])
        return d

    def by_index(self) -> dict[int, EpisodeLabel]:
        return {e.episode_index: e for e in self.episodes}


def sidecar_path(dataset_root: str | Path) -> Path:
    root = Path(dataset_root)
    return root.parent / f"{root.name}.labels.json"


def write_labels(sidecar: LabelsSidecar, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sidecar.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_labels(path: str | Path) -> LabelsSidecar:
    d = json.loads(Path(path).read_text())
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported labels schema {d.get('schema_version')}")
    eps = [EpisodeLabel(**{**e, "defects": tuple(e["defects"])}) for e in d["episodes"]]
    idx = [e.episode_index for e in eps]
    if idx != list(range(len(idx))):
        raise ValueError(f"{path}: episode indices must be 0..n-1 in order")
    return LabelsSidecar(d["dataset"], d["recipe_sha256"], d["seed"], eps, d["schema_version"])
