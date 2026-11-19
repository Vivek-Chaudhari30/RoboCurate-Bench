"""Cleaner plugin API.

A cleaner sees ONLY a label-free DatasetView, the integer episode budget, and a numpy Generator.
It must return exactly `budget` unique episode indices from the view, deterministically for a given rng
state. It never receives labels, eval code, eval seeds or eval results (CLAUDE.md rule 7).

Resource declarations (D-022), validated by `robocurate submit` (`validate_resources`):
  track                      "selection" (select -> ids) or "weighting" (WeightingCleaner, D-020)
  requires_rollouts          int in [0, protocol.max_curation_rollouts]; env seeds come only from
                             protocol.curation_seed_range, via `self.context.rollouts` (curation/rollouts.py)
  requires_reference_policy  the full-data checkpoint of the first train seed, read-only, via
                             `self.context.reference_policy`
  requires_gpu               the prepare stage must run on a GPU host; GPU-h are reported as curation cost
Selections and weights are computed once per (task, budget, replicate) by `robocurate prepare` (D-022). The
prepare stage sets `self.context` (a curation.context.CurationContext) before calling select/frame_progress.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from robocurate.data.episodes import DatasetView, EpisodeData

CURATION_ROLLOUT_CAP = 100  # per task, D-007


TRACKS = ("selection", "weighting")
IMPLEMENTATIONS = ("original", "wrapped", "reimplementation")


class Cleaner(ABC):
    name: str = ""
    version: str = ""
    track: str = "selection"
    requires_rollouts: int = 0
    requires_reference_policy: bool = False
    requires_gpu: bool = False
    implementation: str = "original"  # original | wrapped | reimplementation
    source: str = ""  # citation, paper section or repo@commit
    # "active", or "disabled — <reason>": disabled entrants are refused in grids and excluded from the leaderboard
    status: str = "active"
    requires_dataset_root: bool = False  # reads the LeRobot dataset directory (checker adapters), D-022
    context: Any = None  # curation.context.CurationContext, set by the prepare stage

    @abstractmethod
    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        """Return exactly `budget` unique episode indices from `view`."""

    def resources(self) -> dict[str, Any]:
        return {"track": self.track, "requires_rollouts": self.requires_rollouts,
                "requires_reference_policy": self.requires_reference_policy, "requires_gpu": self.requires_gpu,
                "requires_dataset_root": self.requires_dataset_root}

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "implementation": self.implementation,
                "source": self.source, "status": self.status, **self.resources()}


class WeightingCleaner(Cleaner):
    """Weighting track (D-020): per-frame progress on the FULL dataset, consumed by LeRobot's RA-BC sample
    weighter (lerobot/utils/sample_weighting.py:103, lerobot/rewards/sarm/rabc.py:59). The placebo permutes
    each episode's progress curve across episodes (curation/weights.py)."""

    track = "weighting"

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        raise TypeError(f"{self.name} is a weighting-track entrant; it has no select() (D-020)")

    @abstractmethod
    def frame_progress(self, view: DatasetView, rng: np.random.Generator) -> dict[int, np.ndarray]:
        """episode_index -> per-frame progress in [0, 1], one value per frame of that episode."""


def validate_resources(cleaner: Cleaner, max_rollouts: int = CURATION_ROLLOUT_CAP) -> list[str]:
    """D-022 resource declarations. Returns a list of problems (empty = valid)."""
    out = []
    if cleaner.track not in TRACKS:
        out.append(f"track must be one of {TRACKS}, got {cleaner.track!r}")
    if (cleaner.track == "weighting") != isinstance(cleaner, WeightingCleaner):
        out.append("weighting-track entrants must subclass WeightingCleaner (and only they may)")
    r = cleaner.requires_rollouts
    if not (isinstance(r, int) and not isinstance(r, bool) and 0 <= r <= min(max_rollouts, CURATION_ROLLOUT_CAP)):
        out.append(f"requires_rollouts={r!r} must be an int in [0, {min(max_rollouts, CURATION_ROLLOUT_CAP)}]")
    for flag in ("requires_reference_policy", "requires_gpu"):
        if not isinstance(getattr(cleaner, flag), bool):
            out.append(f"{flag} must be a bool")
    if cleaner.implementation not in IMPLEMENTATIONS:
        out.append(f"implementation must be one of {IMPLEMENTATIONS}")
    if not (cleaner.name and cleaner.version):
        out.append("name and version are required")
    return out


class ScoreCleaner(Cleaner):
    """Score each episode; keep the top `budget` (higher is better).

    Ties are broken by sha256(protocol salt || episode content hash) (D-021), never by episode index; only
    exact-duplicate episodes (identical content, identical key) fall back to episode_index.
    """

    @abstractmethod
    def score(self, episode: EpisodeData) -> float: ...

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        scored = [(-float(self.score(e)), view.tie_break_key(e.episode_index), e.episode_index) for e in view]
        scored.sort()
        return sorted(idx for _, _, idx in scored[:budget])


def budget_count(n_episodes: int, fraction: float) -> int:
    """Episode budget for a fraction. Round half up; at least 1."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return max(1, int(np.floor(n_episodes * fraction + 0.5)))
