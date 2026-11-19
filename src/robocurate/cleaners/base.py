"""Cleaner plugin API.

A cleaner sees ONLY a label-free DatasetView, the integer episode budget, and a numpy Generator.
It must return exactly `budget` unique episode indices from the view, deterministically for a given rng
state. It never receives labels, eval code, eval seeds or eval results (CLAUDE.md rule 7).

Cleaners that roll out policies declare `requires_rollouts` (cap: CURATION_ROLLOUT_CAP, D-007) and may
only use env seeds from protocol.curation_seed_range. No such cleaner exists in this milestone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from robocurate.data.episodes import DatasetView, EpisodeData

CURATION_ROLLOUT_CAP = 100  # per task, D-007


class Cleaner(ABC):
    name: str = ""
    version: str = ""
    requires_rollouts: int = 0

    @abstractmethod
    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        """Return exactly `budget` unique episode indices from `view`."""


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
