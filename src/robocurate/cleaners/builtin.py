"""Built-in cleaners: the random-subset placebo and a length heuristic."""

from __future__ import annotations

import numpy as np

from robocurate.cleaners.base import Cleaner, ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData


class RandomSubset(Cleaner):
    """The placebo: a uniformly random size-`budget` subset, drawn from `rng` (seeded by the subset seed)."""

    name = "random_subset"
    version = "1"

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        ids = np.asarray(view.episode_indices)
        return sorted(int(i) for i in rng.choice(ids, size=budget, replace=False))


class LengthHeuristic(ScoreCleaner):
    """Keep the episodes whose length is nearest the dataset's median episode length.

    Rule: score(e) = -|len(e) - median(len)|. Top-`budget` by score; ties (equal distance, e.g. one episode
    above and one below the median) are broken by lower episode_index. Rationale: pauses/hesitations make
    episodes long and truncated/failed ones short; the median is a robust "typical" length. rng is unused.
    """

    name = "length_heuristic"
    version = "1"

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        self._median = float(np.median(list(view.lengths().values())))
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return -abs(episode.length - self._median)
