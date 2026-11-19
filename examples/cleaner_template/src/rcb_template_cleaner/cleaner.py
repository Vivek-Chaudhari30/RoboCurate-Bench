"""Template cleaner: keep the `budget` episodes with the smallest mean squared action change.

Copy this package, rename it, change `name`, `version` and `score`. Only import from robocurate.cleaners.base and
robocurate.data.episodes (the submit check rejects code that can reach labels, eval code or the oracle).
"""

from __future__ import annotations

import numpy as np

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import EpisodeData


class SmoothestEpisodes(ScoreCleaner):
    name = "template_smoothness"
    version = "1"
    # Resource declarations (D-022). This cleaner needs none of them.
    requires_rollouts = 0  # <= 100; seeds come only from the protocol's curation range
    requires_reference_policy = False
    requires_gpu = False
    implementation = "original"
    source = "examples/cleaner_template"

    def score(self, episode: EpisodeData) -> float:  # higher = keep; ties are broken by the benchmark (D-021)
        a = np.asarray(episode.actions, dtype=np.float64)
        return -float(np.mean(np.diff(a, axis=0) ** 2)) if len(a) > 1 else 0.0
