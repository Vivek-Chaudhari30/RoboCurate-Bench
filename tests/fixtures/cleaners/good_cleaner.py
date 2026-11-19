"""A well-behaved third-party-style cleaner fixture: keeps the smoothest episodes (lowest action jerk)."""

import numpy as np

from robocurate.cleaners.base import ScoreCleaner


class SmoothnessCleaner(ScoreCleaner):
    name = "smoothness_fixture"
    version = "1"

    def score(self, episode):
        a = np.asarray(episode.actions, dtype=float)
        if len(a) < 3:
            return 0.0
        return -float(np.mean(np.abs(np.diff(a, n=2, axis=0))))
