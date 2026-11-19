import random

from robocurate.cleaners.base import Cleaner


class NonDeterministic(Cleaner):
    """Ignores the provided rng and uses global, unseeded randomness."""

    name = "nondeterministic"
    version = "1"

    def select(self, view, budget, rng):
        return sorted(random.sample(view.episode_indices, budget))
