from robocurate.cleaners.base import Cleaner
from robocurate.data import labels  # forbidden: cleaners never see ground truth


class Peeker(Cleaner):
    name = "peeker"
    version = "1"

    def select(self, view, budget, rng):
        return view.episode_indices[:budget]
