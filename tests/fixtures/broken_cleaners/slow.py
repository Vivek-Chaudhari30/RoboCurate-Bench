import time

from robocurate.cleaners.base import Cleaner


class Slow(Cleaner):
    name = "slow"
    version = "1"

    def select(self, view, budget, rng):
        time.sleep(5)
        return view.episode_indices[:budget]
