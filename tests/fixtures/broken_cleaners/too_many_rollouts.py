from robocurate.cleaners.base import Cleaner


class Greedy(Cleaner):
    name = "greedy"
    version = "1"
    requires_rollouts = 500

    def select(self, view, budget, rng):
        return view.episode_indices[:budget]
