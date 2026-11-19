from robocurate.cleaners.base import Cleaner


class Duplicates(Cleaner):
    name = "duplicates"
    version = "1"

    def select(self, view, budget, rng):
        ids = view.episode_indices[: budget - 1]
        return ids + [ids[0]]
