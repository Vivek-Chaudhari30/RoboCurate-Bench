from robocurate.cleaners.base import Cleaner


class WrongCount(Cleaner):
    name = "wrong_count"
    version = "1"

    def select(self, view, budget, rng):
        return view.episode_indices[: budget - 1]
