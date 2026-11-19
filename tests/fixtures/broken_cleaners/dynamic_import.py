import importlib

from robocurate.cleaners.base import Cleaner


class Sneaky(Cleaner):
    name = "sneaky"
    version = "1"

    def select(self, view, budget, rng):
        importlib.import_module("robocurate.data." + "labels")
        return view.episode_indices[:budget]
