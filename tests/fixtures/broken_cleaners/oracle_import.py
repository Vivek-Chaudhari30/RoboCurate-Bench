from robocurate.cleaners.base import Cleaner
from robocurate.runs.oracle import oracle_select  # forbidden


class OracleCaller(Cleaner):
    name = "oracle_caller"
    version = "1"

    def select(self, view, budget, rng):
        return view.episode_indices[:budget]
