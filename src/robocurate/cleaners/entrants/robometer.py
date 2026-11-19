"""Robometer-based entrants (LeRobot reward models). All use `context.progress_model`, which on GPU hosts is
lerobot.rewards.robometer via curation.progress.RobometerProgress (compute_robometer_progress,
lerobot/rewards/robometer/compute_rabc_weights.py:103) behind a cache keyed by dataset hash + model revision, and
in tests/fake runs a deterministic mock. Robometer outputs per-frame progress clamped to [0, 1]
(modeling_robometer.py:279-283).

  robometer_filter v1  selection: keep the top-k episodes by Robometer progress at the episode's final frame
                       ("did the demo reach the goal", as judged by the reward model). Ties: D-021.
  rabc v1              weighting (D-020): per-frame Robometer progress -> LeRobot's RA-BC weights
                       (lerobot/rewards/sarm/rabc.py:262-295, SARM paper arXiv 2509.25358 Eq. 8-9, with the
                       defaults kappa 0.01, epsilon 1e-6; chunk_size = the trained policy's chunk_size,
                       utils/sample_weighting.py:155). LeRobot's own RABCWeights computes the weights in
                       training; this entrant only supplies the progress parquet.
"""

from __future__ import annotations

import numpy as np

from robocurate.cleaners.base import ScoreCleaner, WeightingCleaner
from robocurate.data.episodes import DatasetView, EpisodeData

LEROBOT_SRC = "lerobot 0.6.1 rewards/robometer (Apache-2.0)"


class RobometerFilter(ScoreCleaner):
    name, version = "robometer_filter", "1"
    requires_gpu = True
    implementation = "wrapped"
    source = f"{LEROBOT_SRC}; Robometer reward model lerobot/Robometer-4B"

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        prog = self.context.progress_model.episode_progress(view)
        self._final = {i: float(p[-1]) for i, p in prog.items()}
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._final[episode.episode_index]


class RABC(WeightingCleaner):
    name, version = "rabc", "1"
    requires_gpu = True
    implementation = "wrapped"
    source = f"{LEROBOT_SRC}; RA-BC as in lerobot rewards/sarm/rabc.py (SARM, arXiv 2509.25358 Eq. 8-9)"

    def frame_progress(self, view: DatasetView, rng: np.random.Generator) -> dict[int, np.ndarray]:
        prog = self.context.progress_model.episode_progress(view)
        return {i: np.clip(np.asarray(prog[i], dtype=np.float64), 0.0, 1.0) for i in view.episode_indices}
