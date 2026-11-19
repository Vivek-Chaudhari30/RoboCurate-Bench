"""rcb-score v1 (ours). FROZEN per D-023: any change ships as rcb_score v2 and is reported separately.

Formula (all z-scores are across the episodes of the dataset being curated; population SD; a component with
zero SD contributes 0):

    rcb_score_v1(e) = ( z(P_e) + z(-J_e) + z(-F_e) ) / 3

  P_e  reward-model progress at the episode's final frame (Robometer progress head, clamped to [0, 1];
       context.progress_model). [The m1 proposal named "end-of-episode success probability"; lerobot 0.6.1's
       Robometer.compute_reward exposes only continuous progress or a 0.5-THRESHOLDED success
       (modeling_robometer.py:271-283), so v1 uses final-frame progress.]
  J_e  mean squared action jerk: mean over t and action dims of (a[t+3] - 3a[t+2] + 3a[t+1] - a[t])^2 * fps^6
       (third finite difference); episodes with < 4 frames get J = 0.
  F_e  pause fraction: fraction of transitions with ||s[t+1] - s[t]||_2 < tau, where
       tau = 0.05 * median over ALL transitions of ALL episodes of ||s[t+1] - s[t]||_2.
Keep the top-k by rcb_score_v1; ties per D-021.
"""

from __future__ import annotations

import numpy as np

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData

FORMULA_V1 = "(z(final_progress) + z(-mean_sq_jerk) + z(-pause_fraction)) / 3; tau = 0.05 * median ||ds||"
PAUSE_TAU_FRAC = 0.05


def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return np.zeros_like(x) if sd == 0 else (x - x.mean()) / sd


def mean_sq_jerk(actions: np.ndarray, fps: int) -> float:
    a = np.asarray(actions, dtype=np.float64)
    if len(a) < 4:
        return 0.0
    j = a[3:] - 3 * a[2:-1] + 3 * a[1:-2] - a[:-3]
    return float(np.mean(j**2) * float(fps) ** 6)


def components(view: DatasetView, progress: dict[int, np.ndarray]) -> dict[str, dict[int, float]]:
    ids = view.episode_indices
    steps = {i: np.linalg.norm(np.diff(np.asarray(view.episode(i).state, dtype=np.float64), axis=0), axis=1)
             for i in ids}
    allsteps = np.concatenate([s for s in steps.values() if len(s)]) if ids else np.zeros(0)
    tau = PAUSE_TAU_FRAC * float(np.median(allsteps)) if allsteps.size else 0.0
    return {
        "final_progress": {i: float(progress[i][-1]) for i in ids},
        "mean_sq_jerk": {i: mean_sq_jerk(view.episode(i).actions, view.fps) for i in ids},
        "pause_fraction": {i: float(np.mean(steps[i] < tau)) if len(steps[i]) else 0.0 for i in ids},
    }


def rcb_score_v1(view: DatasetView, progress: dict[int, np.ndarray]) -> dict[int, float]:
    c = components(view, progress)
    ids = view.episode_indices
    P = np.array([c["final_progress"][i] for i in ids])
    J = np.array([c["mean_sq_jerk"][i] for i in ids])
    F = np.array([c["pause_fraction"][i] for i in ids])
    s = (_z(P) + _z(-J) + _z(-F)) / 3.0
    return {i: float(v) for i, v in zip(ids, s, strict=True)}


class RCBScore(ScoreCleaner):
    name, version = "rcb_score", "1"
    requires_gpu = True  # Robometer progress
    implementation = "original"
    source = "RoboCurate-Bench rcb-score v1 (D-023), formula in robocurate/cleaners/entrants/rcb_score.py"

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        self._s = rcb_score_v1(view, self.context.progress_model.episode_progress(view))
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._s[episode.episode_index]
