"""2D point-mass reach. obs = [x, y, gx, gy]; action = 2D velocity in [-1, 1]^2; x += DT * a.
Success when |x - g| < TOL. gymnasium-style API without the gymnasium dependency."""

from __future__ import annotations

import numpy as np

DT, TOL, MAX_STEPS = 0.1, 0.08, 60


class PointReachEnv:
    def __init__(self, max_steps: int = MAX_STEPS):
        self.max_steps = max_steps
        self.x = np.zeros(2)
        self.g = np.zeros(2)
        self.t = 0

    def reset(self, seed: int | None = None):
        r = np.random.default_rng(seed)
        self.x = r.uniform(-1, 1, 2)
        ang = r.uniform(0, 2 * np.pi)
        self.g = self.x + r.uniform(0.6, 1.0) * np.array([np.cos(ang), np.sin(ang)])
        self.t = 0
        return self.obs(), {}

    def obs(self) -> np.ndarray:
        return np.concatenate([self.x, self.g]).astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(2), -1, 1)
        self.x = self.x + DT * a
        self.t += 1
        d = float(np.linalg.norm(self.x - self.g))
        success = d < TOL
        return self.obs(), -d, success, self.t >= self.max_steps, {"is_success": success}


def expert_action(obs: np.ndarray, gain: float = 2.0) -> np.ndarray:
    return np.clip(gain * (obs[2:4] - obs[0:2]), -1, 1).astype(np.float32)


def goal_distance_progress(states: np.ndarray) -> np.ndarray:
    """1 - d_t / d_0 clipped to [0, 1]: the 'reward model' used by mock-based toy sanity tests."""
    d = np.linalg.norm(states[:, 0:2] - states[:, 2:4], axis=1)
    return np.clip(1.0 - d / max(d[0], 1e-6), 0.0, 1.0)
