"""Demo generator with planted defects, all EXECUTED in the env (observations match actions):

  clean   expert (proportional controller)
  pause   expert with a held pose: zero action for k steps mid-episode
  jitter  expert + Gaussian action noise (sigma 0.6) at execution time
  wrong   wrong-direction demo: the expert drives toward the goal mirrored through the start point, so the
          recorded (obs, action) pairs point AWAY from the observed goal (harmful for BC)
"""

from __future__ import annotations

import numpy as np

from robocurate.data.episodes import DatasetView
from toybed.env import PointReachEnv, expert_action

DEFECTS = ("clean", "pause", "jitter", "wrong")


def rollout_demo(seed: int, defect: str = "clean", max_steps: int = 60):
    env = PointReachEnv(max_steps=max_steps)
    obs, _ = env.reset(seed=seed)
    r = np.random.default_rng(seed + 7919)
    start = obs[:2].copy()
    pause_at, pause_k = int(r.integers(2, 5)), int(r.integers(10, 20))
    states, actions = [], []
    t, held = 0, 0
    while True:
        if defect == "wrong":
            fake = obs.copy()
            fake[2:4] = 2 * start - obs[2:4]
            a = expert_action(fake)
        elif defect == "pause" and t >= pause_at and held < pause_k:
            a = np.zeros(2, dtype=np.float32)
            held += 1
        else:
            a = expert_action(obs)
            if defect == "jitter":
                a = np.clip(a + r.normal(0, 0.6, 2), -1, 1).astype(np.float32)
        states.append(obs.copy())
        actions.append(a)
        obs, _, term, trunc, info = env.step(a)
        t += 1
        if (term and defect != "wrong") or trunc or (defect == "wrong" and t >= 25):
            break
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def make_dataset(n_clean: int = 30, n_pause: int = 6, n_jitter: int = 6, n_wrong: int = 8, seed: int = 0,
                 salt: str = "toybed") -> tuple[DatasetView, dict[int, str]]:
    """Shuffled episode order (D-021); returns the label-free view and the hidden defect labels."""
    kinds = ["clean"] * n_clean + ["pause"] * n_pause + ["jitter"] * n_jitter + ["wrong"] * n_wrong
    order = np.random.default_rng(seed).permutation(len(kinds))
    acts, sts, labels = [], [], {}
    for i, j in enumerate(order):
        s, a = rollout_demo(seed * 10_000 + int(j), kinds[j])
        sts.append(s)
        acts.append(a)
        labels[i] = kinds[j]
    view = DatasetView.from_arrays(acts, sts, ["reach the goal"] * len(kinds), fps=10, name="toybed",
                                   tie_break_salt=salt)
    return view, labels


def flatten(view: DatasetView, ids=None):
    ids = view.episode_indices if ids is None else ids
    S = np.concatenate([view.episode(i).state for i in ids])
    A = np.concatenate([view.episode(i).actions for i in ids])
    return S, A


def success_rate(policy, seeds) -> float:
    ok = 0
    for s in seeds:
        env = PointReachEnv()
        obs, _ = env.reset(seed=int(s))
        for _ in range(env.max_steps):
            obs, _, term, trunc, info = env.step(policy.act(obs))
            if term or trunc:
                break
        ok += bool(info["is_success"])
    return ok / len(seeds)
