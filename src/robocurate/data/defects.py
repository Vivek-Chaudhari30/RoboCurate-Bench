"""Defect injection (D-011: defects must be physically consistent).

(a) Post-hoc injectors, applied to recorded episodes. Limited to:
    - pause:      hold frame t for k extra steps. The WHOLE frame (observations AND action) is repeated, so the
                  commanded target and the observed state stay consistent (robot holds still).
    - duplicate:  an exact copy of another episode.
    - wrong_task: the episode's task string is replaced by a wrong one.
    There is deliberately NO post-hoc action noise (rejected in D-011).

(b) Replay-mode perturbations, applied at EXECUTION time in sim: a Perturbation wraps a policy callable so the
    perturbed action is what the env actually executes, and observations follow from it. Jitter, gripper drop
    (failed grasp) and hesitation. Real sim rollouts are out of scope for this milestone; these are exercised
    with a tiny fake env in the tests.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

Frame = dict[str, Any]

# ---------------------------------------------------------------------------- (a) post-hoc


def inject_pause(frames: Sequence[Frame], t: int, k: int) -> list[Frame]:
    """Repeat frame `t` k extra times (obs + action together). Returns a new list (len + k)."""
    if not 0 <= t < len(frames):
        raise ValueError(f"pause onset {t} outside episode of length {len(frames)}")
    if k < 1:
        raise ValueError("pause length k must be >= 1")
    held = [copy.deepcopy(frames[t]) for _ in range(k)]
    return list(frames[: t + 1]) + held + list(frames[t + 1 :])


def inject_wrong_task(frames: Sequence[Frame], wrong_task: str) -> list[Frame]:
    out = []
    for f in frames:
        g = dict(f)
        if g.get("task") == wrong_task:
            raise ValueError("wrong task string equals the original")
        g["task"] = wrong_task
        out.append(g)
    return out


def duplicate_episode(frames: Sequence[Frame]) -> list[Frame]:
    return [copy.deepcopy(f) for f in frames]


# ---------------------------------------------------------------------------- (b) replay-mode

Policy = Callable[[np.ndarray], np.ndarray]


class Perturbation(ABC):
    """Execution-time perturbation. `reset` is called at episode start with a seeded Generator."""

    name: str = ""

    def reset(self, rng: np.random.Generator) -> None:
        self.t = 0

    @abstractmethod
    def apply(self, t: int, obs: np.ndarray, action: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


@dataclass
class Jitter(Perturbation):
    """Gaussian action jitter during a window [onset, onset + duration) on the given dims."""

    sigma: float
    onset: int
    duration: int
    dims: Sequence[int] | None = None
    name: str = "jitter"

    def apply(self, t, obs, action, rng):
        if not self.onset <= t < self.onset + self.duration:
            return action
        a = np.array(action, dtype=float, copy=True)
        idx = slice(None) if self.dims is None else list(self.dims)
        a[idx] = a[idx] + rng.normal(0.0, self.sigma, size=np.shape(a[idx]))
        return a


@dataclass
class GripperDrop(Perturbation):
    """Failed grasp: from `onset` for `duration` steps, force the gripper dims to `open_value`."""

    onset: int
    duration: int
    gripper_dims: Sequence[int]
    open_value: float
    name: str = "gripper_drop"

    def apply(self, t, obs, action, rng):
        if not self.onset <= t < self.onset + self.duration:
            return action
        a = np.array(action, dtype=float, copy=True)
        a[list(self.gripper_dims)] = self.open_value
        return a


@dataclass
class Hesitation(Perturbation):
    """Hold the last executed action for `duration` steps from `onset` (the policy's output is ignored)."""

    onset: int
    duration: int
    name: str = "hesitation"
    _last: np.ndarray | None = field(default=None, repr=False)

    def reset(self, rng):
        super().reset(rng)
        self._last = None

    def apply(self, t, obs, action, rng):
        if self.onset <= t < self.onset + self.duration and self._last is not None:
            return np.array(self._last, copy=True)
        self._last = np.array(action, dtype=float, copy=True)
        return action


class PerturbedPolicy:
    """Wraps a policy callable; the returned action is what the env executes (observations follow)."""

    def __init__(self, policy: Policy, perturbations: Sequence[Perturbation], seed: int):
        self.policy = policy
        self.perturbations = list(perturbations)
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        for p in self.perturbations:
            p.reset(self.rng)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = np.asarray(self.policy(obs), dtype=float)
        for p in self.perturbations:
            a = p.apply(self.t, obs, a, self.rng)
        self.t += 1
        return a


class ReplayPolicy:
    """Open-loop replay of a recorded action sequence (replay mode); holds the last action afterwards."""

    def __init__(self, actions: np.ndarray):
        self.actions = np.asarray(actions, dtype=float)
        self.t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a = self.actions[min(self.t, len(self.actions) - 1)]
        self.t += 1
        return a


def rollout(env, policy: Callable[[np.ndarray], np.ndarray], n_steps: int, seed: int) -> dict[str, np.ndarray]:
    """Minimal gymnasium-style rollout recording (obs_t, executed action_t) pairs."""
    obs, _ = env.reset(seed=seed)
    observations, actions = [], []
    for _ in range(n_steps):
        a = policy(obs)
        observations.append(np.asarray(obs, dtype=float))
        actions.append(np.asarray(a, dtype=float))
        obs, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break
    return {"observation": np.stack(observations), "action": np.stack(actions)}
