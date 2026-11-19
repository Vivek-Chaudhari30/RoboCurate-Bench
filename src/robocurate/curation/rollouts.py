"""Rollout service for cleaners (D-007, D-022).

A cleaner never sees eval seeds or eval code. This service:
  * accepts env seeds ONLY from protocol.curation_seed_range (inclusive) and rejects anything else, eval seeds
    included;
  * enforces protocol.max_curation_rollouts (cumulative per service instance = per prepare cell);
  * never imports robocurate.eval (tested by an import-graph test). Real simulation goes through LeRobot's own
    env factory and rollout loop (`lerobot.envs.factory.make_env` envs/factory.py:58,
    `lerobot.scripts.lerobot_eval.rollout` scripts/lerobot_eval.py:167), not through robocurate's eval wrapper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class RolloutError(RuntimeError):
    pass


@dataclass
class RolloutResult:
    seed: int
    success: bool
    length: int
    states: np.ndarray | None = None
    actions: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)


class RolloutBackend(Protocol):
    def run(self, policy: Any, seeds: Sequence[int]) -> list[RolloutResult]: ...


class RolloutService:
    def __init__(self, backend: RolloutBackend, seed_range: Sequence[int], cap: int,
                 forbidden_ranges: Sequence[tuple[int, int]] = ()):
        lo, hi = int(seed_range[0]), int(seed_range[1])
        if not lo <= hi:
            raise ValueError("seed_range must be [lo, hi]")
        if not 0 <= cap <= 100:
            raise ValueError("cap must be in [0, 100] (D-007)")
        self.backend, self.lo, self.hi, self.cap = backend, lo, hi, int(cap)
        self.forbidden = [(int(a), int(b)) for a, b in forbidden_ranges]  # half-open, e.g. eval seeds
        self.used = 0
        self.seeds_used: list[int] = []

    @classmethod
    def from_protocol(cls, protocol: Any, backend: RolloutBackend, cap: int | None = None) -> RolloutService:
        """cap = min(cleaner's declared requires_rollouts, protocol.max_curation_rollouts)."""
        pcap = int(protocol.max_curation_rollouts)
        forbidden = [protocol.eval_seed_range(), *([protocol.calibration_seed_range()]
                                                   if protocol.calibration_seed_range() else [])]
        return cls(backend, protocol.curation_seed_range, pcap if cap is None else min(cap, pcap), forbidden)

    @property
    def remaining(self) -> int:
        return self.cap - self.used

    def check_seeds(self, seeds: Sequence[int]) -> list[int]:
        seeds = [int(s) for s in seeds]
        for s in seeds:
            if any(a <= s < b for a, b in self.forbidden):
                raise RolloutError(f"seed {s} is a protected (eval/calibration) seed; cleaners may not use it")
            if not self.lo <= s <= self.hi:
                raise RolloutError(f"seed {s} is outside curation_seed_range [{self.lo}, {self.hi}] (D-007)")
        if len(set(seeds)) != len(seeds) or set(seeds) & set(self.seeds_used):
            raise RolloutError("curation seeds must not repeat within a prepare cell")
        if self.used + len(seeds) > self.cap:
            raise RolloutError(f"rollout cap exceeded: {self.used} used + {len(seeds)} requested > cap {self.cap}")
        return seeds

    def rollout(self, policy: Any, seeds: Sequence[int]) -> list[RolloutResult]:
        seeds = self.check_seeds(seeds)
        out = self.backend.run(policy, seeds)
        self.used += len(seeds)
        self.seeds_used += seeds
        return out

    def next_seeds(self, n: int) -> list[int]:
        """The first n unused seeds of the curation range (deterministic)."""
        out, s = [], self.lo
        used = set(self.seeds_used)
        while len(out) < n and s <= self.hi:
            if s not in used:
                out.append(s)
            s += 1
        return out


class GymRolloutBackend:
    """Single-env gymnasium-style backend (toy testbed / fake contexts). env_factory() -> env with
    reset(seed=) -> (obs, info) and step(a) -> (obs, r, terminated, truncated, info); success = info["is_success"]."""

    def __init__(self, env_factory: Callable[[], Any], max_steps: int = 200):
        self.env_factory, self.max_steps = env_factory, max_steps

    def run(self, policy: Any, seeds: Sequence[int]) -> list[RolloutResult]:
        out = []
        for s in seeds:
            env = self.env_factory()
            obs, info = env.reset(seed=int(s))
            if hasattr(policy, "reset"):  # chunked policies clear their action queue per episode (as LeRobot's reset)
                policy.reset()
            states, actions, success = [], [], False
            for _ in range(self.max_steps):
                a = policy.act(obs) if hasattr(policy, "act") else policy(obs)
                states.append(np.asarray(obs, dtype=np.float32))
                actions.append(np.asarray(a, dtype=np.float32))
                obs, _, term, trunc, info = env.step(a)
                success = success or bool(info.get("is_success", False))
                if term or trunc:
                    break
            out.append(RolloutResult(int(s), success, len(actions), np.array(states), np.array(actions)))
        return out


class LeRobotRolloutBackend:
    """Real sim rollouts through LeRobot (GPU hosts only; never executed in the CPU test suite).

    Mirrors how lerobot-eval builds env/policy/processors (scripts/lerobot_eval.py:753-783) and how it reduces
    per-episode success (scripts/lerobot_eval.py:534-560), but with seeds supplied by RolloutService.
    """

    def __init__(self, env_type: str, env_task: str, n_envs: int = 10, device: str = "cuda"):
        self.env_type, self.env_task, self.n_envs, self.device = env_type, env_task, n_envs, device

    def run(self, policy: Any, seeds: Sequence[int]) -> list[RolloutResult]:  # pragma: no cover - GPU only
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.scripts.lerobot_eval import rollout
        from lerobot.utils.constants import OBS_STATE  # "observation.state", utils/constants.py:23

        path = str(policy)  # a checkpoint directory (pretrained_model)
        env_cfg = make_env_config(self.env_type, task=self.env_task)
        pcfg = PreTrainedConfig.from_pretrained(path)
        pcfg.pretrained_path = path
        pcfg.device = self.device
        pol = make_policy(cfg=pcfg, env_cfg=env_cfg)
        pol.eval()
        pre, post = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=path,
                                             preprocessor_overrides={"device_processor": {"device": self.device}})
        env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pcfg)
        out: list[RolloutResult] = []
        seeds = list(seeds)
        for i in range(0, len(seeds), self.n_envs):
            chunk = seeds[i:i + self.n_envs]
            envs = make_env(env_cfg, n_envs=len(chunk))
            env = next(iter(next(iter(envs.values())).values()))
            with torch.no_grad():
                d = rollout(env=env, policy=pol, env_preprocessor=env_pre, env_postprocessor=env_post,
                            preprocessor=pre, postprocessor=post, seeds=chunk, return_observations=True)
            done_idx = torch.argmax(d["done"].to(int), dim=1)
            n_steps = d["done"].shape[1]
            mask = (torch.arange(n_steps)[None, :] <= (done_idx + 1)[:, None]).int()
            succ = (d["success"] * mask).any(dim=1)
            obs_state = d["observation"].get(OBS_STATE)  # (batch, seq + 1, dim), scripts/lerobot_eval.py:189-191
            for j, s in enumerate(chunk):
                L = int(done_idx[j]) + 1
                out.append(RolloutResult(
                    int(s), bool(succ[j]), L,
                    states=None if obs_state is None else obs_state[j, :L].cpu().numpy(),
                    actions=d["action"][j, :L].cpu().numpy()))
            env.close()
        return out
