"""Fake backend: synthetic train + eval with KNOWN effect sizes, for verifying runner/stats/report.

    p_run   = clip(base + effect[condition label] + N(0, sigma_seed), 0, 1)
    success ~ Bernoulli(p_run), n_rollouts times

All randomness is seeded from run_id, so reruns reproduce identical records. Writes kind="fake".
No LeRobot, no data, no GPU.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from robocurate import results
from robocurate.config import ConfigError, is_tbd, load_yaml, sha256_file
from robocurate.report.stats import wilson_ci
from robocurate.runs.planner import Plan
from robocurate.runs.spec import RunSpec


@dataclass
class FakeParams:
    base: float
    sigma_seed: float
    effects: dict[str, float]
    train_seeds: list[int]
    sleep_s: float = 0.0
    path: str | None = None
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def effect(self, label: str) -> float:
        if label not in self.effects:
            raise ConfigError(f"fake params have no effect for condition {label!r}")
        return float(self.effects[label])


def load_fake_params(path: str | Path) -> FakeParams:
    d = load_yaml(path)
    allowed = {"base", "sigma_seed", "effects", "train_seeds", "sleep_s"}
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    return FakeParams(**d, path=str(path), sha256=sha256_file(path))


class FakeBackend:
    name = "fake"
    kind = "fake"

    def __init__(self, plan: Plan, params: FakeParams, command_line: list[str] | None = None):
        self.plan = plan
        self.params = params
        self.command_line = command_line or list(sys.argv)
        self.n_rollouts = plan.protocol.n_eval_rollouts
        self._git = results.git_info()

    def config_hashes(self) -> dict[str, str]:
        return {
            "protocol": self.plan.protocol.sha256,
            "task": self.plan.task.sha256,
            "fake_params": self.params.sha256,
        }

    def gpu_check(self) -> None:  # fake backend never touches a GPU
        return None

    def run(self, spec: RunSpec, rid: str) -> dict[str, Any]:
        t0 = time.time()
        rng = np.random.default_rng(int(rid, 16))
        eff = self.params.effect(spec.condition.label)
        p = float(np.clip(self.params.base + eff + rng.normal(0.0, self.params.sigma_seed), 0.0, 1.0))
        pes = [bool(x) for x in rng.random(self.n_rollouts) < p]
        if self.params.sleep_s:
            time.sleep(self.params.sleep_s)
        k, n = sum(pes), len(pes)
        steps = self.plan.task.raw("steps")
        cfgs = [
            results.ConfigRef(path=str(c.path), sha256=c.sha256)
            for c in (self.plan.protocol, self.plan.task, self.plan.grid)
        ]
        cfgs.append(results.ConfigRef(path=str(self.params.path), sha256=self.params.sha256))
        return results.RunResult(
            run_id=rid,
            spec=spec.to_dict(),
            kind="fake",
            git=self._git,
            configs=cfgs,
            lerobot_version=results.lerobot_version(),
            command_lines=[self.command_line],
            seeds={
                "train_seed": spec.train_seed,
                "subset_seed": self.plan.subset_seed(spec),
                "eval_seed_start": self.plan.protocol.eval_seed_start,
                "fake_rng_seed": int(rid, 16),
            },
            per_episode_success=pes,
            n_episodes=n,
            success_rate=k / n,
            wilson_95ci=list(wilson_ci(k, n)),
            checkpoint_step=None if is_tbd(steps) else int(steps),
            wall_clock_s=time.time() - t0,
            peak_gpu_mem_mb=None,
            hostname=results.hostname(),
            extra={"FAKE": True, "fake_true_rate": p},
        ).to_dict()
