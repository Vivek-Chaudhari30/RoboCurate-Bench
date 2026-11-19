"""Local backend: real LeRobot training + sim evaluation on this machine's GPU.

NOT executed in the CPU-only build milestone. Construction refuses when CUDA_VISIBLE_DEVICES is unset or
empty, so it cannot start by accident while CPU-only rules are in force.

Per run:
  1. Selection on a label-free DatasetView (budget = round-half-up(fraction * n_episodes)):
     full -> all; random -> RandomSubset with rng seeded by protocol.random_subset_seeds[subset_index];
     cleaner -> registry cleaner with rng seeded by selection_seed(task, name, version, budget) -- the SAME
     subset for every train seed (seeds replicate training, not selection); oracle -> runs.oracle (labels).
  2. Materialize the subset with subset-only stats (data/subset.py, D-013).
  3. lerobot-train (train/lerobot_train.py), peak GPU memory polled via nvidia-smi.
  4. lerobot-eval on the FINAL checkpoint only (D-003) with protocol eval seeds; parse per-episode success.
  5. Return a results record (kind "result" or "calibration").
Cleaners with requires_rollouts > 0 are not supported yet (next milestone).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from robocurate import results
from robocurate.cleaners import registry
from robocurate.cleaners.base import budget_count
from robocurate.cleaners.builtin import RandomSubset
from robocurate.config import ConfigError, is_tbd
from robocurate.paths import require_storage_env, scratch_dir
from robocurate.report.stats import wilson_ci
from robocurate.runs import gpu
from robocurate.runs.planner import Plan
from robocurate.runs.spec import RunSpec


class LocalBackendRefused(RuntimeError):
    pass


def selection_seed(task: str, name: str, version: str, budget: float) -> int:
    h = hashlib.sha256(f"{task}|{name}|{version}|{budget}".encode()).hexdigest()
    return int(h[:8], 16)


def require_gpu_env(env: dict[str, str] | None = None) -> None:
    env = dict(os.environ) if env is None else env
    if not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise LocalBackendRefused(
            "CUDA_VISIBLE_DEVICES is empty/unset: the local backend runs GPU training and refuses to start "
            "(GPU use requires an explicit task authorization)."
        )


class LocalBackend:
    name = "local"

    def __init__(self, plan: Plan, kind: str, dataset_repo_id: str, dataset_root: str | Path | None,
                 command_line: list[str] | None = None, device: str = "cuda"):
        require_gpu_env()
        require_storage_env()
        if kind not in ("result", "calibration"):
            raise ValueError("local backend kind must be 'result' or 'calibration'")
        self.kind = kind
        self.plan = plan
        self.repo_id = dataset_repo_id
        self.root = Path(dataset_root) if dataset_root else None  # None -> HF cache (Hub dataset)
        self.device = device
        self.command_line = command_line or list(sys.argv)
        self._view = None
        steps = plan.steps
        if is_tbd(steps):
            raise ConfigError("task.steps is TBD; the local backend needs measured values (D-014)")
        self.steps = int(steps)

    def config_hashes(self) -> dict[str, str]:
        return {"protocol": self.plan.protocol.sha256, "task": self.plan.task.sha256}

    def gpu_check(self) -> None:
        gpu.check_free()

    @property
    def view(self):
        if self._view is None:
            from robocurate.data.hub import load_view

            self._view = load_view(self.repo_id, self.root)
        return self._view

    def select(self, spec: RunSpec) -> list[int]:
        v = self.view
        k = budget_count(v.n_episodes, spec.budget)
        c = spec.condition
        if c.kind == "full":
            return v.episode_indices
        if c.kind == "random":
            return RandomSubset().select(v, k, np.random.default_rng(self.plan.subset_seed(spec)))
        if c.kind == "cleaner":
            cl = registry.get(c.name, c.version)
            if cl.requires_rollouts:
                raise NotImplementedError("rollout-based cleaners are out of scope for this milestone")
            ids = cl.select(v, k, np.random.default_rng(selection_seed(spec.task, c.name, c.version, spec.budget)))
            if len(ids) != k or len(set(ids)) != k or not set(ids) <= set(v.episode_indices):
                raise RuntimeError(f"cleaner {c.name} violated the selection contract")
            return sorted(int(i) for i in ids)
        if c.kind == "oracle":
            from robocurate.data.labels import load_labels, sidecar_path
            from robocurate.runs.oracle import oracle_select

            if self.root is None:
                raise ConfigError("oracle needs a local mixture dataset root (labels sidecar)")
            return oracle_select(load_labels(sidecar_path(self.root)), k, v.episode_indices)
        raise ValueError(c.kind)

    def _run(self, cmd: list[str], log: Path) -> float | None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            with gpu.PeakMemPoller(proc.pid) as poll:
                rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{cmd[0]} exited {rc}; see {log}")
        return poll.peak_mb

    def run(self, spec: RunSpec, rid: str) -> dict[str, Any]:
        from robocurate.data.subset import materialize_subset
        from robocurate.eval.lerobot_eval import build_eval_cmd, parse_eval_info
        from robocurate.train.lerobot_train import (
            SubsetDataset,
            build_train_cmd,
            checkpoint_dir,
            final_checkpoint,
        )

        t0 = time.time()
        p = self.plan.protocol
        run_dir = scratch_dir("runs", rid)
        episodes = self.select(spec)
        sub_root = materialize_subset(self.repo_id, self.root, episodes, run_dir / "subset")
        sub = SubsetDataset(f"{self.repo_id}_train", sub_root)
        save_freq = None
        if self.plan.kind == "calibration":
            sf = self.plan.grid.raw("intermediate_save_freq")
            if is_tbd(sf):
                raise ConfigError("calibration intermediate_save_freq is TBD")
            save_freq = int(sf)
        train_cmd = build_train_cmd(self.plan.task, spec, sub, run_dir / "train", save_freq=save_freq,
                                    steps=self.steps, device=self.device)
        peak_train = self._run(train_cmd, run_dir / "train.log")
        ckpt = final_checkpoint(run_dir / "train", self.steps)
        n = p.n_eval_rollouts
        eval_cmd = build_eval_cmd(ckpt, n, p.eval_seed_start, int(self.plan.task.raw("eval_batch_size")),
                                  env_type=self.plan.task.env_type, env_task=self.plan.task.env_task,
                                  output_dir=run_dir / "eval", device=self.device)
        peak_eval = self._run(eval_cmd, run_dir / "eval.log")
        pes = parse_eval_info(run_dir / "eval", n)
        intermediate = []
        if self.plan.kind == "calibration":  # diagnostics only; results use the final checkpoint (D-003)
            n_mid = int(self.plan.grid.raw("n_intermediate_rollouts"))
            for step in range(save_freq, self.steps, save_freq):
                ck = checkpoint_dir(run_dir / "train", step, self.steps)
                out = run_dir / f"eval_step{step}"
                bs = min(n_mid, int(self.plan.task.raw("eval_batch_size")))
                cmd = build_eval_cmd(ck, n_mid, p.eval_seed_start, bs,
                                     env_type=self.plan.task.env_type, env_task=self.plan.task.env_task,
                                     output_dir=out, device=self.device)
                self._run(cmd, out.with_suffix(".log"))
                mid = parse_eval_info(out, n_mid)
                intermediate.append({"step": step, "n": n_mid, "success_rate": sum(mid) / n_mid})
        k = sum(pes)
        peaks = [x for x in (peak_train, peak_eval) if x is not None]
        return results.RunResult(
            run_id=rid,
            spec=spec.to_dict(),
            kind=self.kind,
            git=results.git_info(),
            configs=[results.ConfigRef(str(c.path), c.sha256)
                     for c in (self.plan.protocol, self.plan.task, self.plan.grid)],
            lerobot_version=results.lerobot_version(),
            command_lines=[self.command_line, train_cmd, eval_cmd],
            seeds={"train_seed": spec.train_seed, "subset_seed": self.plan.subset_seed(spec),
                   "eval_seed_start": p.eval_seed_start, "n_eval_rollouts": n},
            per_episode_success=pes,
            n_episodes=n,
            success_rate=k / n,
            wilson_95ci=list(wilson_ci(k, n)),
            checkpoint_step=self.steps,
            wall_clock_s=time.time() - t0,
            peak_gpu_mem_mb=max(peaks) if peaks else None,
            hostname=results.hostname(),
            extra={"selected_episodes": episodes, "run_dir": str(run_dir), "intermediate": intermediate},
        ).to_dict()
