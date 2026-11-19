"""Local backend: real LeRobot training + sim evaluation on this machine's GPU.

NOT executed in the CPU-only build milestone. Construction refuses when CUDA_VISIBLE_DEVICES is unset or
empty, so it cannot start by accident while CPU-only rules are in force.

Per run:
  1. Selection on a label-free DatasetView (budget = round-half-up(fraction * n_episodes)):
     full -> all; random -> RandomSubset with rng seeded by protocol.random_subset_seeds[subset_index];
     cleaner -> the selection cached by `robocurate prepare` (D-022; the cleaner is never called inside a run,
     so every train seed uses the SAME subset); weighting / weight_placebo -> all episodes, trained with
     LeRobot's RA-BC sample weighting on the prepared progress parquet (weight_placebo: the D-028 permuted
     progress parquet of its permutation index; same RA-BC code, only the progress path differs);
     oracle -> runs.oracle (labels).
  2. Materialize the subset with subset-only stats (data/subset.py, D-013).
  3. lerobot-train (train/lerobot_train.py), peak GPU memory polled via nvidia-smi.
  4. lerobot-eval on the FINAL checkpoint only (D-003) with protocol eval seeds; parse per-episode success.
  5. Return a results record (kind "result" or "calibration").
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from robocurate import results
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


from robocurate.curation.prepare import selection_seed  # noqa: E402,F401  (re-exported; defined with prepare)


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
                 command_line: list[str] | None = None, device: str = "cuda",
                 selections_dir: str | Path | None = None, replicate: int = 0):
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
        self.selections_dir = Path(selections_dir) if selections_dir else scratch_dir("selections", create=False)
        self.replicate = replicate
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

            self._view = load_view(self.repo_id, self.root, self.plan.protocol.tie_break_salt or "")
        return self._view

    def select(self, spec: RunSpec) -> list[int]:
        v = self.view
        k = budget_count(v.n_episodes, spec.budget)
        c = spec.condition
        if c.kind in ("full", "weighting", "weight_placebo"):
            return v.episode_indices
        if c.kind == "random":
            return RandomSubset().select(v, k, np.random.default_rng(self.plan.subset_seed(spec)))
        if c.kind == "cleaner":  # D-022: consume the prepared selection only
            rec = self.record(spec)
            ids = rec["selected_episodes"]
            if len(ids) != k:
                raise RuntimeError(f"prepared selection for {c.name} has {len(ids)} ids, budget is {k}")
            return sorted(int(i) for i in ids)
        if c.kind == "oracle":
            from robocurate.data.labels import load_labels, sidecar_path
            from robocurate.runs.oracle import oracle_select

            if self.root is None:
                raise ConfigError("oracle needs a local mixture dataset root (labels sidecar)")
            return oracle_select(load_labels(sidecar_path(self.root)), k, v.episode_indices)
        raise ValueError(c.kind)

    def record(self, spec: RunSpec) -> dict:
        from robocurate.curation.prepare import load_record

        c = spec.condition
        kind = "weighting" if c.kind in ("weighting", "weight_placebo") else c.kind
        return load_record(self.selections_dir, spec.task, kind, c.name, c.version, spec.budget, self.view,
                           self.replicate)

    def progress_parquet(self, spec: RunSpec) -> Path | None:
        """RA-BC progress parquet: the entrant's for `weighting`, permutation `subset_index` of the D-028 placebo
        for `weight_placebo` (same schema, rewards/sarm/rabc.py:101-126), None otherwise."""
        c = spec.condition
        if c.kind == "weighting":
            return Path(self.record(spec)["progress_parquet"])
        if c.kind == "weight_placebo":
            perms = self.record(spec)["placebo_progress_parquets"]
            key = str(spec.subset_index)
            if key not in perms:
                raise RuntimeError(f"prepared record has no D-028 placebo permutation {key} (has {sorted(perms)})")
            return Path(perms[key])
        return None

    def eval_seed_start(self) -> int:
        """Env-seed start for this plan's evals: protocol eval seeds for result runs, calibration_seed_start for
        calibration runs (D-017, CLAUDE.md rule 10). Calibration never touches eval seeds."""
        p = self.plan.protocol
        if self.plan.kind == "calibration":
            if p.calibration_seed_start is None:
                raise ConfigError("calibration runs need protocol.calibration_seed_start (D-017)")
            return int(p.calibration_seed_start)
        return int(p.eval_seed_start)

    def num_workers(self) -> int | None:
        """D-027: DataLoader workers, the only allowed I/O mitigation (speed, not results). The calibration config
        wins for calibration plans; otherwise the task config; None = LeRobot's default."""
        for cfg in ((self.plan.grid,) if self.plan.kind == "calibration" else ()) + (self.plan.task,):
            raw = getattr(cfg, "raw", None)
            v = raw("num_workers") if raw else None
            if v is not None and not is_tbd(v):
                return int(v)
        return None

    def n_final_rollouts(self) -> int:
        if self.plan.kind == "calibration":
            return int(self.plan.grid.raw("n_final_rollouts"))
        return int(self.plan.protocol.n_eval_rollouts)

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
                                    steps=self.steps, device=self.device,
                                    sample_weighting_progress=self.progress_parquet(spec),
                                    num_workers=self.num_workers())
        peak_train = self._run(train_cmd, run_dir / "train.log")
        ckpt = final_checkpoint(run_dir / "train", self.steps)
        n = self.n_final_rollouts()
        seed0 = self.eval_seed_start()
        eval_cmd = build_eval_cmd(ckpt, n, seed0, int(self.plan.task.raw("eval_batch_size")),
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
                cmd = build_eval_cmd(ck, n_mid, seed0, bs,
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
                   "eval_seed_start": seed0, "n_eval_rollouts": n,
                   "seed_block": "calibration" if self.plan.kind == "calibration" else "eval"},
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
