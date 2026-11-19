"""GPU-phase mixture sources (D-031): policy rollouts from full-data checkpoints and replay-mode scripted episodes,
on CURATION seeds only (top of curation_seed_range, offset per deferred source; data/mixture.py).

  --backend fake   dry run: a stub env and stub policies exercise seeds, offsets, labels and the LeRobot writer.
  --backend local  GPU host. Refuses today with the list of missing pieces (no adapter yet from a LeRobot
                   checkpoint to an obs -> action callable that records camera frames, no scripted transfer-cube
                   policy in gym-aloha 0.1.5 / lerobot 0.6.1). Those are GPU-phase prerequisites
                   (docs/runbook_gpu.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class _StubEnv:
    """2D integrator, success near the origin; enough to exercise the writer and labels (never a benchmark)."""

    def reset(self, seed=None):
        self.r = np.random.default_rng(seed)
        self.x = self.r.uniform(-1, 1, 2).astype(np.float32)
        self.t = 0
        return self.x.copy(), {}

    def step(self, a):
        self.x = (self.x + 0.1 * np.clip(np.asarray(a, dtype=np.float32), -1, 1)).astype(np.float32)
        self.t += 1
        ok = bool(np.linalg.norm(self.x) < 0.1)
        return self.x.copy(), 0.0, ok, self.t >= 40, {"is_success": ok}


def _stub_policy(fraction: float):
    gain = 2.0 * fraction  # later checkpoints are better
    return lambda obs: np.clip(-gain * np.asarray(obs), -1, 1).astype(np.float32)


def checkpoint_paths(m0_run_dir: Path, steps: int, fractions: list[float]) -> dict[float, Path]:
    """pretrained_model dirs of the rollout-source run (lerobot checkpoints/<step:06d>/pretrained_model)."""
    from robocurate.train.lerobot_train import checkpoint_dir

    return {f: checkpoint_dir(m0_run_dir / "train", int(round(f * steps)), steps) for f in fractions}


def generate(recipe_path: str, out_dir: Path, backend: str, m0_run_dir: str | None, steps: int | None,
             protocol_path: str) -> dict[str, Any]:
    from robocurate.config import load_protocol
    from robocurate.data import mixture as M

    recipe = M.Recipe.load(recipe_path)
    proto = load_protocol(protocol_path)
    if backend == "local":
        missing = ["LeRobot checkpoint -> obs/action adapter recording camera frames (rollout writer stores state "
                   "and action only; ACT/Diffusion mixtures need images)"]
        if any(x["source"] == "scripted_replay" for x in recipe.deferred_sources):
            missing.append("a scripted transfer-cube policy (none in gym-aloha 0.1.5 or lerobot 0.6.1)")
        raise SystemExit("gpu_phase_rollouts: local backend not built yet. Missing: " + "; ".join(missing))
    report: dict[str, Any] = {"recipe": recipe.name, "recipe_sha256": recipe.sha256, "sources": {}}
    for spec in recipe.deferred_sources:
        off = M.deferred_seed_offset(recipe, spec["name"])
        if spec["source"] == "policy":
            fr = [float(c["train_fraction"]) for c in spec["checkpoints"]]
            paths = checkpoint_paths(Path(m0_run_dir), int(steps), fr) if m0_run_dir and steps else {}
            srcs = [M.PolicySource(f"ckpt{int(c['train_fraction'] * 100)}", str(paths.get(float(c["train_fraction"]),
                                   f"fake://ckpt{c['train_fraction']}")),
                                   int(round(float(c["train_fraction"]) * (steps or 1000))), int(c["n_episodes"]))
                    for c in spec["checkpoints"]]
            load = lambda src: _stub_policy(src.checkpoint_step / (steps or 1000))  # noqa: E731
            label = "policy"
        else:
            srcs = M.replay_policy_sources(spec, proto, off)
            load = lambda src: _stub_policy(1.0)  # noqa: E731
            label = "scripted"
        eps = M.collect_policy_episodes(srcs, load, _StubEnv, proto, max_steps=40, seed_offset=off)
        root = M.write_policy_dataset(eps, f"robocurate/{recipe.name}_{spec['name']}", out_dir / spec["name"],
                                      fps=10, task="fake", shuffle_seed=recipe.seed, source_label=label)
        report["sources"][spec["name"]] = {
            "root": str(root), "n": len(eps), "seed_range": [min(e.seed for e in eps), max(e.seed for e in eps)],
            "success_rate": float(np.mean([e.success for e in eps])), "seed_offset": off}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu_phase_rollouts")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("fake", "local"), required=True)
    ap.add_argument("--m0-run-dir", help="run dir of the M0 full-data seed-1000 run (rollout-source checkpoints)")
    ap.add_argument("--steps", type=int, help="M0 step count S")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    a = ap.parse_args(argv)
    from robocurate.paths import require_storage_env

    require_storage_env()
    rep = generate(a.recipe, Path(a.out), a.backend, a.m0_run_dir, a.steps, a.protocol)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
