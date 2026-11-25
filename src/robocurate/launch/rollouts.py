"""GPU-phase policy-rollout mixture source (D-031, D-033, D-034).

For each `policy` deferred source of a recipe: the M0 full-data seed-1000 checkpoints at the source's training
fractions (D-033: 10/30/60/100% of S, kept by launch-kit stage e) are rolled out on DATA-GENERATION seeds
(protocol.generation_seed_range, offset per source, D-034) with LeRobot's own eval rollout, recording full
observations including camera frames (robocurate.sim.rollout_recorder), and written as one LeRobot dataset with a
labels sidecar (checkpoint step, success). The replay-mode scripted episodes are produced separately
(scripts/generate_scripted_replay.sh).

  --m0-run-dir DIR --steps S        the seed-1000 full-data run of M0 and its S (checkpoints at round(f * S))
  --fake-checkpoints                dry run: write tiny randomly initialized checkpoints at exactly those paths
                                    first (CPU), and cap episodes / length with --max-episodes / --episode-length
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def checkpoint_paths(m0_run_dir: Path, steps: int, fractions: list[float]) -> dict[float, Path]:
    """D-033 rollout-source checkpoints (lerobot checkpoints/<step>/pretrained_model)."""
    from robocurate.train.lerobot_train import checkpoint_dir

    return {f: checkpoint_dir(m0_run_dir / "train", int(round(f * steps)), steps) for f in fractions}


def write_fake_checkpoints(paths: list[Path], policy_type: str, env_type: str, env_task: str) -> None:
    """Tiny randomly initialized policies saved like lerobot-train saves them (common/train_utils.py:156-166)."""
    import torch
    from lerobot.envs.factory import make_env_config
    from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

    if policy_type != "act":
        raise SystemExit("fake checkpoints are implemented for ACT only")
    env_cfg = make_env_config(env_type, task=env_task)
    for i, p in enumerate(paths):
        torch.manual_seed(i)
        cfg = make_policy_config("act", dim_model=32, n_heads=2, dim_feedforward=64, n_encoder_layers=1,
                                 n_decoder_layers=1, chunk_size=4, n_action_steps=4, use_vae=False,
                                 pretrained_backbone_weights=None, device="cpu")
        pol = make_policy(cfg, env_cfg=env_cfg)
        pre, post = make_pre_post_processors(cfg, dataset_stats=None)
        pol.save_pretrained(p)
        pre.save_pretrained(p)
        post.save_pretrained(p)


def generate(recipe_path: str, task_path: str, out_dir: Path, m0_run_dir: Path, steps: int, protocol_path: str,
             device: str, fake: bool = False, max_episodes: int | None = None,
             episode_length: int | None = None) -> dict[str, Any]:
    from robocurate.config import load_protocol, load_task
    from robocurate.data import mixture as M
    from robocurate.sim import rollout_recorder as RR

    recipe = M.Recipe.load(recipe_path)
    proto = load_protocol(protocol_path)
    task = load_task(task_path)
    report: dict[str, Any] = {"recipe": recipe.name, "recipe_sha256": recipe.sha256, "sources": {}}
    for spec in recipe.deferred_sources:
        if spec["source"] != "policy":
            raise SystemExit(f"{spec['name']}: only policy sources are generated here")
        off = M.deferred_seed_offset(recipe, spec["name"])
        seeds = M.policy_rollout_seeds(proto, int(spec["n_episodes"]), off)
        fr = [float(c["train_fraction"]) for c in spec["checkpoints"]]
        paths = checkpoint_paths(m0_run_dir, steps, fr)
        if fake:
            write_fake_checkpoints([paths[f] for f in fr], task.policy_type, task.env_type, task.env_task)
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SystemExit(f"missing D-033 rollout-source checkpoints: {missing}")
        eps, k = [], 0
        for c in spec["checkpoints"]:
            n = int(c["n_episodes"])
            part = seeds[k:k + n][: max_episodes or n]
            k += n
            step = int(round(float(c["train_fraction"]) * steps))
            eps += RR.record(paths[float(c["train_fraction"])], step, task.env_type, task.env_task, part,
                             protocol=proto, episode_length=episode_length, device=device)
        root = RR.write_rollout_dataset(eps, f"robocurate/{recipe.name}_{spec['name']}", out_dir / spec["name"],
                                        fps=50 if task.env_type == "aloha" else 10, task=task.name,
                                        shuffle_seed=recipe.seed, robot_type=task.env_type)
        report["sources"][spec["name"]] = {
            "root": str(root), "n": len(eps), "seed_range": [min(e.seed for e in eps), max(e.seed for e in eps)],
            "seed_offset": off, "checkpoints": {str(f): str(p) for f, p in paths.items()},
            "success_by_step": {str(s): float(sum(e.success for e in eps if e.checkpoint_step == s)
                                              / max(1, sum(e.checkpoint_step == s for e in eps)))
                                for s in sorted({e.checkpoint_step for e in eps})}}
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu_phase_rollouts")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--task", required=True, help="task config (env, policy type)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--m0-run-dir", required=True)
    ap.add_argument("--steps", type=int, required=True, help="M0 step count S")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fake-checkpoints", action="store_true", help="dry run (CPU): write tiny random checkpoints")
    ap.add_argument("--max-episodes", type=int, help="dry run only: episodes per checkpoint")
    ap.add_argument("--episode-length", type=int, help="dry run only: steps per episode")
    a = ap.parse_args(argv)
    if (a.max_episodes or a.episode_length) and not a.fake_checkpoints:
        raise SystemExit("--max-episodes / --episode-length are for the --fake-checkpoints dry run only")
    from robocurate.paths import require_storage_env

    require_storage_env()
    if sys.platform == "darwin":
        from robocurate.sim.scripted_aloha import ensure_cpu_renderer

        ensure_cpu_renderer()
    rep = generate(a.recipe, a.task, Path(a.out), Path(a.m0_run_dir), a.steps, a.protocol,
                   "cpu" if a.fake_checkpoints else a.device, a.fake_checkpoints, a.max_episodes, a.episode_length)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
