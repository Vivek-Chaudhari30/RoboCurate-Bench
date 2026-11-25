"""GPU-phase mixture sources (D-031): policy rollouts from full-data checkpoints and replay-mode scripted episodes,
on CURATION seeds only (top of curation_seed_range, offset per deferred source; data/mixture.py).

  --backend fake   dry run: a stub env and stub policies exercise seeds, offsets, labels and the LeRobot writer.
  --backend local  GPU host. Loads each pretrained_model dir with load_pretrained_policy and records the env's
                   camera features. Still refuses when the recipe has a scripted_replay source: no scripted
                   transfer-cube policy ships in gym-aloha 0.1.5 / lerobot 0.6.1 (docs/runbook_gpu.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTED_MISSING = "a scripted transfer-cube policy (none in gym-aloha 0.1.5 or lerobot 0.6.1)"


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


def local_missing(recipe: Any) -> list[str]:
    """Pieces that still block --backend local. The checkpoint->camera adapter is not one of them."""
    if any(x["source"] == "scripted_replay" for x in recipe.deferred_sources):
        return [_SCRIPTED_MISSING]
    return []


def load_pretrained_policy(pretrained_dir: str | Path, *, env_cfg: Any, device: str = "cpu"):
    """Load a LeRobot pretrained_model directory into an obs -> action callable.

    ``obs`` is one unbatched gym observation (``agent_pos`` / ``pixels``). The callable returns one
    float32 action. Chunked policies expose ``reset()``; call it at the start of each episode.

    Touchpoints (lerobot 0.6.1), the same ones lerobot-eval uses (scripts/lerobot_eval.py:289-303, :762-783):
      PreTrainedConfig.from_pretrained (configs/policies.py:172)
      make_policy (policies/factory.py:240) — needs env_cfg or dataset metadata
      make_pre_post_processors (policies/factory.py:150)
      make_env_pre_post_processors (envs/factory.py:37)
      preprocess_observation (envs/utils.py:68)
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    path = Path(pretrained_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a pretrained_model directory")
    if device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("refusing CUDA: CUDA_VISIBLE_DEVICES is empty")
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = str(path)
    cfg.device = device
    policy = make_policy(cfg=cfg, env_cfg=env_cfg)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)

    def act(obs: Any) -> np.ndarray:
        if not isinstance(obs, dict) or ("pixels" not in obs and "agent_pos" not in obs):
            raise TypeError("load_pretrained_policy expects a gym observation with 'pixels' and/or 'agent_pos'")
        raw = {
            k: ({kk: np.asarray(vv) for kk, vv in v.items()} if isinstance(v, dict) else np.asarray(v))
            for k, v in obs.items()
        }
        batch = preprocess_observation(raw)
        if "task" not in batch:
            batch["task"] = [""]
        batch = env_pre(batch)
        batch = pre(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = post(action)
        action = env_post({ACTION: action})[ACTION]
        if hasattr(action, "detach"):
            action = action.detach().to("cpu").numpy()
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def reset() -> None:
        policy.reset()

    act.reset = reset  # type: ignore[attr-defined]
    return act


def _unbatch(value: Any) -> Any:
    """Drop the leading env-batch axis that SyncVectorEnv adds (n_envs=1)."""
    if isinstance(value, dict):
        return {k: _unbatch(v) for k, v in value.items()}
    arr = np.asarray(value)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return value


def _success_flag(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    final = info.get("final_info")
    if isinstance(final, dict) and "is_success" in final:
        return bool(np.asarray(final["is_success"]).reshape(-1)[0])
    if isinstance(final, (list, tuple)) and final and isinstance(final[0], dict):
        return bool(final[0].get("is_success", False))
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


class _LeRobotSingleEnv:
    """n_envs=1 LeRobot sim (envs/factory.py:58 make_env) with the reset/step API collect_policy_episodes uses."""

    def __init__(self, env_cfg: Any):
        from lerobot.envs.factory import make_env

        envs = make_env(env_cfg, n_envs=1)
        self.vec = next(iter(next(iter(envs.values())).values()))
        self.max_steps = int(getattr(env_cfg, "episode_length", None) or 400)
        self.fps = int(env_cfg.fps)

    def reset(self, seed: int | None = None):
        obs, info = self.vec.reset(seed=None if seed is None else [int(seed)])
        return _unbatch(obs), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        obs, reward, terminated, truncated, info = self.vec.step(a)
        return (
            _unbatch(obs),
            float(np.asarray(reward).reshape(-1)[0]),
            bool(np.asarray(terminated).reshape(-1)[0]),
            bool(np.asarray(truncated).reshape(-1)[0]),
            {"is_success": _success_flag(info)},
        )

    def close(self) -> None:
        self.vec.close()


def rollout_env_config(env_type: str, env_task: str | None):
    """Build the LeRobot env config for a local rollout.

    ``make_env_config`` (envs/factory.py:26) returns ``cls(**kwargs)``. Passing ``task=None``
    overrides the dataclass default — AlohaEnv.task is ``AlohaInsertion-v0`` (envs/configs.py:150) —
    so a run that only passed ``--env-type`` would build the wrong task. Omit ``task`` unless the
    caller set ``--env-task``.
    """
    from lerobot.envs.factory import make_env_config

    if env_task is None:
        return make_env_config(env_type)
    return make_env_config(env_type, task=env_task)


def generate(recipe_path: str, out_dir: Path, backend: str, m0_run_dir: str | None, steps: int | None,
             protocol_path: str, env_type: str | None = None, env_task: str | None = None,
             device: str = "cpu") -> dict[str, Any]:
    from robocurate.config import load_protocol
    from robocurate.data import mixture as M

    recipe = M.Recipe.load(recipe_path)
    proto = load_protocol(protocol_path)
    sim: _LeRobotSingleEnv | None = None
    if backend == "local":
        missing = local_missing(recipe)
        if missing:
            raise SystemExit("gpu_phase_rollouts: local backend not built yet. Missing: " + "; ".join(missing))
        if not env_type:
            raise SystemExit(
                "gpu_phase_rollouts: --env-type is required to load a LeRobot checkpoint "
                "(make_policy needs the sim env; policies/factory.py:240)"
            )
        if not m0_run_dir or not steps:
            raise SystemExit("gpu_phase_rollouts: --m0-run-dir and --steps are required for local policy rollouts")

        fracs = [
            float(c["train_fraction"])
            for spec in recipe.deferred_sources if spec["source"] == "policy"
            for c in spec["checkpoints"]
        ]
        for ckpt in checkpoint_paths(Path(m0_run_dir), int(steps), fracs).values():
            if not ckpt.is_dir():
                raise FileNotFoundError(f"{ckpt} is not a pretrained_model directory")
        env_cfg = rollout_env_config(env_type, env_task)
        sim = _LeRobotSingleEnv(env_cfg)
        env_factory = lambda sim=sim: sim  # noqa: E731
        load_local = lambda src, env_cfg=env_cfg: load_pretrained_policy(  # noqa: E731
            src.checkpoint, env_cfg=env_cfg, device=device)
        fps, task_name, max_steps = sim.fps, (env_cfg.task or env_type), sim.max_steps
    else:
        env_factory = _StubEnv
        load_local = None
        fps, task_name, max_steps = 10, "fake", 40
    report: dict[str, Any] = {"recipe": recipe.name, "recipe_sha256": recipe.sha256, "sources": {}}
    try:
        for spec in recipe.deferred_sources:
            off = M.deferred_seed_offset(recipe, spec["name"])
            if spec["source"] == "policy":
                fr = [float(c["train_fraction"]) for c in spec["checkpoints"]]
                paths = checkpoint_paths(Path(m0_run_dir), int(steps), fr) if m0_run_dir and steps else {}
                srcs = [M.PolicySource(
                    f"ckpt{int(c['train_fraction'] * 100)}",
                    str(paths.get(float(c["train_fraction"]), f"fake://ckpt{c['train_fraction']}")),
                    int(round(float(c["train_fraction"]) * (steps or 1000))), int(c["n_episodes"]))
                    for c in spec["checkpoints"]]
                load = load_local if load_local is not None else (
                    lambda src: _stub_policy(src.checkpoint_step / (steps or 1000)))  # noqa: E731
                label = "policy"
            else:
                srcs = M.replay_policy_sources(spec, proto, off)
                load = lambda src: _stub_policy(1.0)  # noqa: E731
                label = "scripted"
            root, eps = M.write_policy_episodes(
                srcs, load, env_factory, proto, max_steps=max_steps, seed_offset=off,
                repo_id=f"robocurate/{recipe.name}_{spec['name']}", root=out_dir / spec["name"],
                fps=fps, task=task_name, shuffle_seed=recipe.seed, source_label=label)
            report["sources"][spec["name"]] = {
                "root": str(root), "n": len(eps),
                "seed_range": [min(e.seed for e in eps), max(e.seed for e in eps)],
                "success_rate": float(np.mean([e.success for e in eps])), "seed_offset": off}
    finally:
        if sim is not None:
            sim.close()
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gpu_phase_rollouts")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("fake", "local"), required=True)
    ap.add_argument("--m0-run-dir", help="run dir of the M0 full-data seed-1000 run (rollout-source checkpoints)")
    ap.add_argument("--steps", type=int, help="M0 step count S")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--env-type", help="LeRobot env type for --backend local (aloha, pusht)")
    ap.add_argument(
        "--env-task",
        help="gym task id, e.g. AlohaTransferCube-v0. Omit to keep the env config's default task",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help=(
            "policy device (default: cpu). "
            "scripts/gpu_phase_rollouts.sh passes --device cuda for --backend local. "
            "CUDA_VISIBLE_DEVICES does not select the device; an empty value only refuses cuda"
        ),
    )
    a = ap.parse_args(argv)
    from robocurate.paths import require_storage_env

    require_storage_env()
    rep = generate(a.recipe, Path(a.out), a.backend, a.m0_run_dir, a.steps, a.protocol,
                   env_type=a.env_type, env_task=a.env_task, device=a.device)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
