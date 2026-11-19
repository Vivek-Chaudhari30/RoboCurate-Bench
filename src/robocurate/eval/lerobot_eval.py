"""Command builder and output parser for `lerobot-eval` (lerobot 0.6.1). Never executes anything.

FROZEN once src/robocurate/eval/FROZEN exists (CLAUDE.md). This milestone does not create FROZEN.

How lerobot-eval derives per-episode env seeds (scripts/lerobot_eval.py):
  - eval_main calls set_seed(cfg.seed) (:748) and eval_policy_all(..., start_seed=cfg.seed) (:788-802).
  - make_env(cfg.env, n_envs=cfg.eval.batch_size) (:752-757) builds a vector env with num_envs = batch_size
    (EvalConfig.batch_size; 0 -> auto from CPU count, capped by n_episodes, configs/default.py:110-126).
  - eval_policy runs ceil(n_episodes / num_envs) batches; batch `batch_ix` resets its envs with seeds
        range(start_seed + batch_ix * num_envs, start_seed + (batch_ix + 1) * num_envs)       (:528-533)
    via env.reset(seed=seeds) once at rollout start (rollout(), :223).
  - Results are truncated to the first n_episodes (:662-667).
  => Episode i (0-based) uses env seed  --seed + i,  INDEPENDENT of batch size. The last partial batch also
     runs (and discards) episodes with seeds up to --seed + ceil(n/b)*b - 1, so the protocol's eval seed range
     is [eval_seed_start, eval_seed_start + ceil(n/b)*b) (ProtocolConfig.eval_seed_range).
  - The single-task aloha env yields one task group with task_id 0 (envs/utils.py:487-493), so per-task
    ordering cannot interleave.

Output: <output_dir>/eval_info.json (:819-820), structure (eval_policy_all return value, :1053-1107):
  {"per_task": [{"task_group": str, "task_id": int,
                 "metrics": {"sum_rewards": [...], "max_rewards": [...], "successes": [bool, ...],
                             "video_paths": [...], "predicted_video_paths": [...]}}],
   "per_group": {group: {"avg_sum_reward", "avg_max_reward", "pc_success", "n_episodes", "video_paths",
                         "predicted_video_paths"}},
   "overall":   {"avg_sum_reward", "avg_max_reward", "pc_success", "n_episodes", "eval_s", "eval_ep_s",
                 "video_paths", "predicted_video_paths"}}
  Per-episode SEEDS are not written (eval_one drops them, :877-884); they follow from the rule above.

Video: lerobot 0.6.1 has NO flag to disable video. eval_main hard-codes
  max_episodes_rendered = 0 if cfg.eval.recording else 10   (:785)
so up to 10 episodes are rendered into <output_dir>/videos. `--eval.recording=true` would instead record a
dataset of every rollout, which is worse. We accept the 10 videos; output_dir is under $ROBOCURATE_SCRATCH.
Rendering does not reseed the env, so it does not change outcomes.

Flags emitted (EvalPipelineConfig, configs/eval.py unless noted):
  --policy.path        path field (configs/eval.py:77); WITHOUT it the policy is random-init with only a
                       warning (:55-58), so we always pass it
  --policy.device      configs/policies.py:62 (CLI override applied on top of the pretrained config, :48-52)
  --env.type=aloha     EnvConfig choice; AlohaEnv registered as "aloha" (envs/configs.py:147)
  --env.task           AlohaEnv.task (envs/configs.py:150); gym id gym_aloha/AlohaTransferCube-v0
  --eval.n_episodes    EvalConfig, configs/default.py:95
  --eval.batch_size    EvalConfig, configs/default.py:98
  --seed               :39
  --output_dir         :37
  --job_name           :38
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_BIN = "lerobot-eval"
EVAL_INFO = "eval_info.json"


class EvalOutputError(ValueError):
    pass


def build_eval_cmd(
    checkpoint: str | Path,
    n_episodes: int,
    seed: int,
    batch_size: int,
    *,
    env_type: str,
    env_task: str,
    output_dir: str | Path,
    device: str = "cuda",
) -> list[str]:
    checkpoint = Path(checkpoint)
    if checkpoint.name != "pretrained_model":
        raise ValueError(f"{checkpoint} is not a pretrained_model dir (refusing: would risk random weights)")
    if n_episodes <= 0 or batch_size <= 0:
        raise ValueError("n_episodes and batch_size must be > 0 (batch_size=0 would auto-size from CPU count)")
    output_dir = Path(output_dir)
    return [
        EVAL_BIN,
        f"--policy.path={checkpoint}",
        f"--policy.device={device}",
        f"--env.type={env_type}",
        f"--env.task={env_task}",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size={batch_size}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        f"--job_name={output_dir.name}",
    ]


def episode_seeds(seed: int, n_episodes: int) -> list[int]:
    """Env seed of each evaluated episode (see module docstring)."""
    return [seed + i for i in range(n_episodes)]


def touched_seed_range(seed: int, n_episodes: int, batch_size: int) -> tuple[int, int]:
    """Half-open range of env seeds lerobot-eval resets with, including discarded padding episodes."""
    return (seed, seed + -(-n_episodes // batch_size) * batch_size)


def parse_eval_info(path_or_dict: str | Path | dict, n_episodes: int) -> list[bool]:
    """Per-episode success list from eval_info.json, validated against n_episodes."""
    if isinstance(path_or_dict, dict):
        info = path_or_dict
    else:
        p = Path(path_or_dict)
        if p.is_dir():
            p = p / EVAL_INFO
        info = json.loads(p.read_text())
    try:
        per_task = info["per_task"]
        overall_n = int(info["overall"]["n_episodes"])
    except (KeyError, TypeError) as e:
        raise EvalOutputError(f"unexpected eval_info structure: missing {e}") from e
    if len(per_task) != 1:
        raise EvalOutputError(f"expected exactly 1 task (aloha), got {len(per_task)}")
    succ = per_task[0]["metrics"]["successes"]
    if not all(isinstance(s, bool) for s in succ):
        raise EvalOutputError("successes must be booleans")
    if len(succ) != n_episodes or overall_n != n_episodes:
        raise EvalOutputError(f"expected {n_episodes} episodes, got {len(succ)} (overall {overall_n})")
    pc = float(info["overall"]["pc_success"])
    if abs(pc - 100.0 * sum(succ) / len(succ)) > 1e-6:
        raise EvalOutputError("overall.pc_success inconsistent with per-episode successes")
    return list(succ)
