"""Command builder for `lerobot-train` (lerobot 0.6.1). Builds argv only; never executes anything.

Flags emitted and where they are defined (TrainPipelineConfig, configs/train.py unless noted):
  --dataset.repo_id / --dataset.root   DatasetConfig, configs/default.py:31,37
  --policy.type=<act>                  PreTrainedConfig choice registry; ACTConfig registered as "act"
                                       (policies/act/configuration_act.py:22)
  --policy.push_to_hub=false           configs/policies.py:70 (default True -> would need a repo_id)
  --policy.device=cuda                 configs/policies.py:62
  --seed                               :96
  --steps                              :111   (fixed gradient-step count, D-004)
  --batch_size                         :102
  --save_freq                          :123   (<= 0 disables periodic saves; the final step is ALWAYS saved,
                                               common/train_utils.py:55-63 -> D-003 final checkpoint)
  --save_checkpoint=true               :120
  --env_eval_freq=0                    :113   (no in-training env eval; no --env is passed at all, so eval seeds
                                               are never touched during training, scripts/lerobot_train.py:629,715)
  --output_dir                         :86    (must not exist unless resuming, configs/train.py:259)
  --job_name                           :87
  --wandb.enable=false                 WandBConfig, configs/default.py:82
Weighting track ONLY (D-020; condition kind weighting / weight_placebo):
  --sample_weighting.type=rabc         TrainPipelineConfig.sample_weighting: SampleWeightingConfig | None
                                       (configs/train.py:139; utils/sample_weighting.py:76-100), built by
                                       make_sample_weighter (utils/sample_weighting.py:103-132) at
                                       scripts/lerobot_train.py:406-418; weighted loss sum(w*l)/sum(w) at :162
  --sample_weighting.progress_path     per-frame progress parquet (rewards/sarm/rabc.py:101-126)
  --sample_weighting.head_mode=sparse  reads column progress_sparse (rabc.py:101); kappa/epsilon stay at
                                       LeRobot's defaults 0.01 / 1e-6 (sample_weighting.py:97-98)
Non-weighting conditions emit exactly the pre-weighting command (golden test).
Final checkpoint: <output_dir>/checkpoints/<step:0Nd>/pretrained_model, N = max(6, len(str(steps)))
(common/train_utils.py:44-52, utils/constants.py:47-49).

The training dataset is a MATERIALIZED subset (data/subset.py) so normalization stats come from the subset
only (D-013); --dataset.episodes is deliberately NOT used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robocurate.config import ConfigError, TaskConfig, is_tbd
from robocurate.runs.spec import RunSpec

TRAIN_BIN = "lerobot-train"


@dataclass(frozen=True)
class SubsetDataset:
    repo_id: str
    root: Path


def _num(task: TaskConfig, name: str):
    v = task.raw(name)
    if is_tbd(v):
        raise ConfigError(f"task.{name} is TBD; cannot build a training command (measure it first, D-014)")
    return v


def build_train_cmd(
    task_cfg: TaskConfig,
    spec: RunSpec,
    subset: SubsetDataset,
    out_dir: str | Path,
    *,
    save_freq: int | None = None,
    steps: int | None = None,
    device: str = "cuda",
    sample_weighting_progress: str | Path | None = None,
) -> list[str]:
    """`steps` overrides task.steps (calibration configs carry their own); `save_freq` defaults to steps.

    `sample_weighting_progress` is required for weighting-track conditions and forbidden for all others."""
    weighting = spec.condition.kind in ("weighting", "weight_placebo")
    if weighting != (sample_weighting_progress is not None):
        raise ConfigError("sample weighting is set iff the condition is weighting-track (D-020)")
    steps = int(_num(task_cfg, "steps")) if steps is None else int(steps)
    batch_size = int(_num(task_cfg, "batch_size"))
    if save_freq is None:
        cfg_sf = task_cfg.raw("save_freq")
        save_freq = steps if cfg_sf is None or is_tbd(cfg_sf) else int(cfg_sf)
    if steps <= 0:
        raise ConfigError("steps must be > 0")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; lerobot-train refuses existing output dirs (train.py:259)")
    cmd = [
        TRAIN_BIN,
        f"--dataset.repo_id={subset.repo_id}",
        f"--dataset.root={subset.root}",
        f"--policy.type={task_cfg.policy_type}",
        "--policy.push_to_hub=false",
        f"--policy.device={device}",
        f"--seed={spec.train_seed}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--env_eval_freq=0",
        f"--output_dir={out_dir}",
        f"--job_name={out_dir.name}",
        "--wandb.enable=false",
    ]
    if weighting:
        cmd += ["--sample_weighting.type=rabc", f"--sample_weighting.progress_path={sample_weighting_progress}",
                "--sample_weighting.head_mode=sparse"]
    return cmd


def step_identifier(step: int, total_steps: int) -> str:
    """Mirror of lerobot.common.train_utils.get_step_identifier (common/train_utils.py:44-46)."""
    return f"{step:0{max(6, len(str(total_steps)))}d}"


def checkpoint_dir(out_dir: str | Path, step: int, total_steps: int) -> Path:
    """<out_dir>/checkpoints/<step>/pretrained_model -- the only checkpoint results may use is step == steps."""
    return Path(out_dir) / "checkpoints" / step_identifier(step, total_steps) / "pretrained_model"


def final_checkpoint(out_dir: str | Path, total_steps: int) -> Path:
    return checkpoint_dir(out_dir, total_steps, total_steps)
