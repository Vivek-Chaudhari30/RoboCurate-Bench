"""Task B (D-030): PushT + Diffusion Policy. CPU only.

The PushT env needs the `[pusht]` extra (gym-pusht, pymunk, diffusers, opencv-python; pyproject.toml), which is in
requirements.lock since overnight 3 (item 2), so the env test runs in the default suite."""

from pathlib import Path

import numpy as np
import pytest
from test_wrappers import _flag_ok

from robocurate.config import load_grid, load_task
from robocurate.eval.lerobot_eval import build_eval_cmd
from robocurate.paths import repo_root
from robocurate.runs.spec import Condition, RunSpec
from robocurate.train.lerobot_train import SubsetDataset, build_train_cmd, final_checkpoint

REPO = repo_root()
TASK_B = REPO / "configs/tasks/pusht_diffusion.yaml"
PUSHT_SHA = "7628202a2180972f291ba1bc6723834921e72c19"


def _task_b(tmp_path):
    y = TASK_B.read_text().replace("steps: TBD", "steps: 50000").replace("batch_size: TBD", "batch_size: 64")
    y = y.replace("eval_batch_size: TBD", "eval_batch_size: 50")
    (tmp_path / "task.yaml").write_text(y)
    return load_task(tmp_path / "task.yaml", mode="calibration")


def test_task_b_configs_load():
    t = load_task(TASK_B)
    assert (t.env_type, t.env_task, t.policy_type) == ("pusht", "PushT-v0", "diffusion")
    g = load_grid(REPO / "configs/grids/pusht_diffusion_v0.yaml")
    kinds = [c.kind for c in g.condition_specs()]
    assert g.budgets == [0.5] and "weighting" not in kinds
    assert [c.n_subsets for c in g.condition_specs() if c.kind == "random"] == [8]
    assert "cupid" not in [c.name for c in g.condition_specs()]  # disabled (D-029)


def test_task_b_grid_lists_every_enabled_selection_entrant():
    from robocurate.cleaners import registry

    g = load_grid(REPO / "configs/grids/pusht_diffusion_v0.yaml")
    listed = {c.name for c in g.condition_specs() if c.kind == "cleaner"}
    enabled = {n for n in registry.available() if registry.get(n).track == "selection"
               and registry.disabled_reason(n) is None and n != "random_subset"}  # random_subset = the placebo
    assert listed == enabled


def test_diffusion_train_and_pusht_eval_flags_exist_in_installed_configs(tmp_path):
    import lerobot.policies  # noqa: F401  registers policy config choices (diffusion)
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.train import TrainPipelineConfig

    task = _task_b(tmp_path)
    tcmd = build_train_cmd(task, RunSpec(task.name, Condition.random(), 0.5, 0, 1000),
                           SubsetDataset("robocurate/task_b_v1_train", tmp_path / "s"), tmp_path / "t",
                           num_workers=8)
    assert "--policy.type=diffusion" in tcmd and "--num_workers=8" in tcmd
    ecmd = build_eval_cmd(final_checkpoint(tmp_path / "t", 50000), 200, 300000, 50, env_type="pusht",
                          env_task="PushT-v0", output_dir=tmp_path / "e")
    bad = [f for f in tcmd[1:] if not _flag_ok(TrainPipelineConfig, f, tcmd)]
    bad += [f for f in ecmd[1:] if not _flag_ok(EvalPipelineConfig, f, ecmd, path_fields=("policy",))]
    assert bad == []
    assert not _flag_ok(TrainPipelineConfig, "--policy.chunk_size=8", tcmd)  # why RA-BC is unavailable here


def test_rabc_unavailable_for_diffusion():
    """LeRobot's RA-BC weighter reads policy.config.chunk_size (utils/sample_weighting.py:150-158)."""
    from robocurate.curation.weights import policy_chunk_size

    with pytest.raises(ValueError, match="chunk_size"):
        policy_chunk_size("diffusion")


@pytest.mark.integration
def test_pusht_metadata_loads_on_cpu(scratch_env):
    from robocurate.data.hub import episode_lengths, load_metadata

    m = load_metadata("lerobot/pusht", revision=PUSHT_SHA)
    assert m.info.codebase_version == "v3.0" and m.total_episodes == 206 and m.fps == 10
    assert {"observation.image", "observation.state", "action"} <= set(m.features)
    assert len(episode_lengths(m)) == 206


def test_pusht_env_reset_render_cpu():
    import gym_pusht  # noqa: F401  the [pusht] extra is locked; a missing install must FAIL, not skip
    from lerobot.envs.factory import make_env, make_env_config

    envs = make_env(make_env_config("pusht"), n_envs=1)
    (suite, d), = envs.items()
    venv = d[0]
    obs, _ = venv.reset(seed=[300000])
    assert suite == "pusht" and np.shape(obs["pixels"]) == (1, 96, 96, 3) and np.shape(obs["agent_pos"]) == (1, 2)
    assert np.shape(venv.envs[0].render()) == (384, 384, 3)
    venv.close()


@pytest.mark.slow
def test_task_b_fake_end_to_end(scratch_env, tmp_path, capsys):
    """prepare -> run -> report on the Task B grid with the fake backend (no GPU, no env). ~4 min: prepare runs
    every enabled entrant's real selection on the fake view (DemInf trains its VAEs)."""
    from robocurate.cli import main

    fake = tmp_path / "fake.yaml"
    effects = {"full": 0.0, "random": 0.0, "oracle": 0.25}
    g = load_grid(REPO / "configs/grids/pusht_diffusion_v0.yaml")
    effects.update({f"cleaner:{c.name}": 0.0 for c in g.condition_specs() if c.kind == "cleaner"})
    fake.write_text("base: 0.5\nsigma_seed: 0.06\ntrain_seeds: [0, 1, 2]\nsleep_s: 0\neffects:\n" +
                    "".join(f'  "{k}": {v}\n' for k, v in effects.items()))
    grid = REPO / "configs/grids/pusht_diffusion_v0.yaml"
    sel, res = tmp_path / "sel", tmp_path / "res"
    assert main(["prepare", str(grid), "--backend", "fake", "--selections-dir", str(sel)]) == 0
    assert main(["run", str(grid), "--backend", "fake", "--fake-params", str(fake), "--results-dir", str(res),
                 "--selections-dir", str(sel)]) == 0
    assert main(["report", str(res)]) == 0
    out = capsys.readouterr().out
    assert "pusht_diffusion" in out and "oracle vs placebo" in out
    runs = [p for p in Path(res).glob("*.json") if p.name != "leaderboard.json"]
    assert len(runs) == 19 * 3
