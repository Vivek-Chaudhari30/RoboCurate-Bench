"""LeRobot command builders + the guard against invented flags."""

import copy
import dataclasses
import json
import subprocess
import typing
from pathlib import Path

import pytest

from robocurate.config import ConfigError, load_task
from robocurate.eval.lerobot_eval import (
    EvalOutputError,
    build_eval_cmd,
    episode_seeds,
    parse_eval_info,
    touched_seed_range,
)
from robocurate.runs.spec import Condition, RunSpec
from robocurate.train.lerobot_train import SubsetDataset, build_train_cmd, final_checkpoint

FIX = Path(__file__).parent / "fixtures"
TASK = """
name: aloha_transfer_cube
env_type: aloha
env_task: AlohaTransferCube-v0
policy_type: act
source_datasets: [lerobot/aloha_sim_transfer_cube_human]
train_dataset: robocurate/task_a_v0
steps: {steps}
per_run_gpu_h: TBD
batch_size: 8
eval_batch_size: 50
"""


def _task(tmp_path, steps="100000"):
    p = tmp_path / "task.yaml"
    p.write_text(TASK.format(steps=steps))
    return load_task(p, mode="calibration")


def _train_cmd(tmp_path):
    spec = RunSpec("aloha_transfer_cube", Condition.cleaner("length_heuristic", "1"), 0.5, None, 7)
    sub = SubsetDataset("robocurate/task_a_v0_train", tmp_path / "subset")
    return build_train_cmd(_task(tmp_path), spec, sub, tmp_path / "train_out")


def _eval_cmd(tmp_path):
    ck = final_checkpoint(tmp_path / "train_out", 100000)
    return build_eval_cmd(ck, 200, 100000, 50, env_type="aloha", env_task="AlohaTransferCube-v0",
                          output_dir=tmp_path / "eval_out")


# ------------------------------------------------------------------ builders


def test_train_cmd_contents(tmp_path):
    cmd = _train_cmd(tmp_path)
    assert cmd[0] == "lerobot-train"
    a = dict(x[2:].split("=", 1) for x in cmd[1:])
    assert a["seed"] == "7" and a["steps"] == "100000" and a["save_freq"] == "100000"
    assert a["wandb.enable"] == "false" and a["policy.push_to_hub"] == "false" and a["env_eval_freq"] == "0"
    assert "dataset.episodes" not in a  # D-013: subsets are materialized, never --dataset.episodes
    assert not any(k.startswith("env.") for k in a)  # no env during training -> eval seeds untouched


def test_train_cmd_refuses_tbd_and_existing_dir(tmp_path):
    spec = RunSpec("t", Condition.full(), 1.0, None, 0)
    sub = SubsetDataset("r", tmp_path)
    with pytest.raises(ConfigError, match="TBD"):
        build_train_cmd(_task(tmp_path, "TBD"), spec, sub, tmp_path / "o")
    (tmp_path / "o").mkdir()
    with pytest.raises(FileExistsError):
        build_train_cmd(_task(tmp_path), spec, sub, tmp_path / "o")


def test_final_checkpoint_layout(tmp_path):
    assert final_checkpoint("/o", 100000).as_posix() == "/o/checkpoints/100000/pretrained_model"
    assert final_checkpoint("/o", 5000).as_posix() == "/o/checkpoints/005000/pretrained_model"
    assert final_checkpoint("/o", 1234567).as_posix() == "/o/checkpoints/1234567/pretrained_model"


def test_step_identifier_matches_lerobot():
    from lerobot.common.train_utils import get_step_identifier

    from robocurate.train.lerobot_train import step_identifier

    for s, t in [(5000, 5000), (100000, 100000), (20000, 100000), (1234567, 1234567)]:
        assert step_identifier(s, t) == get_step_identifier(s, t)


def test_eval_cmd_contents(tmp_path):
    a = dict(x[2:].split("=", 1) for x in _eval_cmd(tmp_path)[1:])
    assert a["policy.path"].endswith("checkpoints/100000/pretrained_model")
    assert a["eval.n_episodes"] == "200" and a["eval.batch_size"] == "50" and a["seed"] == "100000"
    with pytest.raises(ValueError, match="random weights"):
        build_eval_cmd(tmp_path, 200, 0, 50, env_type="aloha", env_task="x", output_dir=tmp_path)
    with pytest.raises(ValueError):
        build_eval_cmd(tmp_path / "pretrained_model", 200, 0, 0, env_type="aloha", env_task="x",
                       output_dir=tmp_path)


def test_seed_derivation():
    assert episode_seeds(100000, 3) == [100000, 100001, 100002]
    assert touched_seed_range(100000, 200, 50) == (100000, 100200)
    assert touched_seed_range(100000, 200, 64) == (100000, 100256)


# ------------------------------------------------------------------ eval output parser


def test_parse_eval_info_fixture():
    assert parse_eval_info(FIX / "eval_info_lerobot061.json", 5) == [True, False, True, True, False]


def test_parse_eval_info_rejects():
    base = json.loads((FIX / "eval_info_lerobot061.json").read_text())
    with pytest.raises(EvalOutputError):
        parse_eval_info(base, 6)
    bad = copy.deepcopy(base)
    bad["overall"]["pc_success"] = 80.0
    with pytest.raises(EvalOutputError):
        parse_eval_info(bad, 5)
    bad = copy.deepcopy(base)
    bad["per_task"].append(bad["per_task"][0])
    with pytest.raises(EvalOutputError):
        parse_eval_info(bad, 5)
    with pytest.raises(EvalOutputError):
        parse_eval_info({"per_task": []}, 5)


def test_parser_matches_lerobot_aggregation():
    """Our fixture structure is what eval_policy_all returns (scripts/lerobot_eval.py:1053-1107)."""
    import lerobot.scripts.lerobot_eval as le

    assert le.ACC_KEYS == ("sum_rewards", "max_rewards", "successes", "video_paths", "predicted_video_paths")
    fx = json.loads((FIX / "eval_info_lerobot061.json").read_text())
    assert set(fx) == {"per_task", "per_group", "overall"}
    assert set(fx["per_task"][0]["metrics"]) == set(le.ACC_KEYS)


# ------------------------------------------------------------------ flag existence (anti-invention guard)


def _resolve(cls, parts: list[str], value: str) -> bool:
    """True iff dotted field path exists in the draccus config dataclass tree (choice types resolved)."""
    import draccus

    if not parts:
        return True
    name, rest = parts[0], parts[1:]
    fields = {f.name: f for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
    if name not in fields:
        if name == "type" and not rest and isinstance(cls, type) and issubclass(cls, draccus.ChoiceRegistry):
            return value in cls.get_known_choices()
        return False
    if not rest:
        return True
    hints = typing.get_type_hints(cls)
    t = hints[name]
    cands = [a for a in typing.get_args(t) if a is not type(None)] or [t]
    return any(_resolve(c, rest, value) for c in cands if isinstance(c, type))


def _flag_ok(root_cls, flag: str, cmd: list[str], path_fields=()) -> bool:
    key, value = flag[2:].split("=", 1)
    parts = key.split(".")
    if len(parts) == 2 and parts[0] in path_fields and parts[1] == "path":
        return True
    # choice subclasses: resolve against the selected subclass (policy.type / env.type in the same argv)
    if len(parts) >= 2 and parts[0] in ("policy", "env"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import EnvConfig

        base = PreTrainedConfig if parts[0] == "policy" else EnvConfig
        chosen = [x.split("=", 1)[1] for x in cmd if x.startswith(f"--{parts[0]}.type=")]
        if parts[1] == "type":
            return value in base.get_known_choices()
        if chosen:
            return _resolve(base.get_choice_class(chosen[0]), parts[1:], value)
        if parts[1] == "device" or parts[0] == "policy":
            # --policy.path given: fields resolve against the pretrained config's class, checked via base fields
            return _resolve(base, parts[1:], value)
        return False
    return _resolve(root_cls, parts, value)


def test_flags_exist_in_installed_config_dataclasses(tmp_path):
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.train import TrainPipelineConfig

    tcmd, ecmd = _train_cmd(tmp_path), _eval_cmd(tmp_path)
    bad = [f for f in tcmd[1:] if not _flag_ok(TrainPipelineConfig, f, tcmd)]
    bad += [f for f in ecmd[1:] if not _flag_ok(EvalPipelineConfig, f, ecmd, path_fields=("policy",))]
    assert bad == []


def test_flag_checker_rejects_invented_flags(tmp_path):
    from lerobot.configs.train import TrainPipelineConfig

    cmd = _train_cmd(tmp_path)
    for invented in ["--best_checkpoint=true", "--dataset.normalize_subset=true", "--policy.typo=1",
                     "--policy.type=not_a_policy", "--wandb.enabled=false"]:
        assert not _flag_ok(TrainPipelineConfig, invented, cmd + [invented]), invented


def _help(binary: str) -> str:
    import shutil

    exe = shutil.which(binary)
    assert exe, f"{binary} not on PATH"
    return subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=300).stdout


@pytest.mark.slow
def test_flags_exist_in_help_or_dataclasses(tmp_path):
    """Spec: every emitted flag appears in `--help` output OR in the installed config dataclasses."""
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.train import TrainPipelineConfig

    for binary, cmd, root, pf in [
        ("lerobot-train", _train_cmd(tmp_path), TrainPipelineConfig, ()),
        ("lerobot-eval", _eval_cmd(tmp_path), EvalPipelineConfig, ("policy",)),
    ]:
        text = _help(binary)
        assert "usage" in text.lower()
        for f in cmd[1:]:
            key = f.split("=", 1)[0]
            assert key in text or _flag_ok(root, f, cmd, pf), f"{binary}: {key} not found"


# ------------------------------------------------------------------ GPU etiquette + local backend guard


def test_gpu_busy_rule():
    from robocurate.runs.gpu import GPUProc, decide_busy, parse_compute_apps, parse_utilization

    procs = [(p, m) for p, m in parse_compute_apps("1234, 3000\n5678, 500\n")]
    assert procs == [(1234, 3000.0), (5678, 500.0)]
    assert parse_utilization("35\n") == [35.0]
    me = "vivek"
    assert decide_busy([GPUProc(1234, 3000, "alice")], [0], me)  # foreign > 2 GB
    assert decide_busy([GPUProc(1, 500, "alice")], [50], me)  # foreign + util > 10%
    assert decide_busy([GPUProc(1, 500, "alice")], [5], me) is None
    assert decide_busy([GPUProc(1, 30000, me)], [99], me) is None  # our own processes are fine
    assert decide_busy([], [0], me) is None


def test_local_backend_refuses_without_cuda_visible_devices(monkeypatch):
    from robocurate.runs.backends.local import LocalBackend, LocalBackendRefused, require_gpu_env

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(LocalBackendRefused):
        require_gpu_env()
    with pytest.raises(LocalBackendRefused):
        LocalBackend(plan=None, kind="result", dataset_repo_id="x", dataset_root="/nonexistent")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(LocalBackendRefused):
        require_gpu_env()


def test_selection_seed_stable_and_seed_independent():
    from robocurate.runs.backends.local import selection_seed

    assert selection_seed("t", "c", "1", 0.5) == selection_seed("t", "c", "1", 0.5)
    assert selection_seed("t", "c", "1", 0.5) != selection_seed("t", "c", "2", 0.5)
