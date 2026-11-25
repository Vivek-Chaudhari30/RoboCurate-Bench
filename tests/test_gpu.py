"""GPU tests run by the launch kit's stage c (scripts/m0_pipeline.sh) on hokkaido. Never run on the Mac: they are
marked `gpu` (deselected by the default suite) and skip without CUDA or without the kit's environment variables.

  RCB_GPU_TEST_CKPT     the smoke checkpoint (pretrained_model dir)
  RCB_GPU_TEST_DATASET  the materialized smoke dataset root
  RCB_GPU_TEST_CAL      the calibration config (task, protocol, dataset)
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def _need():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        pytest.skip("no CUDA (GPU tests run only inside the launch kit on the GPU host)")
    keys = ("RCB_GPU_TEST_CKPT", "RCB_GPU_TEST_DATASET", "RCB_GPU_TEST_CAL")
    if not all(os.environ.get(k) for k in keys):
        pytest.skip(f"set {keys} (the launch kit's stage c does)")
    from robocurate.config import load_calibration, load_protocol, load_task, resolve_config_path

    cal = load_calibration(os.environ["RCB_GPU_TEST_CAL"], mode="calibration")
    return (cal, load_task(resolve_config_path(cal.task), mode="calibration"),
            load_protocol(resolve_config_path(cal.protocol), mode="calibration"))


def _exe(name):
    from robocurate.launch.m0 import console_script

    return console_script(name)


def _train_losses(task, cal, out: Path) -> list[float]:
    from robocurate.launch.m0 import default_batch_size
    from robocurate.runs.spec import Condition, RunSpec
    from robocurate.train.lerobot_train import SubsetDataset, build_train_cmd

    spec = RunSpec(task.name, Condition.full(), 1.0, None, int(cal.train_seeds[0]))
    cmd = build_train_cmd(task, spec, SubsetDataset(f"{cal.dataset}_train", Path(os.environ["RCB_GPU_TEST_DATASET"])),
                          out, steps=50, batch_size=default_batch_size(), num_workers=0)
    cmd = [_exe(cmd[0]), *cmd[1:], "--log_freq=10"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    import re

    return [float(x) for x in re.findall(r"loss:([0-9.]+)", p.stdout + p.stderr)]


def test_training_is_seed_deterministic(tmp_path):
    """Same seed, same data -> the same loss trajectory (to 1e-3 relative; cuDNN may reorder reductions)."""
    cal, task, _ = _need()
    a = _train_losses(task, cal, tmp_path / "a")
    b = _train_losses(task, cal, tmp_path / "b")
    assert a and len(a) == len(b)
    assert all(abs(x - y) <= 1e-3 * max(abs(x), 1e-6) for x, y in zip(a, b, strict=True)), (a, b)


def _eval(task, proto, batch, out):
    from robocurate.eval.lerobot_eval import build_eval_cmd, parse_eval_info

    cmd = build_eval_cmd(Path(os.environ["RCB_GPU_TEST_CKPT"]), 25, int(proto.calibration_seed_start), batch,
                         env_type=task.env_type, env_task=task.env_task, output_dir=out)
    p = subprocess.run([_exe(cmd[0]), *cmd[1:]], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    return parse_eval_info(out, 25)


def test_eval_is_batch_invariant(tmp_path):
    """D-017 (ACT tasks): per-episode outcomes must not depend on the eval batch size (env seed = --seed + i)."""
    cal, task, proto = _need()
    if task.policy_type == "diffusion":
        pytest.skip("D-035: diffusion samples noise at eval; batch invariance is not required (fixed batch instead)")
    got = {b: _eval(task, proto, b, tmp_path / f"e{b}") for b in (5, 25)}
    assert got[5] == got[25], json.dumps(got)


def test_eval_is_deterministic_at_the_protocol_batch(tmp_path):
    """D-035 (diffusion tasks): two evals at the protocol-fixed eval batch size give identical per-episode
    outcomes (protocol.yaml eval_batch_size.<task>, written by launch-kit stage d)."""
    cal, task, proto = _need()
    if task.policy_type != "diffusion":
        pytest.skip("ACT tasks are covered by test_eval_is_batch_invariant")
    b = proto.task_eval_batch_size(task.name)
    from robocurate.config import is_tbd

    assert b is not None and not is_tbd(b), "stage d must write eval_batch_size.<task> before stage c checks it"
    a1, a2 = _eval(task, proto, int(b), tmp_path / "r1"), _eval(task, proto, int(b), tmp_path / "r2")
    assert a1 == a2, json.dumps({"run1": a1, "run2": a2})

