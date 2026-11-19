"""GPU launch kit (scripts/m0_pipeline.sh -> robocurate.launch.m0): stage decisions and the fake dry run. CPU only."""

import json
import subprocess
from pathlib import Path

import pytest

from robocurate.launch import m0 as K
from robocurate.paths import repo_root

REPO = repo_root()


def test_startup_refuses_wrong_scratch_and_missing_env(monkeypatch):
    env = {"ROBOCURATE_SCRATCH": "/tmp/x", "HF_HOME": "/tmp/x/hf", "HF_LEROBOT_HOME": "/tmp/x/lerobot",
           "MUJOCO_GL": "egl"}
    with pytest.raises(K.KitError, match="D-027"):
        K.check_startup(env=env)
    with pytest.raises(K.KitError, match="env.sh"):
        K.check_startup(env={k: v for k, v in env.items() if k != "HF_LEROBOT_HOME"})


def test_startup_detects_env_sourced_after_import(monkeypatch, tmp_path):
    """lerobot froze HF_LEROBOT_HOME at import (constants.py:69): a different env value means env.sh came late."""
    from lerobot.utils import constants

    env = {"ROBOCURATE_SCRATCH": str(tmp_path), "HF_HOME": str(tmp_path / "hf"),
           "HF_LEROBOT_HOME": str(tmp_path / "elsewhere"), "MUJOCO_GL": "egl"}
    assert Path(constants.HF_LEROBOT_HOME).resolve() != (tmp_path / "elsewhere").resolve()
    with pytest.raises(K.KitError, match="after Python started"):
        K.check_startup(allow_other_scratch=True, env=env)


class _Busy(K.FakeProbe):
    def gpu_busy(self):
        return "someone else"


def test_stage_a_polls_every_10_min_and_gives_up_after_72h():
    slept = []
    with pytest.raises(K.KitError, match="72"):
        K.stage_a_wait(_Busy(), lambda m: None, sleep=slept.append)
    assert set(slept) == {600} and len(slept) == 72 * 6  # waits the full 72 h, then gives up
    assert K.stage_a_wait(K.FakeProbe(busy_polls=3), lambda m: None, sleep=lambda s: None) == 1800


def test_stage_b2_doubles_only_num_workers_at_most_twice():
    class AlwaysIO(K.FakeProbe):
        def smoke(self, num_workers):
            return K.SmokeResult(0.2, 0.5, 30.0, 8.0, 10.0, num_workers)

    final, hist = K.stage_b(AlwaysIO(), lambda m: None, 4)
    assert [h.num_workers for h in hist] == [4, 8, 16]
    final, hist = K.stage_b(K.FakeProbe(), lambda m: None, 4)
    assert [h.num_workers for h in hist] == [4, 8] and final.num_workers == 8
    assert not K.io_bound(K.SmokeResult(0.2, 0.9, 60.0, 8.0, 1.0, 4))  # busy GPU: not I/O-bound
    assert not K.io_bound(K.SmokeResult(0.2, 0.2, 30.0, 8.0, 1.0, 4))  # little data wait


def test_eval_batch_fastest_under_12gb():
    t = {5: (300, 3), 10: (170, 5), 25: (95, 9), 50: (80, 14)}
    assert K.choose_eval_batch(t) == 25
    with pytest.raises(K.KitError):
        K.choose_eval_batch({50: (1, 13)})


def test_steps_capped_at_2_5h_and_exit_below_25pct():
    r = K.choose_steps(100_000, step_s=0.05, eval_s_per_run=600)
    assert r["S"] == 100_000 and not r["capped"]
    r = K.choose_steps(100_000, step_s=0.12, eval_s_per_run=1200)
    assert r["capped"] and r["S"] % 10 == 0 and r["per_run_h"] <= 2.5
    with pytest.raises(K.KitError, match="25%"):
        K.choose_steps(100_000, step_s=0.5, eval_s_per_run=600)


def test_defaults_read_from_lerobot():
    assert K.default_steps() == 100_000 and K.default_batch_size() == 8 and K.default_num_workers() == 4


def test_write_calibration_values_keeps_comments_and_validates(tmp_path):
    p = tmp_path / "cal.yaml"
    src = (REPO / "configs/calibration/m0_act_transfer_cube_human.yaml").read_text()
    src = src.replace("task: configs/", f"task: {REPO}/configs/")
    src = src.replace("protocol: configs/", f"protocol: {REPO}/configs/")
    p.write_text(src)
    K.write_calibration_values(p, {"steps": 64700, "intermediate_save_freq": 6470, "eval_batch_size": 25,
                                   "batch_size": 8, "num_workers": 8})
    from robocurate.config import load_calibration

    c = load_calibration(p, mode="calibration")
    got = (c.steps, c.intermediate_save_freq, c.eval_batch_size, c.batch_size, c.num_workers)
    assert got == (64700, 6470, 25, 8, 8)
    assert "# D-027" in p.read_text() and p.read_text().count("steps:") == 1


def test_cleanup_keeps_final_and_rollout_checkpoints(tmp_path):
    from robocurate.runs.spec import Condition, RunSpec

    rd = tmp_path / "run"
    for s in (1000, 3000, 6000, 7000, 10000):
        (rd / "train" / "checkpoints" / f"{s:06d}" / "pretrained_model").mkdir(parents=True)
    (rd / "eval" / "videos").mkdir(parents=True)
    (rd / "eval" / "eval_info.json").write_text("{}")
    (rd / "subset").mkdir()
    keep = K.rollout_keep_steps(RunSpec("t", Condition.full(), 1.0, None, 1000), 10000)
    assert keep == (1000, 3000, 6000)
    assert K.rollout_keep_steps(RunSpec("t", Condition.full(), 1.0, None, 1001), 10000) == ()
    K.cleanup_run_dir(rd, 10000, keep)
    left = sorted(int(d.name) for d in (rd / "train" / "checkpoints").iterdir())
    assert left == [1000, 3000, 6000, 10000]
    assert not (rd / "eval" / "videos").exists() and not (rd / "subset").exists()
    assert (rd / "eval" / "eval_info.json").exists()


class _FlakyBackend:
    """Wraps the fake backend; the listed train seeds always fail."""

    def __init__(self, inner, fail_seeds):
        self.inner, self.fail = inner, set(fail_seeds)
        self.name, self.kind = inner.name, inner.kind

    def config_hashes(self):
        return self.inner.config_hashes()

    def gpu_check(self):
        return None

    def run(self, spec, rid):
        if spec.train_seed in self.fail:
            raise RuntimeError("boom")
        return self.inner.run(spec, rid)


def _cal_plan(tmp_path):
    from robocurate.runs.backends.fake import FakeBackend, load_fake_params
    from robocurate.runs.planner import make_calibration_plan

    plan = make_calibration_plan(str(REPO / "configs/calibration/m0_act_transfer_cube_human.yaml"), mode="fake")
    return plan, FakeBackend(plan, load_fake_params(REPO / "configs/fake/m0_dryrun.yaml"))


def test_stage_e_skips_twice_failed_runs_and_stops_after_three(tmp_path, scratch_env):
    plan, fb = _cal_plan(tmp_path)
    logs = []
    st = K.stage_e(plan, _FlakyBackend(fb, {1001}), tmp_path / "res", scratch_env, K.FakeProbe(busy_polls=0),
                   tmp_path / "P.md", K.Committer(tmp_path, False, logs.append), logs.append, sleep=lambda s: None)
    assert len(st.skipped) == 3 and st.stop_reason == "3 runs failed twice"
    assert sum(v == 2 for v in st.failures.values()) == 3 and all(v in (1, 2) for v in st.failures.values())
    assert len(st.done) >= 1 and (tmp_path / "P.md").read_text().count("| M0 ") == len(st.done)


def test_stage_e_caps_and_disk_guard(tmp_path, scratch_env):
    plan, fb = _cal_plan(tmp_path)
    st = K.stage_e(plan, fb, tmp_path / "r1", scratch_env, K.FakeProbe(busy_polls=0), tmp_path / "P.md",
                   K.Committer(tmp_path, False, print), print, sleep=lambda s: None, gpu_h_cap=0.0)
    assert st.stop_reason.startswith("GPU-h cap") and not st.done
    (scratch_env / "big").write_bytes(b"0" * 2_000_000)
    st = K.stage_e(plan, fb, tmp_path / "r2", scratch_env, K.FakeProbe(busy_polls=0), tmp_path / "P.md",
                   K.Committer(tmp_path, False, print), print, sleep=lambda s: None, disk_cap_gb=0.001)
    assert st.stop_reason.startswith("disk guard")
    ticks = iter(range(0, 10**9, 3600 * 80))  # each batch "takes" 80 h
    st = K.stage_e(plan, fb, tmp_path / "r3", scratch_env, K.FakeProbe(busy_polls=0), tmp_path / "P.md",
                   K.Committer(tmp_path, False, print), print, sleep=lambda s: None, clock=lambda: next(ticks))
    assert st.stop_reason.startswith("run-time cap") and len(st.done) == 1


def test_analysis_power_table_and_proposal(tmp_path, scratch_env):
    plan, fb = _cal_plan(tmp_path)
    K.stage_e(plan, fb, tmp_path / "res", scratch_env, K.FakeProbe(busy_polls=0), tmp_path / "P.md",
              K.Committer(tmp_path, False, print), print, sleep=lambda s: None)
    recs = [json.loads(p.read_text()) for p in (tmp_path / "res").glob("*.json")]
    a = K.analyze(recs, m_selection=9, weighting=True)
    assert len(a["power_table"]) == 12 and {(r["n"], r["K"], r["R"]) for r in a["power_table"]} == {
        (n, k, r) for n in (3, 5) for k in (8, 12) for r in (1, 2, 3)}
    assert a["power_table"][0]["runs"] == 1 * 3 * (1 + 8 + 1 + 9 + 1 + 8)
    assert "PROPOSAL ONLY" in a["proposed_design"]["rule"] and a["s_seed_ci95"][0] < a["s_seed"] < a["s_seed_ci95"][1]


def test_kit_dry_run_end_to_end(tmp_path):
    """scripts/m0_pipeline.sh --dry-run: stages a-f with the fake probe/backend, temp scratch, no commits."""
    import os
    import sys

    env = dict(os.environ, ROBOCURATE_SCRATCH=str(tmp_path / "s"), HF_HOME=str(tmp_path / "s/hf"),
               HF_LEROBOT_HOME=str(tmp_path / "s/lerobot"), MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES="",
               PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    (tmp_path / "s").mkdir()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    p = subprocess.run([str(REPO / "scripts/m0_pipeline.sh"), "--task", "b", "--dry-run", "--fake-params",
                        str(REPO / "configs/fake/m0_dryrun.yaml"), "--state-dir", str(tmp_path / "st")],
                       capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0, p.stderr[-3000:]
    assert "[e] done=11" in p.stdout and "finished" in p.stdout and "(dry run, not executed)" in p.stdout
    summary = (tmp_path / "st/m0_summary.md").read_text()
    assert "Proposed (not adopted)" in summary
    assert subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout == head
    real = subprocess.run([str(REPO / "scripts/m0_pipeline.sh"), "--task", "a"], capture_output=True, text=True,
                          env=env)
    assert real.returncode == 2 and "D-027" in real.stderr
