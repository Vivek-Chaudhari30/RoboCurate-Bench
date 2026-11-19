import json
from pathlib import Path

import pytest
from helpers import write_configs

from robocurate import cli, results
from robocurate.paths import repo_root
from robocurate.report import report as R

REPO = repo_root()
FIX = Path(__file__).parent / "fixtures"


def test_plan_real_configs_tbd(capsys):
    assert cli.main(["plan", str(REPO / "configs/grids/task_a_v0.yaml")]) == 0
    out = capsys.readouterr().out
    assert "runs: 6 cells x TBD seeds = TBD" in out
    assert "total GPU-h: TBD" in out and "n_train_seeds" in out


def test_plan_calibration_config(capsys):
    assert cli.main(["plan", str(REPO / "configs/calibration/m0_act_transfer_cube_human.yaml")]) == 0
    out = capsys.readouterr().out
    assert "calibration:" in out and "runs: 1 cells x 3 seeds = 3" in out and "steps per run: TBD" in out


def test_real_configs_load():
    from robocurate.config import load_grid, load_protocol, load_task
    from robocurate.data.mixture import Recipe

    p = load_protocol(REPO / "configs/protocol.yaml")
    assert p.eval_seed_start == 100000 and p.n_eval_rollouts == 200 and p.random_subset_seeds == [2000, 2001, 2002]
    load_task(REPO / "configs/tasks/aloha_transfer_cube.yaml")
    load_grid(REPO / "configs/grids/task_a_v0.yaml")
    Recipe.load(REPO / "configs/mixtures/task_a_v0.yaml")


def test_result_mode_refuses_without_frozen_or_with_tbd(scratch_env, capsys):
    rc = cli.main(["run", str(REPO / "configs/grids/task_a_v0.yaml"), "--backend", "local"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "FROZEN" in err or "TBD" in err


def test_local_backend_refuses_on_cpu(scratch_env, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cal = str(REPO / "configs/calibration/m0_act_transfer_cube_human.yaml")
    assert cli.main(["run", cal, "--backend", "local", "--calibration"]) == 2
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_fake_requires_params(scratch_env, capsys):
    assert cli.main(["run", str(REPO / "configs/grids/task_a_v0.yaml"), "--backend", "fake"]) == 2


def _fake_e2e(tmp_path, scratch_env, **kw):
    cfg = write_configs(tmp_path / "cfg", **kw)
    out = tmp_path / "res"
    assert cli.main(["run", str(cfg["grid"]), "--backend", "fake", "--fake-params", str(cfg["fake"]),
                     "--results-dir", str(out)]) == 0
    return out


def test_fake_e2e_report_recovers_known_effects(tmp_path, scratch_env, capsys):
    out = _fake_e2e(tmp_path, scratch_env, oracle=0.25, length=0.0)
    assert len(list(out.glob("*.json"))) == 36
    assert cli.main(["report", str(out), "--n-boot", "2000"]) == 0
    lb = json.loads((out / "leaderboard.json").read_text())
    b = lb["tasks"][0]["budgets"][0]
    assert lb["fake"] is True
    assert {c["condition"]: c["n_runs"] for c in b["conditions"]} == {
        "full": 6, "random": 18, "oracle": 6, "cleaner:length_heuristic": 6}
    ref = {r["condition"]: r for r in b["reference_vs_placebo"]}
    assert ref["oracle"]["significant"] and ref["oracle"]["diff"] == pytest.approx(0.25, abs=0.08)
    assert not b["cleaner_vs_placebo"][0]["significant"]
    md = (out / "report.md").read_text()
    assert md.startswith(R.FAKE_BANNER) and md.rstrip().endswith(R.FAKE_BANNER)
    assert (out / "FAKE_success_by_condition.png").stat().st_size > 10_000
    assert b["issues"] == []


def test_report_refuses_mixed_kinds(tmp_path, scratch_env):
    out = _fake_e2e(tmp_path, scratch_env)
    p = next(out.glob("*.json"))
    d = json.loads(p.read_text())
    d["kind"] = "calibration"
    p.write_text(json.dumps(d))
    with pytest.raises(R.ReportError, match="mix"):
        R.write_report(out, n_boot=100)


def test_report_flags_missing_placebo_inline():
    recs = []
    from robocurate.report.stats import wilson_ci

    def rec(kind, budget, seed, rate, subset=None, name=None):
        k = int(rate * 200)
        pes = [True] * k + [False] * (200 - k)
        cond = {"kind": kind} if kind != "cleaner" else {"kind": "cleaner", "name": name, "version": "1"}
        return results.RunResult(
            run_id=f"{kind}{subset}{seed}", spec={"task": "t", "condition": cond, "budget": budget,
                                                  "subset_index": subset, "train_seed": seed},
            kind="fake", git=results.GitInfo("x", True), configs=[], lerobot_version="0.6.1",
            command_lines=[[]], seeds={"train_seed": seed}, per_episode_success=pes, n_episodes=200,
            success_rate=k / 200, wilson_95ci=list(wilson_ci(k, 200)), checkpoint_step=None, wall_clock_s=0,
            peak_gpu_mem_mb=None, hostname="h").to_dict()

    for s in range(3):
        recs.append(rec("random", 0.5, s, 0.5, subset=0))
        recs.append(rec("cleaner", 0.5, s, 0.55, name="c"))
    a = R.analyze(recs, n_boot=100)
    issues = a["tasks"][0]["budgets"][0]["issues"]
    assert any("1 random subsets" in i for i in issues) and any("full-data" in i for i in issues)


def test_submit_cli(tmp_path, capsys):
    good = str(FIX / "cleaners/good_cleaner.py")
    assert cli.main(["submit", good, "--out", str(tmp_path / "sel.json")]) == 0
    assert "RESULT: PASS" in capsys.readouterr().out
    assert json.loads((tmp_path / "sel.json").read_text())["cleaner"] == "smoothness_fixture"
    assert cli.main(["submit", str(FIX / "broken_cleaners/label_import.py"), "--out", str(tmp_path / "b.json")]) == 1
    assert not (tmp_path / "b.json").exists()
