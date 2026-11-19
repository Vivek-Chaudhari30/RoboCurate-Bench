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
    assert "runs: 15 cells x TBD seeds = TBD" in out  # 6 selection + 1 weighting + 8 permutations
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


def test_result_mode_refuses_without_frozen_or_with_tbd(scratch_env, tmp_path, capsys):
    from helpers import PROTOCOL

    proto = PROTOCOL.replace("[2000, 2001, 2002]", str(list(range(2000, 2008))))
    cfg = write_configs(tmp_path / "cfg", protocol=proto)
    g = cfg["grid"].read_text().replace("n_subsets: 3", "n_subsets: 8")  # passes the D-019 K check
    cfg["grid"].write_text(g)
    rc = cli.main(["run", str(cfg["grid"]), "--backend", "local"])
    assert rc == 2
    err = capsys.readouterr().err
    assert ("FROZEN" in err or "TBD" in err) and "D-019" not in err


def test_result_mode_refuses_k_below_8(scratch_env, capsys):
    """D-019: result-mode runs need >= 8 random subsets; the launch grid has K = 3."""
    rc = cli.main(["run", str(REPO / "configs/grids/task_a_v0.yaml"), "--backend", "local"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "D-019" in err and "K >= 8" in err


def test_check_result_subsets_unit(tmp_path):
    from robocurate.config import ConfigError, check_result_subsets, load_grid

    def grid(k):
        f = tmp_path / f"g{k}.yaml"
        f.write_text(f"name: g\ntask: t.yaml\nprotocol: p.yaml\nbudgets: [0.5]\nconditions:\n"
                     f"  - {{kind: full}}\n  - {{kind: random, n_subsets: {k}}}\n")
        return load_grid(f)

    check_result_subsets(grid(8))
    check_result_subsets(grid(12))
    for k in (3, 4, 7):
        with pytest.raises(ConfigError, match="D-019"):
            check_result_subsets(grid(k))


def test_local_backend_refuses_on_cpu(scratch_env, monkeypatch, capsys):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cal = str(REPO / "configs/calibration/m0_act_transfer_cube_human.yaml")
    assert cli.main(["run", cal, "--backend", "local", "--calibration"]) == 2
    assert "CUDA_VISIBLE_DEVICES" in capsys.readouterr().err


def test_fake_requires_params(scratch_env, capsys):
    assert cli.main(["run", str(REPO / "configs/grids/task_a_v0.yaml"), "--backend", "fake"]) == 2


def _fake_e2e(tmp_path, scratch_env, **kw):
    cfg = write_configs(tmp_path / "cfg", **kw)
    out, sel = tmp_path / "res", tmp_path / "sel"
    assert cli.main(["prepare", str(cfg["grid"]), "--backend", "fake", "--selections-dir", str(sel)]) == 0
    assert cli.main(["run", str(cfg["grid"]), "--backend", "fake", "--fake-params", str(cfg["fake"]),
                     "--results-dir", str(out), "--selections-dir", str(sel)]) == 0
    return out


def test_fake_run_refuses_without_prepare(tmp_path, scratch_env, capsys):
    """D-022: runs consume selections only from the prepare cache."""
    cfg = write_configs(tmp_path / "cfg")
    rc = cli.main(["run", str(cfg["grid"]), "--backend", "fake", "--fake-params", str(cfg["fake"]),
                   "--results-dir", str(tmp_path / "res"), "--selections-dir", str(tmp_path / "empty")])
    assert rc == 2 and "robocurate prepare" in capsys.readouterr().err


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


def test_report_twice_and_site(tmp_path, scratch_env, capsys):
    """Regression: a second `report` on the same dir loaded leaderboard.json as a run record and crashed."""
    out = _fake_e2e(tmp_path, scratch_env)
    assert cli.main(["report", str(out), "--n-boot", "200"]) == 0
    site = tmp_path / "site"
    assert cli.main(["report", str(out), "--n-boot", "200", "--site", str(site)]) == 0
    page = (site / "index.html").read_text()
    assert page.startswith("<!doctype html>") and "FAKE RESULTS" in page and "<title>FAKE" in page
    assert "http://" not in page and "https://" not in page and "<script" not in page  # self-contained, no requests
    lb = json.loads((site / "leaderboard.json").read_text())
    assert lb["fake"] and lb["tasks"][0]["budgets"][0]["mde"]["df"] == 2  # K = 3 -> df = K - 1 (D-019)
    for c in lb["tasks"][0]["budgets"][0]["conditions"]:
        assert c["cell_id"] in page


def test_reproduce_fake_cell(tmp_path, scratch_env, capsys):
    out = _fake_e2e(tmp_path, scratch_env)
    assert cli.main(["report", str(out), "--n-boot", "100"]) == 0
    lb = json.loads((out / "leaderboard.json").read_text())
    cells = {c["condition"]: c["cell_id"] for c in lb["tasks"][0]["budgets"][0]["conditions"]}
    capsys.readouterr()
    for cond in ("cleaner:length_heuristic", "random"):
        rc = cli.main(["reproduce", cells[cond], "--results-dir", str(out), "--backend", "fake",
                       "--selections-dir", str(tmp_path / "sel")])
        o = capsys.readouterr().out
        assert rc == 0, o
        n = 6 if cond.startswith("cleaner") else 18
        assert f"MATCH {n}/{n}" in o and "robocurate run" in o and "robocurate prepare" in o
        assert "  pytest " not in o  # regression: in-process calls recorded the caller's sys.argv
    assert cli.main(["reproduce", "000000000000", "--results-dir", str(out)]) == 2
