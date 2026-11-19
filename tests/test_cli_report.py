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


def _k8_cfg(tmp_path, weighting=False):
    from helpers import PROTOCOL

    proto = PROTOCOL.replace("[2000, 2001, 2002]", str(list(range(2000, 2008))))
    if weighting:
        proto += "weight_permutation_seeds: [3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007]\n"
    cfg = write_configs(tmp_path / f"cfg{int(weighting)}", protocol=proto)
    g = cfg["grid"].read_text().replace("n_subsets: 3", "n_subsets: 8")
    if weighting:
        g += "  - {kind: weighting, name: rabc, version: \"1\", n_permutations: 8}\n"
    cfg["grid"].write_text(g)
    return cfg


def test_check_result_design_unit(tmp_path, monkeypatch):
    """D-026: result mode needs a balanced design: n_c = n_p = n >= 3, K >= 8 (subsets and weight permutations)."""
    from robocurate.config import ConfigError
    from robocurate.runs.planner import Plan, check_result_design, make_plan, placebo_seed_counts

    grid = str(_k8_cfg(tmp_path)["grid"])
    for seeds in ([0, 1, 2], [0, 1, 2, 3, 4]):
        plan = make_plan(grid, train_seeds_override=seeds)
        check_result_design(plan)
        assert set(placebo_seed_counts(plan).values()) == {len(seeds)} and len(placebo_seed_counts(plan)) == 8
    for seeds in ([0], [0, 1]):  # n < 3
        with pytest.raises(ConfigError, match=r"D-026 requires n >= 3"):
            check_result_design(make_plan(grid, train_seeds_override=seeds))

    plan = make_plan(grid, train_seeds_override=[0, 1, 2])
    full_specs = Plan.specs(plan)

    def drop(pred):
        out, seen = [], set()
        for sp in full_specs:
            if pred(sp) and sp.condition.kind not in seen:
                seen.add(sp.condition.kind)  # drop one run of the first matching unit
                continue
            out.append(sp)
        return out

    # n_c != n_p: the cleaner has one run fewer than every placebo subset (and the reverse)
    for kind in ("cleaner", "random"):
        monkeypatch.setattr(plan, "specs", lambda k=kind: drop(lambda sp: sp.condition.kind == k))
        with pytest.raises(ConfigError, match=r"unbalanced design.*D-026 requires n_c = n_p = n"):
            check_result_design(plan)

    # weighting track: n per permutation, and K >= 8 permutations
    wgrid = str(_k8_cfg(tmp_path, weighting=True)["grid"])
    wplan = make_plan(wgrid, train_seeds_override=[0, 1, 2])
    check_result_design(wplan)
    assert sum(1 for k in placebo_seed_counts(wplan) if k[0].startswith("weight_placebo")) == 8
    with pytest.raises(ConfigError, match="D-026 requires n >= 3"):
        check_result_design(make_plan(wgrid, train_seeds_override=[0, 1]))
    wspecs = Plan.specs(wplan)
    monkeypatch.setattr(wplan, "specs", lambda: [sp for sp in wspecs if not (
        sp.condition.kind == "weight_placebo" and sp.subset_index == 7)])
    with pytest.raises(ConfigError, match=r"K = 7 weight permutations; D-026 requires K >= 8"):
        check_result_design(wplan)


def test_result_mode_guard_d026(tmp_path):
    """D-026 is wired into the result-mode guard after the K check and before the FROZEN / TBD checks."""
    from robocurate.config import ConfigError
    from robocurate.runs.planner import make_plan

    grid = str(_k8_cfg(tmp_path)["grid"])
    with pytest.raises(ConfigError, match="D-026"):
        cli._result_mode_guard(make_plan(grid, train_seeds_override=[0, 1]))
    with pytest.raises(ConfigError) as e:  # balanced n = 3 passes D-026, then hits FROZEN / TBD
        cli._result_mode_guard(make_plan(grid, train_seeds_override=[0, 1, 2]))
    assert "D-026" not in str(e.value) and ("FROZEN" in str(e.value) or "TBD" in str(e.value))


def test_protocol_seed_floor_is_3(tmp_path):
    """D-026 retires D-015's 6-seed floor: a protocol with 3 seeds loads, 2 is refused."""
    from helpers import PROTOCOL

    from robocurate.config import ConfigError, load_protocol

    def proto(seeds):
        f = tmp_path / f"p{len(seeds)}.yaml"
        f.write_text(PROTOCOL.replace("n_train_seeds: TBD", f"n_train_seeds: {len(seeds)}")
                     .replace("\ntrain_seeds: TBD", f"\ntrain_seeds: {seeds}"))
        return f

    assert load_protocol(proto([1, 2, 3])).train_seeds == [1, 2, 3]
    with pytest.raises(ConfigError, match="floor is 3"):
        load_protocol(proto([1, 2]))


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


def test_disabled_entrant_refused_in_grids_and_excluded_from_report(tmp_path, scratch_env, monkeypatch, capsys):
    """CUPID failed its toybed sanity (outcome c): status "disabled — failed sanity"; grids refuse it and the
    report drops any disabled entrant with an issue instead of listing it under its name."""
    from robocurate.cleaners import registry
    from robocurate.config import ConfigError
    from robocurate.runs.planner import make_plan

    assert registry.get("cupid").metadata()["status"] == "disabled — failed sanity"
    assert registry.disabled_reason("cupid") == "disabled — failed sanity"
    assert registry.disabled_reason("length_heuristic") is None
    cfg = write_configs(tmp_path / "cfgc")
    g = cfg["grid"].read_text().replace("name: length_heuristic", "name: cupid")
    cfg["grid"].write_text(g)
    with pytest.raises(ConfigError, match="disabled — failed sanity"):
        make_plan(str(cfg["grid"]))
    assert cli.main(["plan", str(cfg["grid"])]) == 2

    out = _fake_e2e(tmp_path, scratch_env, oracle=0.25, length=0.0)
    monkeypatch.setattr(R, "_disabled", lambda n: "disabled — test" if n == "length_heuristic" else None)
    a = R.analyze(results.load_dir(out), n_boot=200)
    b = a["tasks"][0]["budgets"][0]
    assert not b["cleaner_vs_placebo"] and all(c["condition"] != "cleaner:length_heuristic" for c in b["conditions"])
    assert any("cleaner:length_heuristic excluded: entrant is disabled — test" in i for i in b["issues"])
