import json
import os
import socket

import numpy as np
import pytest
from helpers import write_configs

from robocurate import results
from robocurate.config import TBD, ConfigError
from robocurate.runs import executor
from robocurate.runs.backends.fake import FakeBackend, load_fake_params
from robocurate.runs.planner import make_plan
from robocurate.runs.spec import Condition, RunSpec, run_id

# ------------------------------------------------------------------ spec


def test_condition_validation():
    with pytest.raises(ValueError):
        Condition("best")
    with pytest.raises(ValueError):
        Condition("cleaner", "x")
    with pytest.raises(ValueError):
        Condition("random", "x")
    assert Condition.cleaner("a", 1).label == "cleaner:a"


def test_spec_validation():
    with pytest.raises(ValueError):
        RunSpec("t", Condition.random(), 0.5, None, 0)
    with pytest.raises(ValueError):
        RunSpec("t", Condition.oracle(), 0.5, 1, 0)
    with pytest.raises(ValueError):
        RunSpec("t", Condition.full(), 0.5, None, 0)


def test_run_id_stable_and_sensitive():
    s = RunSpec("t", Condition.cleaner("a", "1"), 0.5, None, 3)
    h = {"protocol": "p", "task": "t"}
    s2 = RunSpec.from_dict(json.loads(s.canonical_json()))
    assert run_id(s, h) == run_id(s2, dict(reversed(h.items())))
    assert run_id(s, h) != run_id(RunSpec("t", Condition.cleaner("a", "2"), 0.5, None, 3), h)
    assert run_id(s, h) != run_id(s, {"protocol": "p2", "task": "t"})
    assert len(run_id(s, h)) == 16


# ------------------------------------------------------------------ planner


def test_plan_tbd_aware(tmp_path):
    cfg = write_configs(tmp_path)
    plan = make_plan(str(cfg["grid"]))
    assert plan.train_seeds is TBD and plan.n_runs is TBD and plan.gpu_hours is TBD
    text = "\n".join(plan.summary_lines())
    assert "TBD" in text and "condition cells: 6" in text
    with pytest.raises(ConfigError):
        plan.specs()


def test_plan_expands_with_override(tmp_path):
    cfg = write_configs(tmp_path)
    plan = make_plan(str(cfg["grid"]), mode="fake", train_seeds_override=[0, 1, 2, 3, 4, 5])
    specs = plan.specs()
    assert len(specs) == 6 * 6 == plan.n_runs
    rnd = [s for s in specs if s.condition.kind == "random"]
    assert sorted({plan.subset_seed(s) for s in rnd}) == [2000, 2001, 2002]
    assert {s.budget for s in specs if s.condition.kind == "full"} == {1.0}


def test_plan_cost_with_numbers(tmp_path):
    task = (
        "name: t\nenv_type: aloha\nenv_task: X\npolicy_type: act\nsource_datasets: []\n"
        "train_dataset: d\nsteps: 1000\nper_run_gpu_h: 0.5\n"
    )
    cfg = write_configs(tmp_path, task=task)
    plan = make_plan(str(cfg["grid"]), mode="fake", train_seeds_override=list(range(6)))
    assert plan.gpu_hours == pytest.approx(36 * 0.5)


def test_result_mode_rejects_override(tmp_path):
    cfg = write_configs(tmp_path)
    with pytest.raises(ConfigError):
        make_plan(str(cfg["grid"]), mode="result", train_seeds_override=[0, 1, 2])


# ------------------------------------------------------------------ fake backend + executor


def _setup(tmp_path, **kw):
    cfg = write_configs(tmp_path / "cfg", **kw)
    params = load_fake_params(cfg["fake"])
    plan = make_plan(str(cfg["grid"]), mode="fake", train_seeds_override=params.train_seeds)
    from robocurate.curation import prepare as P

    sel = tmp_path / "sel"
    view = P.fake_view(plan.protocol)
    P.prepare_plan(plan, view, sel, "fake", P.fake_context_factory(plan.protocol, view, sel))
    return plan, FakeBackend(plan, params, command_line=["pytest"], selections_dir=sel)


def test_fake_deterministic_from_run_id(tmp_path):
    plan, be = _setup(tmp_path)
    s = plan.specs()[0]
    rid = run_id(s, be.config_hashes())
    a, b = be.run(s, rid), be.run(s, rid)
    assert a["per_episode_success"] == b["per_episode_success"]
    results.validate(a)
    assert a["kind"] == "fake" and a["n_episodes"] == 200


def test_fake_effect_sizes_recovered(tmp_path):
    plan, be = _setup(tmp_path, oracle=0.25)
    out = tmp_path / "res"
    executor.execute(plan.specs(), be, out, log=lambda m: None)
    recs = results.load_dir(out)
    by = {}
    for r in recs:
        by.setdefault(Condition(**r["spec"]["condition"]).label, []).append(r["success_rate"])
    assert np.mean(by["oracle"]) - np.mean(by["random"]) == pytest.approx(0.25, abs=0.08)


def test_executor_idempotent(tmp_path):
    plan, be = _setup(tmp_path)
    out = tmp_path / "res"
    r1 = executor.execute(plan.specs(), be, out, log=lambda m: None)
    assert len(r1.done) == 36
    before = {p.name: p.read_text() for p in out.glob("*.json")}
    r2 = executor.execute(plan.specs(), be, out, log=lambda m: None)
    assert r2.done == [] and len(r2.skipped_existing) == 36
    assert before == {p.name: p.read_text() for p in out.glob("*.json")}
    assert not list((out / ".locks").glob("*.lock"))


def test_executor_reclaims_stale_lock_and_skips_live_lock(tmp_path):
    plan, be = _setup(tmp_path)
    out = tmp_path / "res"
    specs = plan.specs()[:2]
    ids = [run_id(s, be.config_hashes()) for s in specs]
    locks = out / ".locks"
    locks.mkdir(parents=True)
    # dead pid on this host -> stale
    (locks / f"{ids[0]}.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": 2**22 + 12345}))
    # live pid (ourselves) -> held by someone else
    (locks / f"{ids[1]}.lock").write_text(json.dumps({"host": socket.gethostname(), "pid": os.getpid()}))
    rep = executor.execute(specs, be, out, log=lambda m: None)
    assert rep.reclaimed_stale == [ids[0]] and rep.done == [ids[0]]
    assert rep.skipped_locked == [ids[1]]


def test_executor_replaces_corrupt_result(tmp_path):
    plan, be = _setup(tmp_path)
    out = tmp_path / "res"
    s = plan.specs()[0]
    rid = run_id(s, be.config_hashes())
    out.mkdir()
    (out / f"{rid}.json").write_text("{truncated")
    rep = executor.execute([s], be, out, log=lambda m: None)
    assert rep.done == [rid] and results.is_valid_file(out / f"{rid}.json")


def test_executor_releases_lock_on_backend_error(tmp_path):
    plan, be = _setup(tmp_path)

    def boom(spec, rid):
        raise RuntimeError("train crashed")

    be.run = boom
    out = tmp_path / "res"
    with pytest.raises(RuntimeError):
        executor.execute(plan.specs()[:1], be, out, log=lambda m: None)
    assert not list((out / ".locks").glob("*.lock"))
    assert not list(out.glob("*.json"))


def test_fake_missing_effect_errors(tmp_path):
    plan, be = _setup(tmp_path)
    be.params.effects.pop("oracle")
    with pytest.raises(ConfigError):
        executor.execute([s for s in plan.specs() if s.condition.kind == "oracle"][:1], be, tmp_path / "r",
                         log=lambda m: None)


def _mock_local_backend(monkeypatch, tmp_path, kind):
    """A LocalBackend whose subprocesses, subset materialization and eval parsing are mocked (no GPU, no LeRobot)."""
    import types

    import robocurate.data.subset as subset_mod
    import robocurate.eval.lerobot_eval as ev
    import robocurate.train.lerobot_train as tr
    from robocurate.runs.backends import local as L

    protocol = types.SimpleNamespace(eval_seed_start=100000, calibration_seed_start=300000, n_eval_rollouts=200,
                                     path=tmp_path / "p.yaml", sha256="0" * 64, task_eval_batch_size=lambda t: None)
    task = types.SimpleNamespace(name="aloha_transfer_cube", env_type="aloha", env_task="AlohaTransferCube-v0",
                                 path=tmp_path / "t.yaml",
                                 sha256="1" * 64, raw=lambda k: {"eval_batch_size": 50, "num_workers": None}[k])
    grid_vals = {"intermediate_save_freq": 500, "n_final_rollouts": 200, "n_intermediate_rollouts": 50,
                 "num_workers": 8, "batch_size": None, "eval_batch_size": None}
    grid = types.SimpleNamespace(path=tmp_path / "g.yaml", sha256="2" * 64, raw=lambda k: grid_vals[k])
    plan = types.SimpleNamespace(protocol=protocol, task=task, grid=grid, kind=kind, subset_seed=lambda spec: None)
    be = L.LocalBackend.__new__(L.LocalBackend)
    be.plan, be.kind, be.steps, be.device, be.command_line = plan, kind, 1000, "cuda", ["robocurate"]
    be.repo_id, be.root, be._view = "x/y", None, types.SimpleNamespace(episode_indices=[0, 1, 2], n_episodes=3)
    cmds = []
    monkeypatch.setattr(L, "scratch_dir", lambda *a, **k: tmp_path / "run")
    monkeypatch.setattr(subset_mod, "materialize_subset", lambda *a, **k: tmp_path / "sub")
    monkeypatch.setattr(tr, "build_train_cmd", lambda *a, **k: ["lerobot-train"])
    monkeypatch.setattr(tr, "final_checkpoint", lambda *a, **k: tmp_path / "pretrained_model")
    monkeypatch.setattr(tr, "checkpoint_dir", lambda d, step, total: tmp_path / f"{step}" / "pretrained_model")
    monkeypatch.setattr(ev, "parse_eval_info", lambda out, n: [True] * n)
    monkeypatch.setattr(be, "_run", lambda cmd, log: cmds.append(cmd))
    return be, cmds


@pytest.mark.parametrize("kind,seed0", [("calibration", 300000), ("result", 100000)])
def test_local_backend_eval_seed_block(monkeypatch, tmp_path, kind, seed0):
    """Regression: calibration runs evaluated on protocol EVAL seeds (D-017 / CLAUDE.md rule 10 violation)."""
    from robocurate.runs.spec import Condition, RunSpec

    be, cmds = _mock_local_backend(monkeypatch, tmp_path, kind)
    rec = be.run(RunSpec("aloha_transfer_cube", Condition.full(), 1.0, None, 0), "rid")
    evals = [c for c in cmds if c[0] == "lerobot-eval"]
    assert evals and all(f"--seed={seed0}" in c for c in evals)
    assert len(evals) == (2 if kind == "calibration" else 1)  # 1000 steps, save every 500 -> one intermediate
    assert rec["seeds"]["eval_seed_start"] == seed0
    assert rec["seeds"]["seed_block"] == ("calibration" if kind == "calibration" else "eval")
