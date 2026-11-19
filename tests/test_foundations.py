import json
from pathlib import Path

import pytest

from robocurate import config as C
from robocurate import paths, results

REPO = paths.repo_root()


# ---------------------------------------------------------------- paths


def test_scratch_unset_refuses():
    with pytest.raises(paths.StorageError, match="unset"):
        paths.scratch_root({})


def test_scratch_inside_repo_refuses():
    with pytest.raises(paths.StorageError, match="overlaps the repo"):
        paths.scratch_root({"ROBOCURATE_SCRATCH": str(REPO / "scratch")}, hostname="mac")


def test_scratch_containing_repo_refuses():
    with pytest.raises(paths.StorageError, match="overlaps the repo"):
        paths.scratch_root({"ROBOCURATE_SCRATCH": str(REPO.parent)}, hostname="mac")


def test_scratch_under_home_ok_on_mac(tmp_path):
    home = tmp_path / "home"
    (home / "robocurate-scratch").mkdir(parents=True)
    root = paths.scratch_root(
        {"ROBOCURATE_SCRATCH": str(home / "robocurate-scratch")}, hostname="vivek-mbp", home=home
    )
    assert root.name == "robocurate-scratch"


def test_scratch_under_home_refused_on_hokkaido(tmp_path):
    home = tmp_path / "home"
    with pytest.raises(paths.StorageError, match="scratch disk"):
        paths.scratch_root({"ROBOCURATE_SCRATCH": str(home / "s")}, hostname="hokkaido", home=home)


def test_scratch_outside_home_ok_on_hokkaido(tmp_path):
    home = tmp_path / "home"
    root = paths.scratch_root({"ROBOCURATE_SCRATCH": str(tmp_path / "data")}, hostname="hokkaido", home=home)
    assert root == (tmp_path / "data").resolve()


def test_require_storage_env_checks_hf_vars(tmp_path):
    s = str(tmp_path / "s")
    env = {"ROBOCURATE_SCRATCH": s, "HF_HOME": s + "/hf"}
    with pytest.raises(paths.StorageError, match="HF_LEROBOT_HOME is unset"):
        paths.require_storage_env(env, hostname="mac")
    env["HF_LEROBOT_HOME"] = str(tmp_path / "elsewhere")
    with pytest.raises(paths.StorageError, match="not inside"):
        paths.require_storage_env(env, hostname="mac")
    env["HF_LEROBOT_HOME"] = s + "/lerobot"
    assert paths.require_storage_env(env, hostname="mac") == Path(s).resolve()


# ---------------------------------------------------------------- config / TBD


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


PROTO = """
eval_seed_start: 100000
n_eval_rollouts: 200
checkpoint_rule: final
n_train_seeds: TBD
train_seeds: TBD
random_subset_seeds: [2000, 2001, 2002]
curation_seed_range: [500000, 599999]
alpha: 0.05
test: welch_holm
"""


def test_tbd_sentinel_is_singleton_and_not_truthy():
    assert C.TBD is C._TBDType()
    assert str(C.TBD) == "TBD"
    with pytest.raises(TypeError):
        bool(C.TBD)


def test_protocol_tbd_plan_mode_returns_sentinel(tmp_path):
    p = C.load_protocol(_write(tmp_path, "p.yaml", PROTO), mode="plan")
    assert p.n_train_seeds is C.TBD
    assert p.eval_seed_start == 100000
    assert p.sha256 == C.sha256_file(tmp_path / "p.yaml")
    assert C.tbd_fields(p) == ["n_train_seeds", "train_seeds"]


def test_protocol_tbd_result_mode_raises(tmp_path):
    p = C.load_protocol(_write(tmp_path, "p.yaml", PROTO), mode="result")
    assert p.alpha == 0.05  # non-TBD fields still readable
    with pytest.raises(C.TBDError):
        _ = p.train_seeds
    with pytest.raises(C.TBDError):
        _ = p.with_mode("result").n_train_seeds
    assert p.with_mode("calibration").n_train_seeds is C.TBD


@pytest.mark.parametrize(
    "patch,msg",
    [
        (("n_eval_rollouts: 200", "n_eval_rollouts: 50"), ">= 200"),
        (("checkpoint_rule: final", "checkpoint_rule: best"), "D-003"),
        (("random_subset_seeds: [2000, 2001, 2002]", "random_subset_seeds: [2000]"), ">= 3"),
        (("n_train_seeds: TBD\ntrain_seeds: TBD", "n_train_seeds: 3\ntrain_seeds: [1, 2, 3]"), "floor is 6"),
        (("curation_seed_range: [500000, 599999]", "curation_seed_range: [100100, 200000]"), "overlap"),
        (("test: welch_holm", "test: binomial"), "D-002"),
        (("alpha: 0.05", "alpha: 0.05\ncalibration_seed_start: 100050"), "calibration seeds"),
        (("alpha: 0.05", "alpha: 0.05\ncalibration_seed_start: 450000"), "calibration seeds"),
    ],
)
def test_protocol_rejects_rule_violations(tmp_path, patch, msg):
    with pytest.raises(C.ConfigError, match=msg):
        C.load_protocol(_write(tmp_path, "p.yaml", PROTO.replace(*patch)))


def test_unknown_keys_rejected(tmp_path):
    with pytest.raises(C.ConfigError, match="unknown keys"):
        C.load_protocol(_write(tmp_path, "p.yaml", PROTO + "best_checkpoint: true\n"))


def test_grid_requires_random_and_full_for_cleaners(tmp_path):
    txt = """
name: g
task: t.yaml
protocol: p.yaml
budgets: [0.5]
conditions:
  - {kind: cleaner, name: x, version: "1"}
  - {kind: full}
"""
    with pytest.raises(C.ConfigError, match="random subsets AND full"):
        C.load_grid(_write(tmp_path, "g.yaml", txt))


def test_grid_random_needs_three_subsets(tmp_path):
    txt = """
name: g
task: t.yaml
protocol: p.yaml
budgets: [0.5]
conditions:
  - {kind: random, n_subsets: 2}
"""
    with pytest.raises(C.ConfigError, match="n_subsets >= 3"):
        C.load_grid(_write(tmp_path, "g.yaml", txt))


# ---------------------------------------------------------------- results


def make_record(**over):
    pes = [True] * 3 + [False] * 7
    from robocurate.report.stats import wilson_ci

    d = results.RunResult(
        run_id="abc123",
        spec={
            "task": "t", "condition": {"kind": "full"}, "budget": 1.0, "subset_index": None, "train_seed": 0
        },
        kind="fake",
        git=results.GitInfo(hash="deadbeef", dirty=True),
        configs=[results.ConfigRef(path="configs/protocol.yaml", sha256="0" * 64)],
        lerobot_version="0.6.1",
        command_lines=[["robocurate", "run"]],
        seeds={"train_seed": 0},
        per_episode_success=pes,
        n_episodes=len(pes),
        success_rate=0.3,
        wilson_95ci=list(wilson_ci(3, 10)),
        checkpoint_step=None,
        wall_clock_s=0.1,
        peak_gpu_mem_mb=None,
        hostname="h",
    ).to_dict()
    d.update(over)
    return d


def test_results_roundtrip_atomic(tmp_path):
    p = results.write_atomic(tmp_path / "r.json", make_record())
    assert results.load(p)["run_id"] == "abc123"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "over,msg",
    [
        ({"kind": "bogus"}, "kind"),
        ({"n_episodes": 11}, "n_episodes"),
        ({"success_rate": 0.5}, "success_rate"),
        ({"wilson_95ci": [0.0, 0.1]}, "wilson"),
        ({"kind": "result"}, "clean git tree"),
        ({"seeds": {}}, "train_seed"),
        ({"per_episode_success": [1, 0]}, "bools"),
    ],
)
def test_results_validate_rejects(over, msg):
    with pytest.raises(results.ResultsValidationError, match=msg):
        results.validate(make_record(**over))


def test_results_result_kind_needs_200(tmp_path):
    from robocurate.report.stats import wilson_ci

    d = make_record(kind="result", git={"hash": "x", "dirty": False}, checkpoint_step=10)
    with pytest.raises(results.ResultsValidationError, match="200"):
        results.validate(d)
    pes = [True] * 100 + [False] * 100
    d.update(per_episode_success=pes, n_episodes=200, success_rate=0.5, wilson_95ci=list(wilson_ci(100, 200)))
    results.validate(d)


def test_write_atomic_refuses_invalid(tmp_path):
    with pytest.raises(results.ResultsValidationError):
        results.write_atomic(tmp_path / "r.json", make_record(kind="nope"))
    assert not (tmp_path / "r.json").exists()


def test_missing_nullable_field_rejected():
    d = make_record()
    del d["peak_gpu_mem_mb"]
    with pytest.raises(results.ResultsValidationError, match="nullable"):
        results.validate(d)


def test_git_info_and_versions():
    g = results.git_info()
    assert len(g.hash) in (40, len("NO_COMMIT"))
    assert results.lerobot_version() == "0.6.1"
    json.dumps(results.hostname())


def test_repo_protocol_calibration_seeds_disjoint():
    """D-017: the repo protocol reserves a calibration seed block disjoint from eval and curation seeds."""
    from robocurate.paths import repo_root

    p = C.load_protocol(repo_root() / "configs/protocol.yaml")
    c0, c1 = p.calibration_seed_range()
    assert c0 == 300000
    e0, e1 = p.eval_seed_range(50)
    assert c1 <= e0 or e1 <= c0
    assert c1 <= p.curation_seed_range[0] or p.curation_seed_range[1] < c0
