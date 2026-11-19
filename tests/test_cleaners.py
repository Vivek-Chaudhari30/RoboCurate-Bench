import json
import pkgutil
import sys
from pathlib import Path

import numpy as np
import pytest

import robocurate.cleaners as cleaners_pkg
from robocurate.cleaners import conformance, importguard, registry
from robocurate.cleaners.base import ScoreCleaner, budget_count
from robocurate.cleaners.builtin import LengthHeuristic, RandomSubset
from robocurate.data.episodes import DatasetView, synthetic_view
from robocurate.data.labels import EpisodeLabel, LabelsSidecar
from robocurate.runs.oracle import oracle_select

FIX = Path(__file__).parent / "fixtures"
BROKEN = FIX / "broken_cleaners"


def _view(lengths):
    acts = [np.zeros((L, 2)) for L in lengths]
    return DatasetView.from_arrays(acts, acts, ["t"] * len(lengths), fps=50)


# ------------------------------------------------------------------ built-ins


def test_random_subset_deterministic_and_valid():
    v = synthetic_view(30)
    a = RandomSubset().select(v, 15, np.random.default_rng(2000))
    b = RandomSubset().select(v, 15, np.random.default_rng(2000))
    c = RandomSubset().select(v, 15, np.random.default_rng(2001))
    assert a == b != c and len(set(a)) == 15 and set(a) <= set(v.episode_indices)


def test_length_heuristic_rule_and_ties():
    # median of [10, 20, 30, 40, 50, 100] = 35 -> distances 25,15,5,5,15,65
    v = _view([10, 20, 30, 40, 50, 100])
    assert LengthHeuristic().select(v, 2, np.random.default_rng(0)) == [2, 3]
    # tie at distance 15 between ep 1 and ep 4 -> D-021 salted content hash decides, not the index
    sel = LengthHeuristic().select(v, 3, np.random.default_rng(0))
    winner = min((1, 4), key=v.tie_break_key)
    assert sel == sorted([2, 3, winner])


def test_score_cleaner_tie_break_by_content_hash():
    """D-021 (replaces the old index tie-break): all-equal scores keep the 4 smallest salted content hashes."""

    class Const(ScoreCleaner):
        name, version = "c", "1"

        def score(self, e):
            return 1.0

    v = synthetic_view(10)
    assert Const().select(v, 4, np.random.default_rng(0)) == sorted(sorted(v.episode_indices, key=v.tie_break_key)[:4])


def test_budget_count():
    assert budget_count(50, 0.5) == 25
    assert budget_count(7, 0.5) == 4
    assert budget_count(3, 0.1) == 1
    with pytest.raises(ValueError):
        budget_count(10, 0)


def test_view_is_read_only():
    v = synthetic_view(3)
    with pytest.raises(AttributeError):
        v.foo = 1
    with pytest.raises(ValueError):
        v.episode(0).actions[0, 0] = 1.0
    assert not hasattr(v, "labels")


# ------------------------------------------------------------------ registry


def test_registry_builtins_and_version_check():
    assert {"random_subset", "length_heuristic"} <= set(registry.available())
    assert registry.get("length_heuristic", "1").name == "length_heuristic"
    with pytest.raises(registry.RegistryError):
        registry.get("length_heuristic", "2")
    with pytest.raises(registry.RegistryError):
        registry.get("no_such_cleaner")
    assert "oracle" not in registry.available()


def test_registry_entry_point_plugin(monkeypatch):
    from importlib import metadata

    ep = metadata.EntryPoint(
        name="length_heuristic_plugin", value="robocurate.cleaners.builtin:LengthHeuristic",
        group=registry.ENTRY_POINT_GROUP,
    )
    monkeypatch.setattr(registry, "plugin_entry_points", lambda: {ep.name: ep})
    assert "length_heuristic_plugin" in registry.available()
    with pytest.raises(registry.RegistryError, match="produced cleaner named"):
        registry.get("length_heuristic_plugin")


# ------------------------------------------------------------------ oracle


def test_oracle_prefers_clean_then_fewer_defects():
    eps = [
        EpisodeLabel(0, "human", "d", 0, ("pause",), "degraded"),
        EpisodeLabel(1, "human", "d", 1, (), "clean"),
        EpisodeLabel(2, "policy", "d", 2, ("jitter", "gripper_drop"), "bad"),
        EpisodeLabel(3, "scripted", "d", 3, (), "clean"),
        EpisodeLabel(4, "human", "d", 4, ("pause", "wrong_task"), "degraded"),
    ]
    lab = LabelsSidecar("d", "0" * 64, 0, eps)
    assert oracle_select(lab, 3) == [0, 1, 3]
    assert oracle_select(lab, 2) == [1, 3]


# ------------------------------------------------------------------ conformance


@pytest.mark.parametrize("target", ["random_subset", "length_heuristic", str(FIX / "cleaners/good_cleaner.py")])
def test_conformance_passes(target, tmp_path):
    rep = conformance.check(target)
    assert rep.passed, rep.lines()
    v = synthetic_view()
    out = conformance.write_selection(rep, v, tmp_path / "sel.json", 0.5)
    d = json.loads(out.read_text())
    assert d["budget"] == 20 and len(d["selected_episode_indices"]) == 20
    assert d["dataset_fingerprint"] == v.fingerprint()


@pytest.mark.parametrize(
    "fixture,failing",
    [
        ("wrong_count.py", "count"),
        ("duplicates.py", "unique"),
        ("nondeterministic.py", "determinism"),
        ("label_import.py", "static"),
        ("oracle_import.py", "static"),
        ("dynamic_import.py", "static"),
        ("too_many_rollouts.py", "rollouts"),
    ],
)
def test_conformance_fails_broken(fixture, failing):
    rep = conformance.check(str(BROKEN / fixture))
    assert not rep.passed
    assert failing in [c.name for c in rep.checks if not c.passed], rep.lines()


def test_conformance_static_failure_never_imports():
    before = set(sys.modules)
    conformance.check(str(BROKEN / "label_import.py"))
    assert not any("_robocurate_submitted_" in m for m in set(sys.modules) - before)


def test_conformance_runtime_limit():
    rep = conformance.check(str(BROKEN / "slow.py"), time_limit_s=0.5)
    assert [c.name for c in rep.checks if not c.passed] == ["runtime"]


def test_write_selection_refuses_failed(tmp_path):
    rep = conformance.check(str(BROKEN / "wrong_count.py"))
    with pytest.raises(ValueError):
        conformance.write_selection(rep, synthetic_view(), tmp_path / "x.json", 0.5)


# ------------------------------------------------------------------ import graph


def _cleaner_modules():
    pkg_path = Path(cleaners_pkg.__file__).parent
    return ["robocurate.cleaners"] + [f"robocurate.cleaners.{m.name}" for m in pkgutil.iter_modules([str(pkg_path)])]


@pytest.mark.parametrize("mod", _cleaner_modules())
def test_import_graph_cleaners_cannot_reach_forbidden(mod):
    assert importguard.module_forbidden_reach(mod) == []


def test_import_graph_detects_transitive_reach():
    # oracle reaches labels directly; a module importing robocurate.runs.oracle must be flagged
    assert importguard.module_forbidden_reach("robocurate.runs.oracle")  # oracle -> data.labels
    assert importguard.forbidden_reach("import robocurate.runs.oracle")
    assert importguard.forbidden_reach("from robocurate.data import labels")
    assert importguard.forbidden_reach("from robocurate import eval")
    assert importguard.forbidden_reach("import robocurate.cleaners.base") == []


def test_data_episodes_is_label_free():
    assert importguard.module_forbidden_reach("robocurate.data.episodes") == []


# ------------------------------------------------------------------ D-021 tie-break


def _two_source_view(seed, n=40, salt="46324cdb15ee2629d87e143bdb0d4fe1"):
    """Fixture: episodes 0..n/2-1 come from source A, the rest from source B (source order = index order)."""
    rng = np.random.default_rng(seed)
    acts = [rng.normal(size=(50, 4)).astype(np.float32) for _ in range(n)]
    v = DatasetView.from_arrays(acts, [a.cumsum(0) for a in acts], ["t"] * n, 50, tie_break_salt=salt)
    source = {i: ("A" if i < n // 2 else "B") for i in range(n)}
    return v, source


class _Flat(ScoreCleaner):
    name, version = "flat", "1"

    def score(self, episode):
        return 0.0


def test_tie_break_uncorrelated_with_source_order():
    """All-equal scores: index tie-break would keep exactly source A; D-021 keeps ~half A."""
    fracs = []
    for seed in range(60):
        v, source = _two_source_view(seed)
        ids = _Flat().select(v, 20, np.random.default_rng(0))
        assert len(set(ids)) == 20
        fracs.append(np.mean([source[i] == "A" for i in ids]))
    assert 0.42 <= float(np.mean(fracs)) <= 0.58
    assert min(fracs) < 1.0  # index tie-break would give 1.0 every time


def test_tie_break_depends_on_salt_and_content_not_index():
    v, _ = _two_source_view(0, salt="a" * 32)
    w, _ = _two_source_view(0, salt="b" * 32)
    assert _Flat().select(v, 20, None) != _Flat().select(w, 20, None)
    # relabelling the same episodes with shifted indices selects the same content
    eps = list(v)
    shifted = DatasetView([type(e)(e.episode_index + 100, e.task, e.actions, e.state, e.timestamps) for e in eps],
                          50, tie_break_salt="a" * 32)
    assert [i + 100 for i in _Flat().select(v, 20, None)] == _Flat().select(shifted, 20, None)
