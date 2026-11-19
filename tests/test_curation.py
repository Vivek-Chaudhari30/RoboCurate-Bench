"""Curation infrastructure (D-020, D-022): rollout service, import graph, reference policy, prepare cache,
weighting-track artifacts and train-command wiring."""

import json
import pkgutil
from pathlib import Path

import numpy as np
import pytest

import robocurate.curation as curation_pkg
from robocurate.cleaners import importguard
from robocurate.cleaners.base import Cleaner, ScoreCleaner, WeightingCleaner, validate_resources
from robocurate.config import load_protocol, load_task
from robocurate.curation import prepare as P
from robocurate.curation.context import ReferencePolicy, ReferencePolicyModified
from robocurate.curation.rollouts import GymRolloutBackend, RolloutError, RolloutService
from robocurate.curation.weights import permute_progress, rabc_frame_weights, write_progress_parquet
from robocurate.data.episodes import synthetic_view
from robocurate.paths import repo_root
from robocurate.runs.spec import Condition, RunSpec
from robocurate.train.lerobot_train import SubsetDataset, build_train_cmd

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")


# ------------------------------------------------------------------ rollout service


def _svc(cap=10):
    return RolloutService.from_protocol(PROTO, P.FakeRolloutBackend(3, 2), cap=cap)


def test_rollouts_reject_eval_and_calibration_seeds():
    svc = _svc()
    for s in (PROTO.eval_seed_start, PROTO.eval_seed_start + 199, PROTO.calibration_seed_start):
        with pytest.raises(RolloutError, match="protected"):
            svc.rollout(None, [s])
    with pytest.raises(RolloutError, match="outside curation_seed_range"):
        svc.rollout(None, [7])
    assert svc.used == 0


def test_rollouts_enforce_cap_and_no_repeats():
    svc = _svc(cap=5)
    lo = PROTO.curation_seed_range[0]
    assert len(svc.rollout(None, [lo, lo + 1, lo + 2])) == 3
    with pytest.raises(RolloutError, match="repeat"):
        svc.rollout(None, [lo])
    with pytest.raises(RolloutError, match="cap"):
        svc.rollout(None, svc.next_seeds(3))
    assert svc.rollout(None, svc.next_seeds(2))[0].seed == lo + 3
    assert svc.remaining == 0


def test_rollout_cap_never_exceeds_protocol():
    assert _svc(cap=1000).cap == PROTO.max_curation_rollouts == 100


def test_gym_backend_runs_toy_env():
    from toybed.env import PointReachEnv

    svc = RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv, max_steps=60), cap=3)
    res = svc.rollout(lambda obs: -obs[:2] * 0.5, svc.next_seeds(2))
    assert len(res) == 2 and all(r.length > 0 for r in res)


# ------------------------------------------------------------------ import graph


def _modules(pkg):
    return [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")] + [pkg.__name__]


def test_curation_cannot_reach_eval_labels_oracle():
    bad = {m: importguard.module_forbidden_reach(m) for m in _modules(curation_pkg)}
    assert {m: r for m, r in bad.items() if r} == {}


def test_cleaners_cannot_reach_eval_labels_oracle():
    import robocurate.cleaners as cl

    bad = {m: importguard.module_forbidden_reach(m) for m in _modules(cl)}
    assert {m: r for m, r in bad.items() if r} == {}


# ------------------------------------------------------------------ reference policy


def test_reference_policy_read_only_and_tamper_detected(tmp_path):
    from robocurate.curation.policy import MLPPolicy

    path = MLPPolicy(3, 2).save(tmp_path / "ckpt")
    rp = ReferencePolicy(path, loader=MLPPolicy.load)
    rp.lock()
    try:
        with pytest.raises(PermissionError):
            (path / "mlp_policy.pt").write_bytes(b"x")
        assert isinstance(rp.load(), MLPPolicy)
    finally:
        rp.unlock()
    rp.verify_unchanged()
    (path / "mlp_policy.pt").write_bytes(b"tampered")
    with pytest.raises(ReferencePolicyModified):
        rp.verify_unchanged()


# ------------------------------------------------------------------ resource declarations


class _RefCleaner(ScoreCleaner):
    """Uses both resources: 2 rollouts per selection, reference-policy loss as the score."""

    name, version = "ref_user", "1"
    requires_reference_policy = True
    requires_rollouts = 4

    def select(self, view, budget, rng):
        self._pol = self.context.reference_policy.load()
        res = self.context.rollouts.rollout(self._pol, self.context.rollouts.next_seeds(2))
        assert all(r.seed >= PROTO.curation_seed_range[0] for r in res)
        return super().select(view, budget, rng)

    def score(self, e):
        from robocurate.curation.policy import make_batch

        return -float(self._pol.loss(make_batch(e.state, e.actions)))


class _Greedy(_RefCleaner):
    """Rolls out once per episode: exceeds its declared cap of 4 on 12 episodes."""

    def score(self, e):
        self.context.rollouts.rollout(self._pol, self.context.rollouts.next_seeds(1))
        return 0.0


class _Weights(WeightingCleaner):
    name, version = "flat_weights", "1"

    def frame_progress(self, view, rng):
        return self.context.progress_model.episode_progress(view)


def test_validate_resources():
    assert validate_resources(_RefCleaner()) == []
    bad = _RefCleaner()
    bad.requires_rollouts = 101
    bad.requires_gpu = "yes"
    assert len(validate_resources(bad)) == 2

    class Liar(Cleaner):
        name, version, track = "liar", "1", "weighting"

        def select(self, v, b, r):
            return []

    assert any("WeightingCleaner" in p for p in validate_resources(Liar()))


# ------------------------------------------------------------------ prepare cache


def test_prepare_selection_cell_cached_and_stale(tmp_path):
    view = synthetic_view(12, tie_break_salt=PROTO.tie_break_salt)
    cell = P.Cell("t", "cleaner", "ref_user", "1", 0.5)
    ctx = P.fake_context_factory(PROTO, view, tmp_path)
    r1 = P.prepare_cell(cell, _RefCleaner(), view, PROTO, ctx(_RefCleaner()), tmp_path, "fake")
    assert not r1["_cached"] and len(r1["selected_episodes"]) == 6
    assert r1["rollouts_used"] == 2 and r1["curation_seeds_used"] == [500000, 500001]
    assert r1["inputs"]["reference_policy_sha256"] and r1["gpu_h"] == 0.0
    r2 = P.prepare_cell(cell, _RefCleaner(), view, PROTO, ctx(_RefCleaner()), tmp_path, "fake")
    assert r2["_cached"] and r2["key"] == r1["key"]
    other = synthetic_view(12, tie_break_salt="f" * 32)
    octx = P.fake_context_factory(PROTO, other, tmp_path)(_RefCleaner())
    rec = P.prepare_cell(cell, _RefCleaner(), other, PROTO, octx, tmp_path, "fake")
    assert not rec["_cached"] and rec["key"] != r1["key"]
    with pytest.raises(P.PrepareError, match="re-run"):  # regression: key used the protocol salt, not the view's
        P.load_record(tmp_path, "t", "cleaner", "ref_user", "1", 0.5, view)


def test_prepare_rollout_cap_enforced_per_cell(tmp_path):
    view = synthetic_view(12)
    cl = _Greedy()
    with pytest.raises(RolloutError, match="cap"):
        P.prepare_cell(P.Cell("t", "cleaner", "ref_user", "1", 0.5), cl, view, PROTO,
                       P.fake_context_factory(PROTO, view, tmp_path)(cl), tmp_path, "fake")


def test_prepare_weighting_cell_writes_rabc_parquets(tmp_path):
    view = synthetic_view(6)
    cell = P.Cell("t", "weighting", "flat_weights", "1", 1.0, 0, 8)
    rec = P.prepare_cell(cell, _Weights(), view, PROTO, P.fake_context_factory(PROTO, view, tmp_path)(_Weights()),
                         tmp_path, "fake")
    import pandas as pd

    df = pd.read_parquet(rec["progress_parquet"])
    assert list(df.columns) == ["index", "episode_index", "frame_index", "progress_sparse"]
    assert len(df) == sum(view.lengths().values()) and df["index"].tolist() == list(range(len(df)))
    assert len(rec["placebo_parquets"]) == 8
    assert rec["inputs"]["weight_permutation_seeds"] == list(range(3000, 3008))


# ------------------------------------------------------------------ weights (D-020 placebo, RA-BC parity)


def test_permute_progress_preserves_pool_and_is_seeded():
    prog = {i: np.linspace(0, 1, 20) ** (i + 1) for i in range(6)}
    a, b, c = permute_progress(prog, 1), permute_progress(prog, 1), permute_progress(prog, 2)
    assert all(np.array_equal(a[i], b[i]) for i in prog)
    assert any(not np.array_equal(a[i], c[i]) for i in prog)
    key = lambda d: sorted(tuple(np.round(v, 12)) for v in d.values())  # noqa: E731
    assert key(a) == key(prog)  # same pool of curves (equal lengths)
    var = {0: np.linspace(0, 1, 10), 1: np.linspace(0, 1, 30) ** 2}
    pv = permute_progress(var, 0)
    assert len(pv[0]) == 10 and len(pv[1]) == 30


def test_rabc_weights_match_lerobot_rabcweights(tmp_path):
    """Our reporting mirror equals LeRobot's own RABCWeights (rewards/sarm/rabc.py) on the same parquet."""
    import torch
    from lerobot.rewards.sarm.rabc import RABCWeights

    rng = np.random.default_rng(0)
    prog = {i: np.clip(np.cumsum(rng.normal(0.02, 0.03, 40)), 0, 1) for i in range(5)}
    path = write_progress_parquet(prog, tmp_path / "p.parquet")
    lr = RABCWeights(path, chunk_size=7, device=torch.device("cpu"))
    ours = rabc_frame_weights(prog, chunk_size=7)
    flat = np.concatenate([ours[i] for i in sorted(ours)])
    idx = np.arange(len(flat))
    w, _ = lr.compute_batch_weights({"index": torch.as_tensor(idx)})
    expect = flat * len(flat) / (flat.sum() + 1e-6)
    np.testing.assert_allclose(w.numpy(), expect, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ train-command wiring


def _task(tmp_path):
    g = json.loads((Path(__file__).parent / "fixtures/golden_train_cmds.json").read_text())
    (tmp_path / "task.yaml").write_text(g["task_yaml"])
    return load_task(tmp_path / "task.yaml", mode="calibration"), g["cmds"]


def test_golden_train_cmds_unchanged_for_non_weighting(tmp_path):
    task, golden = _task(tmp_path)
    for label, cond, b, si in [("full", Condition.full(), 1.0, None), ("random", Condition.random(), 0.5, 1),
                               ("cleaner", Condition.cleaner("length_heuristic", "1"), 0.5, None),
                               ("oracle", Condition.oracle(), 0.5, None)]:
        cmd = build_train_cmd(task, RunSpec("aloha_transfer_cube", cond, b, si, 7),
                              SubsetDataset("robocurate/task_a_v0_train", Path("/SUBSET")), Path("/OUT/train"))
        assert cmd == golden[label], label


def test_weighting_track_adds_only_sample_weighting_flags(tmp_path):
    task, golden = _task(tmp_path)
    sub = SubsetDataset("robocurate/task_a_v0_train", Path("/SUBSET"))
    for cond, si in [(Condition.weighting("rabc", "1"), None), (Condition.weight_placebo("rabc", "1"), 3)]:
        cmd = build_train_cmd(task, RunSpec("aloha_transfer_cube", cond, 1.0, si, 7), sub, Path("/OUT/train"),
                              sample_weighting_progress=Path("/P/progress.parquet"))
        assert cmd[: len(golden["full"])] == golden["full"]
        assert cmd[len(golden["full"]):] == ["--sample_weighting.type=rabc",
                                             "--sample_weighting.progress_path=/P/progress.parquet",
                                             "--sample_weighting.head_mode=sparse"]
    from robocurate.config import ConfigError

    with pytest.raises(ConfigError):
        build_train_cmd(task, RunSpec("t", Condition.weighting("rabc", "1"), 1.0, None, 7), sub, Path("/O"))
    with pytest.raises(ConfigError):
        build_train_cmd(task, RunSpec("t", Condition.full(), 1.0, None, 7), sub, Path("/O"),
                        sample_weighting_progress=Path("/P"))


def test_weighting_flags_exist_in_installed_train_config(tmp_path):
    import lerobot.policies  # noqa: F401  registers the policy config choices (act)
    from lerobot.configs.train import TrainPipelineConfig
    from test_wrappers import _flag_ok

    task, _ = _task(tmp_path)
    cmd = build_train_cmd(task, RunSpec("t", Condition.weighting("rabc", "1"), 1.0, None, 7),
                          SubsetDataset("x/y", Path("/S")), Path("/O/t"), sample_weighting_progress=Path("/P.parquet"))
    assert [f for f in cmd[1:] if not _flag_ok(TrainPipelineConfig, f, cmd)] == []
    assert not _flag_ok(TrainPipelineConfig, "--sample_weighting.typo=1", cmd)
