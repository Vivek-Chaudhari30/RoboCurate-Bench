"""Policy-rollout mixture source (m1-core item 6), end to end on the toybed env with fake policies."""

import numpy as np
import pytest
from toybed.env import PointReachEnv, expert_action

from robocurate.config import ConfigError, load_protocol
from robocurate.data import mixture as M
from robocurate.data.labels import load_labels, sidecar_path
from robocurate.paths import repo_root

PROTO = load_protocol(repo_root() / "configs/protocol.yaml")


def _load(src):
    if src.checkpoint == "fake://expert":
        return expert_action
    return lambda obs: np.zeros(2, dtype=np.float32)  # a 'checkpoint' that never moves


SOURCES = [
    M.PolicySource("expert_late", "fake://expert", 100_000, 4),
    M.PolicySource("expert_jitter", "fake://expert", 100_000, 3,
                   ({"type": "jitter", "sigma": 0.8, "onset": 1, "duration": 30},)),
    M.PolicySource("early_ckpt", "fake://frozen", 10_000, 3),
]


def test_policy_source_end_to_end(tmp_path, scratch_env):
    eps = M.collect_policy_episodes(SOURCES, _load, PointReachEnv, PROTO, max_steps=60)
    assert len(eps) == 10
    lo, hi = PROTO.curation_seed_range
    assert [e.seed for e in eps] == list(range(hi, hi - 10, -1))  # top of the curation range, never eval seeds
    assert all(e.success for e in eps[:4]) and not any(e.success for e in eps[7:])
    clean = M.collect_policy_episodes(SOURCES[:1], _load, PointReachEnv, PROTO, max_steps=60)[0]
    jit = eps[4]
    assert not np.allclose(np.diff(jit.actions, axis=0), 0)  # perturbation executed, observations follow it
    assert np.allclose(jit.states[1:, :2], jit.states[:-1, :2] + 0.1 * np.clip(jit.actions[:-1], -1, 1), atol=1e-5)
    assert clean.actions.shape[1] == 2

    root = M.write_policy_dataset(eps, "rcb/toy_policy_mix", tmp_path / "ds", fps=10, task="reach the goal",
                                  shuffle_seed=7)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("rcb/toy_policy_mix", root=root)
    assert ds.meta.total_episodes == 10 and ds.meta.total_frames == sum(len(e.actions) for e in eps)
    labels = load_labels(sidecar_path(root)).by_index()
    assert {lab.source for lab in labels.values()} == {"policy"}
    steps = [labels[i].checkpoint_step for i in range(10)]
    assert sorted(steps) == sorted([s.checkpoint_step for s in SOURCES for _ in range(s.n_episodes)])
    assert steps != sorted(steps) and steps != sorted(steps, reverse=True)  # shuffled (D-021)
    jl = [lab for lab in labels.values() if lab.perturbations]
    assert len(jl) == 3 and all(lab.defects == ("jitter",) and lab.quality_tier == "bad" for lab in jl)
    assert sum(bool(lab.success) for lab in labels.values()) == sum(e.success for e in eps)
    assert not (root / "labels.json").exists() and sidecar_path(root).parent == root.parent  # beside, not inside


def test_policy_source_rejects_unknown_perturbation():
    with pytest.raises(ConfigError):
        M.make_perturbations(({"type": "teleport"},))
