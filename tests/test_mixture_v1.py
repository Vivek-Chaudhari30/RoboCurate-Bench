"""Mixture recipes v1 (D-031 Task A, Task B proposal): rate-based post-hoc defects, deferred GPU-phase sources,
and the replay-mode generation path tested on the toybed."""

import numpy as np
import pytest
from toybed.env import PointReachEnv, expert_action

from robocurate.config import ConfigError, load_protocol
from robocurate.data import mixture as M
from robocurate.data.labels import load_labels, sidecar_path
from robocurate.paths import repo_root

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")


def test_task_a_v1_recipe_matches_d031():
    r = M.Recipe.load(REPO / "configs/mixtures/task_a_v1.yaml")
    by = {s.source: s.n_episodes for s in r.sources}
    assert by == {"scripted": 40, "human": 50}
    d = {x["name"]: x for x in r.deferred_sources}
    assert d["scripted_replay"]["n_episodes"] == 10  # 20% of 50 scripted
    assert sorted(p["type"] for p in d["scripted_replay"]["perturbations"]) == ["gripper_drop", "jitter"]
    assert [c["train_fraction"] for c in d["policy_rollouts"]["checkpoints"]] == [0.1, 0.3, 0.6, 1.0]
    assert all(c["n_episodes"] == 25 for c in d["policy_rollouts"]["checkpoints"])
    assert sum(by.values()) + r.n_deferred_episodes == 200
    assert r.pause["rate"] == 0.5 and r.duplicate["rate"] == 0.0 and not r.wrong_task  # D-031
    plan = M.plan_mixture(r, {s.dataset: [400] * 50 for s in r.sources})
    assert sum("pause" in p.defects for p in plan) == 45 and not any("duplicate" in p.defects for p in plan)
    assert len(plan) == 90 and all(31 <= p.pause_k <= 60 for p in plan if p.pause_k)


def test_task_b_v1_recipe():
    r = M.Recipe.load(REPO / "configs/mixtures/pusht_diffusion_v1.yaml")
    assert [(s.dataset, s.n_episodes) for s in r.sources] == [("lerobot/pusht", 206)]
    assert r.n_deferred_episodes == 100 and r.pause == {"rate": 0.5, "k_min": 6, "k_max": 12}
    plan = M.plan_mixture(r, {"lerobot/pusht": [120] * 206})
    assert sum("pause" in p.defects for p in plan) == 103


def test_defect_rate_rounds_half_up_and_spec_is_exclusive(tmp_path):
    assert M.defect_count({"rate": 0.5}, 45) == 23 and M.defect_count({"n_episodes": 4}, 45) == 4
    assert M.defect_count({}, 10) == 0
    bad = tmp_path / "r.yaml"
    bad.write_text("name: x\nrepo_id: r/x\nseed: 0\nsources: [{dataset: a/b, source: human, n_episodes: 2}]\n"
                   "defects: {pause: {rate: 0.5, n_episodes: 1, k_min: 1, k_max: 2}}\n")
    with pytest.raises(ConfigError, match="exactly one"):
        M.Recipe.load(bad)


@pytest.mark.parametrize("deferred,msg", [
    ("{name: p, source: policy, n_episodes: 4, generated_in: cpu, checkpoints: [{train_fraction: 1.0, n_episodes: 4}]}",
     "gpu_phase"),
    ("{name: p, source: policy, n_episodes: 5, generated_in: gpu_phase, "
     "checkpoints: [{train_fraction: 1.0, n_episodes: 4}]}", "sum"),
    ("{name: s, source: scripted_replay, n_episodes: 1, generated_in: gpu_phase, "
     "perturbations: [{type: teleport, n_episodes: 1}]}", "unknown perturbation"),
])
def test_deferred_sources_validated(tmp_path, deferred, msg):
    p = tmp_path / "r.yaml"
    p.write_text("name: x\nrepo_id: r/x\nseed: 0\nsources: [{dataset: a/b, source: human, n_episodes: 2}]\n"
                 f"deferred_sources: [{deferred}]\n")
    with pytest.raises(ConfigError, match=msg):
        M.Recipe.load(p)


def test_deferred_sources_never_share_curation_seeds():
    r = M.Recipe.load(REPO / "configs/mixtures/task_a_v1.yaml")
    a = M.policy_rollout_seeds(PROTO, 10, M.deferred_seed_offset(r, "scripted_replay"))
    b = M.policy_rollout_seeds(PROTO, 100, M.deferred_seed_offset(r, "policy_rollouts"))
    lo, hi = PROTO.curation_seed_range
    assert not set(a) & set(b) and a[0] == hi and min(b) >= lo
    e0, e1 = PROTO.eval_seed_range()
    assert not any(e0 <= s < e1 for s in a + b)


TOY_REPLAY = {"name": "scripted_replay", "source": "scripted_replay", "generated_in": "gpu_phase", "n_episodes": 6,
              "perturbations": [{"type": "jitter", "n_episodes": 3, "sigma_range": [0.3, 0.9], "onset": 0,
                                 "duration": 60},
                                {"type": "gripper_drop", "n_episodes": 3, "onset": 2, "duration": 58,
                                 "gripper_dims": [1], "open_value": 0.0}]}


def test_replay_mode_generation_on_toybed(tmp_path, scratch_env):
    """The replay-mode path: scripted expert executed in the env with Perturbations (observations follow the
    executed actions), success + perturbation labels, LeRobot-loadable. Real ALOHA generation: GPU phase."""
    srcs = M.replay_policy_sources(TOY_REPLAY, PROTO, seed_offset=0)
    assert len(srcs) == 6 and all(s.n_episodes == 1 for s in srcs)
    sig = [s.perturbations[0]["sigma"] for s in srcs[:3]]
    assert all(0.3 <= x <= 0.9 for x in sig) and len(set(sig)) == 3
    assert sig == [s.perturbations[0]["sigma"] for s in M.replay_policy_sources(TOY_REPLAY, PROTO, 0)[:3]]
    eps = M.collect_policy_episodes(srcs, lambda src: expert_action, PointReachEnv, PROTO, max_steps=60)
    for e in eps:  # executed in the env: next position = position + DT * clipped executed action
        assert np.allclose(e.states[1:, :2], e.states[:-1, :2] + 0.1 * np.clip(e.actions[:-1], -1, 1), atol=1e-5)
    assert all(np.allclose(e.actions[2:, 1], 0.0) for e in eps[3:])  # dim 1 forced to 0 from t = 2
    root = M.write_policy_dataset(eps, "rcb/toy_replay", tmp_path / "ds", fps=10, task="reach the goal",
                                  shuffle_seed=31, source_label="scripted")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("rcb/toy_replay", root=root)
    assert ds.meta.total_episodes == 6
    labels = load_labels(sidecar_path(root)).by_index()
    assert {lab.source for lab in labels.values()} == {"scripted"}
    assert sorted(lab.defects for lab in labels.values()) == [("gripper_drop",)] * 3 + [("jitter",)] * 3
    assert all(lab.quality_tier == "bad" for lab in labels.values())
    assert sum(bool(lab.success) for lab in labels.values()) == sum(e.success for e in eps)
