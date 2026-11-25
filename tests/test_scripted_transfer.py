"""Scripted Aloha transfer-cube policy: waypoint schedule, unperturbed success, replay-path perturbations.

Image episodes are not produced here. write_policy_dataset records observation.state and action only.
"""

import numpy as np
import pytest

from robocurate.config import load_protocol
from robocurate.data import mixture as M
from robocurate.data.labels import load_labels, sidecar_path
from robocurate.data.scripted_transfer import (
    EPISODE_LENGTH,
    RIGHT_GRIPPER_DIM,
    TransferCubeStateEnv,
    ee_command_sequence,
    quat_from_axis_angle,
    quat_mul,
    scripted_joint_actions,
    scripted_replay_bindings,
)
from robocurate.paths import repo_root

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")


def _scripted_replay():
    recipe = M.Recipe.load(REPO / "configs/mixtures/task_a_v1.yaml")
    spec = next(x for x in recipe.deferred_sources if x["source"] == "scripted_replay")
    offset = M.deferred_seed_offset(recipe, spec["name"])
    return recipe, spec, offset


def test_right_gripper_command_closes_on_the_published_schedule():
    """EE command, before the one-step ctrl delay of the joint recording. No simulation, no noise."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    y_down = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -60.0)
    assert quat_mul(identity, y_down) == pytest.approx(y_down)
    assert y_down[0] == pytest.approx(np.cos(np.deg2rad(-30.0)))
    assert y_down[2] == pytest.approx(np.sin(np.deg2rad(-30.0)))
    ee = ee_command_sequence(np.array([0.1, 0.5, 0.05]))
    assert ee.shape == (EPISODE_LENGTH, 16)
    right = ee[:, 15]
    assert right[0] == pytest.approx(0.0)
    assert right[90] == pytest.approx(1.0)
    assert right[130] == pytest.approx(1.0)
    assert right[150] == pytest.approx(0.5)
    assert right[170] == pytest.approx(0.0)
    assert np.allclose(ee, ee_command_sequence(np.array([0.1, 0.5, 0.05])))


def _replay_state_error(episode) -> float:
    env = TransferCubeStateEnv()
    try:
        obs, _ = env.reset(seed=episode.seed)
        err = 0.0
        for t, action in enumerate(episode.actions):
            err = max(err, float(np.max(np.abs(obs - episode.states[t]))))
            obs, _, _, _, _ = env.step(action)
        return err
    finally:
        env.close()


def _clean_divergence(episode) -> float:
    """How far the recorded states are from executing the unperturbed joint commands."""
    env = TransferCubeStateEnv()
    try:
        obs, _ = env.reset(seed=episode.seed)
        clean = scripted_joint_actions(env.cube_pose)
        err = 0.0
        for t in range(len(episode.actions)):
            err = max(err, float(np.max(np.abs(obs - episode.states[t]))))
            obs, _, terminated, truncated, _ = env.step(clean[t])
            if terminated or truncated:
                break
        return err
    finally:
        env.close()


def test_unperturbed_succeeds_on_scripted_replay_curation_seeds():
    _recipe, spec, offset = _scripted_replay()
    n = int(spec["n_episodes"])
    sources = [M.PolicySource(f"scripted_clean_{i}", "scripted", 0, 1, ()) for i in range(n)]
    env_factory, load_policy = scripted_replay_bindings()
    episodes = M.collect_policy_episodes(
        sources, load_policy, env_factory, PROTO, max_steps=EPISODE_LENGTH, seed_offset=offset
    )
    assert len(episodes) == n
    assert all(e.success for e in episodes)
    assert all(len(e.actions) < EPISODE_LENGTH for e in episodes)  # env ends on reward == 4
    # Joint-command gripper is the previous step's EE command (ctrl recorded in place of measured qpos).
    gripper = scripted_joint_actions(_cube(episodes[0]))[:, RIGHT_GRIPPER_DIM]
    assert gripper[130] == pytest.approx(1.0, abs=1e-6)
    assert gripper[135] == pytest.approx(0.9, abs=1e-6)
    assert gripper[152] == pytest.approx(0.475, abs=1e-6)


def _cube(episode):
    env = TransferCubeStateEnv()
    try:
        env.reset(seed=episode.seed)
        return env.cube_pose.copy()
    finally:
        env.close()


def test_replay_perturbations_match_executed_actions_and_labels(tmp_path, scratch_env):
    recipe, spec, offset = _scripted_replay()
    sources = M.replay_policy_sources(spec, PROTO, offset)
    assert len(sources) == 10
    sigmas = [s.perturbations[0]["sigma"] for s in sources if s.perturbations[0]["type"] == "jitter"]
    assert len(sigmas) == 5 and all(0.173 <= x <= 0.726 for x in sigmas)
    env_factory, load_policy = scripted_replay_bindings()
    episodes = M.collect_policy_episodes(
        sources, load_policy, env_factory, PROTO, max_steps=EPISODE_LENGTH, seed_offset=offset
    )
    by_kind = {}
    for episode in episodes:
        kind = episode.perturbation_names[0]
        by_kind.setdefault(kind, []).append(episode)
        assert np.isfinite(episode.states).all() and np.isfinite(episode.actions).all()
        assert _replay_state_error(episode) == 0.0
    assert sorted(by_kind) == ["gripper_drop", "jitter"]
    for episode in by_kind["jitter"][:1] + by_kind["gripper_drop"][:1]:
        assert _clean_divergence(episode) > 0.1
    for episode in by_kind["gripper_drop"]:
        assert len(episode.actions) > 135
        assert np.allclose(episode.actions[135:, RIGHT_GRIPPER_DIM], 1.0)
    root = M.write_policy_dataset(
        episodes, "robocurate/task_a_v1_scripted_replay_state", tmp_path / "ds", fps=50,
        task="transfer the cube", shuffle_seed=recipe.seed, source_label="scripted",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("robocurate/task_a_v1_scripted_replay_state", root=root)
    assert ds.meta.total_episodes == len(episodes)
    labels = load_labels(sidecar_path(root)).by_index()
    assert {lab.source for lab in labels.values()} == {"scripted"}
    assert {lab.quality_tier for lab in labels.values()} == {"bad"}
    assert sorted(lab.defects for lab in labels.values()) == [("gripper_drop",)] * 5 + [("jitter",)] * 5
    assert sum(bool(lab.success) for lab in labels.values()) == sum(bool(e.success) for e in episodes)


def test_local_rollout_backend_blocks_on_camera_writer(tmp_path):
    from robocurate.launch.rollouts import generate

    with pytest.raises(SystemExit, match="camera-frame") as exc:
        generate(
            str(REPO / "configs/mixtures/task_a_v1.yaml"), tmp_path, "local", None, None,
            str(REPO / "configs/protocol.yaml"),
        )
    assert "none in gym-aloha" not in str(exc.value)
