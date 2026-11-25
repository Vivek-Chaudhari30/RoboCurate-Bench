"""Scripted ALOHA transfer-cube generation (robocurate.sim.scripted_aloha; ACT port) and CPU rendering. CPU only.

Fixture tests/fixtures/scripted_ee_790000_mujoco233.npz = stage 1 (end-effector rollout) of generation seed 790000
produced under ACT's simulator (mujoco 2.3.3 / dm_control 1.0.9), where ACT's own code succeeds; stage 2 (replay,
recording, success) runs here in gym-aloha 0.1.5 / mujoco 3.8.1."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from robocurate.sim import scripted_aloha as SA

FIX = Path(__file__).parent / "fixtures/scripted_ee_790000_mujoco233.npz"


def test_quaternions_match_pyquaternion_semantics():
    q = SA.quat_from_axis_angle([0.0, 1.0, 0.0], -60)  # pyquaternion quaternion.py:238-258
    np.testing.assert_allclose(q, [np.cos(np.deg2rad(-30)), 0, np.sin(np.deg2rad(-30)), 0], atol=1e-15)
    np.testing.assert_allclose(SA.quat_mul([1, 0, 0, 0], q), q)
    i, j, k = np.eye(4)[1], np.eye(4)[2], np.eye(4)[3]
    np.testing.assert_allclose(SA.quat_mul(i, j), k)  # Hamilton product (quaternion.py:580-587)
    np.testing.assert_allclose(SA.quat_mul(j, i), -k)
    np.testing.assert_allclose(SA.quat_from_axis_angle([0, 0, 2.0], 90), SA.quat_from_axis_angle([0, 0, 1.0], 90))


def test_policy_waypoints_follow_act():
    obs = {"mocap_pose_right": np.array([0.317, 0.5, 0.295, 1, 0, 0, 0.0]),
           "mocap_pose_left": np.array([-0.317, 0.5, 0.295, 1, 0, 0, 0.0]),
           "env_state": np.array([0.1, 0.5, 0.05, 1, 0, 0, 0.0])}
    pol = SA.PickAndTransferPolicy()
    acts = np.array([pol(obs) for _ in range(400)])
    assert acts.shape == (400, 16)
    np.testing.assert_allclose(acts[0, 8:11], obs["mocap_pose_right"][:3])  # t = 0: sleep pose
    np.testing.assert_allclose(acts[90, 8:11], [0.1, 0.5, 0.05 + 0.08])  # t = 90: approach the cube (+8 cm)
    np.testing.assert_allclose(acts[130, 8:11], [0.1, 0.5, 0.05 - 0.015])  # t = 130: go down
    assert acts[130, 15] == 1.0 and acts[170, 15] == 0.0  # right gripper open, then closed (scripted_policy.py:97-98)
    np.testing.assert_allclose(acts[220, 8:11], SA.MEET_XYZ)


def _egl_display_missing(exc: BaseException) -> bool:
    """True only for dm_control's missing-EGL import failure.

    CI sets MUJOCO_GL=egl (dm_control/_render/__init__.py:21-22, 73-90). With no EGL display the import
    dies in PyOpenGL (`'NoneType' object has no attribute 'eglQueryString'`) or, once libEGL loads, in
    dm_control/_render/pyopengl/egl_renderer.py:74-75 (`Cannot initialize a headless EGL display`).
    """
    pending, seen = [exc], set()
    while pending:
        cur = pending.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        msg = str(cur)
        if "eglQueryString" in msg or "Cannot initialize a headless EGL display" in msg:
            return True
        if cur.__cause__ is not None:
            pending.append(cur.__cause__)
        if cur.__context__ is not None and cur.__context__ is not cur.__cause__:
            pending.append(cur.__context__)
    return False


def _skip_if_egl_backend_unavailable() -> None:
    """Skip when MUJOCO_GL=egl cannot create a display. Any other backend, and a working EGL import, run on."""
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        return
    try:
        from dm_control import _render  # noqa: F401  reads MUJOCO_GL and imports that backend
    except Exception as exc:
        if _egl_display_missing(exc):
            pytest.skip(f"MUJOCO_GL=egl backend failed to initialize: {exc}")
        raise


def test_egl_display_missing_matches_the_runner_failure_only():
    assert _egl_display_missing(AttributeError("'NoneType' object has no attribute 'eglQueryString'"))
    assert _egl_display_missing(ImportError("Cannot initialize a headless EGL display."))
    wrapped = RuntimeError("dm_control import")
    wrapped.__cause__ = AttributeError("'NoneType' object has no attribute 'eglQueryString'")
    assert _egl_display_missing(wrapped)
    assert not _egl_display_missing(AttributeError("'NoneType' object has no attribute 'joint_traj'"))
    assert not _egl_display_missing(AssertionError("joint trajectories differ"))


def test_ee_stage_is_deterministic_per_seed_and_uses_the_seeded_pose():
    _skip_if_egl_backend_unavailable()
    from gym_aloha.utils import sample_box_pose

    a, b, c = SA.rollout_ee(700000), SA.rollout_ee(700000), SA.rollout_ee(700001)
    assert np.array_equal(a.joint_traj, b.joint_traj) and np.array_equal(a.ee_targets, b.ee_targets)
    np.testing.assert_allclose(a.box_pose[:7], sample_box_pose(700000))  # = AlohaEnv.reset(seed)'s BOX_POSE
    assert not np.allclose(a.box_pose, c.box_pose)
    assert a.joint_traj.shape == (SA.EPISODE_LEN + 1, 14)


def test_cpu_render_refuses_hardware_backends(monkeypatch):
    from robocurate.sim import cpu_render

    monkeypatch.setenv("MUJOCO_GL", "glfw")
    with pytest.raises(cpu_render.CPURenderError, match="disable"):
        cpu_render.install()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="the Apple software renderer exists only on macOS")
def test_replay_matches_scripted_path_and_succeeds():
    """Stage 2 of a stage-1 trajectory from ACT's simulator: success in the benchmark env, gripper path within
    ACT-derived tolerances (p95 <= cube half-size 0.02 m, median <= ACT's EE noise scale 0.01 m)."""
    if os.environ.get("MUJOCO_GL", "").lower() not in ("disable", "off", "false", "0", "disabled"):
        pytest.skip("needs MUJOCO_GL=disable at interpreter start (CPU software renderer)")
    ee, ver = SA.load_ee(FIX)
    assert ver == "2.3.3" and ee.ee_success
    ep = SA.replay(ee)
    assert ep.success and ep.max_reward == SA.MAX_REWARD
    assert ep.meta["path_dev_p95_m"] <= SA.PATH_P95_TOL and ep.meta["path_dev_median_m"] <= SA.PATH_MEDIAN_TOL
    assert ep.images.shape == (400, 480, 640, 3) and ep.actions.shape == (400, 14) and ep.states.shape == (400, 14)
    from dm_control import _render

    assert _render.BACKEND == "apple-software-cgl" and not _render.USING_GPU


def test_dataset_writer_labels_sidecar(tmp_path, scratch_env):
    """write_dataset: LeRobot-loadable, Hub-identical features, labels carry source, perturbation type +
    parameters and success; the sidecar sits beside the root."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robocurate.data.labels import load_labels, sidecar_path

    rng = np.random.default_rng(0)

    def ep(seed, pert, ok):
        return SA.ReplayEpisode(seed, rng.normal(size=(6, 14)).astype(np.float32),
                                rng.integers(0, 255, size=(6, 480, 640, 3), dtype=np.uint8),
                                rng.normal(size=(6, 14)).astype(np.float32), ok, 4.0 if ok else 2.0, pert)

    eps = [ep(700000, [{"type": "jitter", "sigma": 0.643, "onset": 0, "duration": 400}], False),
           ep(700001, [{"type": "gripper_drop", "onset": 135, "duration": 265, "gripper_dims": [13],
                        "open_value": 1.0}], True),
           ep(700002, [], True)]
    root = SA.write_dataset(eps, "rcb/test_replay", tmp_path / "ds", shuffle_seed=31)
    ds = LeRobotDataset("rcb/test_replay", root=root)
    assert ds.meta.total_episodes == 3 and ds.meta.total_frames == 18 and ds.meta.fps == 50
    assert {"observation.images.top", "observation.state", "action", "next.done"} <= set(ds.meta.features)
    lab = load_labels(sidecar_path(root)).by_index()
    assert not (root / "labels.json").exists() and sidecar_path(root).parent == root.parent
    by_seed = {lab[i].source_episode: lab[i] for i in lab}
    assert by_seed[700000].defects == ("jitter",) and by_seed[700000].quality_tier == "bad"
    assert json.loads(by_seed[700000].perturbation_params)[0]["sigma"] == 0.643
    assert by_seed[700001].defects == ("gripper_drop",) and by_seed[700001].success is True
    assert by_seed[700002].defects == () and by_seed[700002].quality_tier == "clean"
    assert {lab[i].source for i in lab} == {"scripted"}
    assert [lab[i].source_episode for i in range(3)] != [700000, 700001, 700002]  # shuffled (D-021)
