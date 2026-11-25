"""Scripted policy for gym_aloha/AlohaTransferCube-v0.

gym-aloha 0.1.5 and lerobot 0.6.1 do not ship a scripted transfer-cube policy. The joint env is
registered (gym_aloha/__init__.py:13-21, max_episode_steps 300) and consumes 14-dim absolute joint
targets with normalized grippers (tasks/sim.py:40-54, 0 close, 1 open). The end-effector task the
original waypoints need is shipped (tasks/sim_end_effector.py:139 TransferCubeEndEffectorTask; the
mocap observation is marked "used in scripted policy" at :123) but AlohaEnv refuses to build it
(env.py:120-121 raises NotImplementedError before the EE xml).

This module follows the standard two-stage recording procedure: roll end-effector waypoints in the
shipped EE task, convert each EE observation into a 14-dim joint command (arm qpos, gripper command
in place of measured gripper qpos), and replay those commands in AlohaTransferCube-v0. Perturbations
are applied only by data.defects.PerturbedPolicy at replay time (D-011). There is no extra action noise.

Camera frames are not recorded. tasks/sim.py:92-94 renders three cameras inside get_observation; the
state env below replaces that method so a CPU test does not render. write_policy_dataset stores
observation.state and action only. Image episodes stay blocked on that writer.

Episode length is 400, matching lerobot's AlohaEnv.episode_length (envs/configs.py:152) and
gym_kwargs max_episode_steps (envs/configs.py:187). The registered 300-step cap ends before the
left gripper closes at waypoint t=310. gymnasium make() applies the override
(gymnasium/envs/registration.py:653, :812-816).
"""

from __future__ import annotations

import ctypes.util
import itertools
import os
from dataclasses import dataclass

import numpy as np

# MuJoCo reads this on first import. Rendering is skipped; egl is the task's required backend.
os.environ.setdefault("MUJOCO_GL", "egl")

# One token per env.reset, unique for the process. Not id(env): a collected env's id can be
# reused for the next TransferCubeStateEnv, and that env's episode_id is 1 again after reset.
_RESET_TOKENS = itertools.count(1)

EPISODE_LENGTH = 400
RIGHT_GRIPPER_DIM = 13  # gym_aloha/constants.py ACTIONS[13] == "right_arm_gripper"
STATE_DIM = 14

# Start mocap of BimanualViperXEndEffectorTask.initialize_robots (sim_end_effector.py:66-70).
_LEFT_MOCAP = np.array([-0.31718881, 0.5, 0.29525084, 1.0, 0.0, 0.0, 0.0])
_RIGHT_MOCAP = np.array([0.31718881, 0.49999888, 0.29525084, 1.0, 0.0, 0.0, 0.0])

_JOINT_CACHE: dict[bytes, np.ndarray] = {}


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, both (w, x, y, z). Matches pyquaternion.Quaternion.__mul__."""
    aw, ax, ay, az = (float(x) for x in np.asarray(a, dtype=np.float64).reshape(4))
    bw, bx, by, bz = (float(x) for x in np.asarray(b, dtype=np.float64).reshape(4))
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_from_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    """Unit quaternion (w, x, y, z) for a right-handed rotation of `degrees` about `axis`."""
    half = np.deg2rad(degrees) / 2.0
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    xyz = np.sin(half) * axis
    return np.array([np.cos(half), xyz[0], xyz[1], xyz[2]], dtype=np.float64)


@dataclass
class _Waypoint:
    t: int
    xyz: np.ndarray
    quat: np.ndarray
    gripper: float


def _lerp(curr: _Waypoint, nxt: _Waypoint, t: int) -> np.ndarray:
    frac = (t - curr.t) / (nxt.t - curr.t)
    xyz = curr.xyz + (nxt.xyz - curr.xyz) * frac
    quat = curr.quat + (nxt.quat - curr.quat) * frac
    grip = curr.gripper + (nxt.gripper - curr.gripper) * frac
    return np.concatenate([xyz, quat, [grip]])


class _EeTransferPolicy:
    """Open-loop end-effector waypoints for a bimanual cube handoff. No action noise."""

    def __init__(self) -> None:
        self.step_count = 0
        self._ready = False
        self.left: list[_Waypoint] = []
        self.right: list[_Waypoint] = []
        self.curr_left: _Waypoint | None = None
        self.curr_right: _Waypoint | None = None

    def _trajectory(self, obs: dict) -> None:
        left0 = np.asarray(obs["mocap_pose_left"], dtype=np.float64)
        right0 = np.asarray(obs["mocap_pose_right"], dtype=np.float64)
        box_xyz = np.asarray(obs["env_state"], dtype=np.float64)[:3]
        pick_quat = quat_mul(right0[3:], quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), -60.0))
        meet_left = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), 90.0)
        meet = np.array([0.0, 0.5, 0.25])
        self.left = [
            _Waypoint(0, left0[:3].copy(), left0[3:].copy(), 0.0),
            _Waypoint(100, meet + np.array([-0.1, 0.0, -0.02]), meet_left.copy(), 1.0),
            _Waypoint(260, meet + np.array([0.02, 0.0, -0.02]), meet_left.copy(), 1.0),
            _Waypoint(310, meet + np.array([0.02, 0.0, -0.02]), meet_left.copy(), 0.0),
            _Waypoint(360, meet + np.array([-0.1, 0.0, -0.02]), np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
            _Waypoint(400, meet + np.array([-0.1, 0.0, -0.02]), np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
        ]
        self.right = [
            _Waypoint(0, right0[:3].copy(), right0[3:].copy(), 0.0),
            _Waypoint(90, box_xyz + np.array([0.0, 0.0, 0.08]), pick_quat.copy(), 1.0),
            _Waypoint(130, box_xyz + np.array([0.0, 0.0, -0.015]), pick_quat.copy(), 1.0),
            _Waypoint(170, box_xyz + np.array([0.0, 0.0, -0.015]), pick_quat.copy(), 0.0),
            _Waypoint(200, meet + np.array([0.05, 0.0, 0.0]), pick_quat.copy(), 0.0),
            _Waypoint(220, meet.copy(), pick_quat.copy(), 0.0),
            _Waypoint(310, meet.copy(), pick_quat.copy(), 1.0),
            _Waypoint(360, meet + np.array([0.1, 0.0, 0.0]), pick_quat.copy(), 1.0),
            _Waypoint(400, meet + np.array([0.1, 0.0, 0.0]), pick_quat.copy(), 1.0),
        ]
        self._ready = True

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._ready:
            self._trajectory(obs)
        if self.left[0].t == self.step_count:
            self.curr_left = self.left.pop(0)
        if self.right[0].t == self.step_count:
            self.curr_right = self.right.pop(0)
        if self.curr_left is None or self.curr_right is None:
            raise RuntimeError("scripted EE policy stepped before its first waypoint")
        left = _lerp(self.curr_left, self.left[0], self.step_count)
        right = _lerp(self.curr_right, self.right[0], self.step_count)
        action = np.concatenate([left, right])
        self.step_count += 1
        return action


def ee_command_sequence(box_xyz: np.ndarray, n: int = EPISODE_LENGTH) -> np.ndarray:
    """n end-effector commands (16,) from the fixed start mocap. No simulation and no noise."""
    obs = {
        "mocap_pose_left": _LEFT_MOCAP,
        "mocap_pose_right": _RIGHT_MOCAP,
        "env_state": np.concatenate([np.asarray(box_xyz, dtype=np.float64), [1.0, 0.0, 0.0, 0.0]]),
    }
    policy = _EeTransferPolicy()
    return np.stack([policy(obs) for _ in range(n)])


def _joint_command(obs: dict) -> np.ndarray:
    """14-dim joint target: measured arm qpos, commanded gripper (ctrl) in place of measured gripper qpos."""
    from gym_aloha.constants import normalize_puppet_gripper_position

    q = np.asarray(obs["qpos"], dtype=np.float64).copy()
    ctrl = obs["gripper_ctrl"]
    q[6] = normalize_puppet_gripper_position(ctrl[0])
    q[RIGHT_GRIPPER_DIM] = normalize_puppet_gripper_position(ctrl[2])
    return q


def _roll_ee_joint_actions(cube_pose: np.ndarray) -> np.ndarray:
    """Roll the EE waypoints at this cube pose and return (EPISODE_LENGTH, 14) joint commands."""
    _prepare_mujoco_import()
    import collections

    from dm_control import mujoco
    from dm_control.rl import control
    from gym_aloha.constants import ASSETS_DIR, DT
    from gym_aloha.tasks.sim_end_effector import (
        BimanualViperXEndEffectorTask,
        TransferCubeEndEffectorTask,
    )

    class _FixedCube(TransferCubeEndEffectorTask):
        def __init__(self, pose: np.ndarray):
            super().__init__(random=False)
            self._pose = np.asarray(pose, dtype=np.float64)

        def initialize_episode(self, physics):
            # Same as TransferCubeEndEffectorTask.initialize_episode (sim_end_effector.py:144-153)
            # except the cube pose is the joint env's pose, not an unseeded sample_box_pose().
            self.initialize_robots(physics)
            box_start = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[box_start : box_start + 7], self._pose)
            BimanualViperXEndEffectorTask.initialize_episode(self, physics)

        def get_observation(self, physics):
            # sim_end_effector.py:113-132 without the three camera renders (:120-122).
            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            obs["qvel"] = self.get_qvel(physics)
            obs["env_state"] = self.get_env_state(physics)
            obs["mocap_pose_left"] = np.concatenate(
                [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
            ).copy()
            obs["mocap_pose_right"] = np.concatenate(
                [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
            ).copy()
            obs["gripper_ctrl"] = physics.data.ctrl.copy()
            return obs

    xml = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
    physics = mujoco.Physics.from_xml_path(str(xml))
    # Shipped weld is soft (solimp 0.25, xml equality). MuJoCo 3.8 then misses the cube.
    # Stiffer only in this trajectory generator; AlohaTransferCube-v0 is unchanged.
    physics.model.eq_solimp[:] = np.array([0.95, 0.99, 0.001, 0.5, 2.0])
    physics.model.eq_solref[:] = np.array([0.005, 1.0])
    task = _FixedCube(cube_pose)
    ee = control.Environment(
        physics, task, time_limit=float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False
    )
    ts = ee.reset()
    policy = _EeTransferPolicy()
    actions = []
    for _ in range(EPISODE_LENGTH):
        actions.append(_joint_command(ts.observation))
        ts = ee.step(policy(ts.observation))
    return np.stack(actions)


def scripted_joint_actions(cube_pose: np.ndarray) -> np.ndarray:
    """(EPISODE_LENGTH, 14) joint commands for this cube pose. Cached by the pose bytes. No action noise."""
    pose = np.asarray(cube_pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"cube pose must be (7,), got {pose.shape}")
    key = pose.tobytes()
    cached = _JOINT_CACHE.get(key)
    if cached is None:
        cached = _roll_ee_joint_actions(pose)
        _JOINT_CACHE[key] = cached
    return cached.copy()


def new_reset_token() -> int:
    """Process-unique id for one env.reset. Never reused, unlike id(env)."""
    return next(_RESET_TOKENS)


def _prepare_mujoco_import() -> None:
    """dm_control loads its GL backend at import (_render/__init__.py:73-90).

    CI sets MUJOCO_GL=egl (.github/workflows/ci.yml). With no libEGL that import raises
    (OpenGL.raw.EGL). These rollouts never render, so use the no-renderer backend
    (constants.NO_RENDERER includes 'disable') only when libEGL is absent.
    """
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        return
    if ctypes.util.find_library("EGL") is None:
        os.environ["MUJOCO_GL"] = "disable"


def _skip_camera_renders(task) -> None:
    """Replace task.get_observation so AlohaEnv.step does not render (tasks/sim.py:92-94)."""

    def get_observation(physics, _task=task):
        blank = np.zeros((2, 2, 3), dtype=np.uint8)
        return {
            "qpos": _task.get_qpos(physics),
            "qvel": _task.get_qvel(physics),
            "env_state": _task.get_env_state(physics),
            "images": {"top": blank, "angle": blank, "vis": blank},
        }

    task.get_observation = get_observation


class TransferCubeStateEnv:
    """AlohaTransferCube-v0 with a 14-dim qpos observation and no camera frames.

    `info["is_success"]` is reward == 4 (env.py:180). `cube_pose` is the 7-dim box pose after reset
    (tasks/sim.py:121-122), which the scripted policy reads. Declared action Box is [-1, 1]
    (env.py action_space) but the task executes absolute joint targets, so this wrapper does not clip.
    """

    def __init__(self, max_episode_steps: int = EPISODE_LENGTH):
        _prepare_mujoco_import()
        import gym_aloha  # noqa: F401  (registers gym_aloha/AlohaTransferCube-v0)
        import gymnasium as gym

        self._gym = gym.make(
            "gym_aloha/AlohaTransferCube-v0",
            obs_type="pixels_agent_pos",
            disable_env_checker=True,
            max_episode_steps=max_episode_steps,
        )
        task = self._gym.unwrapped._env.task
        _skip_camera_renders(task)
        self._task = task
        self.cube_pose: np.ndarray | None = None
        self.episode_id = 0
        self.reset_token = 0

    def _qpos(self) -> np.ndarray:
        physics = self._gym.unwrapped._env.physics
        return np.asarray(self._task.get_qpos(physics), dtype=np.float32)

    def reset(self, seed=None):
        self._gym.reset(seed=seed)
        pose = np.asarray(self._task.get_env_state(self._gym.unwrapped._env.physics), dtype=np.float64).copy()
        if pose.shape != (7,):
            raise RuntimeError(f"transfer-cube env_state should be (7,), got {pose.shape}")
        self.cube_pose = pose
        self.episode_id += 1
        self.reset_token = new_reset_token()
        return self._qpos(), {"is_success": False}

    def step(self, action):
        _obs, reward, terminated, truncated, info = self._gym.step(np.asarray(action, dtype=np.float64))
        return self._qpos(), reward, terminated, truncated, info

    def close(self) -> None:
        self._gym.close()


class EnvSlot:
    """Filled by the env factory before the policy is called (collect_policy_episodes order)."""

    env: TransferCubeStateEnv | None = None


class ScriptedTransferPolicy:
    """obs -> 14-dim joint command. The trajectory is fixed by the cube pose of the reset env.

    One instance is reused across episodes. A new env.reset_token regenerates the trajectory
    and restarts the step index. The token is not id(env): collect_policy_episodes builds a
    new env per episode and drops the previous one, and CPython can reuse that id while the
    new env's episode_id is 1 again. The returned array is a copy; PerturbedPolicy is what
    mutates the executed action (D-011).
    """

    def __init__(self, slot: EnvSlot):
        self._slot = slot
        self._token: int | None = None
        self._actions: np.ndarray | None = None
        self._t = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        env = self._slot.env
        if env is None or env.cube_pose is None:
            raise RuntimeError("scripted transfer policy called before the env was reset")
        if env.reset_token != self._token:
            self._actions = scripted_joint_actions(env.cube_pose)
            self._t = 0
            self._token = env.reset_token
        assert self._actions is not None
        t = min(self._t, len(self._actions) - 1)
        self._t += 1
        return self._actions[t].copy()


def scripted_replay_bindings():
    """(env_factory, load_policy) for data.mixture.collect_policy_episodes."""
    slot = EnvSlot()

    def env_factory() -> TransferCubeStateEnv:
        env = TransferCubeStateEnv()
        slot.env = env
        return env

    def load_policy(_source) -> ScriptedTransferPolicy:
        return ScriptedTransferPolicy(slot)

    return env_factory, load_policy
