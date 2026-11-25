"""Scripted ALOHA transfer-cube demonstrations, generated on CPU (replay mode, D-011 / D-031).

Ported from the original ACT code, tonyzhaozh/act@742c753c0d4a5d87076c8f69e5628c79a8cc5488:
  * `BasePolicy` / `PickAndTransferPolicy` from scripted_policy.py:12-104 (waypoints, linear interpolation of
    position, quaternion and gripper; open-loop after the first step). Noise injection is not ported: ACT records
    with inject_noise=False (record_sim_episodes.py:30).
  * The recording procedure from record_sim_episodes.py:46-151: roll the end-effector policy out in the mocap env,
    take the joint trajectory, replace the gripper joints with the commanded (normalized) gripper control
    (:75-82), replay the joint trajectory in the joint-control env with the same box pose (:91-109), and keep
    max_timesteps = len(joint_traj) - 1 (obs_t, action_t) pairs (:141-155). Success = the replay's maximum reward
    equals the task's max_reward 4 (:111-118).

ACT is MIT-licensed; its notice is reproduced here as the license requires:

    MIT License

    Copyright (c) 2023 Tony Z. Zhao

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Mapping to gym-aloha 0.1.5 (the env LeRobot evaluates in):
  * Mocap env: gym_aloha's AlohaEnv refuses "end_effector_transfer_cube" (env.py:120-124 raise
    NotImplementedError), so the env is built the way that code would: assets
    bimanual_viperx_end_effector_transfer_cube.xml + TransferCubeEndEffectorTask in a dm_control
    control.Environment with control_timestep DT (env.py:133-135; ACT ee_sim_env.py:42 uses time_limit=20).
    gym-aloha samples the cube pose UNSEEDED there (tasks/sim_end_effector.py:148, sample_box_pose()); here it is
    sample_box_pose(seed), the same pose AlohaEnv.reset(seed) puts in BOX_POSE for the joint env (env.py:159-160),
    which plays the role of ACT's BOX_POSE[0] = subtask_info (record_sim_episodes.py:94). The mocap env renders no
    images (ACT never records them).
  * Joint env: gym_aloha AlohaEnv("transfer_cube", obs_type="pixels_agent_pos") (env.py:112-115, 143-147):
    observation.state = agent_pos (qpos, 14), observation.images.top = the 480x640 top camera, action = the
    14-D joint target, as in lerobot/aloha_sim_transfer_cube_scripted.
  * Quaternions without the pyquaternion dependency: axis-angle and Hamilton product exactly as pyquaternion
    (KieranWynn/pyquaternion@99025c17 quaternion.py:238-258 and :389-391, :580-587).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

EPISODE_LEN = 400  # ACT constants.py:9 (sim_transfer_cube_scripted episode_len)
MAX_REWARD = 4  # gym_aloha tasks/sim_end_effector.py:142
MEET_XYZ = np.array([0.0, 0.5, 0.25])  # scripted_policy.py:83
# Replay-vs-scripted-path tolerances, from ACT's code/assets: p95 <= the cube half-size (bimanual_viperx_ee_transfer_
# cube.xml geom size 0.02) and median <= ACT's end-effector noise scale 0.01 (scripted_policy.py:56).
PATH_P95_TOL, PATH_MEDIAN_TOL = 0.02, 0.01


# ---------------------------------------------------------------------------- quaternions (pyquaternion semantics)


def quat_from_axis_angle(axis, degrees: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    mag_sq = float(np.dot(axis, axis))
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    if abs(1.0 - mag_sq) > 1e-12:
        axis = axis / np.sqrt(mag_sq)
    theta = np.deg2rad(degrees) / 2.0
    return np.concatenate([[np.cos(theta)], axis * np.sin(theta)])


def quat_mul(a, b) -> np.ndarray:
    q = np.asarray(a, dtype=float)
    m = np.array([[q[0], -q[1], -q[2], -q[3]],
                  [q[1], q[0], -q[3], q[2]],
                  [q[2], q[3], q[0], -q[1]],
                  [q[3], -q[2], q[1], q[0]]])
    return m @ np.asarray(b, dtype=float)


# ---------------------------------------------------------------------------- the scripted policy (ACT port)


class BasePolicy:
    def __init__(self):
        self.step_count = 0
        self.left_trajectory: list[dict] | None = None
        self.right_trajectory: list[dict] | None = None

    def generate_trajectory(self, observation: dict) -> None:
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        xyz = curr_waypoint["xyz"] + (next_waypoint["xyz"] - curr_waypoint["xyz"]) * t_frac
        quat = curr_waypoint["quat"] + (next_waypoint["quat"] - curr_waypoint["quat"]) * t_frac
        gripper = curr_waypoint["gripper"] + (next_waypoint["gripper"] - curr_waypoint["gripper"]) * t_frac
        return xyz, quat, gripper

    def __call__(self, observation: dict) -> np.ndarray:
        if self.step_count == 0:  # generate the trajectory at the first timestep, then open-loop execution
            self.generate_trajectory(observation)
        if self.left_trajectory[0]["t"] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left = self.left_trajectory[0]
        if self.right_trajectory[0]["t"] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right = self.right_trajectory[0]
        lx, lq, lg = self.interpolate(self.curr_left_waypoint, next_left, self.step_count)
        rx, rq, rg = self.interpolate(self.curr_right_waypoint, next_right, self.step_count)
        self.step_count += 1
        return np.concatenate([lx, lq, [lg], rx, rq, [rg]])


class PickAndTransferPolicy(BasePolicy):
    def generate_trajectory(self, observation: dict) -> None:
        init_right = np.asarray(observation["mocap_pose_right"], dtype=float)
        init_left = np.asarray(observation["mocap_pose_left"], dtype=float)
        box_xyz = np.asarray(observation["env_state"], dtype=float)[:3]
        gripper_pick_quat = quat_mul(init_right[3:], quat_from_axis_angle([0.0, 1.0, 0.0], -60))
        meet_left_quat = quat_from_axis_angle([1.0, 0.0, 0.0], 90)
        m = MEET_XYZ
        self.left_trajectory = [
            {"t": 0, "xyz": init_left[:3], "quat": init_left[3:], "gripper": 0},  # sleep
            {"t": 100, "xyz": m + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat, "gripper": 1},  # approach meet
            {"t": 260, "xyz": m + np.array([0.02, 0, -0.02]), "quat": meet_left_quat, "gripper": 1},  # move to meet
            {"t": 310, "xyz": m + np.array([0.02, 0, -0.02]), "quat": meet_left_quat, "gripper": 0},  # close gripper
            {"t": 360, "xyz": m + np.array([-0.1, 0, -0.02]), "quat": np.array([1.0, 0, 0, 0]), "gripper": 0},  # left
            {"t": 400, "xyz": m + np.array([-0.1, 0, -0.02]), "quat": np.array([1.0, 0, 0, 0]), "gripper": 0},  # stay
        ]
        self.right_trajectory = [
            {"t": 0, "xyz": init_right[:3], "quat": init_right[3:], "gripper": 0},  # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat, "gripper": 1},  # approach
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat, "gripper": 1},  # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat, "gripper": 0},  # close
            {"t": 200, "xyz": m + np.array([0.05, 0, 0]), "quat": gripper_pick_quat, "gripper": 0},  # approach meet
            {"t": 220, "xyz": m, "quat": gripper_pick_quat, "gripper": 0},  # move to meet position
            {"t": 310, "xyz": m, "quat": gripper_pick_quat, "gripper": 1},  # open gripper
            {"t": 360, "xyz": m + np.array([0.1, 0, 0]), "quat": gripper_pick_quat, "gripper": 1},  # move right
            {"t": 400, "xyz": m + np.array([0.1, 0, 0]), "quat": gripper_pick_quat, "gripper": 1},  # stay
        ]


# ---------------------------------------------------------------------------- environments


def _ee_task_class():
    from gym_aloha.tasks.sim_end_effector import BimanualViperXEndEffectorTask, TransferCubeEndEffectorTask
    from gym_aloha.utils import sample_box_pose

    class SeededTransferCubeEETask(TransferCubeEndEffectorTask):
        """gym-aloha's mocap task with the cube pose drawn from the episode seed (see module docstring)."""

        def __init__(self, seed: int):
            super().__init__(random=None)
            self.box_seed = int(seed)

        def initialize_episode(self, physics):
            self.initialize_robots(physics)
            cube_pose = sample_box_pose(self.box_seed)
            i = physics.model.name2id("red_box_joint", "joint")
            np.copyto(physics.data.qpos[i:i + 7], cube_pose)
            BimanualViperXEndEffectorTask.initialize_episode(self, physics)

        def get_observation(self, physics):  # as the parent, without the three (unrecorded) camera renders
            import collections

            obs = collections.OrderedDict()
            obs["qpos"] = self.get_qpos(physics)
            obs["qvel"] = self.get_qvel(physics)
            obs["env_state"] = self.get_env_state(physics)
            obs["mocap_pose_left"] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
            obs["mocap_pose_right"] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()
            obs["gripper_ctrl"] = physics.data.ctrl.copy()
            obs["gripper_link_xpos"] = np.stack([physics.named.data.xpos["vx300s_left/gripper_link"],
                                                 physics.named.data.xpos["vx300s_right/gripper_link"]]).copy()
            return obs

    return SeededTransferCubeEETask


def make_ee_env(seed: int):
    from dm_control import mujoco
    from dm_control.rl import control
    from gym_aloha.constants import ASSETS_DIR, DT

    physics = mujoco.Physics.from_xml_path(str(ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"))
    return control.Environment(physics, _ee_task_class()(seed), time_limit=20, control_timestep=DT,
                               n_sub_steps=None, flat_observation=False)


# ---------------------------------------------------------------------------- generation


@dataclass
class EEResult:
    seed: int
    joint_traj: np.ndarray  # (EPISODE_LEN + 1, 14), grippers replaced by the normalized commanded control
    box_pose: np.ndarray
    ee_success: bool
    ee_targets: np.ndarray  # (EPISODE_LEN, 2, 3) commanded mocap positions (left, right)
    gripper_xpos: np.ndarray  # (EPISODE_LEN, 2, 3) achieved gripper-link positions after each step


def rollout_ee(seed: int) -> EEResult:
    """record_sim_episodes.py:50-84 on the seeded mocap env."""
    from gym_aloha.constants import normalize_puppet_gripper_position

    env = make_ee_env(seed)
    ts = env.reset()
    episode = [ts]
    policy = PickAndTransferPolicy()
    targets, xpos = [], []
    for _ in range(EPISODE_LEN):
        action = policy(ts.observation)
        targets.append(np.stack([action[:3], action[8:11]]))
        ts = env.step(action)
        episode.append(ts)
        xpos.append(ts.observation["gripper_link_xpos"])
    ee_success = max(float(t.reward) for t in episode[1:]) == MAX_REWARD
    joint_traj = np.array([np.array(t.observation["qpos"], dtype=float) for t in episode])
    for joint, t in zip(joint_traj, episode, strict=True):
        ctrl = t.observation["gripper_ctrl"]
        joint[6] = normalize_puppet_gripper_position(ctrl[0])  # :79-82
        joint[6 + 7] = normalize_puppet_gripper_position(ctrl[2])
    return EEResult(int(seed), joint_traj, np.array(episode[0].observation["env_state"], dtype=float), ee_success,
                    np.array(targets), np.array(xpos))


@dataclass
class ReplayEpisode:
    seed: int
    states: np.ndarray  # (T, 14) observation.state (qpos before each action)
    images: np.ndarray | None  # (T, 480, 640, 3) uint8 top camera, or None
    actions: np.ndarray  # (T, 14) executed actions (perturbed where a perturbation is active)
    success: bool
    max_reward: float
    perturbations: list[dict] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def replay(ee: EEResult, perturbations: list[dict] | None = None, record_images: bool = True) -> ReplayEpisode:
    """record_sim_episodes.py:91-155: replay the joint trajectory in gym-aloha's joint env with the same cube pose.
    Replay-mode perturbations (D-011) act on the executed action, so observations follow it."""
    ensure_cpu_renderer()
    from gym_aloha.env import AlohaEnv

    from robocurate.data import defects as D
    from robocurate.data.mixture import make_perturbations

    env = AlohaEnv("transfer_cube", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=ee.seed)
    from gym_aloha.tasks.sim import BOX_POSE

    if not np.allclose(BOX_POSE[0], ee.box_pose[:7], atol=1e-9):
        raise RuntimeError(f"seed {ee.seed}: joint-env cube pose differs from the mocap env's")
    specs = [dict(p) for p in (perturbations or [])]
    pol = D.PerturbedPolicy(D.ReplayPolicy(ee.joint_traj), make_perturbations(tuple(dict(p) for p in specs)),
                            seed=ee.seed)
    states, images, actions, rewards, xpos = [], [], [], [], []
    physics = env._env.physics
    for _ in range(len(ee.joint_traj)):  # "this will increase episode length by 1" (:103)
        a = np.asarray(pol(obs["agent_pos"]), dtype=float)
        states.append(np.array(obs["agent_pos"], dtype=np.float32))
        if record_images:
            images.append(obs["pixels"]["top"])
        actions.append(a.astype(np.float32))
        obs, r, _, _, _ = env.step(a)
        rewards.append(float(r))
        xpos.append(np.stack([physics.named.data.xpos["vx300s_left/gripper_link"],
                              physics.named.data.xpos["vx300s_right/gripper_link"]]).copy())
    max_r = max(rewards)
    n = len(ee.joint_traj) - 1  # truncate to max_timesteps (:141-146)
    # replay step k (action joint_traj[k]) reaches the joint state the EE stage had after its step k-1
    m = min(len(ee.gripper_xpos), len(xpos) - 1)
    d = np.linalg.norm(np.asarray(xpos[1:m + 1]) - ee.gripper_xpos[:m], axis=-1)  # (steps, 2 arms), metres
    return ReplayEpisode(ee.seed, np.stack(states[:n]), np.stack(images[:n]) if record_images else None,
                         np.stack(actions[:n]), max_r == MAX_REWARD, max_r, specs,
                         {"ee_success": ee.ee_success, "box_pose": ee.box_pose[:7].tolist(),
                          "path_dev_median_m": float(np.median(d)), "path_dev_p95_m": float(np.percentile(d, 95)),
                          "path_dev_max_m": float(d.max())})


def ensure_cpu_renderer() -> None:
    """On macOS, render on the Apple Software Renderer (sim/cpu_render.py); elsewhere dm_control's MUJOCO_GL
    backend is used as configured (osmesa = CPU, egl/glfw = GPU)."""
    import sys

    if sys.platform != "darwin":
        return
    from dm_control import _render

    if _render.BACKEND != "apple-software-cgl":
        from robocurate.sim.cpu_render import install

        install()


def generate(seed: int, perturbations: list[dict] | None = None, record_images: bool = True) -> ReplayEpisode:
    return replay(rollout_ee(seed), perturbations, record_images)


# ---------------------------------------------------------------------------- two-stage generation (CLI)
#
# Stage 1 (end-effector rollout) must run under ACT's pinned simulator, mujoco 2.3.3 / dm_control 1.0.9
# (tonyzhaozh/act conda_env.yaml): ACT's own unmodified ee_sim_env + PickAndTransferPolicy succeed 5/5 there and
# 0/5 under mujoco 3.8.1 (PROGRESS.md, overnight 3 item 3). Stage 2 (joint replay + recording, success) runs in the
# benchmark env (gym-aloha 0.1.5, mujoco 3.8.1, the CPU software renderer on macOS).

HUB_TASK = "Pick up the cube with the right arm and transfer it to the left arm."  # aloha_sim_transfer_cube_scripted
ACT_COMMIT = "742c753c0d4a5d87076c8f69e5628c79a8cc5488"


def save_ee(ee: EEResult, path) -> None:
    import mujoco

    np.savez(path, seed=ee.seed, joint_traj=ee.joint_traj, box_pose=ee.box_pose, ee_success=ee.ee_success,
             ee_targets=ee.ee_targets, gripper_xpos=ee.gripper_xpos, mujoco_version=mujoco.__version__,
             act_commit=ACT_COMMIT)


def load_ee(path) -> tuple[EEResult, str]:
    z = np.load(path, allow_pickle=False)
    return (EEResult(int(z["seed"]), z["joint_traj"], z["box_pose"], bool(z["ee_success"]), z["ee_targets"],
                     z["gripper_xpos"]), str(z["mujoco_version"]))


def write_dataset(episodes: list[ReplayEpisode], repo_id: str, root, shuffle_seed: int, source: str = "scripted",
                  checkpoint: str = "scripted_policy") -> Any:
    """LeRobot dataset with the features of lerobot/aloha_sim_transfer_cube_scripted (top-camera video, 14-D state
    and action, next.done on the last frame, fps 50, robot aloha) + a labels sidecar beside the root."""
    import json
    from pathlib import Path

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robocurate.data.labels import TIERS, EpisodeLabel, LabelsSidecar, sidecar_path, write_labels
    from robocurate.data.mixture import DEFECT_TIER

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"{root} exists")
    names = {"motors": ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle",
                        "left_wrist_rotate", "left_gripper", "right_waist", "right_shoulder", "right_elbow",
                        "right_forearm_roll", "right_wrist_angle", "right_wrist_rotate", "right_gripper"]}
    features = {"observation.images.top": {"dtype": "video", "shape": (480, 640, 3),
                                           "names": ["height", "width", "channel"]},
                "observation.state": {"dtype": "float32", "shape": (14,), "names": names},
                "action": {"dtype": "float32", "shape": (14,), "names": names},
                "next.done": {"dtype": "bool", "shape": (1,), "names": None}}
    ds = LeRobotDataset.create(repo_id=repo_id, fps=50, features=features, root=root, robot_type="aloha",
                               use_videos=True)
    order = np.random.default_rng(shuffle_seed).permutation(len(episodes))
    labels = []
    try:
        for out_i, j in enumerate(order):
            ep = episodes[int(j)]
            n = len(ep.actions)
            for t in range(n):
                ds.add_frame({"task": HUB_TASK, "observation.images.top": ep.images[t],
                              "observation.state": ep.states[t], "action": ep.actions[t],
                              "next.done": np.array([t == n - 1])})
            ds.save_episode()
            kinds = tuple(sorted({p["type"] for p in ep.perturbations}))
            labels.append(EpisodeLabel(out_i, source, checkpoint, ep.seed, defects=kinds,
                                       quality_tier=max([DEFECT_TIER[d] for d in kinds] or ["clean"], key=TIERS.index),
                                       success=bool(ep.success), perturbations=kinds,
                                       perturbation_params=json.dumps(ep.perturbations, sort_keys=True)))
    finally:
        ds.finalize()
    write_labels(LabelsSidecar(dataset=repo_id, recipe_sha256="scripted-replay", seed=shuffle_seed, episodes=labels),
                 sidecar_path(root))
    params = [{"out_index": int(k), "seed": episodes[int(j)].seed, "perturbations": episodes[int(j)].perturbations,
               "success": bool(episodes[int(j)].success), "max_reward": episodes[int(j)].max_reward,
               **episodes[int(j)].meta} for k, j in enumerate(order)]
    (root.parent / f"{root.name}.generation.json").write_text(json.dumps(params, indent=1))
    return root


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(prog="scripted_aloha")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ee", help="stage 1: end-effector rollouts -> <out>/ee_<seed>.npz (run under mujoco 2.3.3)")
    e.add_argument("--seeds", type=int, nargs="+", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "ee":
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for s in a.seeds:
            ee = rollout_ee(s)
            save_ee(ee, out / f"ee_{s}.npz")
            rows.append({"seed": s, "ee_success": ee.ee_success})
        print(json.dumps(rows))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
