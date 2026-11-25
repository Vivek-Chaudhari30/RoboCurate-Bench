"""Camera frames on policy-rollout episodes (C1). Stub camera + stub policy, no ACT/Diffusion checkpoint."""

import numpy as np
import pytest

from robocurate.config import load_protocol
from robocurate.data import mixture as M
from robocurate.launch.rollouts import generate, load_pretrained_policy, local_missing
from robocurate.paths import repo_root

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")


class _TopCamEnv:
    """Gym-style obs: pixels['top'] + agent_pos. Three steps, then truncated."""

    def reset(self, seed=None):
        self.x = np.random.default_rng(seed).uniform(-0.5, 0.5, 2).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        img = np.zeros((32, 48, 3), np.uint8)
        img[..., 0] = 30 + 40 * self.t
        img[:8, :8, 1] = 220
        img[:, -4:, 2] = 15 * self.t
        return {"pixels": {"top": img.copy()}, "agent_pos": self.x.copy()}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:2], -1, 1)
        self.x = (self.x + 0.1 * a).astype(np.float32)
        self.t += 1
        return self._obs(), 0.0, False, self.t >= 3, {"is_success": False}


def _stub_policy(obs):
    return np.clip(-np.asarray(obs["agent_pos"], dtype=np.float32), -1, 1)


class _PushTPixels:
    def reset(self, seed=None):
        self.x = np.zeros(2, np.float32)
        return {"pixels": np.zeros((16, 16, 3), np.uint8), "agent_pos": self.x.copy()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, True, False, {"is_success": False}


def test_stub_camera_episode_round_trips(tmp_path, scratch_env):
    """A stub policy's camera frames are stored and LeRobotDataset reloads the episode."""
    src = M.PolicySource("stub", "fake://stub", 1, 1)
    eps = M.collect_policy_episodes([src], lambda _src: _stub_policy, _TopCamEnv, PROTO, max_steps=3)
    assert len(eps) == 1
    ep = eps[0]
    assert set(ep.cameras) == {"observation.images.top"}
    assert ep.cameras["observation.images.top"].shape == (3, 32, 48, 3)
    assert ep.cameras["observation.images.top"].dtype == np.uint8
    assert ep.states.shape == (3, 2) and ep.actions.shape == (3, 2)

    root = M.write_policy_dataset(eps, "rcb/stub_camera", tmp_path / "ds", fps=10, task="transfer", shuffle_seed=0)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("rcb/stub_camera", root=root, video_backend="pyav")
    ft = ds.meta.features["observation.images.top"]
    assert ft["dtype"] == "video" and tuple(ft["shape"]) == (32, 48, 3)
    assert ds.meta.total_episodes == 1 and ds.meta.total_frames == 3
    for i in range(3):
        item = ds[i]
        state = item["observation.state"]
        action = item["action"]
        state = state.numpy() if hasattr(state, "numpy") else np.asarray(state)
        action = action.numpy() if hasattr(action, "numpy") else np.asarray(action)
        np.testing.assert_allclose(np.asarray(state, np.float32).reshape(-1), ep.states[i], atol=1e-5)
        np.testing.assert_allclose(np.asarray(action, np.float32).reshape(-1), ep.actions[i], atol=1e-5)
        got = M._to_hwc_uint8(item["observation.images.top"])
        # LeRobot's default encoder is lossy (SVT-AV1, YUV420). The frame still has to be this camera.
        np.testing.assert_allclose(got, ep.cameras["observation.images.top"][i], atol=20)
        assert got[:8, :8, 1].mean() > 180


def test_single_pixels_array_is_observation_image():
    """PushT's pixels array is the observation.image feature (envs/configs.py features_map)."""
    src = M.PolicySource("stub", "fake://stub", 1, 1)
    ep = M.collect_policy_episodes(
        [src], lambda _src: (lambda obs: np.zeros(2, np.float32)), _PushTPixels, PROTO, max_steps=1,
    )[0]
    assert set(ep.cameras) == {"observation.image"}
    assert ep.cameras["observation.image"].shape == (1, 16, 16, 3)


def test_local_backend_does_not_list_the_camera_adapter(tmp_path):
    task_a = str(REPO / "configs/mixtures/task_a_v1.yaml")
    with pytest.raises(SystemExit) as ei:
        generate(task_a, tmp_path / "a", "local", None, None, str(REPO / "configs/protocol.yaml"))
    msg = str(ei.value)
    assert "scripted transfer-cube policy" in msg
    assert "camera" not in msg.lower() and "adapter" not in msg.lower()
    assert local_missing(M.Recipe.load(task_a)) == [
        "a scripted transfer-cube policy (none in gym-aloha 0.1.5 or lerobot 0.6.1)"
    ]

    task_b = str(REPO / "configs/mixtures/pusht_diffusion_v1.yaml")
    assert local_missing(M.Recipe.load(task_b)) == []
    with pytest.raises(SystemExit) as ei:
        generate(task_b, tmp_path / "b", "local", None, None, str(REPO / "configs/protocol.yaml"))
    msg = str(ei.value)
    assert "--env-type" in msg
    assert "camera" not in msg.lower() and "adapter" not in msg.lower()


def test_load_pretrained_policy_requires_a_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pretrained_policy(tmp_path / "no-such-pretrained-model", env_cfg=object())
