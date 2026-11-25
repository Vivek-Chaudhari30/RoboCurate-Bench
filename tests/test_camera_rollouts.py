"""Camera frames on policy-rollout episodes (C1). Stub camera + stub policy, no ACT/Diffusion checkpoint."""

import os
import stat
import subprocess

import numpy as np
import pytest

from robocurate.config import load_protocol
from robocurate.data import mixture as M
from robocurate.data.labels import load_labels, sidecar_path
from robocurate.launch.rollouts import generate, load_pretrained_policy, local_missing, main, rollout_env_config
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


def test_write_policy_episodes_releases_frames_before_the_next(tmp_path, scratch_env, monkeypatch):
    """Each episode's camera arrays are dropped before the next rollout starts."""
    seen: list[M.PolicyEpisode] = []
    copies: list[np.ndarray] = []
    real = M._rollout_one

    def wrapped(*args, **kwargs):
        assert all(not ep.cameras for ep in seen), "a previous episode still holds camera frames"
        ep = real(*args, **kwargs)
        assert sum(a.nbytes for a in ep.cameras.values()) > 0
        copies.append(ep.cameras["observation.images.top"][0].copy())
        seen.append(ep)
        return ep

    monkeypatch.setattr(M, "_rollout_one", wrapped)
    srcs = [
        M.PolicySource("early", "fake://early", 10, 2),
        M.PolicySource("late", "fake://late", 90, 2),
    ]
    root, kept = M.write_policy_episodes(
        srcs, lambda _src: _stub_policy, _TopCamEnv, PROTO, max_steps=3, seed_offset=0,
        repo_id="rcb/stream_cam", root=tmp_path / "ds", fps=10, task="transfer", shuffle_seed=7,
    )
    assert len(seen) == 4
    assert all(not ep.cameras for ep in seen)
    assert all(not ep.cameras for ep in kept)
    order = np.random.default_rng(7).permutation(4)
    steps = [10, 10, 90, 90]
    labels = load_labels(sidecar_path(root)).by_index()
    assert [labels[i].checkpoint_step for i in range(4)] == [steps[int(j)] for j in order]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("rcb/stream_cam", root=root, video_backend="pyav")
    assert ds.meta.total_episodes == 4 and ds.meta.total_frames == 12
    got = M._to_hwc_uint8(ds[0]["observation.images.top"])
    np.testing.assert_allclose(got, copies[0], atol=20)
    assert got[:8, :8, 1].mean() > 180


def test_generate_writes_each_episode_instead_of_collecting_them(tmp_path, scratch_env, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("generate must not collect every episode before writing")

    monkeypatch.setattr(M, "collect_policy_episodes", boom)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        "name: tiny_stream\n"
        "repo_id: robocurate/tiny_stream\n"
        "seed: 3\n"
        "sources: []\n"
        "deferred_sources:\n"
        "  - name: policy_rollouts\n"
        "    source: policy\n"
        "    generated_in: gpu_phase\n"
        "    n_episodes: 3\n"
        "    checkpoints:\n"
        "      - {train_fraction: 0.10, n_episodes: 2}\n"
        "      - {train_fraction: 1.00, n_episodes: 1}\n"
    )
    rep = generate(str(recipe), tmp_path / "out", "fake", None, None, str(REPO / "configs/protocol.yaml"))
    assert rep["sources"]["policy_rollouts"]["n"] == 3


def test_omitted_env_task_keeps_the_aloha_default():
    """Passing task=None would override AlohaInsertion-v0. Omitting the kwarg keeps it."""
    from lerobot.envs.factory import make_env_config

    assert make_env_config("aloha", task=None).task is None
    assert rollout_env_config("aloha", None).task == "AlohaInsertion-v0"
    assert rollout_env_config("aloha", "AlohaTransferCube-v0").task == "AlohaTransferCube-v0"


def test_device_help_does_not_claim_cuda_visible_devices_selects_it(capsys):
    with pytest.raises(SystemExit) as ei:
        main(["--help"])
    assert ei.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "default: cpu" in text
    assert "--device cuda" in text
    assert "CUDA_VISIBLE_DEVICES does not select the device" in text
    assert "unless a GPU task sets CUDA_VISIBLE_DEVICES" not in text


class _StateEnv:
    """State only, three steps. Actions are whatever the policy returns."""

    def reset(self, seed=None):
        self.t = 0
        self.x = np.zeros(2, np.float32)
        return self.x.copy(), {}

    def step(self, action):
        self.t += 1
        self.x = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        return self.x.copy(), 0.0, False, self.t >= 3, {"is_success": False}


def _torch_noise_policy(_obs):
    import torch

    return torch.randn(2).numpy().astype(np.float32)


def _write_noise(root, shuffle_seed):
    srcs = [M.PolicySource("noise", "fake://noise", 1, 2)]
    return M.write_policy_episodes(
        srcs, lambda _src: _torch_noise_policy, _StateEnv, PROTO, max_steps=3,
        repo_id="rcb/noise_order", root=root, fps=10, task="noise", shuffle_seed=shuffle_seed,
    )


def test_episode_actions_do_not_depend_on_rollout_order(tmp_path, scratch_env):
    """Same env seed, same actions, whether that episode is rolled out first or second."""
    assert not np.array_equal(
        np.random.default_rng(0).permutation(2), np.random.default_rng(3).permutation(2),
    )
    _, first = _write_noise(tmp_path / "a", shuffle_seed=0)
    _, second = _write_noise(tmp_path / "b", shuffle_seed=3)
    by_seed_a = {ep.seed: ep.actions for ep in first}
    by_seed_b = {ep.seed: ep.actions for ep in second}
    assert [ep.seed for ep in first] != [ep.seed for ep in second]
    assert set(by_seed_a) == set(by_seed_b)
    for seed, actions in by_seed_a.items():
        np.testing.assert_array_equal(actions, by_seed_b[seed])
    seeds = list(by_seed_a)
    assert not np.array_equal(by_seed_a[seeds[0]], by_seed_a[seeds[1]])


def test_failed_second_episode_does_not_block_retry(tmp_path, scratch_env, monkeypatch):
    calls = {"n": 0, "fail": True}
    real = M._rollout_one

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["fail"] and calls["n"] == 2:
            raise RuntimeError("episode 2 failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(M, "_rollout_one", wrapped)
    srcs = [M.PolicySource("p", "fake://p", 1, 2)]
    root = tmp_path / "out" / "policy"
    kwargs = dict(
        sources=srcs, load_policy=lambda _src: (lambda _obs: np.zeros(2, np.float32)),
        env_factory=_StateEnv, protocol=PROTO,
        max_steps=3, repo_id="rcb/retry", root=root, fps=10, task="retry", shuffle_seed=0,
    )
    with pytest.raises(RuntimeError, match="episode 2 failed"):
        M.write_policy_episodes(**kwargs)
    assert not root.exists()
    assert not sidecar_path(root).exists()
    assert not (root.parent / f".{root.name}.partial").exists()

    calls["fail"] = False
    calls["n"] = 0
    written, kept = M.write_policy_episodes(**kwargs)
    assert written == root and root.is_dir() and len(kept) == 2
    assert sidecar_path(root).is_file()
    with pytest.raises(FileExistsError):
        M.write_policy_episodes(**kwargs)


def test_gpu_phase_script_selects_cuda_for_local(tmp_path):
    script = REPO / "scripts" / "gpu_phase_rollouts.sh"
    text = script.read_text()
    assert "--device cuda" in text
    bindir = tmp_path / "bin"
    bindir.mkdir()
    out = tmp_path / "args"
    stub = bindir / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$STUB_OUT"\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["ROBOCURATE_SCRATCH"] = str(tmp_path / "scratch")
    env["HF_LEROBOT_HOME"] = str(tmp_path / "lerobot")
    env["STUB_OUT"] = str(out)

    def run(*args: str) -> list[str]:
        subprocess.run([str(script), *args], check=True, env=env)
        return out.read_text().splitlines()

    local = run("--recipe", "r", "--out", "o", "--backend", "local", "--env-type", "aloha")
    assert local[-2:] == ["--device", "cuda"]
    explicit = run("--backend", "local", "--device", "cpu")
    assert "cuda" not in explicit
    assert explicit[-1] == "cpu"
    equals = run("--backend=local", "--device=cpu")
    assert "cuda" not in equals
    fake = run("--backend", "fake")
    assert "cuda" not in fake

