"""Policy-rollout recorder with camera frames (robocurate.sim.rollout_recorder, launch/rollouts.py). CPU only;
on macOS the env renders on the Apple Software Renderer (conftest sets MUJOCO_GL=disable)."""

import json
import sys

import pytest

from robocurate.config import load_protocol
from robocurate.paths import repo_root

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="CPU rendering path is macOS-only here")


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    from robocurate.launch.rollouts import write_fake_checkpoints
    from robocurate.sim.scripted_aloha import ensure_cpu_renderer

    ensure_cpu_renderer()
    p = tmp_path_factory.mktemp("m0") / "train/checkpoints/000002/pretrained_model"
    write_fake_checkpoints([p], "act", "aloha", "AlohaTransferCube-v0")
    return p


def test_recorder_refuses_non_generation_seeds(tiny_ckpt):
    from robocurate.sim import rollout_recorder as RR

    with pytest.raises(ValueError, match="D-034"):
        RR.record(tiny_ckpt, 2, "aloha", "AlohaTransferCube-v0", [500000], protocol=PROTO, episode_length=4)


@pytest.mark.slow
def test_recorder_writes_lerobot_dataset_with_camera_frames(tiny_ckpt, tmp_path, scratch_env):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from robocurate.data.labels import load_labels, sidecar_path
    from robocurate.sim import rollout_recorder as RR

    eps = RR.record(tiny_ckpt, 2, "aloha", "AlohaTransferCube-v0", [700010, 700011], protocol=PROTO,
                    episode_length=8)
    assert [e.seed for e in eps] == [700010, 700011]
    assert all(e.images["observation.images.top"].shape == (8, 480, 640, 3) and e.actions.shape == (8, 14)
               for e in eps)
    root = RR.write_rollout_dataset(eps, "rcb/test_rollouts", tmp_path / "ds", fps=50, task="transfer cube",
                                    shuffle_seed=7, robot_type="aloha")
    ds = LeRobotDataset("rcb/test_rollouts", root=root)
    assert ds.meta.total_episodes == 2 and ds.meta.total_frames == 16
    assert {"observation.images.top", "observation.state", "action"} <= set(ds.meta.features)
    assert tuple(ds[0]["observation.images.top"].shape) == (3, 480, 640)
    lab = load_labels(sidecar_path(root)).by_index()
    assert {lab[i].source for i in lab} == {"policy"} and {lab[i].checkpoint_step for i in lab} == {2}
    assert all(lab[i].success is not None for i in lab) and sorted(lab[i].source_episode for i in lab) == [700010,
                                                                                                         700011]


@pytest.mark.slow
def test_recorder_perturbation_is_executed_and_labelled(tiny_ckpt):
    from robocurate.sim import rollout_recorder as RR

    pert = [{"type": "gripper_drop", "onset": 0, "duration": 6, "gripper_dims": [13], "open_value": 1.0}]
    (ep,) = RR.record(tiny_ckpt, 2, "aloha", "AlohaTransferCube-v0", [700012], protocol=PROTO, episode_length=6,
                      perturbations=pert)
    assert (ep.actions[:, 13] == 1.0).all() and ep.perturbations == pert


@pytest.mark.slow
def test_gpu_phase_rollouts_dry_run_with_fake_d033_checkpoints(tmp_path, scratch_env):
    """launch/rollouts.py on task_a_v1: D-033 checkpoint paths, D-034 seeds from offset 10, one short episode per
    checkpoint with fake (tiny random) checkpoints."""
    from robocurate.launch.rollouts import generate

    rep = generate(str(REPO / "configs/mixtures/task_a_v1.yaml"), str(REPO / "configs/tasks/aloha_transfer_cube.yaml"),
                   tmp_path / "out", tmp_path / "m0run", 1000, str(REPO / "configs/protocol.yaml"), "cpu",
                   fake=True, max_episodes=1, episode_length=4)
    src = rep["sources"]["policy_rollouts"]
    assert src["n"] == 4 and src["seed_offset"] == 10 and src["seed_range"][0] == 700010
    assert sorted(src["checkpoints"]) == ["0.1", "0.3", "0.6", "1.0"]
    assert all(f"/{s:06d}/pretrained_model" in src["checkpoints"][k]
               for k, s in zip(["0.1", "0.3", "0.6", "1.0"], [100, 300, 600, 1000], strict=True))
    assert json.loads(json.dumps(src["success_by_step"])).keys() == {"100", "300", "600", "1000"}
