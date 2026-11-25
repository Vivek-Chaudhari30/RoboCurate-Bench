"""`robocurate prepare --backend local` (overnight 3 item 5): CPU entrants on a real LeRobot dataset, GPU entrants with
tiny or mocked models on CPU, missing resources refused. No GPU."""

import tempfile

import numpy as np
import pytest
from toybed.demos import flatten, make_dataset
from toybed.env import PointReachEnv, goal_distance_progress

from robocurate.cleaners import registry
from robocurate.config import load_protocol
from robocurate.curation import prepare as P
from robocurate.curation.policy import MLPPolicy, train_bc
from robocurate.curation.progress import MockProgressModel
from robocurate.curation.rollouts import GymRolloutBackend
from robocurate.data.episodes import synthetic_view
from robocurate.paths import repo_root

REPO = repo_root()
PROTO = load_protocol(REPO / "configs/protocol.yaml")


@pytest.fixture(scope="module")
def real_dataset(tmp_path_factory):
    """A small real LeRobot dataset on disk (no labels), loaded back through hub.load_view as prepare does."""
    from robocurate.data.hub import load_view
    from robocurate.data.lerobot_write import write_view_dataset

    root = write_view_dataset(synthetic_view(12, tie_break_salt=PROTO.tie_break_salt),
                              tmp_path_factory.mktemp("ds") / "mix", repo_id="rcb/local_mix")
    return root, load_view("rcb/local_mix", root, PROTO.tie_break_salt)


@pytest.mark.parametrize("name", ["length_heuristic", "kinema"])
def test_cpu_entrants_run_on_a_real_dataset(real_dataset, tmp_path, name):
    root, view = real_dataset
    cl = registry.get(name)
    fac = P.local_context_factory(PROTO, P.LocalResources(root), tmp_path)
    rec = P.prepare_cell(P.Cell("t", "cleaner", name, "1", 0.5), cl, view, PROTO, fac(cl), tmp_path, "local")
    assert rec["backend"] == "local" and len(rec["selected_episodes"]) == 6 and rec["gpu_h"] == 0.0


def test_gpu_entrants_with_mocked_progress_on_cpu(real_dataset, tmp_path, monkeypatch):
    root, view = real_dataset
    res = P.LocalResources(root, device="cpu", progress_model=MockProgressModel())
    fac = P.local_context_factory(PROTO, res, tmp_path)
    for name in ("robometer_filter", "rcb_score"):
        cl = registry.get(name)
        rec = P.prepare_cell(P.Cell("t", "cleaner", name, "1", 0.5), cl, view, PROTO, fac(cl), tmp_path, "local")
        assert len(rec["selected_episodes"]) == 6
    cl = registry.get("rabc")
    rec = P.prepare_cell(P.Cell("t", "weighting", "rabc", "1", 1.0, 0, 8), cl, view, PROTO, fac(cl), tmp_path,
                         "local", rabc_chunk_size=5)
    assert len(rec["placebo_progress_parquets"]) == 8
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    cuda = P.local_context_factory(PROTO, P.LocalResources(root, device="cuda",
                                                           progress_model=MockProgressModel()), tmp_path / "c")
    cl = registry.get("robometer_filter")
    with pytest.raises(P.PrepareError, match="requires a GPU"):
        P.prepare_cell(P.Cell("t", "cleaner", "robometer_filter", "1", 0.5), cl, view, PROTO, cuda(cl), tmp_path / "c",
                       "local")


def test_reference_policy_entrants_with_toy_models_on_cpu(tmp_path):
    view, labels = make_dataset()
    d = tempfile.mkdtemp()
    ref = train_bc(MLPPolicy(4, 2), *flatten(view), steps=200, seed=0).save(d + "/ref")
    res = P.LocalResources(tmp_path, device="cpu", rollout_backend=GymRolloutBackend(PointReachEnv),
                           reference_policy_path=ref, reference_loader=MLPPolicy.load,
                           progress_model=MockProgressModel(lambda e: goal_distance_progress(np.asarray(e.state))))
    fac = P.local_context_factory(PROTO, res, tmp_path)
    for name in ("gradsim_return", "demo_score"):
        cl = registry.get(name)
        rec = P.prepare_cell(P.Cell("toy", "cleaner", name, "1", 0.5), cl, view, PROTO, fac(cl), tmp_path, "local")
        assert len(rec["selected_episodes"]) == view.n_episodes // 2 and rec["rollouts_used"] == 100
        assert all(500000 <= s <= 599999 for s in rec["curation_seeds_used"])  # curation seeds only (D-007)


def test_missing_resources_are_refused(real_dataset, tmp_path):
    root, view = real_dataset
    fac = P.local_context_factory(PROTO, P.LocalResources(root), tmp_path)
    for name, msg in (("robometer_filter", "progress model"), ("demo_score", "rollout backend"),
                      ("gradsim_return", "rollout backend")):
        with pytest.raises(P.PrepareError, match=msg):
            fac(registry.get(name))


def test_checkpoint_dir_uses_pretrained_path_not_the_policy_repr():
    """A loaded policy's str() is its repr. The checkpoint is config.pretrained_path."""
    from robocurate.curation.rollouts import RolloutError, checkpoint_dir

    class Policy:
        def __init__(self):
            self.config = type("Cfg", (), {"pretrained_path": "/ckpt/pretrained_model"})()

        def eval(self):
            return self

    pol = Policy()
    assert checkpoint_dir("/ckpt/pretrained_model") == "/ckpt/pretrained_model"
    assert checkpoint_dir(pol) == "/ckpt/pretrained_model"
    assert checkpoint_dir(pol) != str(pol)
    with pytest.raises(RolloutError, match="pretrained_path"):
        checkpoint_dir(type("Bare", (), {"config": object()})())


def test_cli_reference_policy_loader_returns_a_policy(real_dataset, tmp_path, scratch_env, monkeypatch):
    """--reference-policy must load a policy. `lambda p: str(p)` skips the missing-loader PrepareError and
    then crashes gradsim_return on policy.eval() (and demo_score rolls that string out)."""
    from pathlib import Path

    from robocurate.cleaners.entrants.demo_score import DemoScore
    from robocurate.cli import main
    from robocurate.curation import rollouts as R

    root, _ = real_dataset
    ckpt = tmp_path / "pretrained_model"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}\n")
    seen = {}

    class Policy:
        def __init__(self):
            self.config = type("Cfg", (), {"pretrained_path": str(ckpt)})()
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

    def load_policy(path, env_type, env_task, device):
        assert Path(path) == ckpt and env_type == "aloha" and env_task == "AlohaTransferCube-v0"
        assert device == "cpu"
        return Policy()

    def select(self, view, budget, rng):
        pol = self.context.reference_policy.load()
        pol.eval()  # gradsim_return.py:103; a path string has no eval
        getattr(pol, "policy_family", None)  # gradsim_return.py:49
        seen["policy"] = pol
        return sorted(e.episode_index for e in view)[:budget]

    monkeypatch.setattr(R, "load_lerobot_policy", load_policy)
    monkeypatch.setattr(DemoScore, "select", select)
    grid, task = tmp_path / "g.yaml", tmp_path / "t.yaml"
    task.write_text((REPO / "configs/tasks/aloha_transfer_cube.yaml").read_text().replace(
        "train_dataset: robocurate/task_a_v1", "train_dataset: rcb/local_mix"))
    grid.write_text(
        f"name: g\ntask: {task}\nprotocol: {REPO}/configs/protocol.yaml\nbudgets: [0.5]\nconditions:\n"
        "  - {kind: full}\n  - {kind: random, n_subsets: 3}\n"
        "  - {kind: cleaner, name: demo_score, version: '1'}\n")
    assert main(["prepare", str(grid), "--backend", "local", "--dataset-root", str(root),
                 "--device", "cpu", "--reference-policy", str(ckpt), "--only", "demo_score",
                 "--selections-dir", str(tmp_path / "sel")]) == 0
    pol = seen["policy"]
    assert not isinstance(pol, str) and pol.evaluated and pol.config.pretrained_path == str(ckpt)


def test_cli_prepare_local_on_real_dataset(real_dataset, tmp_path, scratch_env, capsys):
    from robocurate.cli import main

    root, _ = real_dataset
    grid = tmp_path / "g.yaml"
    task = tmp_path / "t.yaml"
    task.write_text((REPO / "configs/tasks/aloha_transfer_cube.yaml").read_text().replace(
        "train_dataset: robocurate/task_a_v1", "train_dataset: rcb/local_mix"))
    grid.write_text(f"name: g\ntask: {task}\nprotocol: {REPO}/configs/protocol.yaml\nbudgets: [0.5]\nconditions:\n"
                    "  - {kind: full}\n  - {kind: random, n_subsets: 3}\n"
                    "  - {kind: cleaner, name: length_heuristic, version: '1'}\n"
                    "  - {kind: cleaner, name: kinema, version: '1'}\n")
    assert main(["prepare", str(grid), "--backend", "local", "--dataset-root", str(root),
                 "--selections-dir", str(tmp_path / "sel")]) == 0
    out = capsys.readouterr().out
    assert "length_heuristic v1 [selection] budget 0.5: 6 episodes" in out and "kinema v1" in out
