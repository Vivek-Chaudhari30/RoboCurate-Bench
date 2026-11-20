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
