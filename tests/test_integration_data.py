"""Integration tests (CPU, network). Run with the storage env sourced:  source scripts/env.sh"""

import shutil
import uuid

import numpy as np
import pytest

from robocurate.paths import StorageError, require_storage_env

pytestmark = pytest.mark.integration

HUMAN = "lerobot/aloha_sim_transfer_cube_human"
WRONG = "Insert the peg into the socket."

RECIPE = f"""
name: it6
repo_id: robocurate/it6
seed: 3
sources:
  - {{dataset: {HUMAN}, source: human, n_episodes: 5}}
wrong_task_strings: ["{WRONG}"]
defects:
  pause: {{n_episodes: 1, k_min: 10, k_max: 20}}
  duplicate: {{n_episodes: 1}}
  wrong_task: {{n_episodes: 1}}
"""


@pytest.fixture(scope="module")
def scratch():
    try:
        return require_storage_env()
    except StorageError as e:
        pytest.fail(f"integration tests need the storage env (source scripts/env.sh): {e}")


@pytest.fixture(scope="module")
def workdir(scratch):
    d = scratch / "tests" / f"it-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d)  # own run dir under $ROBOCURATE_SCRATCH


@pytest.fixture(scope="module")
def mixture(workdir):
    from robocurate.data.mixture import build_mixture

    rp = workdir / "recipe.yaml"
    rp.write_text(RECIPE)
    root = build_mixture(rp, root=workdir / "robocurate" / "it6")
    return rp, root


def test_load_human_metadata(scratch):
    from robocurate.data.hub import episode_lengths, load_metadata

    m = load_metadata(HUMAN)
    assert m.total_episodes == 50 and m.fps == 50
    assert {"action", "observation.state", "observation.images.top"} <= set(m.features)
    assert len(episode_lengths(m)) == 50
    assert str(m.root).startswith(str(scratch))


def test_mixture_reloads_with_lerobot(mixture):
    from robocurate.data.hub import load_dataset, load_view
    from robocurate.data.labels import load_labels, sidecar_path
    from robocurate.data.mixture import Recipe, plan_mixture

    rp, root = mixture
    ds = load_dataset("robocurate/it6", root=root)
    assert ds.num_episodes == 6
    lab = load_labels(sidecar_path(root))
    assert [e.episode_index for e in lab.episodes] == list(range(6))
    assert sorted(d for e in lab.episodes for d in e.defects) == ["duplicate", "pause", "wrong_task"]
    assert not (root / "labels.json").exists() and sidecar_path(root).parent == root.parent

    # labels sidecar matches a fresh plan of the same recipe (deterministic)
    from robocurate.data.hub import episode_lengths, load_metadata

    plan = plan_mixture(Recipe.load(rp), {HUMAN: episode_lengths(load_metadata(HUMAN))})
    assert [(p.source_episode, p.defects) for p in plan] == [(e.source_episode, e.defects) for e in lab.episodes]

    view = load_view("robocurate/it6", root=root)
    by = lab.by_index()
    for p in plan:
        ep = view.episode(p.out_index)
        if "pause" in p.defects:
            assert ep.length == 400 + p.pause_k
            held = ep.actions[p.pause_t : p.pause_t + p.pause_k + 1]
            assert np.allclose(held, held[0])
        else:
            assert ep.length == 400
        if "wrong_task" in p.defects:
            assert ep.task == WRONG
        else:
            assert ep.task.startswith("Pick up the cube")
        if "duplicate" in p.defects:
            assert np.array_equal(ep.actions, view.episode(p.duplicate_of).actions)
        assert by[p.out_index].source_episode == p.source_episode

    # images survive the round trip (video encoded) and decode to the right shape
    item = ds[0]
    assert tuple(item["observation.images.top"].shape) == (3, 480, 640)


def test_subset_stats_come_from_subset_only(mixture, workdir):
    """D-013: materialized subset's stats equal stats of the kept episodes, not of the full dataset."""
    from robocurate.data.hub import load_dataset
    from robocurate.data.subset import materialize_subset

    _, root = mixture
    keep = [0, 2, 5]
    sub_root = materialize_subset("robocurate/it6", root, keep, workdir / "subset")
    sub = load_dataset("robocurate/it6_train", root=sub_root)
    full = load_dataset("robocurate/it6", root=root)
    assert sub.num_episodes == 3

    rows = full.hf_dataset.select_columns(["episode_index", "action"]).with_format("numpy")[:]
    mask = np.isin(rows["episode_index"].reshape(-1), keep)
    expect_mean = rows["action"][mask].mean(0)
    expect_min = rows["action"][mask].min(0)
    np.testing.assert_allclose(np.asarray(sub.meta.stats["action"]["mean"]), expect_mean, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.asarray(sub.meta.stats["action"]["min"]), expect_min, rtol=1e-5, atol=1e-6)
    assert not np.allclose(np.asarray(full.meta.stats["action"]["mean"]), expect_mean, atol=1e-6)
