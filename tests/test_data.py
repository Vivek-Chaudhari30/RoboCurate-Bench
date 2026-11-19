import numpy as np
import pytest

from robocurate.config import ConfigError
from robocurate.data import defects as D
from robocurate.data.labels import LabelsSidecar, load_labels, sidecar_path, write_labels
from robocurate.data.mixture import Recipe, labels_for, plan_json, plan_mixture

RECIPE = """
name: t
repo_id: robocurate/test_mix
seed: {seed}
sources:
  - {{dataset: a/scripted, source: scripted, n_episodes: 10}}
  - {{dataset: a/human, source: human, n_episodes: 10}}
wrong_task_strings: ["Insert the peg into the socket."]
defects:
  pause: {{n_episodes: 3, k_min: 20, k_max: 60}}
  duplicate: {{n_episodes: 2}}
  wrong_task: {{n_episodes: 2}}
"""
LENGTHS = {"a/scripted": [400] * 50, "a/human": [400] * 50}


def _recipe(tmp_path, seed=0, text=RECIPE):
    p = tmp_path / f"r{seed}.yaml"
    p.write_text(text.format(seed=seed))
    return Recipe.load(p)


# ------------------------------------------------------------------ mixture plan


def test_plan_counts_and_labels(tmp_path):
    r = _recipe(tmp_path)
    plan = plan_mixture(r, LENGTHS)
    assert len(plan) == 22
    defects = [d for p in plan for d in p.defects]
    assert defects.count("pause") == 3 and defects.count("duplicate") == 2 and defects.count("wrong_task") == 2
    assert [p.out_index for p in plan] == list(range(22))
    for p in plan:
        if "duplicate" in p.defects:
            orig = plan[p.duplicate_of]
            assert (orig.source_dataset, orig.source_episode) == (p.source_dataset, p.source_episode)
            assert orig.defects == ()
        if "pause" in p.defects:
            assert 80 <= p.pause_t < 320 and 20 <= p.pause_k <= 60
    lab = labels_for(r, plan)
    assert {e.quality_tier for e in lab.episodes} == {"clean", "degraded"}


def test_plan_byte_identical_and_seed_sensitive(tmp_path):
    a = plan_mixture(_recipe(tmp_path), LENGTHS)
    b = plan_mixture(_recipe(tmp_path), LENGTHS)
    c = plan_mixture(_recipe(tmp_path, seed=1), LENGTHS)
    assert plan_json(a) == plan_json(b) != plan_json(c)
    r = _recipe(tmp_path)
    p1 = write_labels(labels_for(r, a), tmp_path / "l1.json")
    p2 = write_labels(labels_for(r, b), tmp_path / "l2.json")
    assert p1.read_bytes() == p2.read_bytes()


def test_recipe_rejects_action_noise(tmp_path):
    text = RECIPE.replace("  wrong_task: {{n_episodes: 2}}", "  action_noise: {{sigma: 0.1}}")
    with pytest.raises(ConfigError, match="D-011"):
        _recipe(tmp_path, text=text)


def test_plan_rejects_too_many_defects(tmp_path):
    text = RECIPE.replace("pause: {{n_episodes: 3", "pause: {{n_episodes: 30")
    with pytest.raises(ConfigError):
        plan_mixture(_recipe(tmp_path, text=text), LENGTHS)


# ------------------------------------------------------------------ labels sidecar


def test_sidecar_lives_beside_dataset(tmp_path):
    root = tmp_path / "datasets" / "robocurate" / "mix"
    assert sidecar_path(root) == tmp_path / "datasets" / "robocurate" / "mix.labels.json"
    assert root not in sidecar_path(root).parents


def test_labels_roundtrip(tmp_path):
    r = _recipe(tmp_path)
    lab = labels_for(r, plan_mixture(r, LENGTHS))
    p = write_labels(lab, tmp_path / "x.labels.json")
    back = load_labels(p)
    assert isinstance(back, LabelsSidecar) and back.to_dict() == lab.to_dict()


# ------------------------------------------------------------------ post-hoc defects


def _frames(n=10):
    return [{"task": "t", "action": np.full(2, i, float), "observation.state": np.full(2, i, float)} for i in range(n)]


def test_pause_holds_obs_and_action_together():
    out = D.inject_pause(_frames(), t=4, k=3)
    assert len(out) == 13
    held = out[4:8]
    for f in held:
        assert np.array_equal(f["action"], held[0]["action"])
        assert np.array_equal(f["observation.state"], held[0]["observation.state"])
    assert out[8]["action"][0] == 5
    with pytest.raises(ValueError):
        D.inject_pause(_frames(), t=10, k=1)


def test_wrong_task_and_duplicate():
    w = D.inject_wrong_task(_frames(3), "other")
    assert {f["task"] for f in w} == {"other"}
    dup = D.duplicate_episode(_frames(3))
    assert all(np.array_equal(a["action"], b["action"]) for a, b in zip(dup, _frames(3), strict=True))
    with pytest.raises(ValueError):
        D.inject_wrong_task(_frames(2), "t")


def test_no_post_hoc_action_noise_api():
    assert not any("noise" in n for n in dir(D) if not n.startswith("_"))


# ------------------------------------------------------------------ replay-mode perturbations (fake env)


class PointEnv:
    """Tiny position-controlled env: state moves toward the commanded target by `gain` each step."""

    def __init__(self, dim=3, gain=0.5, horizon=40):
        self.dim, self.gain, self.horizon = dim, gain, horizon

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.normal(size=self.dim)
        self.t = 0
        return self.x.copy(), {}

    def step(self, a):
        self.x = self.x + self.gain * (np.asarray(a, float) - self.x)
        self.t += 1
        return self.x.copy(), 0.0, False, self.t >= self.horizon, {}


def _consistent(traj, gain=0.5):
    """obs[t+1] must equal the env response to the EXECUTED action[t]."""
    o, a = traj["observation"], traj["action"]
    pred = o[:-1] + gain * (a[:-1] - o[:-1])
    return np.allclose(pred, o[1:])


def test_perturbations_are_execution_time_consistent():
    target = np.ones(3)
    base = lambda obs: target  # noqa: E731
    perts = [
        D.Jitter(sigma=0.2, onset=5, duration=10),
        D.GripperDrop(onset=12, duration=5, gripper_dims=[2], open_value=-1.0),
        D.Hesitation(onset=25, duration=6),
    ]
    traj = D.rollout(PointEnv(), D.PerturbedPolicy(base, perts, seed=7), 40, seed=3)
    assert _consistent(traj)
    a = traj["action"]
    assert np.allclose(a[:5], target) and not np.allclose(a[5:15], target)
    assert np.allclose(a[12:17, 2], -1.0)
    assert np.allclose(a[25:31], a[24])  # hesitation holds the previous executed action


def test_perturbed_policy_deterministic_and_replay():
    perts = [D.Jitter(sigma=0.3, onset=0, duration=40)]
    env = PointEnv()
    t1 = D.rollout(env, D.PerturbedPolicy(lambda o: np.zeros(3), perts, seed=1), 40, seed=0)
    t2 = D.rollout(env, D.PerturbedPolicy(lambda o: np.zeros(3), perts, seed=1), 40, seed=0)
    assert np.array_equal(t1["action"], t2["action"])
    # replay mode: re-executing recorded actions open loop from the same env seed reproduces observations
    t3 = D.rollout(env, D.ReplayPolicy(t1["action"]), 40, seed=0)
    assert np.allclose(t3["observation"], t1["observation"])


def test_mixture_order_shuffled_by_recipe_seed():
    """D-021: source labels are not monotone in output index (the builder shuffles by recipe seed)."""
    from robocurate.data.mixture import Recipe, SourceSpec, plan_mixture

    r = Recipe(name="m", repo_id="x/m", seed=7,
               sources=(SourceSpec(source="human", dataset="h", n_episodes=20),
                        SourceSpec(source="scripted", dataset="s", n_episodes=20)))
    plan = plan_mixture(r, {"h": [400] * 20, "s": [400] * 20})
    src = [p.source for p in plan]
    assert src != sorted(src) and src != sorted(src, reverse=True)
    first_half_human = sum(s == "human" for s in src[:20])
    assert 5 <= first_half_human <= 15
    import dataclasses

    r2 = dataclasses.replace(r, seed=8)
    assert [p.source for p in plan_mixture(r2, {"h": [400] * 20, "s": [400] * 20})] != src
