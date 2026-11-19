"""Launch entrants (m1-core item 5): registration + metadata, `submit` conformance, toybed sanity with published
defaults (failures are REPORTED, never tuned away), checker mappings, frozen rcb-score v1, method-math checks."""

import hashlib
import json
import tempfile

import numpy as np
import pytest
import torch
from toybed.demos import flatten, make_dataset
from toybed.env import PointReachEnv, goal_distance_progress

from robocurate.cleaners import conformance, registry
from robocurate.cleaners.base import validate_resources
from robocurate.config import load_protocol
from robocurate.curation.context import CurationContext, ReferencePolicy
from robocurate.curation.policy import MLPPolicy, train_bc
from robocurate.curation.progress import CachedProgress, MockProgressModel
from robocurate.curation.rollouts import GymRolloutBackend, RolloutService
from robocurate.curation.weights import rabc_frame_weights
from robocurate.data.episodes import synthetic_view
from robocurate.paths import repo_root

PROTO = load_protocol(repo_root() / "configs/protocol.yaml")
LAUNCH = ["lerobot_doctor", "kinema", "roboaudit", "rda", "robometer_filter", "rabc", "rcb_score", "demo_score",
          "deminf", "cupid"]
KINDS = ("clean", "pause", "jitter", "wrong")


def test_all_launch_entrants_registered_with_metadata():
    for n in LAUNCH:
        c = registry.get(n)
        assert validate_resources(c) == [], n
        m = c.metadata()
        assert m["implementation"] in ("wrapped", "reimplementation", "original") and m["source"], n
    assert registry.get("rabc").track == "weighting"
    assert {registry.get(n).implementation for n in ("demo_score", "deminf", "cupid")} == {"reimplementation"}


# ------------------------------------------------------------------ toybed fixtures


@pytest.fixture(scope="module")
def toy():
    view, labels = make_dataset()
    d = tempfile.mkdtemp()
    ref = train_bc(MLPPolicy(4, 2), *flatten(view), steps=400, seed=0).save(d + "/ref")
    return view, labels, ref


def _ctx(cleaner, ref, progress=None):
    return CurationContext(
        rollouts=RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv), cap=cleaner.requires_rollouts)
        if cleaner.requires_rollouts else None,
        reference_policy=ReferencePolicy(ref, MLPPolicy.load) if cleaner.requires_reference_policy else None,
        progress_model=progress)


def _kept(name, toy, progress=None):
    view, labels, ref = toy
    cl = registry.get(name)
    cl.context = _ctx(cl, ref, progress)
    ids = cl.select(view, len(labels) // 2, np.random.default_rng(0))
    frac = {k: np.mean([i in ids for i in labels if labels[i] == k]) for k in KINDS}
    return ids, frac


TOY_PROGRESS = MockProgressModel(lambda e: goal_distance_progress(np.asarray(e.state)))


def test_sanity_robometer_filter(toy):
    _, frac = _kept("robometer_filter", toy, TOY_PROGRESS)
    assert frac["wrong"] == 0.0 and frac["clean"] > 0.5


def test_sanity_rcb_score(toy):
    _, frac = _kept("rcb_score", toy, TOY_PROGRESS)
    assert frac["wrong"] < frac["clean"] and frac["jitter"] < frac["clean"]


def test_sanity_rabc_weights(toy):
    view, labels, ref = toy
    cl = registry.get("rabc")
    cl.context = _ctx(cl, ref, TOY_PROGRESS)
    w = rabc_frame_weights(cl.frame_progress(view, np.random.default_rng(0)), chunk_size=5)
    mean = {k: np.mean([w[i].mean() for i in labels if labels[i] == k]) for k in KINDS}
    assert mean["wrong"] < mean["clean"]


def test_sanity_demo_score(toy):
    _, frac = _kept("demo_score", toy)
    assert frac["wrong"] < frac["clean"]


@pytest.mark.slow
def test_sanity_deminf(toy):
    """Published default (2 VAEs x 50k Adam steps) takes ~220 s on this Mac CPU: over the 60 s toy budget."""
    _, frac = _kept("deminf", toy)
    assert frac["wrong"] < frac["clean"]


@pytest.mark.xfail(strict=True, reason="CUPID with published defaults (d=4000, lambda=0, 100 rollouts) does not "
                   "down-rank the planted wrong-direction toy demos; investigated, outcome (c): disabled, not tuned "
                   "(PROGRESS.md)")
def test_sanity_cupid(toy):
    _, frac = _kept("cupid", toy)
    assert frac["wrong"] < frac["clean"]


# ------------------------------------------------------------------ method math


def test_ksg_handles_exact_duplicates_regression():
    """Regression: held poses give a zero k-NN radius; the self-inclusive count was 0 and digamma(0) = NaN."""
    from robocurate.cleaners.entrants.deminf import ksg_scores

    rng = np.random.default_rng(0)
    zs, za = rng.normal(size=(40, 3)), rng.normal(size=(40, 2))
    zs[:10], za[:10] = zs[0], za[0]  # ten identical samples
    s = ksg_scores(zs, za)
    assert np.all(np.isfinite(s))


def test_ksg_detects_dependence():
    from robocurate.cleaners.entrants.deminf import ksg_scores

    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 2))
    dep = ksg_scores(x, x[:, :1] + 0.01 * rng.normal(size=(400, 1))).mean()
    ind = ksg_scores(x, rng.normal(size=(400, 1))).mean()
    assert dep > ind


def test_cupid_per_sample_grads_match_autograd():
    from robocurate.cleaners.entrants.cupid import per_sample_projected_grads

    pol = MLPPolicy(3, 2, hidden=5)
    n = sum(p.numel() for p in pol.parameters())
    rng = np.random.default_rng(0)
    S, A = rng.normal(size=(7, 3)).astype(np.float32), rng.normal(size=(7, 2)).astype(np.float32)
    got = per_sample_projected_grads(pol, S, A, torch.eye(n))
    for i in range(7):
        pol.zero_grad()
        pol.loss({"observation.state": torch.as_tensor(S[i:i + 1]), "action": torch.as_tensor(A[i:i + 1])}).backward()
        ref = torch.cat([p.grad.reshape(-1) for p in pol.parameters()])
        torch.testing.assert_close(got[i], ref, rtol=1e-5, atol=1e-6)


def test_cupid_influence_sign_on_linear_toy():
    """A demo whose gradient points the way successful rollouts' gradients point gets positive influence."""
    from robocurate.cleaners.entrants.cupid import per_sample_projected_grads

    pol = MLPPolicy(1, 1, hidden=4, seed=3)
    n = sum(p.numel() for p in pol.parameters())
    P = torch.eye(n)
    S = np.linspace(-1, 1, 30, dtype=np.float32)[:, None]
    good, bad = S.copy(), -S.copy()  # 'good' demos act a = s, 'bad' act a = -s
    phi = per_sample_projected_grads(pol, np.concatenate([S, S]), np.concatenate([good, bad]), P)
    Hinv = torch.linalg.pinv((phi.T @ phi).double()).float()
    u = per_sample_projected_grads(pol, S, good, P).sum(0)  # successful rollouts behaved like 'good'
    infl = phi @ (Hinv @ u)
    assert infl[:30].sum() > infl[30:].sum()


# ------------------------------------------------------------------ rcb-score v1 frozen (D-023)




def _rcb_digest():
    from robocurate.cleaners.entrants.rcb_score import FORMULA_V1, rcb_score_v1

    v = synthetic_view(12, seed=3)
    s = rcb_score_v1(v, MockProgressModel().episode_progress(v))
    payload = json.dumps({"formula": FORMULA_V1, "scores": {k: round(x, 10) for k, x in s.items()}}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def test_rcb_score_v1_frozen():
    """D-023: rcb_score v1 is frozen. If this digest changes, ship the change as v2 instead."""
    assert _rcb_digest() == "77c312122589684e5f9dbc776c673bce1fb05814bd21dac8dce9581780567b78"


def test_rcb_score_components():
    from robocurate.cleaners.entrants.rcb_score import mean_sq_jerk

    a = np.arange(10, dtype=float)[:, None] ** 3  # constant third difference = 6
    assert mean_sq_jerk(a, fps=1) == pytest.approx(36.0)
    assert mean_sq_jerk(np.zeros((3, 2)), fps=50) == 0.0


# ------------------------------------------------------------------ checker mappings (tools not installed here)


def test_checker_mappings():
    from robocurate.cleaners.entrants.checkers import RDA, Kinema, LerobotDoctor, RoboAudit, combine

    assert LerobotDoctor.episode_ranks(["2/5 episode(s) flagged", "Episode 3: too short (4 frames, <1.0s); 2 dropped "
                                        "frame(s)", "Episode 7: NaN in action"]) == {3: (2, 2.0), 7: (2, 1.0)}
    assert Kinema.episode_ranks([(1, 2), (1, 3), (2, 1), (None, 3), (4, 4), (5, 2)]) == {
        1: (2, 2.0), 2: (0, 0.0), 5: (1, 1.0)}
    rep = {"dispositions": {"0": "keep", "1": "exclude", "2": "review"},
           "excluded_episodes": [{"episode_index": 1, "checks": ["A1", "V2"]}],
           "review_episodes": [{"episode_index": 2, "checks": ["V3"]}]}
    assert RoboAudit.episode_ranks(rep) == {0: (0, 0.0), 1: (2, 2.0), 2: (1, 1.0)}
    assert RDA.episode_ranks({0: {"verdict": "PASS", "combined_score": 0.3}, 1: {"verdict": "EXCLUDE",
                              "combined_score": None, "behavior_severity": 2.0}, 2: {"verdict": "UNVERIFIABLE"}}) == {
        0: (0, 0.3), 1: (2, 2.0), 2: (1, 0.0)}
    assert combine(0, 5.0) > combine(1, 0.0) > combine(2, 0.0) > combine(2, 3.0)


def test_checker_needs_dataset_root():
    cl = registry.get("kinema")
    cl.context = CurationContext()
    with pytest.raises(RuntimeError, match="dataset_root"):
        cl.select(synthetic_view(6), 3, np.random.default_rng(0))


# ------------------------------------------------------------------ progress cache + conformance


def test_progress_cache_keyed_by_data_and_revision(tmp_path):
    c = CachedProgress(MockProgressModel(), tmp_path)
    v = synthetic_view(5)
    a = c.episode_progress(v)
    b = c.episode_progress(v)
    assert c.hits == 1 and all(np.array_equal(a[i], b[i]) for i in a)
    assert c.path(v) != c.path(synthetic_view(5, seed=1))


@pytest.mark.parametrize("name", ["robometer_filter", "rabc", "rcb_score", "demo_score", "cupid"])
def test_entrant_passes_submit_conformance(name):
    rep = conformance.check(name, synthetic_view(12), time_limit_s=120)
    assert rep.passed, rep.lines()


@pytest.mark.slow
def test_deminf_passes_submit_conformance():
    rep = conformance.check("deminf", synthetic_view(12), time_limit_s=900)
    assert rep.passed, rep.lines()


def test_cupid_influence_matches_leave_one_out_refit():
    """CUPID's influence formula (per-sample grads from `per_sample_projected_grads` with identity projection,
    score_z = grad l_z^T H^-1 u, u = sum_tau R(tau) sum grad l(s', a')) against brute-force leave-one-out refits
    of a linear least-squares policy with the EXACT Hessian. Upweighting z changes the target
    T = sum R log pi(a'|s') = -sum R l(s', a') by +score_z (Eq. 2 sign), so removing z changes it by ~ -score_z."""
    from torch.func import functional_call, hessian

    from robocurate.cleaners.entrants.cupid import per_sample_projected_grads, performance_influence

    class Lin(torch.nn.Module):
        def __init__(self, W):
            super().__init__()
            self.W = torch.nn.Parameter(torch.as_tensor(W, dtype=torch.float32))

        def forward(self, s):
            return s @ self.W

        def loss(self, b):
            return ((self(b["observation.state"]) - b["action"]) ** 2).mean(-1).mean()

    rng = np.random.default_rng(0)
    N, ds, da = 200, 3, 2
    X = rng.normal(size=(N, ds))
    Wt = rng.normal(size=(ds, da))
    Y = X @ Wt + 0.3 * rng.normal(size=(N, da))
    Y[:5] += 3.0  # a few corrupted labels
    fit = lambda idx: np.linalg.lstsq(X[idx], Y[idx], rcond=None)[0]  # noqa: E731  minimizes the sum loss exactly
    W = fit(np.arange(N))
    pol = Lin(W)
    Xs = rng.normal(size=(60, ds))  # "rollout" states with sampled actions and +-1 returns
    As = Xs @ Wt + 0.3 * rng.normal(size=(60, da))
    R = np.where(rng.uniform(size=60) < 0.5, 1.0, -1.0)
    eye = torch.eye(ds * da)
    G = per_sample_projected_grads(pol, X.astype(np.float32), Y.astype(np.float32), eye)
    Gr = per_sample_projected_grads(pol, Xs.astype(np.float32), As.astype(np.float32), eye)
    u = (torch.as_tensor(R, dtype=torch.float32)[:, None] * Gr).sum(0)
    params = {k: v.detach() for k, v in pol.named_parameters()}
    St, At = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    H = hessian(lambda p: ((functional_call(pol, p, (St,)) - At) ** 2).mean(-1).sum())(params)["W"]["W"]
    score = performance_influence(G, u, H.reshape(ds * da, ds * da)).numpy()

    def T(Wm):
        return float(-(R * ((Xs @ Wm - As) ** 2).mean(-1)).sum())

    loo = np.array([T(fit(np.delete(np.arange(N), z))) - T(W) for z in range(N)])
    assert np.corrcoef(loo, -score)[0, 1] > 0.99
    assert np.mean(np.sign(loo) == np.sign(-score)) > 0.95
