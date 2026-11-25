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
    """A demo whose gradient points the way successful rollouts' gradients point gets positive influence.

    The old fixture was MLPPolicy(1, 1, hidden=4) on 30 linspace states (33 parameters). A 1-D input only
    excites a few of those directions, so phi.T @ phi had condition number ~3.2e11. torch.linalg.pinv of
    that matrix flipped the comparison across BLAS: one runner got tensor(13.0770) > tensor(33.3154) and
    failed, another passed. This fixture is a linear map on a centered full-rank design. Good actions are
    the first state coordinate, bad actions are its negation, and the rollout target matches the good
    actions. The gram-matrix condition number stays small, so the inequality is the influence sign.
    """
    from robocurate.cleaners.entrants.cupid import per_sample_projected_grads

    class _LinearToy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.w = torch.nn.Parameter(torch.randn(4, generator=g) * 0.05)
            self.b = torch.nn.Parameter(torch.zeros(()))

        def loss(self, batch: dict) -> torch.Tensor:
            state = batch["observation.state"]
            action = batch["action"].reshape(-1)
            return ((state @ self.w + self.b - action) ** 2).mean()

    dim, n = 4, 32
    design = np.linalg.qr(np.random.default_rng(0).normal(size=(n, dim)), mode="reduced")[0]
    states = design.astype(np.float32)
    states -= states.mean(axis=0, keepdims=True)
    good, bad = states[:, :1].copy(), -states[:, :1]
    pol = _LinearToy()
    projection = torch.eye(sum(p.numel() for p in pol.parameters()))
    both_states = np.concatenate([states, states])
    both_actions = np.concatenate([good, bad])
    phi = per_sample_projected_grads(pol, both_states, both_actions, projection)
    gram = (phi.T @ phi).double()
    cond = torch.linalg.cond(gram).item()
    assert cond < 1e3, cond  # the old fixture was ~3.2e11; a large cond is the failure mode to catch
    rollout = per_sample_projected_grads(pol, states, good, projection).sum(0)
    influence = phi @ (torch.linalg.pinv(gram).float() @ rollout)
    assert influence[:n].sum() > influence[n:].sum()


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


# ------------------------------------------------------------------ checkers, installed ([checkers] extra)
CHECKER_MODULES = {"lerobot_doctor": "lerobot_doctor", "kinema": "kinema", "roboaudit": "roboaudit", "rda": "rda"}


@pytest.fixture(scope="module")
def toy_dataset_root(tmp_path_factory):
    from robocurate.data.lerobot_write import write_view_dataset

    view, labels = make_dataset()
    return view, labels, write_view_dataset(view, tmp_path_factory.mktemp("toyds") / "toybed")


@pytest.mark.parametrize("name", list(CHECKER_MODULES))
def test_checker_toybed_sanity(name, toy_dataset_root):
    """Runs the real tool on the toybed LeRobot dataset (no labels sidecar). Sanity: some planted defect type is
    kept less often than clean demos (pauses, the defect every checker targets). Reported, not tuned."""
    pytest.importorskip(CHECKER_MODULES[name])
    view, labels, root = toy_dataset_root
    cl = registry.get(name)
    cl.context = CurationContext(dataset_root=root)
    ids = set(cl.select(view, len(labels) // 2, np.random.default_rng(0)))
    frac = {k: np.mean([i in ids for i in labels if labels[i] == k]) for k in KINDS}
    assert frac["pause"] < frac["clean"], frac


@pytest.mark.slow
@pytest.mark.parametrize("name", list(CHECKER_MODULES))
def test_checker_conformance_with_installed_tool(name):
    pytest.importorskip(CHECKER_MODULES[name])
    rep = conformance.check(name)
    assert rep.passed, rep.lines()


def test_fake_context_dataset_root_has_no_labels(tmp_path):
    """Checkers read a dataset directory: the fake one written for conformance carries no labels sidecar."""
    from robocurate.curation.prepare import fake_dataset_root

    root = fake_dataset_root(synthetic_view(6), tmp_path)
    assert (root / "meta" / "info.json").exists()
    assert not list(root.rglob("*labels*"))


# ------------------------------------------------------------------ CUPID in its native setting (D-029)
# Diffusion toybed, fixed BEFORE the first CUPID run: 240 clean + 60 wrong-direction demos (a distinct mode),
# ToyDiffusionPolicy trained 4000 steps (batch 256, lr 1e-3, seed 0). Pass rule (task prompt): AUROC >= 0.8 for the
# planted demos being ranked most harmful.

DIFF_TOY = {"n_clean": 240, "n_pause": 0, "n_jitter": 0, "n_wrong": 60, "seed": 0}


def _auroc_low(scores: dict, labels: dict, pos: str = "wrong", neg: str = "clean") -> float:
    """P(score of a planted demo < score of a clean demo), ties 1/2 (Mann-Whitney AUROC)."""
    w = np.array([scores[i] for i in labels if labels[i] == pos])
    c = np.array([scores[i] for i in labels if labels[i] == neg])
    return float(np.mean(w[:, None] < c[None, :]) + 0.5 * np.mean(w[:, None] == c[None, :]))


def test_auroc_helper():
    lab = {0: "wrong", 1: "wrong", 2: "clean", 3: "clean"}
    assert _auroc_low({0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0}, lab) == 1.0
    assert _auroc_low({0: 2.0, 1: 3.0, 2: 0.0, 3: 1.0}, lab) == 0.0
    assert _auroc_low({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, lab) == 0.5


def test_cupid_diffusion_square_features_match_autograd():
    """vmap features == per-sample autograd of the official "square" output averaged over the same timesteps."""
    from toybed.diffusion import ToyDiffusionPolicy

    from robocurate.cleaners.entrants import cupid as CU

    pol = ToyDiffusionPolicy(seed=1)
    n = sum(p.numel() for p in pol.denoiser.parameters())
    rng = np.random.default_rng(0)
    obs = torch.as_tensor(rng.normal(size=(5, 4)).astype(np.float32))
    ch = torch.as_tensor(rng.uniform(-1, 1, size=(5, 8, 2)).astype(np.float32))
    P = torch.eye(n)
    got = CU.diffusion_square_features(pol, obs, ch, P, torch.Generator().manual_seed(7))
    g = torch.Generator().manual_seed(7)  # replay the same draws (one chunk of <= DIFF_CHUNK samples)
    ts = torch.randint(0, pol.num_train_timesteps, (5, CU.NUM_TIMESTEPS), generator=g)
    nz = torch.randn((5, CU.NUM_TIMESTEPS, 8, 2), generator=g)
    for i in range(5):
        pol.denoiser.zero_grad()
        noisy = pol.add_noise(ch[i].expand(CU.NUM_TIMESTEPS, -1, -1), nz[i], ts[i])
        pred = pol.denoiser(noisy, ts[i], obs[i].expand(CU.NUM_TIMESTEPS, -1))
        (pred**2).mean(dim=(1, 2)).mean().backward()
        ref = torch.cat([p.grad.reshape(-1) for p in pol.denoiser.parameters()])
        torch.testing.assert_close(got[i], ref, rtol=1e-4, atol=1e-6)


@pytest.fixture(scope="module")
def diffusion_toy():
    from toybed.diffusion import train_diffusion

    view, labels = make_dataset(**DIFF_TOY)
    d = tempfile.mkdtemp()
    ref = train_diffusion(view, seed=0).save(d + "/ref")
    return view, labels, ref


@pytest.mark.slow
def test_diffusion_toybed_premise(diffusion_toy):
    """The planted wrong-direction mode lowers closed-loop success (2026-11-19: 0.075 vs 1.000, seed 0)."""
    from toybed.diffusion import ToyDiffusionPolicy, diffusion_success_rate, train_diffusion

    view, labels, ref = diffusion_toy
    seeds = list(range(900000, 900200))  # disjoint from demo seeds (< 10^4) and curation seeds (5x10^5)
    with_wrong = diffusion_success_rate(ToyDiffusionPolicy.load(ref), seeds)
    without = diffusion_success_rate(train_diffusion(view, [i for i in labels if labels[i] != "wrong"], seed=0), seeds)
    assert without - with_wrong >= 0.3, (with_wrong, without)


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="D-029: CUPID with the official diffusion setting (square output, 64 timesteps, "
                   "d=4000, lambda=0, m=100, net/sum_of_sum) ranks the planted mode at AUROC ~0.52 < 0.8; stays "
                   "disabled, not tuned (PROGRESS.md 'CUPID diffusion toybed')")
def test_sanity_cupid_diffusion(diffusion_toy):
    from toybed.diffusion import ToyDiffusionPolicy

    view, labels, ref = diffusion_toy
    cl = registry.get("cupid")
    cl.context = CurationContext(
        rollouts=RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv), cap=cl.requires_rollouts),
        reference_policy=ReferencePolicy(ref, ToyDiffusionPolicy.load))
    cl.select(view, view.n_episodes // 2, np.random.default_rng(0))
    assert cl.diagnostics["family"] == "diffusion" and cl.diagnostics["n_rollouts"] == 100
    assert _auroc_low(cl._psi, labels) >= 0.8


# ------------------------------------------------------------------ gradsim-return (D-036), both toybeds


def _gradsim_auroc(view, labels, ref, loader):
    cl = registry.get("gradsim_return")
    cl.context = CurationContext(
        rollouts=RolloutService.from_protocol(PROTO, GymRolloutBackend(PointReachEnv), cap=cl.requires_rollouts),
        reference_policy=ReferencePolicy(ref, loader))
    cl.select(view, view.n_episodes // 2, np.random.default_rng(0))
    assert cl.diagnostics["n_rollouts"] == 100
    return _auroc_low(cl._psi, labels), cl.diagnostics


def test_gradsim_return_has_no_tunable_parameters_and_credits():
    from robocurate.cleaners.entrants import gradsim_return as G

    c = registry.get("gradsim_return")
    assert c.implementation == "original" and "2506.19121" in c.source and "2002.08484" in c.source
    assert {n for n in dir(G) if n.isupper()} == {"CHUNK", "N_ROLLOUTS"}  # batching + CUPID m=100; no lambda / d
    assert not hasattr(G, "LAMBDA_REG") and not hasattr(G, "PROJ_DIM")


def test_gradsim_return_toybed_mlp(toy):
    """ACT-style unimodal (MLP) toybed: report the AUROC; enablement follows it (D-036)."""
    view, labels, ref = toy
    auc, diag = _gradsim_auroc(view, labels, ref, MLPPolicy.load)
    assert diag["family"] == "unimodal"
    fams = registry.get("gradsim_return").enabled_policy_families
    assert ("act" in fams) == (auc >= 0.8), (auc, fams)


@pytest.mark.slow
def test_gradsim_return_toybed_diffusion(diffusion_toy):
    from toybed.diffusion import ToyDiffusionPolicy

    view, labels, ref = diffusion_toy
    auc, diag = _gradsim_auroc(view, labels, ref, ToyDiffusionPolicy.load)
    assert diag["family"] == "diffusion"
    fams = registry.get("gradsim_return").enabled_policy_families
    assert ("diffusion" in fams) == (auc >= 0.8), (auc, fams)
