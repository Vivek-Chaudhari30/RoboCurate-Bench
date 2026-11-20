"""CUPID-style performance influence (reimplementation). Agia et al., "CUPID: Curating Data your Robot Loves with
Influence Functions", CoRL 2025, arXiv 2506.19121v2. Official code
github.com/agiachris/cupid@2941eba9427a3398d3cbe26b10c01e3ece56bd18 (MIT) is built into the Diffusion Policy
codebase and is not pip-installable, so this is a reimplementation that is generic over any policy exposing
`loss(batch)` (per-sample BC loss) and `parameters()`.

Paper -> implementation:
  * Action influence (Eq. 2): Psi_a((s',a'),(s,a)) = -grad log pi(a'|s')^T H_bc^-1 grad l(s,a). With a Gaussian BC
    policy l = -log pi + const, so Psi_a = grad l(s',a')^T H^-1 grad l(s,a).
  * Performance influence (Def. 1, Prop. 1 Eq. 3, Alg. 1): Psi(xi) ~ sum_tau R(tau) sum_{(s',a') in tau}
    sum_{(s,a) in xi} Psi_a, with R = +1 success / -1 failure (Sec. 4). The reference code aggregates exactly so
    ("net" = successes minus failures, "sum_of_sum"; diffusion_policy/common/error_util.py:17-19, 311-330),
    dropping the paper's 1/m and 1/H factors (rank-preserving).
  * TRAK approximation (App. A.1-A.2): per-sample gradients randomly projected to d = 4000
    (scripts/train/train_trak.sh:309), Gauss-Newton H = Phi^T Phi over training samples, no damping
    (lambda_reg = 0.0, train_trak.sh:311; pseudo-inverse when singular). Model output function = the per-sample
    BC loss (our reading of the code's "square" output, modelout_functions.py:116-119).
  * Rollouts: m = 100 in simulation (Sec. 6, App. B.4) = requires_rollouts 100 (the D-007 cap).
  * Selection (Eqs. 4-5; ranking as in notebooks/data_curation.ipynb cell 5): keep the top-k demos by Psi;
    filtering drops the lowest-influence demos. Ties D-021.
Benchmark adaptation (declared): the policy is rolled out STOCHASTICALLY as N(mu(s), sigma^2 I) with sigma the RMS
BC residual of the reference policy on the dataset (the homoscedastic-Gaussian MLE). A deterministic rollout
policy makes grad l(s', mu(s')) = 0 for an MSE policy, so the influence would vanish; CUPID's diffusion policy is
stochastic by construction. The LeRobot/ACT path needs a loss(batch) adapter for PreTrainedPolicy (not built).

Diffusion path (D-029, native setting; used when the reference policy declares policy_family == "diffusion"):
  * Model output = the official "square" function (diffusion_policy/data_attribution/modelout_functions.py:116-119
    @2941eba): f = mean(eps_theta(noisy_chunk, t, obs)^2) over the action chunk, with fresh Gaussian noise and the
    chunk noised by the scheduler (:79-82); Q = 1 (get_out_to_loss_grad, :128-136).
  * Per-sample feature = gradient of f w.r.t. the denoiser weights averaged over num_timesteps = 64 timesteps drawn
    uniformly from [0, num_train_timesteps) (train_trak.sh:316; train_trak_diffusion.py:175-182 and 214-221;
    gradient_computers.py:110-128 `_average_vectorize`), projected to d = 4000 (train_trak.sh:309), H = Phi^T Phi
    with lambda_reg = 0 (:311). Training samples and rollout targets use the same output function.
  * Scores: train-target TRAK score phi_target^T H^-1 phi_train; trajectory scores aggregated with sum_of_sum and
    the "net" demo-quality metric (error_util.py:17-19, 299-330): successes minus failures. Higher = keep.
  * Rollouts: the diffusion policy is sampled as is (stochastic by construction); m = 100.
  The policy object supplies `denoiser` (nn.Module(noisy, t, obs)), `alphas_cumprod`, `num_train_timesteps`,
  `make_samples(states, actions) -> (obs, chunks)` and `reseed(seed)`. Only the toybed policy implements this; a
  LeRobot DiffusionPolicy adapter is not built.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData

PROJ_DIM, LAMBDA_REG, N_ROLLOUTS, CHUNK = 4000, 0.0, 100, 512
NUM_TIMESTEPS = 64  # train_trak.sh:316
DIFF_CHUNK = 64


class _LossWrap(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.p = policy

    def forward(self, batch):
        return self.p.loss(batch)


def per_sample_projected_grads(policy: nn.Module, states: np.ndarray, actions: np.ndarray, P: torch.Tensor | None
                               ) -> torch.Tensor:
    """(N, d): random projections of per-sample loss gradients (torch.func vmap over samples); P=None returns the
    full flattened gradients (N, n_params)."""
    wrap = _LossWrap(policy).eval()
    params = {k: v.detach() for k, v in wrap.named_parameters()}

    def lf(prm, s, a):
        return functional_call(wrap, prm, ({"observation.state": s[None], "action": a[None]},))

    g = vmap(grad(lf), in_dims=(None, 0, 0))
    S = torch.as_tensor(np.array(states, dtype=np.float32))
    A = torch.as_tensor(np.array(actions, dtype=np.float32))
    out = []
    for i in range(0, len(S), CHUNK):
        gi = g(params, S[i:i + CHUNK], A[i:i + CHUNK])
        flat = torch.cat([gi[k].reshape(len(S[i:i + CHUNK]), -1) for k in params], dim=1)
        out.append(flat if P is None else flat @ P)
    return torch.cat(out) if out else torch.zeros(0, 0 if P is None else P.shape[1])


def performance_influence(train_feats: torch.Tensor, target_sum: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Per-training-sample performance influence (Eq. 2-3): score_z = phi_z^T H^+ u, with u = sum_tau R(tau)
    sum_{(s',a')} phi(s', a'). Positive = upweighting z raises sum R log pi(a'|s'). H^+ = pseudo-inverse
    (lambda_reg = 0, App. A.2)."""
    return train_feats @ (torch.linalg.pinv(H.double()).float() @ target_sum)


def _projection(n_params: int, seed: int) -> torch.Tensor:
    return torch.randn(n_params, PROJ_DIM, generator=torch.Generator().manual_seed(seed)) / PROJ_DIM**0.5


def diffusion_square_features(policy, obs: torch.Tensor, chunks: torch.Tensor, P: torch.Tensor | None,
                              gen: torch.Generator) -> torch.Tensor:
    """(N, d): projected per-sample gradients of the official "square" output, averaged over NUM_TIMESTEPS random
    timesteps with fresh noise (modelout_functions.py:79-119; gradient_computers.py:110-128)."""
    den = policy.denoiser
    params = {k: v.detach() for k, v in den.named_parameters()}
    abar = policy.alphas_cumprod.detach()

    def f(prm, o, c, ts, nz):
        ab = abar[ts].reshape(-1, 1, 1)
        noisy = ab.sqrt() * c.unsqueeze(0) + (1 - ab).sqrt() * nz
        pred = functional_call(den, prm, (noisy, ts, o.unsqueeze(0).expand(len(ts), -1)))
        return (pred**2).mean(dim=(1, 2)).mean()  # mean over the chunk, then over timesteps (= averaged grads)

    g = vmap(grad(f), in_dims=(None, 0, 0, 0, 0))
    out = []
    for i in range(0, len(obs), DIFF_CHUNK):
        o, c = obs[i:i + DIFF_CHUNK], chunks[i:i + DIFF_CHUNK]
        ts = torch.randint(0, int(policy.num_train_timesteps), (len(o), NUM_TIMESTEPS), generator=gen)
        nz = torch.randn((len(o), NUM_TIMESTEPS, *c.shape[1:]), generator=gen)
        gi = g(params, o, c, ts, nz)
        flat = torch.cat([gi[k].reshape(len(o), -1) for k in params], dim=1)
        out.append(flat if P is None else flat @ P)
    return torch.cat(out) if out else torch.zeros(0, 0 if P is None else P.shape[1])


def cupid_diffusion_scores(policy, view: DatasetView, rollouts, seed: int) -> tuple[dict[int, float], dict]:
    """Per-episode CUPID demo-quality scores (net, sum_of_sum) for a diffusion policy."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    n_params = sum(p.numel() for p in policy.denoiser.parameters())
    P = _projection(n_params, seed)
    eps = list(view)
    obs_l, chunk_l = zip(*[policy.make_samples(e.state, e.actions) for e in eps], strict=True)
    phi = diffusion_square_features(policy, torch.cat(obs_l), torch.cat(chunk_l), P, gen)  # (N, d)
    H = phi.T @ phi + LAMBDA_REG * torch.eye(PROJ_DIM)
    n = min(N_ROLLOUTS, rollouts.remaining)
    res = rollouts.rollout(policy.reseed(seed), rollouts.next_seeds(n))
    u = torch.zeros(PROJ_DIM)
    for r in res:
        if r.length == 0:
            continue
        o, c = policy.make_samples(r.states, r.actions)
        u += (1.0 if r.success else -1.0) * diffusion_square_features(policy, o, c, P, gen).sum(0)
    per_frame = performance_influence(phi, u, H).numpy()
    bounds = np.cumsum([0] + [e.length for e in eps])
    psi = {e.episode_index: float(per_frame[bounds[j]:bounds[j + 1]].sum()) for j, e in enumerate(eps)}
    return psi, {"family": "diffusion", "n_rollouts": len(res), "n_success": int(sum(r.success for r in res)),
                 "n_train_samples": int(len(phi)), "n_params": int(n_params)}


class StochasticPolicy:
    """a ~ N(mu(s), sigma^2 I) for rollouts (see module docstring)."""

    def __init__(self, policy, sigma: float, seed: int):
        self.policy, self.sigma, self.rng = policy, float(sigma), np.random.default_rng(seed)

    def act(self, obs):
        mu = self.policy.act(obs)
        return (mu + self.rng.normal(0.0, self.sigma, np.shape(mu))).astype(np.float32)


class Cupid(ScoreCleaner):
    name, version = "cupid", "1"
    requires_rollouts = N_ROLLOUTS
    requires_reference_policy = True
    implementation = "reimplementation"
    # Toybed sanity fails on the MLP toybed (outcome (c)) and on the diffusion toybed in its native setting
    # (D-029: AUROC 0.51-0.52 < 0.8). Stays disabled for every policy family (PROGRESS.md).
    status = "disabled — failed sanity"
    source = ("Agia et al. 2025, arXiv 2506.19121v2 Def. 1, Eqs. 2-5, App. A; ref code (MIT) "
              "github.com/agiachris/cupid@2941eba")

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        ctx = self.context
        policy = ctx.reference_policy.load()
        policy.eval()
        seed = int(rng.integers(2**31))
        if getattr(policy, "policy_family", None) == "diffusion":  # native setting (D-029)
            self._psi, self.diagnostics = cupid_diffusion_scores(policy, view, ctx.rollouts, seed)
            return super().select(view, budget, rng)
        n_params = sum(p.numel() for p in policy.parameters())
        P = _projection(n_params, seed)
        eps = list(view)
        S = np.concatenate([e.state for e in eps]).astype(np.float32)
        A = np.concatenate([e.actions for e in eps]).astype(np.float32)
        with torch.no_grad():
            resid = policy(torch.as_tensor(S)) - torch.as_tensor(A)
        sigma = float(resid.pow(2).mean().sqrt())
        phi = per_sample_projected_grads(policy, S, A, P)  # (N, d)
        H = phi.T @ phi + LAMBDA_REG * torch.eye(PROJ_DIM)
        n = min(N_ROLLOUTS, ctx.rollouts.remaining)
        res = ctx.rollouts.rollout(StochasticPolicy(policy, sigma, seed), ctx.rollouts.next_seeds(n))
        u = torch.zeros(PROJ_DIM)
        for r in res:
            if r.length == 0:
                continue
            gt = per_sample_projected_grads(policy, r.states, r.actions, P).sum(0)
            u += (1.0 if r.success else -1.0) * gt
        per_frame = performance_influence(phi, u, H).numpy()
        bounds = np.cumsum([0] + [e.length for e in eps])
        self._psi = {e.episode_index: float(per_frame[bounds[j]:bounds[j + 1]].sum()) for j, e in enumerate(eps)}
        self.diagnostics = {"sigma": sigma, "n_rollouts": len(res), "n_success": int(sum(r.success for r in res))}
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._psi[episode.episode_index]
