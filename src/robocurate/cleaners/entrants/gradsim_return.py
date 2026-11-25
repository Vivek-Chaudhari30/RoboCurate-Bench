"""gradsim-return (D-036): return-weighted gradient similarity, the pre-registered influence entrant.

Credits. The objective is CUPID's (Agia et al., CoRL 2025, arXiv 2506.19121; official code agiachris/cupid@2941eba):
rollouts of the reference policy weighted by their return R = +1 success / -1 failure ("net", sum_of_sum;
diffusion_policy/common/error_util.py:17-19, 299-330), with CUPID's model-output function per policy family. The
comparison is TracIn's (Pruthi et al., NeurIPS 2020, arXiv 2002.08484): influence as a plain dot product of
gradients, with no inverse Hessian. Bibliographic entries: docs/references.bib (agia2025cupid, pruthi2020tracin).

Computation (exactly the gradient-similarity limit recorded in PROGRESS.md "CUPID diffusion toybed": H ~ lambda I
gave AUROC 1.000; here the dot product is exact, without the random projection):
  g(s, a)  = gradient w.r.t. the policy parameters of CUPID's model output at a sample:
               diffusion policies: the official "square" output, mean over the action chunk of eps_theta^2, averaged
                 over 64 random timesteps with fresh noise (modelout_functions.py:116-119; train_trak.sh:316);
               other (unimodal, ACT-style) policies: the per-sample BC loss (the CUPID entrant's MLP/ACT reading).
  u        = sum over m = 100 rollouts tau of R(tau) * sum_{(s', a') in tau} g(s', a')  (CUPID Def. 1 / Eq. 3, m = 100
             in simulation, Sec. 6), rolled out on curation seeds (D-007, D-022).
  score(e) = sum_{(s, a) in episode e} g(s, a) . u;  keep the top-k episodes (ties D-021).
No Hessian, no Gram matrix, no damping, no projection: it has no tunable parameter of its own (timesteps, m and the
return weighting are CUPID's published values). Rollout policy as in the CUPID entrant: diffusion policies are
sampled as is; unimodal policies as N(mu(s), sigma^2 I) with sigma the RMS BC residual (cupid.py docstring).

Enablement (D-036): only for policy families whose toybed AUROC (planted wrong-direction demos ranked most harmful)
is >= 0.8 (tests/test_entrants.py, PROGRESS.md overnight 3 item 6).
"""

from __future__ import annotations

import numpy as np
import torch

from robocurate.cleaners.base import ScoreCleaner
from robocurate.cleaners.entrants.cupid import (
    N_ROLLOUTS,
    StochasticPolicy,
    diffusion_square_features,
    per_sample_projected_grads,
)
from robocurate.data.episodes import DatasetView, EpisodeData

CHUNK = 256


def _flat_grads_mlp(policy, states, actions) -> torch.Tensor:
    return per_sample_projected_grads(policy, states, actions, None)


def gradsim_scores(policy, view: DatasetView, rollouts, seed: int) -> tuple[dict[int, float], dict]:
    eps = list(view)
    diffusion = getattr(policy, "policy_family", None) == "diffusion"
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    n = min(N_ROLLOUTS, rollouts.remaining)
    if diffusion:
        res = rollouts.rollout(policy.reseed(seed), rollouts.next_seeds(n))

        def feats(states, actions):
            o, c = policy.make_samples(states, actions)
            return diffusion_square_features(policy, o, c, None, gen)
        sigma = None
    else:
        S = np.concatenate([e.state for e in eps]).astype(np.float32)
        A = np.concatenate([e.actions for e in eps]).astype(np.float32)
        with torch.no_grad():
            sigma = float((policy(torch.as_tensor(S)) - torch.as_tensor(A)).pow(2).mean().sqrt())
        res = rollouts.rollout(StochasticPolicy(policy, sigma, seed), rollouts.next_seeds(n))

        def feats(states, actions):
            return _flat_grads_mlp(policy, np.asarray(states, np.float32), np.asarray(actions, np.float32))
    u = None
    for r in res:
        if r.length == 0:
            continue
        g = feats(r.states, r.actions).sum(0) * (1.0 if r.success else -1.0)
        u = g if u is None else u + g
    if u is None:
        raise RuntimeError("gradsim-return: no non-empty rollouts")
    psi = {}
    for e in eps:
        if diffusion:  # the diffusion sampler builds action chunks from the whole episode (padding at its end)
            psi[e.episode_index] = float((feats(e.state, e.actions) @ u).sum())
        else:
            psi[e.episode_index] = sum(float((feats(e.state[i:i + CHUNK], e.actions[i:i + CHUNK]) @ u).sum())
                                       for i in range(0, e.length, CHUNK))
    return psi, {"family": "diffusion" if diffusion else "unimodal", "n_rollouts": len(res),
                 "n_success": int(sum(r.success for r in res)), "sigma": sigma}


class GradsimReturn(ScoreCleaner):
    name, version = "gradsim_return", "1"
    requires_rollouts = N_ROLLOUTS
    requires_reference_policy = True
    implementation = "original"
    # D-036: enabled per policy family only after passing that family's toybed (AUROC >= 0.8). Measured
    # 2026-11-20 (PROGRESS.md overnight 3 item 6): diffusion toybed AUROC 1.000 -> enabled; MLP/ACT-style toybed
    # AUROC 0.383 -> not enabled for ACT.
    enabled_policy_families: tuple[str, ...] = ("diffusion",)
    source = ("D-036; objective from CUPID (Agia et al. 2025, arXiv 2506.19121, agiachris/cupid@2941eba), "
              "gradient-similarity comparison from TracIn (Pruthi et al. 2020, arXiv 2002.08484)")

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        ctx = self.context
        policy = ctx.reference_policy.load()
        policy.eval()
        self._psi, self.diagnostics = gradsim_scores(policy, view, ctx.rollouts, int(rng.integers(2**31)))
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._psi[episode.episode_index]
