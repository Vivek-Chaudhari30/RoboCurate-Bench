"""DemInf (reimplementation). Hejna et al., "Robot Data Curation with Mutual Information Estimators", RSS 2025,
arXiv 2502.08623v3. Official code github.com/jhejna/demonstration-information@232151defb258b5ceb0a6b62644574b57d981b58
(MIT) is JAX/TF and not pip-installable, so this is a reimplementation of the paper + reference code.

  * Representations (Sec. V-A, Table I "RoboMimic state"): separate VAEs for state and action, z_s = 12, z_a = 6,
    action chunk 1, MLP 2x512, beta 0.05, Adam lr 1e-4, batch 256, 50k steps
    (configs/quality/vae_robomimic_state.py:32-35, 91, 94, 104). We embed with the posterior mean.
  * Estimator: KSG (Kraskov et al. 2004), per-sample I(s_i, a_i) ~ -psi(n_s + 1) - psi(n_a + 1) (Sec. V-B, App. D),
    joint distance = max of the two L2 latent distances; counts use strict "<" and include the point itself
    (scripts/quality/quality_estimators.py:39-55); k in {5, 6, 7}, averaged (Table I; quality_estimators.py:165).
  * Batching (estimate_quality.py:39; quality_estimators.py:171): distances within batches of 1024; 4 repeats,
    each keeping a random 50% of every episode's timesteps.
  * Per-sample scores are clipped to the 1st-99th percentile and z-scored (quality_estimators.py:89-90), then
    averaged per episode: S(tau) = (1/T) sum_t I_t (Sec. V-C).
  * Selection: paper keeps S(tau) > kappa; the selection track fixes the budget, so we keep the top-k
    (ties D-021).
The constant psi(k) + psi(N) terms are dropped: they do not change rankings within a batch.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import digamma
from torch import nn

from robocurate.cleaners.base import ScoreCleaner
from robocurate.data.episodes import DatasetView, EpisodeData

Z_S, Z_A, HIDDEN, BETA, LR, BATCH, STEPS = 12, 6, 512, 0.05, 1e-4, 256, 50_000
KS, KSG_BATCH, REPEATS, KEEP_FRACTION = (5, 6, 7), 1024, 4, 0.5


class VAE(nn.Module):
    def __init__(self, d: int, z: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 2 * z))
        self.dec = nn.Sequential(nn.Linear(z, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, d))

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.enc(x).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = ((self.dec(z) - x) ** 2).sum(-1).mean()
        kl = (-0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(-1)).mean()
        return rec + BETA * kl

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x).chunk(2, -1)[0]


def train_vae(X: np.ndarray, z: int, steps: int, seed: int) -> tuple[VAE, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.as_tensor((X - mu) / sd, dtype=torch.float32)
    vae = VAE(X.shape[1], z)
    opt = torch.optim.Adam(vae.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(Xt), (min(BATCH, len(Xt)),), generator=g)
        loss = vae.loss(Xt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return vae, mu, sd


def ksg_scores(zs: np.ndarray, za: np.ndarray, ks=KS) -> np.ndarray:
    """Per-sample KSG terms (up to constants) for one batch, averaged over ks."""
    ds = np.linalg.norm(zs[:, None] - zs[None], axis=-1)
    da = np.linalg.norm(za[:, None] - za[None], axis=-1)
    dj = np.maximum(ds, da)
    out = np.zeros(len(zs))
    for k in ks:
        eps = np.sort(dj, axis=1)[:, k]  # k-th neighbour (column 0 is self)
        # KSG: psi(n + 1) with n = neighbours strictly closer than eps EXCLUDING self. The reference code counts
        # with self included (equal whenever eps > 0); this form also handles eps == 0 (exact duplicates, e.g.
        # held poses), where the self-inclusive count is 0 and digamma(0) is NaN.
        self_in = (eps > 0).astype(int)
        n_s = (ds < eps[:, None]).sum(1) - self_in
        n_a = (da < eps[:, None]).sum(1) - self_in
        out += -digamma(n_s + 1) - digamma(n_a + 1)
    return out / len(ks)


class DemInf(ScoreCleaner):
    name, version = "deminf", "1"
    implementation = "reimplementation"
    source = ("Hejna et al. 2025, arXiv 2502.08623v3 Sec. V, App. D, Table I; ref code (MIT) "
              "github.com/jhejna/demonstration-information@232151d")
    vae_steps = STEPS  # paper default; never changed for results

    def select(self, view: DatasetView, budget: int, rng: np.random.Generator) -> list[int]:
        eps = list(view)
        S = np.concatenate([e.state for e in eps]).astype(np.float32)
        A = np.concatenate([e.actions for e in eps]).astype(np.float32)
        ep_of = np.concatenate([np.full(e.length, e.episode_index) for e in eps])
        seed = int(rng.integers(2**31))
        vs, ms, ss = train_vae(S, Z_S, self.vae_steps, seed)
        va, ma, sa = train_vae(A, Z_A, self.vae_steps, seed + 1)
        zs = vs.embed(torch.as_tensor((S - ms) / ss)).numpy()
        za = va.embed(torch.as_tensor((A - ma) / sa)).numpy()
        tot, cnt = np.zeros(len(S)), np.zeros(len(S))
        for _ in range(REPEATS):
            keep = np.concatenate([rng.permutation(np.flatnonzero(ep_of == e.episode_index))
                                   [: max(1, int(KEEP_FRACTION * e.length))] for e in eps])
            keep = rng.permutation(keep)
            for b in range(0, len(keep), KSG_BATCH):
                idx = keep[b:b + KSG_BATCH]
                if len(idx) <= max(KS):
                    continue
                tot[idx] += ksg_scores(zs[idx], za[idx])
                cnt[idx] += 1
        m = cnt > 0
        x = tot[m] / cnt[m]
        lo, hi = np.percentile(x, [1, 99])
        x = np.clip(x, lo, hi)
        x = (x - x.mean()) / (x.std() + 1e-8)
        per = {}
        for e, v in zip(ep_of[m], x, strict=True):
            per.setdefault(int(e), []).append(v)
        self._s = {e.episode_index: float(np.mean(per.get(e.episode_index, [-np.inf]))) for e in eps}
        return super().select(view, budget, rng)

    def score(self, episode: EpisodeData) -> float:
        return self._s[episode.episode_index]
