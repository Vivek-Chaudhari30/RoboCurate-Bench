"""Null-calibration and power simulation for the primary test (D-016 as amended by D-019). CPU only.

Model per grid: each of C null cleaners picks ONE subset with effect a_c ~ N(0, sigma_sub); each of K placebo
subsets has a_k ~ N(0, sigma_sub); run rate = clip(base + effect + a + N(0, sigma_seed), 0, 1), measured as
Binomial(n_rollouts)/n_rollouts. The C cleaners share one placebo per grid (as in a real report).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from robocurate.report import stats as S

SIGMA_SUB = (0.0, 0.02, 0.04, 0.06)
SIGMA_SEED = (0.03, 0.06)
DESIGNS = ((4, 2, 3), (8, 2, 3), (8, 1, 3))  # (K, n_p, n_c)


def simulate(rng, n, sigma_sub, sigma_seed, K, n_p, n_c, n_cleaners=7, base=0.5, effects=None, n_rollouts=200):
    e = 0.0 if effects is None else np.asarray(effects)[None, :, None]
    pc = np.clip(base + e + rng.normal(0, sigma_sub, (n, n_cleaners, 1))
                 + rng.normal(0, sigma_seed, (n, n_cleaners, n_c)), 0, 1)
    pp = np.clip(base + rng.normal(0, sigma_sub, (n, K, 1)) + rng.normal(0, sigma_seed, (n, K, n_p)), 0, 1)
    return rng.binomial(n_rollouts, pc) / n_rollouts, rng.binomial(n_rollouts, pp) / n_rollouts


@dataclass(frozen=True)
class Cell:
    sigma_sub: float
    sigma_seed: float
    K: int
    n_p: int
    n_c: int
    n_grids: int
    fwer: float
    power15: float | None = None
    power15_analytic: float | None = None

    @property
    def mc_se(self) -> float:
        return math.sqrt(self.fwer * (1 - self.fwer) / self.n_grids)

    @property
    def upper95(self) -> float:
        """One-sided-style upper 95% MC bound (normal approx, p + 1.96 SE)."""
        return self.fwer + 1.959963984540054 * self.mc_se


def _fwer(c, p, alpha=0.05, chunk=50_000) -> float:
    hits = 0
    for i in range(0, c.shape[0], chunk):
        _, _, _, pv, _, _ = S.subset_test_arrays(c[i:i + chunk], p[i:i + chunk])
        hits += int(S.holm_reject(pv, alpha).any(-1).sum())
    return hits / c.shape[0]


def null_cell(sigma_sub, sigma_seed, K, n_p, n_c, n_grids, seed=0, chunk=50_000, **kw) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(0, n_grids, chunk):
        m = min(chunk, n_grids - i)
        c, p = simulate(rng, m, sigma_sub, sigma_seed, K, n_p, n_c, **kw)
        hits += round(_fwer(c, p) * m)
    return hits / n_grids


def power_cell(effect, sigma_sub, sigma_seed, K, n_p, n_c, n_grids, n_cleaners=7, seed=1, **kw) -> float:
    rng = np.random.default_rng(seed)
    eff = np.zeros(n_cleaners)
    eff[0] = effect
    c, p = simulate(rng, n_grids, sigma_sub, sigma_seed, K, n_p, n_c, n_cleaners=n_cleaners, effects=eff, **kw)
    _, _, _, pv, _, _ = S.subset_test_arrays(c, p)
    return float(S.holm_reject(pv)[:, 0].mean())


def grid(n_small: int = 40_000, n_large: int = 200_000, n_power: int = 20_000, seed: int = 0,
         power: bool = True) -> list[Cell]:
    """K < 8 cells get n_small grids; K >= 8 cells (two-tier bound) get n_large."""
    out = []
    for ss, sd, (K, n_p, n_c) in itertools.product(SIGMA_SUB, SIGMA_SEED, DESIGNS):
        n = n_large if K >= 8 else n_small
        f = null_cell(ss, sd, K, n_p, n_c, n, seed=seed)
        pw = pa = None
        if power:
            pw = power_cell(0.15, ss, sd, K, n_p, n_c, n_power, seed=seed + 1)
            sw = S.run_level_sigma(sd, 0.5, 200)
            pa = S.subset_power(0.15, ss, sw, S.Design(n_c=n_c, K=K, n_p=n_p))
        out.append(Cell(ss, sd, K, n_p, n_c, n, f, pw, pa))
    return out


def markdown(cells: list[Cell]) -> str:
    L = ["| sigma_sub | sigma_seed | K, n_p, n_c | grids | FWER | MC SE | upper 95% | bound | power 15pp (MC) "
         "| power 15pp (analytic) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        bound = 0.06 if c.K >= 8 else 0.07
        val = c.upper95 if c.K >= 8 else c.fwer
        ok = "ok" if val <= bound else "**FAIL**"
        L.append(f"| {c.sigma_sub:.2f} | {c.sigma_seed:.2f} | {c.K}, {c.n_p}, {c.n_c} | {c.n_grids} | {c.fwer:.4f} "
                 f"| {c.mc_se:.4f} | {c.upper95:.4f} | {'upper95' if c.K >= 8 else 'FWER'} <= {bound} {ok} "
                 f"| {'—' if c.power15 is None else f'{c.power15:.3f}'} "
                 f"| {'—' if c.power15_analytic is None else f'{c.power15_analytic:.3f}'} |")
    return "\n".join(L)
