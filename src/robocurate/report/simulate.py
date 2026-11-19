"""Null-calibration and power simulation for the primary test (D-026). CPU only.

Model per grid: each of C null entrants picks ONE subset with effect a_c ~ N(0, sigma_sub); each of K placebo
subsets has a_k ~ N(0, sigma_sub); run rate = clip(base + effect + a + N(0, sigma_seed), 0, 1), measured as
Binomial(n_rollouts)/n_rollouts. The C entrants share one placebo per grid (as in a real report).

Main table (result-eligible, D-026): balanced n_c = n_p = n in {3, 5}, K in {8, 12}, tested with the D-026 exact
balanced test. Informational table: the old unbalanced designs (B-003/B-004) under the old D-016/D-019
variance-components test; not result-eligible.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from robocurate.report import stats as S

SIGMA_SUB = (0.0, 0.02, 0.04, 0.06)
SIGMA_SEED = (0.03, 0.06)
BALANCED = ((8, 3), (8, 5), (12, 3), (12, 5))  # (K, n): result-eligible under D-026
UNBALANCED = ((8, 1, 3), (8, 2, 3), (12, 2, 3), (8, 2, 5), (12, 2, 5))  # (K, n_p, n_c): informational only
ELIGIBLE_BOUND = 0.06  # upper 95% MC bound on the Holm FWER, every result-eligible cell
EFFECT = 0.15  # power column (informational)


def result_eligible(K: int, n_p: int, n_c: int) -> bool:
    from robocurate.config import RESULT_MIN_SUBSETS, SEED_FLOOR

    return K >= RESULT_MIN_SUBSETS and n_p == n_c and n_c >= SEED_FLOOR


def simulate(rng, n, sigma_sub, sigma_seed, K, n_p, n_c, n_cleaners=7, base=0.5, effects=None, n_rollouts=200):
    e = 0.0 if effects is None else np.asarray(effects)[None, :, None]
    pc = np.clip(base + e + rng.normal(0, sigma_sub, (n, n_cleaners, 1))
                 + rng.normal(0, sigma_seed, (n, n_cleaners, n_c)), 0, 1)
    pp = np.clip(base + rng.normal(0, sigma_sub, (n, K, 1)) + rng.normal(0, sigma_seed, (n, K, n_p)), 0, 1)
    return rng.binomial(n_rollouts, pc) / n_rollouts, rng.binomial(n_rollouts, pp) / n_rollouts


def _pvals(c, p, test: str):
    if test == "d026":
        return S.balanced_test_arrays(c[:, None], p[:, None])[3]
    return S.subset_test_arrays(c, p)[3]  # old D-016/D-019 variance-components test


@dataclass(frozen=True)
class Cell:
    sigma_sub: float
    sigma_seed: float
    K: int
    n_p: int
    n_c: int
    n_grids: int
    fwer: float
    test: str = "d026"
    power15: float | None = None
    power15_analytic: float | None = None

    @property
    def mc_se(self) -> float:
        return math.sqrt(self.fwer * (1 - self.fwer) / self.n_grids)

    @property
    def upper95(self) -> float:
        """Upper 95% MC bound (normal approx, p + 1.96 SE)."""
        return self.fwer + 1.959963984540054 * self.mc_se

    @property
    def eligible(self) -> bool:
        return self.test == "d026" and result_eligible(self.K, self.n_p, self.n_c)

    @property
    def verdict(self) -> str | None:
        """'ok' / 'FAIL' for result-eligible cells; None (no pass/fail) for informational cells."""
        if not self.eligible:
            return None
        return "ok" if self.upper95 <= ELIGIBLE_BOUND else "FAIL"


def null_cell(sigma_sub, sigma_seed, K, n_p, n_c, n_grids, seed=0, chunk=50_000, test="d026", **kw) -> float:
    rng = np.random.default_rng(seed)
    hits = 0
    for i in range(0, n_grids, chunk):
        m = min(chunk, n_grids - i)
        c, p = simulate(rng, m, sigma_sub, sigma_seed, K, n_p, n_c, **kw)
        hits += int(S.holm_reject(_pvals(c, p, test)).any(-1).sum())
    return hits / n_grids


def power_cell(effect, sigma_sub, sigma_seed, K, n_p, n_c, n_grids, n_cleaners=7, seed=1, test="d026", **kw) -> float:
    rng = np.random.default_rng(seed)
    eff = np.zeros(n_cleaners)
    eff[0] = effect
    c, p = simulate(rng, n_grids, sigma_sub, sigma_seed, K, n_p, n_c, n_cleaners=n_cleaners, effects=eff, **kw)
    return float(S.holm_reject(_pvals(c, p, test))[:, 0].mean())


def grid(n_grids: int = 200_000, n_power: int = 20_000, seed: int = 0, power: bool = True,
         informational: bool = False) -> list[Cell]:
    """Main D-026 table (balanced cells); informational=True gives the old unbalanced cells (old test)."""
    out = []
    if informational:
        designs = [(K, n_p, n_c, "d016") for K, n_p, n_c in UNBALANCED]
    else:
        designs = [(K, n, n, "d026") for K, n in BALANCED]
    for ss, sd, (K, n_p, n_c, test) in itertools.product(SIGMA_SUB, SIGMA_SEED, designs):
        f = null_cell(ss, sd, K, n_p, n_c, n_grids, seed=seed, test=test)
        pw = pa = None
        if power:
            pw = power_cell(EFFECT, ss, sd, K, n_p, n_c, n_power, seed=seed + 1, test=test)
            sw = S.run_level_sigma(sd, 0.5, 200)
            pa = (S.balanced_power(EFFECT, ss, sw, n_c, K) if test == "d026"
                  else S.subset_power(EFFECT, ss, sw, S.Design(n_c=n_c, K=K, n_p=n_p)))
        out.append(Cell(ss, sd, K, n_p, n_c, n_grids, f, test, pw, pa))
    return out


def markdown(cells: list[Cell]) -> str:
    L = ["| sigma_sub | sigma_seed | K, n_p, n_c | test | eligibility | grids | FWER | MC SE | upper 95% | criterion "
         "| power 15pp (MC) | power 15pp (analytic) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if c.eligible:
            elig, crit = "result-eligible", f"upper95 <= {ELIGIBLE_BOUND} {c.verdict}".replace("FAIL", "**FAIL**")
        else:
            elig, crit = "not result-eligible (informational)", "— (no pass/fail)"
        L.append(f"| {c.sigma_sub:.2f} | {c.sigma_seed:.2f} | {c.K}, {c.n_p}, {c.n_c} | {c.test} | {elig} "
                 f"| {c.n_grids} | {c.fwer:.4f} | {c.mc_se:.4f} | {c.upper95:.4f} | {crit} "
                 f"| {'—' if c.power15 is None else f'{c.power15:.3f}'} "
                 f"| {'—' if c.power15_analytic is None else f'{c.power15_analytic:.3f}'} |")
    return "\n".join(L)
