"""Null-calibration and power grid for the D-016 subset-variance test (CPU only; PROGRESS.md B-002).

Simulation: each null cleaner picks ONE subset whose effect a_c ~ N(0, sigma_sub); each of K placebo subsets
has a_k ~ N(0, sigma_sub); run rate = clip(base + a + N(0, sigma_seed)), then Binomial(n_rollouts)/n_rollouts.
7 cleaners share one placebo per grid. Reports Holm family-wise FPR (MC SE) and power for a 15pp effect on one
cleaner, for the test as specified in D-016 ("spec") and two candidate df rules that are NOT adopted:
  C: when s_sub^2 is truncated at 0, use df = K-1 instead of the pooled within df
  F: always use df = K-1
Usage: python scripts/d016_null_grid.py [--n 40000]
"""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
from scipy import stats as sps

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


def pvalues(c, p):
    K = p.shape[-2]
    _, t, df, pv, s_sub2, _ = S.subset_test_arrays(c, p)
    trunc = (s_sub2 == 0)[:, None]
    return {"spec": pv,
            "C": 2 * sps.t.sf(np.abs(t), np.where(trunc, K - 1.0, df)),
            "F": 2 * sps.t.sf(np.abs(t), np.full_like(t, K - 1.0))}


def grid(n: int, seed: int = 0):
    rows = []
    eff = np.zeros(7)
    eff[0] = 0.15
    for ss, sd, (K, n_p, n_c) in itertools.product(SIGMA_SUB, SIGMA_SEED, DESIGNS):
        c, p = simulate(np.random.default_rng(seed), n, ss, sd, K, n_p, n_c)
        fwer = {k: float(S.holm_reject(v).any(-1).mean()) for k, v in pvalues(c, p).items()}
        c, p = simulate(np.random.default_rng(seed + 1), n // 2, ss, sd, K, n_p, n_c, effects=eff)
        power = {k: float(S.holm_reject(v)[:, 0].mean()) for k, v in pvalues(c, p).items()}
        rows.append({"sigma_sub": ss, "sigma_seed": sd, "K": K, "n_p": n_p, "n_c": n_c, "fwer": fwer,
                     "fwer_mc_se": {k: math.sqrt(max(v, 1e-12) * (1 - v) / n) for k, v in fwer.items()},
                     "power15": power})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000, help="simulated grids per cell")
    a = ap.parse_args()
    print("| s_sub | s_seed | K,n_p,n_c | FWER spec (MC SE) | FWER C | FWER F | power15 spec | C | F |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in grid(a.n):
        f, se, pw = r["fwer"], r["fwer_mc_se"], r["power15"]
        flag = " **>0.07**" if f["spec"] > 0.07 else ""
        print(f"| {r['sigma_sub']:.2f} | {r['sigma_seed']:.2f} | {r['K']},{r['n_p']},{r['n_c']} | "
              f"{f['spec']:.4f} ({se['spec']:.4f}){flag} | {f['C']:.4f} | {f['F']:.4f} | "
              f"{pw['spec']:.3f} | {pw['C']:.3f} | {pw['F']:.3f} |")


if __name__ == "__main__":
    main()
