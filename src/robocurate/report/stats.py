"""Statistics for RoboCurate-Bench (D-002).

Unit of analysis = training run. Each run contributes ONE number: its success rate over the eval
rollouts of its final checkpoint. Rollouts are never pooled across runs as independent samples.

Primary test: Welch t-test of a cleaner's run-level success rates vs ALL placebo runs (all random
subsets x seeds) at the same budget; Holm-Bonferroni across cleaners within a task (and budget).
Secondary: bootstrap CI on the difference of means, resampling runs (not rollouts).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------- Wilson


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default).

    Used only to describe ONE run's eval (rollouts within a run are i.i.d. given the checkpoint).
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return (lo, hi)


# ---------------------------------------------------------------------------- Welch


@dataclass(frozen=True)
class WelchResult:
    diff: float  # mean(a) - mean(b)
    t: float
    df: float
    p: float  # two-sided


def welch_arrays(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized two-sided Welch t-test along the last axis.

    a: (..., n1), b: (..., n2) broadcastable on leading axes. Returns (diff, t, df, p).
    If both groups have zero variance: p = 1 when the means are equal, else p = 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.shape[-1], b.shape[-1]
    if n1 < 2 or n2 < 2:
        raise ValueError("Welch t-test needs >= 2 runs per group")
    m1, m2 = a.mean(-1), b.mean(-1)
    v1, v2 = a.var(-1, ddof=1), b.var(-1, ddof=1)
    s1, s2 = v1 / n1, v2 / n2
    se2 = s1 + s2
    diff = m1 - m2
    with np.errstate(divide="ignore", invalid="ignore"):
        t = diff / np.sqrt(se2)
        df = se2**2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
        p = 2.0 * sps.t.sf(np.abs(t), df)
    degenerate = se2 == 0
    p = np.where(degenerate, np.where(diff == 0, 1.0, 0.0), p)
    t = np.where(degenerate, np.where(diff == 0, 0.0, np.where(diff > 0, np.inf, -np.inf)), t)
    df = np.where(degenerate, float(n1 + n2 - 2), df)
    return diff, t, df, p


def welch_t_test(a: Sequence[float], b: Sequence[float]) -> WelchResult:
    diff, t, df, p = welch_arrays(np.asarray(a), np.asarray(b))
    return WelchResult(float(diff), float(t), float(df), float(p))


# ---------------------------------------------------------------------------- Holm


def holm_adjust(pvals: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down), along the last axis. Vectorized."""
    p = np.asarray(pvals, dtype=float)
    m = p.shape[-1]
    order = np.argsort(p, axis=-1, kind="stable")
    ps = np.take_along_axis(p, order, axis=-1)
    mult = m - np.arange(m)
    adj_sorted = np.maximum.accumulate(np.minimum(ps * mult, 1.0), axis=-1)
    adj = np.empty_like(adj_sorted)
    np.put_along_axis(adj, order, adj_sorted, axis=-1)
    return adj


def holm_reject(pvals: Sequence[float] | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return holm_adjust(pvals) <= alpha


# ---------------------------------------------------------------------------- bootstrap


def bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b), resampling RUNS within each group."""
    rng = rng if rng is not None else np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 1 or b.size < 1:
        raise ValueError("need >= 1 run per group")
    ia = rng.integers(0, a.size, size=(n_boot, a.size))
    ib = rng.integers(0, b.size, size=(n_boot, b.size))
    d = a[ia].mean(1) - b[ib].mean(1)
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------- power / MDE


def run_level_sigma(sigma_seed: float, p: float, n_rollouts: int) -> float:
    """SD of a run's measured success rate: seed-to-seed spread plus binomial eval noise."""
    return math.sqrt(sigma_seed**2 + p * (1 - p) / n_rollouts)


def mde(
    sigma_run: float,
    n_cleaner_runs: int,
    n_placebo_runs: int,
    alpha: float = 0.05,
    power: float = 0.80,
    n_comparisons: int = 1,
) -> float:
    """Minimum detectable effect (absolute success-rate difference) for the Welch test.

    Assumes equal run-level SD in both groups. t-based approximation with Welch-Satterthwaite df:
        MDE = (t_{1 - a/2, df} + t_{power, df}) * sigma_run * sqrt(1/n1 + 1/n2)
    with a = alpha / n_comparisons (Bonferroni = Holm's first, strictest step; conservative).
    """
    if n_cleaner_runs < 2 or n_placebo_runs < 2:
        raise ValueError("need >= 2 runs per group")
    n1, n2 = n_cleaner_runs, n_placebo_runs
    s1, s2 = sigma_run**2 / n1, sigma_run**2 / n2
    df = (s1 + s2) ** 2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
    a = alpha / n_comparisons
    return float((sps.t.ppf(1 - a / 2, df) + sps.t.ppf(power, df)) * math.sqrt(s1 + s2))


def welch_power(
    effect: float,
    sigma_run: float,
    n_cleaner_runs: int,
    n_placebo_runs: int,
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> float:
    """Approximate power (noncentral t) of the two-sided Welch test for a given effect."""
    n1, n2 = n_cleaner_runs, n_placebo_runs
    s1, s2 = sigma_run**2 / n1, sigma_run**2 / n2
    df = (s1 + s2) ** 2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
    a = alpha / n_comparisons
    crit = sps.t.ppf(1 - a / 2, df)
    nc = effect / math.sqrt(s1 + s2)
    return float(sps.nct.sf(crit, df, nc) + sps.nct.cdf(-crit, df, nc))


# ---------------------------------------------------------------------------- primary comparison


@dataclass(frozen=True)
class CleanerComparison:
    name: str
    n_cleaner_runs: int
    n_placebo_runs: int
    cleaner_mean: float
    placebo_mean: float
    diff: float
    t: float
    df: float
    p_raw: float
    p_holm: float
    significant: bool
    boot_ci: tuple[float, float]


def compare_cleaners_to_placebo(
    cleaner_runs: Mapping[str, Sequence[float]],
    placebo_runs: Sequence[float],
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> list[CleanerComparison]:
    """Primary analysis for one task x budget.

    cleaner_runs: name -> run-level success rates (one float per training run).
    placebo_runs: run-level success rates of ALL random-subset runs (all subsets x seeds).
    Holm correction across the cleaners in `cleaner_runs`.
    """
    names = list(cleaner_runs)
    if not names:
        return []
    pl = np.asarray(placebo_runs, dtype=float)
    welch = [welch_t_test(cleaner_runs[n], pl) for n in names]
    p_adj = holm_adjust([w.p for w in welch])
    rng = np.random.default_rng(seed)
    out = []
    for n, w, pa in zip(names, welch, p_adj, strict=True):
        cr = np.asarray(cleaner_runs[n], dtype=float)
        out.append(
            CleanerComparison(
                name=n,
                n_cleaner_runs=int(cr.size),
                n_placebo_runs=int(pl.size),
                cleaner_mean=float(cr.mean()),
                placebo_mean=float(pl.mean()),
                diff=w.diff,
                t=w.t,
                df=w.df,
                p_raw=w.p,
                p_holm=float(pa),
                significant=bool(pa <= alpha),
                boot_ci=bootstrap_diff_ci(cr, pl, n_boot=n_boot, rng=rng),
            )
        )
    return out
