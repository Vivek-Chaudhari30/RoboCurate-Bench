"""Statistics for RoboCurate-Bench (D-002 as amended by D-016 and D-019).

Unit of analysis = training run. Each run contributes ONE number: its success rate over the eval
rollouts of its final checkpoint. Rollouts are never pooled across runs as independent samples.

Primary test (D-016): a deterministic cleaner yields ONE subset, so its mean carries irreducible
subset-to-subset variance. Statistic t = (mean_c - mean_p) / SE with

    SE^2 = s_sub^2 (1 + 1/K) + s_seed^2 (1/n_c + 1/(K n_p))

mean_p = mean of the K random-subset means. s_sub^2 comes from a one-way random-effects ANOVA on the
placebo runs (truncated at 0); s_seed^2 is the within-condition variance pooled over the replicates of
ALL conditions (each cleaner, oracle, full, each random subset). df = K - 1 always (D-019: the between-subset
term dominates the SE and is estimated from K subset means); Holm across cleaners.
Unbalanced placebo designs use the general forms (n0 for the ANOVA, sum 1/n_i / K^2 for the mean).

Secondary diagnostics (labeled as such in reports): Welch t-test of a cleaner's runs vs all placebo runs
(D-002, withdrawn as primary by D-016: FPR 0.42 at sigma_sub = 0.04), and a run-level bootstrap CI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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


def _two_sided_nct_power(crit: float, df: float, nc: float) -> float:
    """P(|T| > crit), T ~ noncentral t(df, nc). Each tail as an upper-tail sf (P(T < -c | nc) =
    P(T > c | -nc)); scipy's nct.cdf returns NaN deep in the tail. A NaN tail is ~0 and is dropped."""
    tails = [sps.nct.sf(crit, df, nc), sps.nct.sf(crit, df, -nc)]
    return float(sum(0.0 if math.isnan(x) else x for x in tails))


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
    if sigma_run == 0:
        return 0.0  # zero run-to-run spread: any nonzero difference is detectable
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
    return _two_sided_nct_power(crit, df, nc)


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
    """Welch of each cleaner's runs vs ALL placebo runs, Holm across cleaners (D-002).

    Ignores subset-to-subset variance (anti-conservative when sigma_sub > 0). Still wired into `robocurate
    report` because the D-016 replacement failed its null calibration at K=4 (PROGRESS.md B-002); D-016
    makes this a secondary diagnostic once B-002 is resolved.
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


# ---------------------------------------------------------------------------- D-016 primary test


@dataclass(frozen=True)
class VarianceComponents:
    """One-way random-effects ANOVA on placebo runs + within-condition variance pooled over all conditions."""

    s_sub2: float  # truncated at 0
    s_sub2_raw: float  # (MSB - s_seed2) / n0, may be negative
    s_seed2: float  # pooled within-condition variance (seed + eval binomial noise)
    msb: float  # between-subset mean square of the placebo runs
    df_b: int  # K - 1
    df_w: int  # pooled within df
    n0: float  # effective runs per subset (= n_p when balanced)
    K: int
    inv_n_sum: float  # sum_i 1/n_i over placebo subsets (balanced: K / n_p)

    @property
    def truncated(self) -> bool:
        return self.s_sub2_raw <= 0.0

    @property
    def s_sub(self) -> float:
        return math.sqrt(self.s_sub2)

    @property
    def s_seed(self) -> float:
        return math.sqrt(self.s_seed2)


def _within_ss(groups: Sequence[Sequence[float]]) -> tuple[float, int]:
    ss, df = 0.0, 0
    for g in groups:
        g = np.asarray(g, dtype=float)
        if g.size >= 2:
            ss += float(((g - g.mean()) ** 2).sum())
            df += int(g.size - 1)
    return ss, df


def variance_components(
    placebo_by_subset: Sequence[Sequence[float]],
    other_groups: Sequence[Sequence[float]] = (),
) -> VarianceComponents:
    """placebo_by_subset: run rates per random subset (K lists). other_groups: run rates of every other
    condition (each cleaner, oracle, full); they contribute only within-condition replicates to s_seed."""
    subs = [np.asarray(g, dtype=float) for g in placebo_by_subset if len(g) > 0]
    K = len(subs)
    if K < 2:
        raise ValueError("D-016 test needs >= 2 random subsets")
    n_i = np.array([g.size for g in subs], dtype=float)
    N = n_i.sum()
    means = np.array([g.mean() for g in subs])
    grand = float((n_i * means).sum() / N)
    msb = float((n_i * (means - grand) ** 2).sum() / (K - 1))
    n0 = float((N - (n_i**2).sum() / N) / (K - 1))
    ss, df_w = _within_ss(list(subs) + [list(g) for g in other_groups])
    if df_w < 1:
        raise ValueError("D-016 test needs >= 1 within-condition replicate (some condition with >= 2 runs)")
    s_seed2 = ss / df_w
    raw = (msb - s_seed2) / n0
    return VarianceComponents(max(0.0, raw), raw, s_seed2, msb, K - 1, df_w, n0, K, float((1.0 / n_i).sum()))


@dataclass(frozen=True)
class SubsetTestResult:
    diff: float
    se: float
    t: float
    df: float
    p: float


def subset_test(
    cleaner_runs: Sequence[float],
    vc: VarianceComponents,
    placebo_mean: float,
    *,
    own_subset_var: bool = True,
) -> SubsetTestResult:
    """D-016 statistic for one condition vs the placebo.

    placebo_mean = mean of the K subset means. own_subset_var=False drops the condition's own s_sub^2
    term (used only for the full-data reference, which is not a draw from the subset distribution).
    """
    c = np.asarray(cleaner_runs, dtype=float)
    n_c = c.size
    if n_c < 1:
        raise ValueError("need >= 1 run for the condition")
    a = (1.0 if own_subset_var else 0.0) + 1.0 / vc.K  # coefficient of s_sub^2
    h = 1.0 / n_c + vc.inv_n_sum / vc.K**2  # coefficient of s_seed^2
    diff = float(c.mean() - placebo_mean)
    se2 = vc.s_sub2 * a + vc.s_seed2 * h
    df = float(vc.df_b)  # D-019: always K - 1
    if se2 <= 0:
        p = 1.0 if diff == 0 else 0.0
        t = 0.0 if diff == 0 else math.copysign(math.inf, diff)
        return SubsetTestResult(diff, 0.0, t, df, p)
    se = math.sqrt(se2)
    t = diff / se
    return SubsetTestResult(diff, se, t, df, float(2.0 * sps.t.sf(abs(t), df)))


def subset_test_arrays(cleaners: np.ndarray, placebo: np.ndarray, extra: np.ndarray | None = None):
    """Vectorized balanced D-016 test (for simulation). cleaners: (..., C, n_c); placebo: (..., K, n_p);
    extra: optional (..., G, n_g) other conditions contributing within replicates only.

    Returns (diff, t, df, p, s_sub2, s_seed2) with diff/t/df/p of shape (..., C).
    """
    cleaners = np.asarray(cleaners, dtype=float)
    placebo = np.asarray(placebo, dtype=float)
    K, n_p = placebo.shape[-2:]
    n_c = cleaners.shape[-1]
    if K < 2:
        raise ValueError("D-016 test needs >= 2 random subsets")
    sub_means = placebo.mean(-1)
    pm = sub_means.mean(-1)
    msb = n_p * sub_means.var(-1, ddof=1)
    ss = ((placebo - sub_means[..., None]) ** 2).sum((-1, -2))
    ss = ss + ((cleaners - cleaners.mean(-1, keepdims=True)) ** 2).sum((-1, -2))
    df_w = K * (n_p - 1) + cleaners.shape[-2] * (n_c - 1)
    if extra is not None:
        extra = np.asarray(extra, dtype=float)
        ss = ss + ((extra - extra.mean(-1, keepdims=True)) ** 2).sum((-1, -2))
        df_w += extra.shape[-2] * (extra.shape[-1] - 1)
    if df_w < 1:
        raise ValueError("D-016 test needs >= 1 within-condition replicate")
    s_seed2 = ss / df_w
    s_sub2 = np.maximum((msb - s_seed2) / n_p, 0.0)
    se2 = s_sub2 * (1.0 + 1.0 / K) + s_seed2 * (1.0 / n_c + 1.0 / (K * n_p))
    df = np.full(se2.shape, float(K - 1))  # D-019
    with np.errstate(divide="ignore", invalid="ignore"):
        diff = cleaners.mean(-1) - pm[..., None]
        t = diff / np.sqrt(se2)[..., None]
        p = 2.0 * sps.t.sf(np.abs(t), K - 1)
    deg = (se2 <= 0)[..., None]
    p = np.where(deg, np.where(diff == 0, 1.0, 0.0), p)
    return diff, t, np.broadcast_to(df[..., None], diff.shape), p, s_sub2, s_seed2


@dataclass(frozen=True)
class Design:
    """Grid design for the D-016 power analysis. R = mixture replicates (each with its own cleaner subset
    and K placebo subsets; the mixture main effect cancels because cleaner and placebo share a mixture)."""

    n_c: int  # seeds per cleaner (per mixture replicate)
    K: int  # random subsets (per mixture replicate)
    n_p: int  # seeds per random subset
    R: int = 1
    n_cleaners: int = 7
    n_full: int = 0  # full-data seeds (contribute within df only)

    @property
    def df_b(self) -> int:
        return self.R * (self.K - 1)

    @property
    def df_w(self) -> int:
        return self.R * (self.K * (self.n_p - 1) + self.n_cleaners * (self.n_c - 1) + max(0, self.n_full - 1))

    def runs(self) -> int:
        return self.R * (self.n_cleaners * self.n_c + self.K * self.n_p + self.n_full)


def design_se(sigma_sub: float, sigma_seed: float, d: Design) -> tuple[float, float]:
    """True SE of (mean_c - mean_p) under the design, and the test's df = R(K - 1) (D-019)."""
    a = 1.0 + 1.0 / d.K
    h = 1.0 / d.n_c + 1.0 / (d.K * d.n_p)
    se2 = (sigma_sub**2 * a + sigma_seed**2 * h) / d.R
    return math.sqrt(se2), float(d.df_b)


def subset_power(effect: float, sigma_sub: float, sigma_seed: float, d: Design, alpha: float = 0.05,
                 n_comparisons: int | None = None) -> float:
    """Approximate power of the D-016/D-019 test (noncentral t at df = R(K-1); Bonferroni alpha/m =
    Holm's strictest step, conservative). sigma_seed includes eval binomial noise (use run_level_sigma)."""
    m = d.n_cleaners if n_comparisons is None else n_comparisons
    se, df = design_se(sigma_sub, sigma_seed, d)
    if se == 0:
        return 1.0 if effect != 0 else alpha / m
    crit = sps.t.ppf(1 - alpha / m / 2, df)
    nc = effect / se
    return _two_sided_nct_power(crit, df, nc)


def subset_mde(sigma_sub: float, sigma_seed: float, d: Design, alpha: float = 0.05, power: float = 0.80,
               n_comparisons: int | None = None) -> float:
    """Smallest effect with the requested power under the D-016 test."""
    from scipy.optimize import brentq

    se, _ = design_se(sigma_sub, sigma_seed, d)
    if se == 0:
        return 0.0
    f = lambda e: subset_power(e, sigma_sub, sigma_seed, d, alpha, n_comparisons) - power  # noqa: E731
    hi = se
    while f(hi) < 0:
        hi *= 2
    return float(brentq(f, 0.0, hi, xtol=1e-6))


@dataclass(frozen=True)
class SubsetComparison:
    name: str
    n_cleaner_runs: int
    n_placebo_runs: int
    n_subsets: int
    cleaner_mean: float
    placebo_mean: float
    diff: float
    se: float
    t: float
    df: float
    p_raw: float
    p_holm: float
    significant: bool
    test: str = "d016"
    boot_ci: tuple[float, float] | None = None
    welch_secondary: dict = field(default_factory=dict)


def compare_cleaners_d016(
    cleaner_runs: Mapping[str, Sequence[float]],
    placebo_by_subset: Sequence[Sequence[float]],
    other_groups: Sequence[Sequence[float]] = (),
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[list[SubsetComparison], VarianceComponents]:
    """PRIMARY analysis for one task x budget (D-016), Holm across the cleaners in `cleaner_runs`.

    other_groups: run rates of reference conditions (oracle, full); within replicates only.
    The Welch test and the run-level bootstrap CI are attached as labeled secondary diagnostics.
    """
    vc = variance_components(placebo_by_subset, list(cleaner_runs.values()) + list(other_groups))
    pl_flat = np.concatenate([np.asarray(g, dtype=float) for g in placebo_by_subset])
    pm = float(np.mean([np.mean(g) for g in placebo_by_subset if len(g)]))
    names = list(cleaner_runs)
    tests = [subset_test(cleaner_runs[n], vc, pm) for n in names]
    p_adj = holm_adjust([r.p for r in tests]) if names else np.array([])
    rng = np.random.default_rng(seed)
    welch = {w.name: w for w in compare_cleaners_to_placebo(cleaner_runs, pl_flat, alpha, n_boot, seed)} \
        if names and all(len(cleaner_runs[n]) >= 2 for n in names) and pl_flat.size >= 2 else {}
    out = []
    for n, r, pa in zip(names, tests, p_adj, strict=True):
        cr = np.asarray(cleaner_runs[n], dtype=float)
        w = welch.get(n)
        out.append(SubsetComparison(
            name=n, n_cleaner_runs=int(cr.size), n_placebo_runs=int(pl_flat.size), n_subsets=vc.K,
            cleaner_mean=float(cr.mean()), placebo_mean=pm, diff=r.diff, se=r.se, t=r.t, df=r.df, p_raw=r.p,
            p_holm=float(pa), significant=bool(pa <= alpha),
            boot_ci=bootstrap_diff_ci(cr, pl_flat, n_boot=n_boot, rng=rng),
            welch_secondary={} if w is None else {"t": w.t, "df": w.df, "p_raw": w.p_raw, "p_holm": w.p_holm,
                                                  "significant": w.significant},
        ))
    return out, vc
