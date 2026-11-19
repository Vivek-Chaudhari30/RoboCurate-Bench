import math
import time

import numpy as np
import pytest
from scipy import stats as sps

from robocurate.report import stats as S

# ------------------------------------------------------------------ Wilson (known values)


def test_wilson_known_values():
    # Reference: Newcombe (1998) / statsmodels proportion_confint(method="wilson")
    lo, hi = S.wilson_ci(81, 263)
    assert lo == pytest.approx(0.2553, abs=1e-4)
    assert hi == pytest.approx(0.3662, abs=1e-4)
    lo, hi = S.wilson_ci(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-4)


def test_wilson_edges():
    assert S.wilson_ci(0, 200)[0] == 0.0
    assert S.wilson_ci(200, 200)[1] == 1.0
    assert S.wilson_ci(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)
    with pytest.raises(ValueError):
        S.wilson_ci(3, 0)
    with pytest.raises(ValueError):
        S.wilson_ci(11, 10)


# ------------------------------------------------------------------ Welch


def test_welch_matches_scipy():
    a = [0.62, 0.55, 0.71, 0.58]
    b = [0.40, 0.52, 0.47, 0.44, 0.39, 0.50, 0.48, 0.45, 0.42]
    r = S.welch_t_test(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert r.t == pytest.approx(ref.statistic)
    assert r.p == pytest.approx(ref.pvalue)
    assert r.diff == pytest.approx(np.mean(a) - np.mean(b))


def test_welch_known_value():
    # Hand computation: a=[1,2,3] (var 1), b=[4,5,6,7] (var 5/3); diff=-3.5,
    # se=sqrt(1/3 + 5/12)=0.866025 -> t=-4.041452; df=0.75^2/((1/3)^2/2 + (5/12)^2/3)=4.959184
    r = S.welch_t_test([1, 2, 3], [4, 5, 6, 7])
    assert r.t == pytest.approx(-4.041452, abs=1e-5)
    assert r.df == pytest.approx(4.959184, abs=1e-5)
    assert r.p == pytest.approx(2 * sps.t.sf(4.041452, 4.959184), abs=1e-6)


def test_welch_degenerate_and_small():
    assert S.welch_t_test([0.5, 0.5], [0.5, 0.5, 0.5]).p == 1.0
    assert S.welch_t_test([0.6, 0.6], [0.5, 0.5, 0.5]).p == 0.0
    with pytest.raises(ValueError):
        S.welch_t_test([0.5], [0.4, 0.3])


def test_welch_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 3))
    b = rng.normal(size=(5, 9))
    _, t, _, p = S.welch_arrays(a, b)
    for i in range(5):
        ref = sps.ttest_ind(a[i], b[i], equal_var=False)
        assert t[i] == pytest.approx(ref.statistic)
        assert p[i] == pytest.approx(ref.pvalue)


# ------------------------------------------------------------------ Holm


def test_holm_known_values():
    # Holm (1979) worked example style: m=4
    p = [0.01, 0.04, 0.03, 0.005]
    adj = S.holm_adjust(p)
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone: .02,.03,.06,.06
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert S.holm_reject(p, 0.05).tolist() == [True, False, False, True]


def test_holm_caps_at_one_and_vectorizes():
    adj = S.holm_adjust(np.array([[0.5, 0.9], [0.01, 0.02]]))
    assert adj[0].tolist() == [1.0, 1.0]
    assert adj[1].tolist() == pytest.approx([0.02, 0.02])


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_resamples_runs_and_covers_diff():
    a = [0.70, 0.74, 0.66]
    b = [0.50, 0.52, 0.47, 0.49, 0.53, 0.51]
    lo, hi = S.bootstrap_diff_ci(a, b, n_boot=20000, rng=np.random.default_rng(0))
    d = np.mean(a) - np.mean(b)
    assert lo < d < hi
    # Run-level resampling: CI half-width must reflect run spread, not 200-rollout binomial noise.
    assert hi - lo > 0.04


def test_bootstrap_constant_groups_zero_width():
    lo, hi = S.bootstrap_diff_ci([0.6, 0.6], [0.4, 0.4, 0.4], n_boot=500)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


# ------------------------------------------------------------------ MDE / power


def test_mde_matches_power():
    m = S.mde(0.06, 3, 9, alpha=0.05)
    assert S.welch_power(m, 0.06, 3, 9) == pytest.approx(0.80, abs=0.03)


def test_mde_known_value_and_monotone():
    # equal n=5 vs 15, sigma=.06: df = 6.4... ; (t.975 + t.80) * .06 * sqrt(1/5+1/15)
    n1, n2, s = 5, 15, 0.06
    s1, s2 = s * s / n1, s * s / n2
    df = (s1 + s2) ** 2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
    ref = (sps.t.ppf(0.975, df) + sps.t.ppf(0.8, df)) * math.sqrt(s1 + s2)
    assert S.mde(s, n1, n2) == pytest.approx(ref)
    assert S.mde(s, 10, 30) < S.mde(s, 5, 15) < S.mde(s, 3, 9)
    assert S.mde(s, 5, 15, n_comparisons=7) > S.mde(s, 5, 15)


def test_run_level_sigma():
    assert S.run_level_sigma(0.06, 0.5, 200) == pytest.approx(math.sqrt(0.0036 + 0.00125))


# ------------------------------------------------------------------ primary comparison


def test_compare_cleaners_to_placebo_shapes_and_holm():
    rng = np.random.default_rng(3)
    placebo = rng.normal(0.5, 0.05, 9)
    cl = {"good": rng.normal(0.8, 0.05, 3), "null": rng.normal(0.5, 0.05, 3)}
    out = {c.name: c for c in S.compare_cleaners_to_placebo(cl, placebo, n_boot=2000)}
    assert out["good"].significant and not out["null"].significant
    assert out["good"].n_placebo_runs == 9 and out["good"].n_cleaner_runs == 3
    assert out["good"].p_holm >= out["good"].p_raw
    lo, hi = out["good"].boot_ci
    assert lo < out["good"].diff < hi


# ------------------------------------------------------------------ simulation calibration


def simulate_run_rates(rng, base, effect, sigma_seed, n_rollouts, shape):
    """Run-level rates: per-run true rate with seed noise, then Bernoulli draws over rollouts."""
    true = np.clip(base + effect + rng.normal(0.0, sigma_seed, size=shape), 0.0, 1.0)
    return rng.binomial(n_rollouts, true) / n_rollouts


def null_grid_fwer(n_sims=1000, n_cleaners=7, n_seeds=3, n_subsets=3, sigma_seed=0.06, n_rollouts=200,
                   base=0.5, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    cleaners = simulate_run_rates(rng, base, 0.0, sigma_seed, n_rollouts, (n_sims, n_cleaners, n_seeds))
    placebo = simulate_run_rates(rng, base, 0.0, sigma_seed, n_rollouts, (n_sims, 1, n_subsets * n_seeds))
    _, _, _, p = S.welch_arrays(cleaners, placebo)
    rej = S.holm_reject(p, alpha)
    return rej.any(axis=-1).mean()


WELCH_SEEDS = 6  # D-015's historical floor for Welch (secondary diagnostic since D-016; floor retired by D-026)


def test_null_fwer_calibration():
    """Welch (secondary diagnostic) null calibration at D-015's historical 6 seeds.

    20000 grids instead of 1000: at a true FWER of ~0.06, 1000 grids have MC sd ~0.0075, too coarse to
    test a 0.07 bound reliably. More grids make the test stricter, not weaker.
    """
    t0 = time.time()
    fwer = null_grid_fwer(n_sims=20000, n_seeds=WELCH_SEEDS)
    assert fwer <= 0.07, f"Holm family-wise false-positive rate {fwer:.4f} > 0.07"
    assert time.time() - t0 < 60


def test_three_seeds_is_not_enough():
    """Regression guard for D-015: with 3 seeds Welch + Holm exceeds the 0.07 null FWER bound."""
    assert null_grid_fwer(n_sims=20000, n_seeds=3) > 0.07


@pytest.mark.slow
@pytest.mark.parametrize("base", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", [1, 2])
def test_null_fwer_calibration_robust(base, seed):
    assert null_grid_fwer(n_sims=20000, n_seeds=WELCH_SEEDS, base=base, seed=seed) <= 0.07


def power_detect_rate(effect=0.20, sigma_seed=0.05, n_seeds=5, n_subsets=3, n_cleaners=7, n_sims=1000,
                      n_rollouts=200, base=0.5, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    eff = np.zeros(n_cleaners)
    eff[0] = effect  # one real effect in a family of 7 cleaners (others null)
    cleaners = simulate_run_rates(rng, base, eff[None, :, None], sigma_seed, n_rollouts,
                                  (n_sims, n_cleaners, n_seeds))
    placebo = simulate_run_rates(rng, base, 0.0, sigma_seed, n_rollouts, (n_sims, 1, n_subsets * n_seeds))
    _, _, _, p = S.welch_arrays(cleaners, placebo)
    return S.holm_reject(p, alpha)[:, 0].mean()


def test_power_20pp_5_seeds():
    t0 = time.time()
    rate = power_detect_rate()
    assert rate >= 0.70, f"20pp effect detected only {rate:.3f} of the time"
    assert time.time() - t0 < 60


def test_power_20pp_at_seed_floor():
    assert power_detect_rate(n_seeds=WELCH_SEEDS) >= 0.70


def test_mde_zero_sigma_regression():
    """Regression: mde() raised ZeroDivisionError when all placebo runs had identical success rates."""
    assert S.mde(0.0, 3, 3) == 0.0


# ------------------------------------------------------------------ D-016 subset-variance test


def _balanced(rng, K=4, n_p=2, n_c=3, C=3):
    return rng.uniform(0.3, 0.7, (C, n_c)), rng.uniform(0.3, 0.7, (K, n_p))


def test_variance_components_known_values():
    # placebo subsets [.5,.7], [.3,.5]: means .6,.4; grand .5; MSB = 2*(.01+.01)/1 = .04
    # within SS: .02+.02 (placebo) + cleaner [.6,.6,.9] -> .06 ; df = 1+1+2 = 4 -> s_seed2 = .1/4 = .025
    vc = S.variance_components([[0.5, 0.7], [0.3, 0.5]], [[0.6, 0.6, 0.9]])
    assert vc.msb == pytest.approx(0.04)
    assert vc.s_seed2 == pytest.approx(0.025)
    assert vc.df_w == 4 and vc.df_b == 1 and vc.n0 == pytest.approx(2.0)
    assert vc.s_sub2_raw == pytest.approx((0.04 - 0.025) / 2)
    assert not vc.truncated


def test_variance_components_truncates_at_zero():
    vc = S.variance_components([[0.4, 0.6], [0.4, 0.6], [0.5, 0.5]])
    assert vc.s_sub2_raw < 0 and vc.s_sub2 == 0.0 and vc.truncated


def test_subset_test_se_formula_balanced():
    """SE^2 = s_sub^2 (1 + 1/K) + s_seed^2 (1/n_c + 1/(K n_p)) (D-016)."""
    rng = np.random.default_rng(4)
    for _ in range(20):
        cl, pl = _balanced(rng)
        out, vc = S.compare_cleaners_d016({f"c{i}": cl[i] for i in range(len(cl))}, list(pl), n_boot=10)
        K, n_p = pl.shape
        se2 = vc.s_sub2 * (1 + 1 / K) + vc.s_seed2 * (1 / cl.shape[1] + 1 / (K * n_p))
        for o in out:
            assert o.se**2 == pytest.approx(se2)
            assert o.diff == pytest.approx(o.cleaner_mean - pl.mean())


def test_subset_test_vectorized_matches_scalar():
    rng = np.random.default_rng(7)
    cl, pl = rng.uniform(0.3, 0.7, (50, 5, 3)), rng.uniform(0.3, 0.7, (50, 4, 2))
    _, t, df, p, _, _ = S.subset_test_arrays(cl, pl)
    for i in range(50):
        out, _ = S.compare_cleaners_d016({f"c{j}": cl[i, j] for j in range(5)}, list(pl[i]), n_boot=10)
        for j, o in enumerate(out):
            assert o.p_raw == pytest.approx(p[i, j], abs=1e-12)
            assert o.df == df[i, j] == 3  # D-019: df = K - 1


def test_subset_test_unbalanced_and_full_reference():
    vc = S.variance_components([[0.5, 0.7, 0.6], [0.3, 0.5], [0.45]], [[0.6, 0.62]])
    assert vc.inv_n_sum == pytest.approx(1 / 3 + 1 / 2 + 1)
    with_own = S.subset_test([0.6, 0.62], vc, 0.5)
    without = S.subset_test([0.6, 0.62], vc, 0.5, own_subset_var=False)
    assert without.se < with_own.se


def test_subset_test_needs_two_subsets_and_replicates():
    with pytest.raises(ValueError):
        S.variance_components([[0.5, 0.6]])
    with pytest.raises(ValueError):
        S.variance_components([[0.5], [0.6]])


def test_subset_power_and_mde_consistent():
    d = S.Design(n_c=3, K=8, n_p=2)
    m = S.subset_mde(0.03, 0.05, d)
    assert S.subset_power(m, 0.03, 0.05, d) == pytest.approx(0.80, abs=1e-4)
    # more random subsets shrink the MDE; more cleaner seeds cannot remove the sigma_sub term
    assert S.subset_mde(0.03, 0.05, S.Design(3, 12, 2)) < m
    big_nc = S.Design(n_c=1000, K=8, n_p=2)
    assert S.design_se(0.03, 0.05, big_nc)[0] > 0.03
    assert S.subset_mde(0.03, 0.05, S.Design(3, 8, 2, R=3)) < m


def test_design_runs():
    assert S.Design(n_c=3, K=8, n_p=2, R=2, n_cleaners=7, n_full=3).runs() == 2 * (21 + 16 + 3)


@pytest.mark.slow
def test_primary_test_null_calibration():
    """Hard null calibration of the primary test (D-026), 7 null entrants, 200 rollouts, 200 000 grids per cell.

    Every result-eligible cell (balanced n in {3, 5}, K in {8, 12}, sigma_sub x sigma_seed grid): upper 95% Monte
    Carlo bound on the Holm FWER <= 0.06. The old unbalanced designs (old test) are printed as an informational
    table with no pass/fail. Tables with MC SEs are printed (pytest -s).
    """
    from robocurate.report.simulate import grid, markdown

    cells = grid(power=False)
    print("\n" + markdown(cells))
    info = grid(power=False, informational=True)
    print("\nInformational (not result-eligible):\n" + markdown(info))
    assert len(cells) == 32 and all(c.eligible for c in cells)
    assert all(c.verdict is None for c in info)
    fails = [c for c in cells if c.verdict == "FAIL"]
    assert not fails, f"calibration fails: {fails}"


def test_simulate_eligibility_labels():
    from robocurate.report.simulate import Cell, markdown, result_eligible

    assert result_eligible(8, 3, 3) and result_eligible(12, 5, 5)
    assert not result_eligible(8, 2, 3) and not result_eligible(8, 2, 2) and not result_eligible(4, 3, 3)
    cells = [Cell(0.04, 0.06, 8, 3, 3, 200_000, 0.041), Cell(0.04, 0.06, 8, 1, 3, 200_000, 0.0711, "d016"),
             Cell(0.04, 0.06, 12, 5, 5, 200_000, 0.0595), Cell(0.04, 0.06, 8, 2, 5, 200_000, 0.03, "d016")]
    assert [c.verdict for c in cells] == ["ok", None, "FAIL", None]
    md = markdown(cells)
    assert "| 8, 1, 3 | d016 | not result-eligible (informational) | 200000 | 0.0711" in md and "no pass/fail" in md
    assert md.count("| result-eligible |") == 2 and "**FAIL**" in md


# ------------------------------------------------------------------ D-026 exact balanced test


def test_balanced_test_by_hand():
    placebo = [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.6, 0.6, 0.6], [0.3, 0.5, 0.7]]
    entrant = [0.8, 0.7, 0.9]
    means = np.array([0.6, 0.5, 0.6, 0.5])
    K, n = 4, 3
    msb = n * means.var(ddof=1)
    se = math.sqrt((1 + 1 / K) * msb / n)
    r = S.balanced_test(entrant, placebo)
    assert r.balanced and r.df == K - 1
    assert r.diff == pytest.approx(0.8 - 0.55) and r.se == pytest.approx(se)
    assert r.p == pytest.approx(2 * sps.t.sf(abs((0.8 - 0.55) / se), K - 1))
    assert not S.balanced_test([0.8, 0.7], placebo).balanced  # n_c != n_p is flagged


def test_balanced_test_arrays_match_scalar():
    rng = np.random.default_rng(3)
    c = rng.uniform(0.3, 0.7, (5, 7, 3))
    p = rng.uniform(0.3, 0.7, (5, 8, 3))
    diff, t, df, pv = S.balanced_test_arrays(c[:, None], p[:, None])
    for g in range(5):
        for j in range(7):
            r = S.balanced_test(c[g, j], p[g])
            assert pv[g, j] == pytest.approx(r.p) and t[g, j] == pytest.approx(r.t) and df[g, j] == 7


def test_balanced_test_is_exact_under_normal_model():
    """Under the normal model (no clipping, no binomial) the single-test rejection rate equals alpha."""
    rng = np.random.default_rng(0)
    G, K, n = 200_000, 8, 3
    a_c = rng.normal(0, 0.05, (G, 1, 1, 1))
    a_p = rng.normal(0, 0.05, (G, 1, K, 1))
    c = a_c + rng.normal(0, 0.08, (G, 1, 1, n))
    p = a_p + rng.normal(0, 0.08, (G, 1, K, n))
    pv = S.balanced_test_arrays(c, p)[3][:, 0]
    rate = (pv <= 0.05).mean()
    assert abs(rate - 0.05) < 4 * math.sqrt(0.05 * 0.95 / G)


def test_balanced_power_matches_simulation():
    rng = np.random.default_rng(1)
    G, K, n, R, eff, ss, sd = 100_000, 8, 3, 2, 0.12, 0.04, 0.06
    c = eff + rng.normal(0, ss, (G, R, 1, 1)) + rng.normal(0, sd, (G, R, 1, n))
    p = rng.normal(0, ss, (G, R, K, 1)) + rng.normal(0, sd, (G, R, K, n))
    pv = S.balanced_test_arrays(c, p)[3][:, 0]
    mc = (pv <= 0.05 / 7).mean()
    assert S.balanced_power(eff, ss, sd, n, K, R) == pytest.approx(mc, abs=0.01)
    mde = S.balanced_mde(ss, sd, n, K, R)
    assert S.balanced_power(mde, ss, sd, n, K, R) == pytest.approx(0.8, abs=1e-4)
    assert S.balanced_mde(ss, sd, n, K, 3) < mde < S.balanced_mde(ss, sd, n, K, 1)


def test_compare_entrants_d026_primary_with_secondaries():
    rng = np.random.default_rng(2)
    placebo = [list(0.5 + rng.normal(0, 0.03) + rng.normal(0, 0.04, 3)) for _ in range(8)]
    ents = {"good": list(0.75 + rng.normal(0, 0.04, 3)), "null": list(0.5 + rng.normal(0, 0.04, 3))}
    out, vc = S.compare_entrants_d026(ents, placebo, n_boot=500)
    by = {c.name: c for c in out}
    assert all(c.test == "d026" and c.balanced and c.df == 7 for c in out)
    assert by["good"].significant and not by["null"].significant
    assert set(by["good"].vc_secondary) >= {"p_holm", "se"} and "p_holm" in by["good"].welch_secondary
    assert by["good"].p_holm >= by["good"].p_raw and vc.K == 8


def test_power_large_noncentrality_regression():
    """Regression: scipy nct.cdf returned NaN deep in the tail, so subset_mde's root finder crashed (R=3)."""
    d = S.Design(n_c=3, K=8, n_p=2, R=3)
    assert S.subset_power(0.2, 0.03, 0.05, d) == pytest.approx(1.0, abs=1e-3)
    assert 0 < S.subset_mde(0.03, 0.05, d) < 0.2


def test_preregistered_outcome_labels():
    """docs/preregistration.md section 8."""
    from robocurate.report.stats import outcome_label, t_ci95

    lo, hi = t_ci95(0.1, 0.02, 7)
    assert lo < 0.1 < hi and abs((hi - 0.1) - 2.3646 * 0.02) < 1e-3
    assert outcome_label(0.2, (0.1, 0.3), True, 0.15) == "positive"
    assert outcome_label(-0.2, (-0.3, -0.1), True, 0.15) == "negative"
    assert outcome_label(0.01, (-0.05, 0.07), False, 0.08) == "no detectable effect"
    assert outcome_label(0.05, (-0.1, 0.2), False, 0.18) == "inconclusive"
