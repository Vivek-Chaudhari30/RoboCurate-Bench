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


def test_null_fwer_calibration():
    """Task-spec null calibration at the protocol seed floor (D-015: 6 seeds).

    20000 grids instead of 1000: at a true FWER of ~0.06, 1000 grids have MC sd ~0.0075, too coarse to
    test a 0.07 bound reliably. More grids make the test stricter, not weaker.
    """
    from robocurate.config import SEED_FLOOR

    t0 = time.time()
    fwer = null_grid_fwer(n_sims=20000, n_seeds=SEED_FLOOR)
    assert fwer <= 0.07, f"Holm family-wise false-positive rate {fwer:.4f} > 0.07"
    assert time.time() - t0 < 60


def test_three_seeds_is_not_enough():
    """Regression guard for D-015: with 3 seeds Welch + Holm exceeds the 0.07 null FWER bound."""
    assert null_grid_fwer(n_sims=20000, n_seeds=3) > 0.07


@pytest.mark.slow
@pytest.mark.parametrize("base", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("seed", [1, 2])
def test_null_fwer_calibration_robust(base, seed):
    from robocurate.config import SEED_FLOOR

    assert null_grid_fwer(n_sims=20000, n_seeds=SEED_FLOOR, base=base, seed=seed) <= 0.07


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
    from robocurate.config import SEED_FLOOR

    assert power_detect_rate(n_seeds=SEED_FLOOR) >= 0.70
