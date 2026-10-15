"""Lead-lag tools: clock lags, VAR, cross-correlation, net lead, confidence intervals."""
import numpy as np
import pandas as pd
import pytest

from src.leadlag import _shift, cross_correlation, fit_var, mean_with_ci, net_lead


def minutes(n):
    return pd.date_range("2026-01-05 09:30", periods=n, freq="min", tz="America/New_York")


def lead_pair(n=60_000, late=0.2, noise=0.3, seed=0):
    """L reacts to news at once; F reacts (1 - late) at once and `late` one minute later."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    e_lag = np.r_[0.0, e[:-1]]
    return pd.DataFrame({"L": e + noise * rng.standard_normal(n),
                         "F": (1 - late) * e + late * e_lag + noise * rng.standard_normal(n)},
                        index=minutes(n)) * 5e-4


def test_clock_lag_does_not_jump_gaps():
    idx = pd.DatetimeIndex(["2026-01-05 09:30", "2026-01-05 09:31", "2026-01-05 09:33"])
    s = pd.Series([1.0, 2.0, 3.0], index=idx)
    lagged = _shift(s, 1, "1min")
    assert np.isnan(lagged.iloc[0]) and lagged.iloc[1] == 1.0
    assert np.isnan(lagged.iloc[2])               # 09:32 is missing: no value, not 09:31's


def test_var_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    A = np.array([[0.2, 0.1], [0.3, 0.0]])
    x = np.zeros((40_000, 2))
    for t in range(1, len(x)):
        x[t] = A @ x[t - 1] + rng.standard_normal(2)
    res = fit_var(pd.DataFrame(x, columns=["a", "b"], index=minutes(len(x))), max_lags=4, freq="1min")
    assert res.k_ar == 1                          # BIC picks the true lag length
    assert np.allclose(res.coefs[0], A, atol=0.02)


def test_cross_correlation_peaks_at_the_true_lag():
    r = lead_pair(late=0.5)
    ccf = cross_correlation(r["L"], r["F"], max_lag=3, freq="1min")
    lagged = ccf.drop(index=0)
    assert lagged["corr"].idxmax() == 1           # corr(L_t, F_t+1): L moves first
    assert lagged.loc[1, "t_robust"] > 10


@pytest.mark.parametrize("late", [0.0, 0.1, 0.3])
def test_net_lead_recovers_a_planted_lead(late):
    nl = net_lead(lead_pair(late=late), "L", "F")["net_lead_L_pp"]
    assert abs(nl - 100 * late) < 2.0             # within 2 percentage points


def test_net_lead_flips_sign_when_roles_swap():
    r = lead_pair(late=0.2)
    assert net_lead(r, "L", "F")["net_lead_L_pp"] > 15
    assert net_lead(r, "F", "L")["net_lead_F_pp"] < -15


def test_mean_with_ci():
    ci = mean_with_ci([1.0, 2.0, 3.0, 4.0])
    se = np.std([1, 2, 3, 4], ddof=1) / 2
    assert ci["mean"] == 2.5 and ci["n"] == 4
    assert np.isclose(ci["se"], se)
    assert np.isclose(ci["ci_low"], 2.5 - 1.96 * se) and np.isclose(ci["ci_high"], 2.5 + 1.96 * se)
