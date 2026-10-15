"""
A toy market where WE KNOW the truth - used to check our tools before trusting
them on real data.  (Professional habit: never trust an estimator you haven't
seen recover a known answer.)

One stream of news e_t hits two assets:

    leader:    r1_t = e_t + noise
    follower:  r2_t = beta * ( a * e_t  +  (1 - a) * e_{t-1} ) + noise

a = 1.0 -> the follower absorbs the news instantly (no lead-lag)
a = 0.8 -> 80% of the news reaches the follower in the same minute and
           20% one minute later => the leader "leads" by one bar
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_pair(n: int = 50_000, a: float = 0.8, beta: float = 1.0,
                  noise: float = 0.3, garch: bool = True, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Volatility clustering with a GARCH(1,1): calm periods and storms, like real markets.
    sigma = np.ones(n)
    if garch:
        omega, alpha, b = 0.02, 0.08, 0.90         # long-run variance = 1.0
        z = rng.standard_normal(n)
        var = np.empty(n)
        var[0] = 1.0
        for t in range(1, n):
            var[t] = omega + alpha * (np.sqrt(var[t - 1]) * z[t - 1]) ** 2 + b * var[t - 1]
        sigma = np.sqrt(var)
        e = sigma * z
    else:
        e = rng.standard_normal(n)

    e_lag = np.r_[0.0, e[:-1]]
    r1 = e + noise * sigma * rng.standard_normal(n)
    r2 = beta * (a * e + (1 - a) * e_lag) + noise * sigma * rng.standard_normal(n)

    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="min", tz="America/New_York")
    # scale to realistic 1-minute return sizes (~0.05%)
    return pd.DataFrame({"LEADER": r1, "FOLLOWER": r2}, index=idx) * 5e-4


def simulate_same_asset_pair(w: float, kappa: float, noise_a: float, noise_b: float,
                             n_days: int = 60, seed: int = 11) -> pd.DataFrame:
    """Two prices of the SAME asset (like NQ futures and the QQQ ETF), regular
    hours only, returned as LOG prices.

        A = efficient price + noise_a
        B absorbs a share w of each piece of news immediately, then closes a
          fraction kappa of its remaining gap to the efficient price each minute,
          + noise_b

    w = 1  -> no lead (B is as fast as A)
    w < 1  -> A leads (the smaller w, the stronger the lead)
    """
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=n_days)
    idx = pd.DatetimeIndex(np.concatenate(
        [pd.date_range(d + pd.Timedelta("9h30min"), periods=390, freq="min").values for d in days]
    )).tz_localize("America/New_York")
    n = len(idx)
    e = rng.standard_normal(n) * 3e-4
    m = np.cumsum(e)
    x = np.empty(n)
    x[0] = m[0]
    for t in range(1, n):
        x[t] = x[t - 1] + w * e[t] + kappa * (m[t - 1] - x[t - 1])
    return pd.DataFrame({"A": 9 + m + rng.standard_normal(n) * noise_a,
                         "B": 6 + x + rng.standard_normal(n) * noise_b}, index=idx)
