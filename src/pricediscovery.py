"""
Price discovery between two prices of the SAME asset - e.g. NQ futures and the
QQQ ETF, which both track the Nasdaq-100.

Arbitrage keeps the two prices tied together: they can wander, but never far
apart for long. In statistics this is called COINTEGRATION, and the right model
is a Vector Error Correction Model (VECM), here in log prices p:

    dp_t = alpha * z_{t-1}  +  sum_i Gamma_i * dp_{t-i}  +  e_t,     z = p_A - p_B

z is the gap between the two prices. alpha says who CORRECTS when the gap opens:
the market that does not need to move is the one where the news arrived first.

Three standard measures of each market's share of price discovery (0 to 1):

  CS   Gonzalo & Granger (1995) component share. Built from alpha alone.
  IS   Hasbrouck (1995) information share: the share of the variance of the
       common "efficient price" innovations that comes from each market. It
       depends on the order the two variables are listed in (Cholesky), so we
       report the lower and upper bound and their midpoint.
  ILS  Putnins (2013) information leadership share. IS and CS are both pushed
       around by how NOISY each market is, not only by who is first. Combining
       them cancels the noise part and isolates leadership: who moves FIRST.

Practical details:
  * z is demeaned day by day: the futures-ETF basis (interest, dividends) changes
    slowly and jumps on dividend days; demeaning per day removes that.
  * Lags are taken on the clock (t - 1 minute), never across days.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .leadlag import _shift


def vecm_price_discovery(logp: pd.DataFrame, lags: int = 3, freq: str = "1min") -> dict:
    """Estimate the VECM and return IS (bounds + midpoint), CS and ILS for the
    FIRST column of `logp` (the second column's shares are 1 minus these)."""
    a, b = logp.columns
    dp = logp.diff()
    gap = logp.index.to_series().diff()
    dp[gap > pd.Timedelta(freq)] = np.nan                    # no returns across gaps

    day = logp.index.normalize()
    z = logp[a] - logp[b]
    z = z - z.groupby(day).transform("mean")                  # remove each day's basis

    X = {"z_lag1": _shift(z, 1, freq)}
    for i in range(1, lags + 1):
        X[f"d_{a}_lag{i}"] = _shift(dp[a], i, freq)
        X[f"d_{b}_lag{i}"] = _shift(dp[b], i, freq)
    df = pd.concat([dp[[a, b]], pd.DataFrame(X, index=logp.index)], axis=1).dropna()
    cols = list(X)
    Y = df[[a, b]].to_numpy()
    Xm = np.column_stack([np.ones(len(df)), df[cols].to_numpy()])
    B, *_ = np.linalg.lstsq(Xm, Y, rcond=None)                # B[regressor, equation]
    U = Y - Xm @ B
    omega = U.T @ U / (len(df) - Xm.shape[1])                 # residual covariance

    alpha = B[1 + cols.index("z_lag1")]                       # (alpha_a, alpha_b)
    gamma_sum = np.zeros((2, 2))                              # sum of Gamma_i[equation, variable]
    for i in range(1, lags + 1):
        gamma_sum[:, 0] += B[1 + cols.index(f"d_{a}_lag{i}")]
        gamma_sum[:, 1] += B[1 + cols.index(f"d_{b}_lag{i}")]

    # Component share: from the vector orthogonal to alpha
    alpha_perp = np.array([alpha[1], -alpha[0]])
    cs = alpha_perp / alpha_perp.sum()

    # Hasbrouck information share: long-run impact row psi, then Cholesky in both orders
    psi = alpha_perp / (alpha_perp @ (np.eye(2) - gamma_sum) @ np.ones(2))
    shares = []
    for order in ([0, 1], [1, 0]):
        P = np.eye(2)[order]
        F = np.linalg.cholesky(P @ omega @ P.T)
        contrib = (P @ psi @ F) ** 2
        shares.append(P.T @ (contrib / contrib.sum()))
    is_lo = np.minimum(shares[0], shares[1])
    is_hi = np.maximum(shares[0], shares[1])
    is_mid = (is_lo + is_hi) / 2

    # Putnins information leadership share (uses the IS midpoints)
    il = abs((is_mid[0] / is_mid[1]) * (cs[1] / cs[0]))
    ils = il ** 2 / (1 + il ** 2)

    return {"n_obs": len(df), "alpha_" + a: alpha[0], "alpha_" + b: alpha[1],
            "CS_" + a: cs[0], "IS_low_" + a: is_lo[0], "IS_high_" + a: is_hi[0],
            "IS_mid_" + a: is_mid[0], "ILS_" + a: ils}
