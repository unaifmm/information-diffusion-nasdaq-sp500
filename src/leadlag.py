"""
Lead-lag and information-diffusion toolkit.

Three tools, from simplest to richest:

1. cross_correlation  - is x today correlated with y k bars LATER?  (who moves first)
2. granger_test       - does the PAST of x help forecast y, beyond y's own past?
3. diffusion_curve    - after news hits x, what share of y's total reaction
                        happens in the same bar, 1 bar later, 2 bars later...?
                        (built from a VAR + generalized impulse responses)

Every function takes a `freq` argument for intraday data (e.g. "1min").  When it
is given, "k bars later" is computed on the CLOCK (t + k minutes), so we never
accidentally pair 15:59 today with 09:30 tomorrow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _shift(s: pd.Series, k: int, freq: str | None) -> pd.Series:
    """Value of s at t-k (k>0 = lag, k<0 = lead), aligned to s's own index."""
    if freq is None:
        return s.shift(k)
    return s.shift(k, freq=freq).reindex(s.index)


# ---------------------------------------------------------------------------
# 1) Cross-correlation function
# ---------------------------------------------------------------------------
def cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 10,
                      freq: str | None = None) -> pd.DataFrame:
    """corr(x_t, y_{t+k}) for k = -max_lag ... +max_lag.

    Reading the result:
      k = 0            -> same-bar co-movement (always large for NQ vs ES)
      k > 0 and significant -> x moves first, y follows  => "x leads y"
      k < 0 and significant -> y leads x

    Two significance measures:
      t_classic = corr * sqrt(n)   (textbook; assumes i.i.d. data)
      t_robust  = sum(x*y) / sqrt(sum((x*y)^2))
                  (Dalla, Giraitis & Phillips 2022) - stays valid when volatility
                  clusters, which it always does in markets. |t| > 1.96 ~ 5% level.
    """
    xd = (x - x.mean()).rename("x")
    yd = (y - y.mean()).rename("y")
    rows = []
    for k in range(-max_lag, max_lag + 1):
        pair = pd.concat([xd, _shift(yd, -k, freq)], axis=1).dropna()
        a, b = pair["x"].to_numpy(), pair["y"].to_numpy()
        ab = a * b
        corr = ab.sum() / np.sqrt((a * a).sum() * (b * b).sum())
        rows.append({
            "lag": k,
            "corr": corr,
            "t_classic": corr * np.sqrt(len(a)),
            "t_robust": ab.sum() / np.sqrt((ab * ab).sum()),
            "n": len(a),
        })
    return pd.DataFrame(rows).set_index("lag")


# ---------------------------------------------------------------------------
# 2) Granger causality (with robust standard errors)
# ---------------------------------------------------------------------------
def granger_test(data: pd.DataFrame, cause: str, effect: str, lags: int,
                 freq: str | None = None) -> dict:
    """Regress effect_t on a constant, `lags` lags of effect and `lags` lags of cause:

        effect_t = c + sum_i a_i*effect_{t-i} + sum_i b_i*cause_{t-i} + e_t

    H0: all b_i = 0  ("the past of `cause` adds no forecasting power").
    A small p-value (< 0.05) rejects H0 => `cause` Granger-causes `effect`.

    NB: "Granger causality" means PREDICTIVE power, not true causality.
    HAC (Newey-West) standard errors make the test robust to volatility clustering.
    """
    X = {}
    for i in range(1, lags + 1):
        X[f"{effect}_lag{i}"] = _shift(data[effect], i, freq)
        X[f"{cause}_lag{i}"] = _shift(data[cause], i, freq)
    df = pd.concat([data[effect].rename("y"), pd.DataFrame(X)], axis=1).dropna()
    res = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags})
    hypothesis = ", ".join(f"{cause}_lag{i} = 0" for i in range(1, lags + 1))
    test = res.f_test(hypothesis)
    return {
        "cause": cause,
        "effect": effect,
        "lags": lags,
        "F": float(np.squeeze(test.fvalue)),
        "p_value": float(np.squeeze(test.pvalue)),
        "sum_of_cause_coefs": float(sum(res.params[f"{cause}_lag{i}"] for i in range(1, lags + 1))),
        "r2": float(res.rsquared),
        "n": int(res.nobs),
    }


# ---------------------------------------------------------------------------
# 3) VAR + generalized impulse responses -> "diffusion curve"
# ---------------------------------------------------------------------------
class SimpleVAR:
    """Minimal fitted-VAR container (coefficients, residual covariance, names)
    with the one extra thing we need: the moving-average representation."""

    def __init__(self, names, intercept, coefs, sigma_u, nobs):
        self.names = list(names)
        self.intercept = intercept          # (k,)
        self.coefs = coefs                  # (p, k, k): coefs[i-1][eq, var] for lag i
        self.sigma_u = sigma_u              # (k, k) residual covariance
        self.k_ar = coefs.shape[0]
        self.nobs = nobs

    def ma_rep(self, maxn: int) -> np.ndarray:
        """Phi_h: effect of a shock today on every variable h bars later."""
        k = len(self.names)
        phi = np.zeros((maxn + 1, k, k))
        phi[0] = np.eye(k)
        for h in range(1, maxn + 1):
            for j in range(1, min(h, self.k_ar) + 1):
                phi[h] += self.coefs[j - 1] @ phi[h - j]
        return phi


def fit_var(data: pd.DataFrame, max_lags: int = 10, ic: str = "bic",
            freq: str | None = None) -> SimpleVAR:
    """Fit a Vector AutoRegression by OLS: every variable is regressed on the
    lags of ALL variables. The number of lags p is chosen by an information
    criterion (BIC punishes extra lags harder than AIC -> simpler model).

    With `freq`, lags are taken on the clock (t - i minutes), so a regression
    never links 15:59 today to 09:30 tomorrow or one hour's last bar to the
    next day's first one. All candidate p use the same sample, as they must
    for a fair comparison."""
    names = list(data.columns)
    k = len(names)
    lagged = {f"{c}_lag{i}": _shift(data[c], i, freq)
              for i in range(1, max_lags + 1) for c in names}
    full = pd.concat([data, pd.DataFrame(lagged, index=data.index)], axis=1).dropna()
    Y = full[names].to_numpy()
    n = len(Y)
    if n < 50 * (k * max_lags + 1):
        raise ValueError(f"not enough observations ({n}) for a VAR with {max_lags} lags")

    def ols(p):
        X = np.column_stack([np.ones(n)] + [full[[f"{c}_lag{i}" for c in names]].to_numpy()
                                            for i in range(1, p + 1)])
        B, *_ = np.linalg.lstsq(X, Y, rcond=None)
        U = Y - X @ B
        return B, U

    best_p, best_ic, best = None, np.inf, None
    for p in range(1, max_lags + 1):
        B, U = ols(p)
        sigma_ml = U.T @ U / n
        penalty = (np.log(n) if ic == "bic" else 2.0) * k * (k * p + 1) / n
        crit = np.log(np.linalg.det(sigma_ml)) + penalty
        if crit < best_ic:
            best_p, best_ic, best = p, crit, (B, U)
    B, U = best
    coefs = np.stack([B[1 + (i - 1) * k: 1 + i * k].T for i in range(1, best_p + 1)])
    sigma_u = U.T @ U / (n - (k * best_p + 1))
    return SimpleVAR(names, B[0], coefs, sigma_u, n)


def generalized_irf(var_res, horizon: int = 10) -> np.ndarray:
    """Generalized impulse responses (Pesaran & Shin 1998).

    girf[h, i, j] = response of variable i, h bars after a one-standard-deviation
    shock to variable j.  Unlike the usual (Cholesky) IRF, the result does NOT
    depend on the order in which you list the variables - important here because
    NQ and ES move together so strongly within the same bar.
    """
    phi = var_res.ma_rep(horizon)              # (horizon+1, k, k) MA coefficients
    sigma = np.asarray(var_res.sigma_u)        # residual covariance matrix
    scale = np.sqrt(np.diag(sigma))
    return np.einsum("hik,kj->hij", phi, sigma) / scale[None, None, :]


def diffusion_curve(var_res, shock: str, response: str, horizon: int = 10) -> pd.DataFrame:
    """How the information in a shock to `shock` reaches `response` over time.

    cumulative[h] = total price reaction of `response` up to h bars after the shock
    share[h]      = cumulative[h] / cumulative[horizon]  (fraction of the long-run
                    reaction already absorbed after h bars)

    share[0] close to 1  -> information is absorbed within the same bar (fast diffusion)
    share[0] clearly < 1 -> part of the information arrives later (slow diffusion)
    """
    names = list(var_res.names)
    i, j = names.index(response), names.index(shock)
    g = generalized_irf(var_res, horizon)[:, i, j]
    cum = np.cumsum(g)
    return pd.DataFrame({"irf": g, "cumulative": cum, "share": cum / cum[-1]},
                        index=pd.RangeIndex(horizon + 1, name="h"))


# ---------------------------------------------------------------------------
# Everything for one pair, in one call
# ---------------------------------------------------------------------------
def analyze_pair(returns: pd.DataFrame, a: str, b: str, freq: str | None = None,
                 max_lag: int = 10, var_max_lags: int = 10, horizon: int = 10) -> dict:
    """Run all three tools on returns[a] and returns[b] and summarise.

    Summary keys use the generic names A (= a) and B (= b) so that results for
    different pairs fit in one table."""
    pair = returns[[a, b]]
    ccf = cross_correlation(pair[a], pair[b], max_lag=max_lag, freq=freq)
    var_res = fit_var(pair, max_lags=var_max_lags, freq=freq)
    p = max(var_res.k_ar, 1)
    g_ab = granger_test(pair, cause=a, effect=b, lags=p, freq=freq)
    g_ba = granger_test(pair, cause=b, effect=a, lags=p, freq=freq)
    d_ab = diffusion_curve(var_res, shock=a, response=b, horizon=horizon)
    d_ba = diffusion_curve(var_res, shock=b, response=a, horizon=horizon)
    summary = {
        "A": a, "B": b,
        "n_obs": int(ccf.loc[0, "n"]),
        "corr_same_bar": ccf.loc[0, "corr"],
        "corr(A_t,B_t+1)": ccf.loc[1, "corr"],
        "t_robust(A_t,B_t+1)": ccf.loc[1, "t_robust"],
        "corr(B_t,A_t+1)": ccf.loc[-1, "corr"],
        "t_robust(B_t,A_t+1)": ccf.loc[-1, "t_robust"],
        "var_lags_bic": p,
        "granger_p_A->B": g_ab["p_value"],
        "granger_p_B->A": g_ba["p_value"],
        "same_bar_share_B_after_A_shock": d_ab["share"].iloc[0],
        "same_bar_share_A_after_B_shock": d_ba["share"].iloc[0],
    }
    return {"summary": summary, "ccf": ccf, "granger": [g_ab, g_ba],
            "diffusion": {f"{a}->{b}": d_ab, f"{b}->{a}": d_ba}, "var": var_res}


def verdict(summary: dict, alpha: float = 0.01) -> str:
    """Plain-English reading of analyze_pair()'s summary.

    Two different questions:
      * statistical significance - is the effect distinguishable from zero?
        (with hundreds of thousands of minutes, even tiny effects are)
      * economic significance - is it BIG enough to matter? Here: how much of
        the reaction is NOT absorbed within the same bar.
    """
    a, b = summary["A"], summary["B"]
    p_ab, p_ba = summary["granger_p_A->B"], summary["granger_p_B->A"]
    late_b = 1 - summary["same_bar_share_B_after_A_shock"]
    late_a = 1 - summary["same_bar_share_A_after_B_shock"]
    if p_ab >= alpha and p_ba >= alpha:
        direction = (f"Granger: neither market's past adds forecasting power beyond the other's "
                     f"own past ({a}->{b} p={p_ab:.2g}, {b}->{a} p={p_ba:.2g}).")
    elif p_ab < alpha and p_ba >= alpha:
        direction = (f"{a} leads {b}: {a}'s past helps forecast {b} beyond {b}'s own past "
                     f"(p={p_ab:.2g}); not the other way round (p={p_ba:.2g}).")
    elif p_ba < alpha and p_ab >= alpha:
        direction = (f"{b} leads {a}: {b}'s past helps forecast {a} beyond {a}'s own past "
                     f"(p={p_ba:.2g}); not the other way round (p={p_ab:.2g}).")
    else:
        direction = (f"Feedback both ways ({a}->{b} p={p_ab:.2g}, {b}->{a} p={p_ba:.2g}); "
                     f"compare the sizes below.")
    size = (f"Share of the reaction that arrives AFTER the first bar: "
            f"{b} after a {a} shock {late_b:+.1%}, {a} after a {b} shock {late_a:+.1%}.")
    if max(abs(late_a), abs(late_b)) < 0.05:
        size += " => under 5%: information diffuses essentially within one bar."
    return direction + "\n   " + size


def diffusion_row(returns: pd.DataFrame, a: str, b: str, freq: str | None = "1min",
                  max_lag: int = 5, var_max_lags: int = 10, **labels) -> dict:
    """analyze_pair() condensed into one tidy row, with the two numbers we
    report most:

    late_%_<b>_after_<a>_shock : share of b's total reaction to an a-shock that
                                 arrives AFTER the first bar. Negative = the first
                                 bar over-shoots and partly reverses (bid-ask
                                 bounce), which hits both directions alike.
    net_lead_<a>_pp            : late_%(b after a) - late_%(a after b). The bounce
                                 cancels out: > 0 means a moves first.
    """
    s = analyze_pair(returns, a, b, freq=freq, max_lag=max_lag,
                     var_max_lags=var_max_lags)["summary"]
    late_b = 100 * (1 - s["same_bar_share_B_after_A_shock"])
    late_a = 100 * (1 - s["same_bar_share_A_after_B_shock"])
    return {
        **labels,
        "n_obs": s["n_obs"],
        "corr_same_bar": s["corr_same_bar"],
        f"corr({a}_t,{b}_t+1)": s["corr(A_t,B_t+1)"],
        f"t({a}_t,{b}_t+1)": s["t_robust(A_t,B_t+1)"],
        f"corr({b}_t,{a}_t+1)": s["corr(B_t,A_t+1)"],
        f"t({b}_t,{a}_t+1)": s["t_robust(B_t,A_t+1)"],
        f"granger_p_{a}->{b}": s["granger_p_A->B"],
        f"granger_p_{b}->{a}": s["granger_p_B->A"],
        "var_lags": s["var_lags_bic"],
        f"late_%_{b}_after_{a}_shock": late_b,
        f"late_%_{a}_after_{b}_shock": late_a,
        f"net_lead_{a}_pp": late_b - late_a,
    }


def net_lead(returns: pd.DataFrame, a: str, b: str, freq: str | None = "1min",
             var_max_lags: int = 10) -> dict:
    """Only the VAR-based numbers (no cross-correlations, no Granger tests).

    Much faster than diffusion_row(), so it can be run hundreds of times, e.g.
    once per month to measure how precise the yearly estimates are."""
    pair = returns[[a, b]]
    var_res = fit_var(pair, max_lags=var_max_lags, freq=freq)
    share_b = diffusion_curve(var_res, shock=a, response=b)["share"].iloc[0]
    share_a = diffusion_curve(var_res, shock=b, response=a)["share"].iloc[0]
    clean = pair.dropna()
    return {
        "n_obs": len(clean),
        "corr_same_bar": float(clean[a].corr(clean[b])),
        "var_lags": var_res.k_ar,
        f"late_%_{b}_after_{a}_shock": 100 * (1 - share_b),
        f"late_%_{a}_after_{b}_shock": 100 * (1 - share_a),
        f"net_lead_{a}_pp": 100 * (share_a - share_b),
    }


def mean_with_ci(values, z: float = 1.96) -> dict:
    """Average of several independent estimates (e.g. one per month) and its
    95% confidence interval: mean +/- 1.96 * sd / sqrt(n)."""
    v = pd.Series(values, dtype=float).dropna()
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean": v.mean(), "se": se, "ci_low": v.mean() - z * se,
            "ci_high": v.mean() + z * se, "t_stat": v.mean() / se if se else np.nan,
            "n": n}


def fresh_cross_correlation(x: pd.Series, y: pd.Series, fresh: pd.Series, max_lag: int = 5,
                            freq: str = "1s") -> pd.DataFrame:
    """corr(x_t, y_{t+k}) using only starting seconds where BOTH markets traded.

    Why: on a previous-tick grid, a market that did not trade in second t shows
    its move one second late even if it was not slow at all - a fake lead
    (09_one_second.py shows a copy of NQ that simply trades less often 'lags' NQ).
    Requiring a trade in both markets in the earlier second removes that.
      k > 0: sample = seconds t where both traded; corr(x_t, y_{t+k})
      k < 0: sample = seconds s where both traded; corr(y_s, x_{s+|k|})
    t_robust as in cross_correlation()."""
    rows = []
    for k in range(-max_lag, max_lag + 1):
        first, second = (x, y) if k >= 0 else (y, x)
        later = _shift(second, -abs(k), freq)
        pair = pd.concat([first[fresh], later[fresh]], axis=1, keys=["a", "b"]).dropna()
        a = (pair["a"] - pair["a"].mean()).to_numpy()
        b = (pair["b"] - pair["b"].mean()).to_numpy()
        ab = a * b
        rows.append({"lag": k, "corr": ab.sum() / np.sqrt((a * a).sum() * (b * b).sum()),
                     "t_robust": ab.sum() / np.sqrt((ab * ab).sum()), "n": len(a)})
    return pd.DataFrame(rows).set_index("lag")
