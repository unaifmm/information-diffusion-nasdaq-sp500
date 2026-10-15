"""
STEP 6 - How sure are we, and why? Precision and mechanism for NQ vs ES.

05_nq_vs_es.py gives ONE number per year. A single number hides how noisy it
is. Here we:

  A) Estimate the net lead separately for EVERY month (about 125 months x 2
     sessions). The spread of the monthly estimates within a year gives a 95%
     confidence interval for that year:  mean +/- 1.96 * sd / sqrt(months).
     If the interval contains 0, we cannot say either market leads.

  B) Test an explanation (the "liquidity hypothesis"): a market where less
     money trades updates its price less often, so it should look like the
     follower. NQ traded ~8x less money than ES in 2016 and ~1.8x less in 2026.
     If the hypothesis is right, ES's lead should shrink as NQ's share of the
     combined dollar volume grows.
       net_lead_month = a + b * NQ_share_of_dollar_volume_month (+ trend) + error
     A positive b supports the hypothesis. Standard errors are HAC (robust to
     autocorrelation between neighbouring months).

    python 06_nq_es_robustness.py        (about 2-4 minutes)

Outputs
  figures/06_net_lead_ci.png, 06_liquidity_mechanism.png
  results/06_monthly_net_lead.csv, 06_net_lead_by_year_ci.csv,
          06_net_lead_by_period_ci.csv, 06_liquidity_regression.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config
from src.data import log_returns, rth_mask
from src.leadlag import mean_with_ci, net_lead
from src.lse import load_lse
from src.plots import AQUA, BLUE, INK, INK_2, save, set_style

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()

px = load_lse(["NQ", "ES"])
vol = load_lse(["NQ", "ES"], field="volume").reindex(px.index)
r = log_returns(px, max_gap="1min")
rth = rth_mask(r.index)
month = r.index.year * 100 + r.index.month                  # e.g. 202403
dollar = pd.DataFrame({n: px[n] * vol[n] * config.POINT_VALUE[n] for n in ("NQ", "ES")})

# ---------------------------------------------------------------------------
# A) One estimate per month and session
# ---------------------------------------------------------------------------
rows = []
for m in np.unique(month):
    in_month = month == m
    for session, mask in (("regular", rth), ("overnight", ~rth)):
        sel = in_month & mask
        sub = r[sel]
        if sub.dropna().shape[0] < 2_000:
            continue
        est = net_lead(sub, "NQ", "ES", freq="1min")
        dv = dollar[sel].sum()
        rows.append({
            "month": f"{m // 100}-{m % 100:02d}", "year": m // 100, "session": session,
            **est,
            "NQ_share_of_dollar_volume": dv["NQ"] / (dv["NQ"] + dv["ES"]),
            "ES_tick_vs_NQ_tick": px["NQ"][sel].median() / px["ES"][sel].median(),
            # realised volatility of NQ in this session-month: sqrt(sum of squared 1-min returns)
            "NQ_realised_vol_month_%": 100 * np.sqrt((sub["NQ"].dropna() ** 2).sum()),
        })
    if (m % 100) == 12:
        print(f"[{time.time() - T0:5.0f}s] {m // 100} done", flush=True)
monthly = pd.DataFrame(rows)
monthly.to_csv(config.RESULTS / "06_monthly_net_lead.csv", index=False)

# Yearly mean +/- 95% CI from the monthly estimates
yearly = []
for (year, session), d in monthly.groupby(["year", "session"]):
    yearly.append({"year": year, "session": session, **mean_with_ci(d["net_lead_NQ_pp"])})
yearly = pd.DataFrame(yearly)
yearly.to_csv(config.RESULTS / "06_net_lead_by_year_ci.csv", index=False)

periods = {"2016-2019": (2016, 2019), "2020-2021": (2020, 2021), "2022-2026": (2022, 2026),
           "all years": (2016, 2026)}
by_period = []
for name, (y0, y1) in periods.items():
    for session in ("regular", "overnight"):
        d = monthly[(monthly["year"].between(y0, y1)) & (monthly["session"] == session)]
        by_period.append({"period": name, "session": session, **mean_with_ci(d["net_lead_NQ_pp"])})
by_period = pd.DataFrame(by_period)
by_period.to_csv(config.RESULTS / "06_net_lead_by_period_ci.csv", index=False)

print("\nNet lead of NQ over ES (percentage points), mean of monthly estimates with 95% CI:")
print(by_period.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# B) Liquidity hypothesis: regression on NQ's share of dollar volume
# ---------------------------------------------------------------------------
reg_rows = []
reg_models = {}
for session in ("regular", "overnight"):
    d = monthly[monthly["session"] == session].reset_index(drop=True)
    trend = (pd.to_datetime(d["month"]) - pd.Timestamp("2016-01-01")).dt.days / 365.25
    for label, X in (("share only", d[["NQ_share_of_dollar_volume"]]),
                     ("share + time trend", d[["NQ_share_of_dollar_volume"]].assign(trend_years=trend))):
        res = sm.OLS(d["net_lead_NQ_pp"], sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        reg_models[(session, label)] = res
        for var in X.columns:
            reg_rows.append({"session": session, "model": label, "variable": var,
                             "coef": res.params[var], "se": res.bse[var], "t": res.tvalues[var],
                             "p_value": res.pvalues[var], "n_months": int(res.nobs),
                             "r2": res.rsquared})
reg = pd.DataFrame(reg_rows)
reg.to_csv(config.RESULTS / "06_liquidity_regression.csv", index=False)
print("\nLiquidity regression (monthly net lead on NQ's share of dollar volume; HAC standard errors):")
print(reg.round(3).to_string(index=False))
print("\nReading it: a positive, significant coefficient on NQ_share_of_dollar_volume means ES's"
      "\nlead was larger when NQ traded relatively less money. If it stays significant after adding"
      "\nthe time trend, liquidity explains more than the general passage of time.")

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
ax = axes[0]
for session, color, offset, marker in (("regular", INK, -0.12, "o"), ("overnight", AQUA, 0.12, "s")):
    d = yearly[yearly["session"] == session]
    ax.errorbar(d["year"] + offset, d["mean"], yerr=1.96 * d["se"], fmt=marker, color=color,
                ecolor=color, elinewidth=1.2, capsize=3, markersize=5, label=f"{session} hours")
ax.axhline(0, color=INK_2, linewidth=0.8)
ax.set_title("Net lead of NQ over ES by year: mean of monthly estimates with 95% confidence interval")
ax.set_ylabel("percentage points (> 0: NQ first)")
ax.set_xticks(sorted(yearly["year"].unique()))
ax.legend(loc="lower right", fontsize=8)
ax.set_xlabel("year   (an interval that crosses 0 means no market can be said to lead that year)")

ax = axes[1]
d = monthly[monthly["session"] == "regular"].copy()
d["date"] = pd.to_datetime(d["month"])
ax.scatter(d["date"], d["net_lead_NQ_pp"], s=10, color="#b7d3f6", label="one month")
ax.plot(d["date"], d["net_lead_NQ_pp"].rolling(12, min_periods=6).mean(), color=BLUE,
        label="12-month average")
ax.axhline(0, color=INK_2, linewidth=0.8)
ax.set_title("Regular hours: every monthly estimate (shows how noisy one month is)")
ax.set_ylabel("percentage points")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, "06_net_lead_ci.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
d = monthly[monthly["session"] == "regular"]
sc = axes[0].scatter(100 * d["NQ_share_of_dollar_volume"], d["net_lead_NQ_pp"], c=d["year"],
                     cmap="Blues", vmin=2013, vmax=2026, s=16, edgecolor="none")
res = reg_models[("regular", "share only")]
xs = np.linspace(d["NQ_share_of_dollar_volume"].min(), d["NQ_share_of_dollar_volume"].max(), 50)
axes[0].plot(100 * xs, res.params["const"] + res.params["NQ_share_of_dollar_volume"] * xs,
             color=INK, linewidth=1.2)
axes[0].axhline(0, color=INK_2, linewidth=0.8)
axes[0].set_xlabel("NQ's share of NQ+ES dollar volume (%)")
axes[0].set_ylabel("net lead of NQ (pp)")
axes[0].set_title(f"Regular hours, one dot per month (slope t = {res.tvalues['NQ_share_of_dollar_volume']:.1f})")
cb = fig.colorbar(sc, ax=axes[0], ticks=range(2016, 2027, 2))
cb.set_label("year")
for session, color in (("regular", BLUE), ("overnight", AQUA)):
    dd = monthly[monthly["session"] == session]
    axes[1].plot(pd.to_datetime(dd["month"]), 100 * dd["NQ_share_of_dollar_volume"], color=color,
                 label=f"{session} hours")
axes[1].set_title("NQ's share of the money traded in NQ + ES")
axes[1].set_ylabel("%")
axes[1].legend(loc="upper left", fontsize=8)
fig.suptitle("Liquidity hypothesis: does the thinner market follow?", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "06_liquidity_mechanism.png")
print(f"\n[{time.time() - T0:5.0f}s] finished")
