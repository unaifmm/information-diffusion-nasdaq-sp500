"""
STEP 7 - Futures vs ETF: on the SAME index, where is news priced first?

NQ futures and the QQQ ETF both track the Nasdaq-100; ES futures and the SPY
ETF both track the S&P 500. Arbitrageurs keep each pair tied together, but the
news can still show up in one of them first. Hasbrouck (2003) found that, around
the year 2000, the E-mini futures did most of the price discovery.

Regular hours only (the ETFs trade 09:30-16:00), 1-minute bars, May 2016 -> today.

  A) Net lead of the futures over the ETF, estimated month by month and averaged
     per year with a 95% confidence interval (same method as 06).
  B) The textbook measures - Hasbrouck information share (IS), Gonzalo-Granger
     component share (CS), Putnins leadership share (ILS) - from a VECM, month
     by month. 02_validate_simulation.py shows these are NOT reliable at 1-minute
     bars (IS can only be bounded between ~0 and ~1). We report them so the reader
     can see that on real data too, not to draw conclusions from them.

    python 07_futures_vs_etf.py        (about 2 minutes)

Outputs
  figures/07_futures_vs_etf.png
  results/07_monthly.csv, 07_by_year_ci.csv, 07_by_period_ci.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import log_returns, regular_hours
from src.leadlag import mean_with_ci, net_lead
from src.lse import load_lse
from src.plots import BLUE, INK_2, ORANGE, save, set_style
from src.pricediscovery import vecm_price_discovery

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()
PAIRS = [("NQ", "QQQ", BLUE), ("ES", "SPY", ORANGE)]

px = regular_hours(load_lse(["NQ", "QQQ", "ES", "SPY"], how="outer"))
month = px.index.year * 100 + px.index.month

rows = []
for fut, etf, _ in PAIRS:
    pair_px = px[[fut, etf]].dropna()
    pair_month = pair_px.index.year * 100 + pair_px.index.month
    r = log_returns(pair_px, max_gap="1min")
    for m in np.unique(pair_month):
        sel = pair_month == m
        if sel.sum() < 2_000:
            continue
        nl = net_lead(r[sel], fut, etf, freq="1min")
        vecm = vecm_price_discovery(np.log(pair_px[sel]), lags=3, freq="1min")
        rows.append({"pair": f"{fut} vs {etf}", "month": f"{m // 100}-{m % 100:02d}", "year": m // 100,
                     "n_obs": nl["n_obs"], "corr_same_minute": nl["corr_same_bar"],
                     "late_%_ETF_after_futures_shock": nl[f"late_%_{etf}_after_{fut}_shock"],
                     "late_%_futures_after_ETF_shock": nl[f"late_%_{fut}_after_{etf}_shock"],
                     "net_lead_futures_pp": nl[f"net_lead_{fut}_pp"],
                     "IS_low_futures": vecm[f"IS_low_{fut}"], "IS_high_futures": vecm[f"IS_high_{fut}"],
                     "CS_futures": vecm[f"CS_{fut}"], "ILS_futures": vecm[f"ILS_{fut}"]})
    print(f"[{time.time() - T0:5.0f}s] {fut} vs {etf}: {int((np.array([r_['pair'] for r_ in rows]) == f'{fut} vs {etf}').sum())} months", flush=True)
monthly = pd.DataFrame(rows)
monthly.to_csv(config.RESULTS / "07_monthly.csv", index=False)

yearly = []
for (pair, year), d in monthly.groupby(["pair", "year"]):
    yearly.append({"pair": pair, "year": year, **mean_with_ci(d["net_lead_futures_pp"]),
                   "median_corr_same_minute": d["corr_same_minute"].median(),
                   "median_IS_low": d["IS_low_futures"].median(),
                   "median_IS_high": d["IS_high_futures"].median(),
                   "median_CS": d["CS_futures"].median(),
                   "share_of_months_CS_outside_0_1": ((d["CS_futures"] < 0) | (d["CS_futures"] > 1)).mean()})
yearly = pd.DataFrame(yearly)
yearly.to_csv(config.RESULTS / "07_by_year_ci.csv", index=False)

periods = {"2016-2019": (2016, 2019), "2020-2021": (2020, 2021), "2022-2026": (2022, 2026),
           "all years": (2016, 2026)}
by_period = []
for name, (y0, y1) in periods.items():
    for pair in monthly["pair"].unique():
        d = monthly[(monthly["pair"] == pair) & monthly["year"].between(y0, y1)]
        by_period.append({"pair": pair, "period": name, **mean_with_ci(d["net_lead_futures_pp"])})
by_period = pd.DataFrame(by_period)
by_period.to_csv(config.RESULTS / "07_by_period_ci.csv", index=False)

print("\nNet lead of the FUTURES over the ETF (pp), mean of monthly estimates with 95% CI:")
print(by_period.round(2).to_string(index=False))
print("\nTextbook measures, medians across months (for reference only - see 02 Part 2):")
print(yearly[["pair", "year", "median_corr_same_minute", "median_IS_low", "median_IS_high", "median_CS",
              "share_of_months_CS_outside_0_1"]].round(2).to_string(index=False))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
for i, (fut, etf, color) in enumerate(PAIRS):
    d = yearly[yearly["pair"] == f"{fut} vs {etf}"]
    off = -0.12 if i == 0 else 0.12
    axes[0].errorbar(d["year"] + off, d["mean"], yerr=1.96 * d["se"], fmt="o", color=color, ecolor=color,
                     elinewidth=1.2, capsize=3, markersize=5, label=f"{fut} futures vs {etf} ETF")
    axes[1].vlines(d["year"] + off, d["median_IS_low"], d["median_IS_high"], color=color, linewidth=7,
                   alpha=0.8, label=f"{fut} vs {etf}: range of the futures' IS")
axes[0].axhline(0, color=INK_2, linewidth=0.8)
axes[0].set_title("Net lead of the futures over the ETF (mean of monthly estimates, 95% CI)")
axes[0].set_ylabel("percentage points (> 0: futures first)")
axes[0].legend(loc="upper right", fontsize=8)
axes[1].axhline(0.5, color=INK_2, linewidth=0.8, linestyle="--")
axes[1].set_ylim(-0.02, 1.02)
axes[1].set_title("Textbook Hasbrouck information share of the futures: only known to lie in this range")
axes[1].set_ylabel("futures' share")
axes[1].set_xlabel("year   (at 1-minute bars the range covers almost 0 to 1, so it cannot say who leads)")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
axes[1].set_xticks(sorted(yearly["year"].unique()))
fig.tight_layout()
save(fig, "07_futures_vs_etf.png")
print(f"\n[{time.time() - T0:5.0f}s] finished")
