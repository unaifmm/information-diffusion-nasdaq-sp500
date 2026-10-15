"""
STEP 4 - Market anatomy: what do these four markets look like?

Before testing who leads whom, describe the markets (every good research
report does this first). Four views, all from London Strategic Edge data:

  A) Data coverage: how many bars and days we have, per instrument and year.
  B) Daily picture: growth, relative strength, correlation, beta and volatility
     of QQQ (Nasdaq-100) vs SPY (S&P 500), plus a data cross-check: a future and
     the ETF on the same index must move together almost perfectly every day.
  C) Intraday picture: when during the futures day do NQ and ES trade, and
     when do they move? (volume and volatility minute by minute)
  D) Liquidity and tick size by year: how much money trades in each future, and
     how coarse each one's price grid is compared with its price.

    python 04_market_anatomy.py        (about 1 minute)

Outputs
  figures/04_daily_overview.png, 04_intraday_profile.png, 04_liquidity_and_tick.png
  results/04_data_coverage.csv, 04_daily_summary.csv, 04_daily_crosscheck.csv,
          04_liquidity_by_year.csv
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from src import config
from src.data import daily_close, log_returns, rth_mask
from src.lse import load_lse
from src.plots import AQUA, BLUE, INK_2, ORANGE, save, set_style

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
TRADING_DAYS = 252
NAMES = ["NQ", "ES", "QQQ", "SPY"]

close = load_lse(NAMES, field="close", how="outer")
volume = load_lse(NAMES, field="volume", how="outer")

# ---------------------------------------------------------------------------
# A) Data coverage
# ---------------------------------------------------------------------------
cov = []
for name in NAMES:
    s = close[name].dropna()
    for year, part in s.groupby(s.index.year):
        cov.append({"instrument": name, "year": year, "bars": len(part),
                    "days": part.index.normalize().nunique(),
                    "regular_hours_bars": int(rth_mask(part.index).sum())})
coverage = pd.DataFrame(cov)
coverage.to_csv(config.RESULTS / "04_data_coverage.csv", index=False)
print("Bars per year (all hours):")
print(coverage.pivot(index="year", columns="instrument", values="bars").fillna(0).astype(int), "\n")

# ---------------------------------------------------------------------------
# B) Daily picture (close = last regular-session minute, normally 15:59)
# ---------------------------------------------------------------------------
daily = daily_close(close)
# ETF days only: futures also trade on some US holidays when the ETFs are closed,
# and those empty ETF days would break every rolling window below.
etf = daily[["QQQ", "SPY"]].dropna()
rd = np.log(etf).diff().dropna()


def summary(p: pd.Series) -> dict:
    p = p.dropna()
    x = np.log(p).diff().dropna()
    years = (p.index[-1] - p.index[0]).days / 365.25
    dd = p / p.cummax() - 1
    cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
    vol = x.std() * np.sqrt(TRADING_DAYS)
    return {"start": p.index[0].date(), "end": p.index[-1].date(),
            "CAGR_%": 100 * cagr, "ann_vol_%": 100 * vol, "return_per_unit_vol": cagr / vol,
            "max_drawdown_%": 100 * dd.min(),
            "worst_day_%": 100 * (np.exp(x.min()) - 1), "worst_day": x.idxmin().date(),
            "best_day_%": 100 * (np.exp(x.max()) - 1), "best_day": x.idxmax().date(),
            "skewness": x.skew(), "excess_kurtosis": x.kurt()}


summ = pd.DataFrame({n: summary(daily[n]) for n in ["QQQ", "SPY"]}).T
summ.to_csv(config.RESULTS / "04_daily_summary.csv")
print("QQQ vs SPY, daily (price return, dividends not included):")
print(summ.round(2).to_string(), "\n")

# Cross-check: future vs ETF on the same index, same 15:59 minute -> should be ~identical
check = []
for fut, etf_name in (("NQ", "QQQ"), ("ES", "SPY")):
    pair = np.log(daily[[fut, etf_name]].dropna()).diff().dropna()
    diff = (pair[fut] - pair[etf_name]).abs()
    check.append({"pair": f"{fut} vs {etf_name}", "days": len(pair),
                  "daily_return_corr": pair.corr().iloc[0, 1],
                  "median_abs_gap_bp": 1e4 * diff.median(),
                  "days_gap_over_0.5%": int((diff > 0.005).sum()),
                  "largest_gap_%": 100 * diff.max(), "largest_gap_day": diff.idxmax().date()})
crosscheck = pd.DataFrame(check)
crosscheck.to_csv(config.RESULTS / "04_daily_crosscheck.csv", index=False)
print("Data cross-check, future vs ETF on the same index (daily returns at 15:59):")
print(crosscheck.round(4).to_string(index=False), "\n")

W = 63
q, s_ = rd["QQQ"], rd["SPY"]
roll_corr = q.rolling(W).corr(s_)
roll_beta = q.rolling(W).cov(s_) / s_.rolling(W).var()
realised = rd[["QQQ", "SPY"]].rolling(21).std() * np.sqrt(TRADING_DAYS) * 100
growth = etf / etf.iloc[0]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
ax = axes[0, 0]
for n, color, label in (("QQQ", BLUE, "QQQ (Nasdaq-100)"), ("SPY", ORANGE, "SPY (S&P 500)")):
    ax.plot(growth.index, growth[n], color=color, label=label)
    ax.annotate(f"x{growth[n].iloc[-1]:.1f}", (growth.index[-1], growth[n].iloc[-1]),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
ax.set_yscale("log")
ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 6, 8]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_title(f"Growth of $1 since {growth.index[0]:%b %Y} (price only, log scale)")
ax.legend(loc="upper left", fontsize=8)
ax = axes[0, 1]
rel = growth["QQQ"] / growth["SPY"]
ax.plot(rel.index, rel, color=BLUE)
ax.axhline(1, color=INK_2, linewidth=0.8, linestyle="--")
ax.set_title("Relative strength QQQ / SPY (rising = Nasdaq-100 outperforming)")
ax = axes[1, 0]
ax.plot(roll_corr.index, roll_corr, color=INK_2, label=f"{W}-day correlation")
ax.plot(roll_beta.index, roll_beta, color=BLUE, label=f"{W}-day beta of QQQ on SPY")
ax.axhline(1, color=INK_2, linewidth=0.6, linestyle=":")
ax.set_title("How tightly they move: rolling correlation and beta")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
ax = axes[1, 1]
for n, color in (("QQQ", BLUE), ("SPY", ORANGE)):
    ax.plot(realised.index, realised[n], color=color, linewidth=1.1, label=n)
ax.set_title("Realised volatility, 21-day rolling (annualised %)")
ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Daily picture: Nasdaq-100 vs S&P 500 through their ETFs",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "04_daily_overview.png")

# ---------------------------------------------------------------------------
# C) Intraday profile of the futures, last 3 years (today's market)
# ---------------------------------------------------------------------------
recent_start = close.index.max() - pd.DateOffset(years=3)
fut_px = close.loc[close.index >= recent_start, ["NQ", "ES"]].dropna()
fut_vol = volume.loc[fut_px.index, ["NQ", "ES"]]
fut_r = log_returns(fut_px, max_gap="1min")
# minutes since the 18:00 session open (18:00 -> 0, 17:59 next day -> 1439)
mos = ((fut_px.index.hour * 60 + fut_px.index.minute) - 18 * 60) % (24 * 60)
day = fut_px.index.normalize()
share = fut_vol.div(fut_vol.groupby(fut_px.index.date).transform("sum"))   # share of the day's volume
vol_profile = (share * 100).groupby(mos).mean()
abs_profile = (fut_r.abs() * 1e4).groupby(mos).mean()                      # mean |return| in bp

fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)
for n, color in (("NQ", BLUE), ("ES", ORANGE)):
    axes[0].plot(vol_profile.index, vol_profile[n], color=color, linewidth=1, label=n)
    axes[1].plot(abs_profile.index, abs_profile[n], color=color, linewidth=1, label=n)
axes[0].set_ylabel("% of the day's volume")
axes[0].set_title("When do the futures trade? Share of daily volume in each minute (last 3 years)")
axes[1].set_ylabel("mean |1-min return| (bp)")
axes[1].set_title("When do they move? Average size of a 1-minute move")
events = {"08:30 US data": 14 * 60 + 30, "09:30 cash open": 15 * 60 + 30,
          "10:00 data": 16 * 60, "14:00 FOMC": 20 * 60, "16:00 cash close": 22 * 60}
for ax in axes:
    ax.legend(loc="upper left", fontsize=8)
    for label, x in events.items():
        ax.axvline(x, color=INK_2, linewidth=0.6, linestyle=":")
ymax = axes[1].get_ylim()[1]
for label, x in events.items():
    axes[1].text(x + 5, ymax * 0.97, label, fontsize=7, color=INK_2, va="top", rotation=90)
hours = list(range(0, 24 * 60, 120))
axes[1].set_xticks(hours)
axes[1].set_xticklabels([f"{(18 + h // 60) % 24:02d}:00" for h in hours])
axes[1].set_xlabel("New York time (futures session runs 18:00 -> 17:00)")
fig.tight_layout()
save(fig, "04_intraday_profile.png")

# ---------------------------------------------------------------------------
# D) Liquidity and relative tick size by year
# ---------------------------------------------------------------------------
fut_all = close[["NQ", "ES"]].dropna(how="all")
rows = []
for n in ("NQ", "ES"):
    px_n, vol_n = close[n].dropna(), volume[n].reindex(close[n].dropna().index)
    dollar = px_n * vol_n * config.POINT_VALUE[n]
    rth = rth_mask(px_n.index)
    for year in sorted(set(px_n.index.year)):
        y = px_n.index.year == year
        daily_rth = dollar[y & rth].groupby(px_n.index[y & rth].date).sum()
        daily_on = dollar[y & ~rth].groupby(px_n.index[y & ~rth].date).sum()
        rows.append({"instrument": n, "year": year,
                     "median_daily_$bn_regular": daily_rth.median() / 1e9,
                     "median_daily_$bn_overnight": daily_on.median() / 1e9,
                     "median_contracts_per_min_regular": vol_n[y & rth].median(),
                     "median_price": px_n[y].median(),
                     "tick_in_bp": 1e4 * config.TICK_SIZE[n] / px_n[y].median()})
liq = pd.DataFrame(rows)
liq.to_csv(config.RESULTS / "04_liquidity_by_year.csv", index=False)
print("Liquidity and relative tick size by year:")
print(liq.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
for n, color in (("NQ", BLUE), ("ES", ORANGE)):
    d = liq[liq["instrument"] == n]
    axes[0].plot(d["year"], d["median_daily_$bn_regular"], color=color, marker="o", markersize=4,
                 label=f"{n}, regular hours")
    axes[0].plot(d["year"], d["median_daily_$bn_overnight"], color=color, marker="o", markersize=4,
                 linestyle="--", label=f"{n}, overnight")
    axes[1].plot(d["year"], d["tick_in_bp"], color=color, marker="o", markersize=4, label=n)
axes[0].set_title("Money traded per day (median, $ billions)")
axes[0].set_ylabel("$ bn per day")
axes[0].legend(loc="upper left", fontsize=8)
axes[1].set_title("Relative tick size: one tick as basis points of the price")
axes[1].set_ylabel("bp per tick (lower = finer price grid)")
axes[1].legend(loc="upper right", fontsize=8)
for ax in axes:
    ax.set_xlabel("year")
fig.suptitle("Liquidity and price grid of NQ and ES futures", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "04_liquidity_and_tick.png")
