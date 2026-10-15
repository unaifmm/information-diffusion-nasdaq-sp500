"""
STEP 5 - Ten years of minute-by-minute information diffusion, NQ vs ES futures
(London Strategic Edge 1-minute bars, 2016 -> today).

    python 05_nq_vs_es.py

Questions
  A) Year by year: what share of the reaction arrives AFTER the first minute,
     in the regular session and overnight? Which market leads?
  B) Hour by hour: is diffusion slower in the quiet overnight hours?
  C) Time scale: does the lead-lag that is left at 1 minute vanish at 2, 5, 15
     minutes?
  D) Which minutes moved the most? (preview of the event study)

How to read the numbers
  late_% (one per direction) = share of a market's total reaction to the other's
  shock that arrives AFTER the first minute. Negative values mean the first
  minute OVER-shoots and partly reverses - typical of bid-ask bounce in thin
  markets, which hits both directions alike.
  net lead (pp) = late_%(ES after NQ) - late_%(NQ after ES). The bounce cancels
  out in the difference: positive = NQ leads ES, negative = ES leads NQ.

Outputs
  figures/05_nq_es_by_year.png, 05_nq_es_by_hour.png, 05_nq_es_by_timescale.png
  results/05_nq_es_by_year.csv, 05_nq_es_by_hour.csv, 05_nq_es_by_timescale.csv,
  results/05_largest_moves.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import log_returns, rth_mask
from src.leadlag import diffusion_row
from src.lse import load_lse
from src.plots import BLUE, INK_2, ORANGE, save, set_style

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()

# ---------------------------------------------------------------------------
# Data: aligned 1-minute closes. Inner join = only minutes where BOTH traded.
# ---------------------------------------------------------------------------
px = load_lse(["NQ", "ES"])
r = log_returns(px, max_gap="1min")
rth = rth_mask(r.index)
print(f"{len(px):,} common minutes, {px.index[0]} -> {px.index[-1]}")
print(f"{r.dropna().shape[0]:,} usable 1-minute return pairs "
      f"({rth.sum():,} regular hours, {(~rth).sum():,} overnight)\n")


def measure(sub: pd.DataFrame, freq: str = "1min", **labels) -> dict:
    """All the numbers for NQ vs ES on one slice of the data (see diffusion_row)."""
    return diffusion_row(sub, "NQ", "ES", freq=freq, **labels)


def tick(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A) Year by year, regular hours vs overnight
# ---------------------------------------------------------------------------
rows = []
for year in sorted(set(r.index.year)):
    for session, mask in (("regular", rth), ("overnight", ~rth)):
        sub = r[(r.index.year == year) & mask]
        if sub.dropna().shape[0] < 5_000:
            continue
        rows.append(measure(sub, year=year, session=session))
        tick(f"year {year} {session}: done")
by_year = pd.DataFrame(rows)
by_year.to_csv(config.RESULTS / "05_nq_es_by_year.csv", index=False)
print("\nBy year (late_% = share of the reaction arriving after the first minute):")
print(by_year[["year", "session", "n_obs", "corr_same_bar", "late_%_ES_after_NQ_shock",
               "late_%_NQ_after_ES_shock", "net_lead_NQ_pp", "granger_p_NQ->ES", "granger_p_ES->NQ"]]
      .round(4).to_string(index=False))
print("\nNOTE: 03_validate_real_noise.py shows that Granger tests say 'NQ leads' in 11 of 11"
      "\nyears even when there is NO lead, only because ES's tick is coarser than NQ's."
      "\nSo draw conclusions from net_lead_NQ_pp (validated), not from the Granger p-values.")


def two_direction_lines(ax, df, x, title, xlabel):
    ax.plot(df[x], df["late_%_ES_after_NQ_shock"], color=BLUE, marker="o", markersize=4,
            label="ES reacting to an NQ shock")
    ax.plot(df[x], df["late_%_NQ_after_ES_shock"], color=ORANGE, marker="o", markersize=4,
            label="NQ reacting to an ES shock")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("% of reaction after 1st bar")
    ax.legend(loc="best", fontsize=8)


cols = ["late_%_ES_after_NQ_shock", "late_%_NQ_after_ES_shock"]
lo, hi = min(by_year[cols].min().min(), 0), max(by_year[cols].max().max(), 0)
pad = 0.1 * (hi - lo) + 0.1
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
for ax, session in zip(axes[0], ("regular", "overnight")):
    d = by_year[by_year["session"] == session]
    two_direction_lines(ax, d, "year", f"{session.capitalize()} hours: share of the reaction "
                        "that arrives late", "")
    ax.set_ylim(lo - pad, hi + pad)
for session, style in (("regular", "-"), ("overnight", "--")):
    d = by_year[by_year["session"] == session]
    axes[1, 0].plot(d["year"], d["net_lead_NQ_pp"], color=INK_2, linestyle=style, marker="o",
                    markersize=4, label=f"{session} hours")
    axes[1, 1].plot(d["year"], d["corr_same_bar"], color=INK_2, linestyle=style, marker="o",
                    markersize=4, label=f"{session} hours")
axes[1, 0].axhline(0, color=INK_2, linewidth=0.8)
axes[1, 0].set_title("Net lead of NQ over ES (above 0 = NQ moves first)")
axes[1, 0].set_ylabel("percentage points")
axes[1, 1].set_title("Same-minute correlation of NQ and ES returns")
axes[1, 1].set_ylabel("correlation")
for ax in axes[1]:
    ax.set_xlabel("year")
    ax.legend(loc="best", fontsize=8)
fig.suptitle("NQ vs ES futures, 1-minute bars: information diffusion year by year",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "05_nq_es_by_year.png")

# ---------------------------------------------------------------------------
# B) Hour of day (New York time), all years pooled, in session order 18:00 -> 16:00
# ---------------------------------------------------------------------------
rows = []
for hour in list(range(18, 24)) + list(range(0, 17)):
    sub = r[r.index.hour == hour]
    if sub.dropna().shape[0] < 5_000:
        continue
    rows.append(measure(sub, hour=hour))
    tick(f"hour {hour:02d}:00 done")
by_hour = pd.DataFrame(rows)
by_hour["position"] = range(len(by_hour))
by_hour.to_csv(config.RESULTS / "05_nq_es_by_hour.csv", index=False)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
two_direction_lines(axes[0], by_hour, "position", "By hour of day (New York time, all years): "
                    "share of the reaction that arrives late", "")
axes[1].bar(by_hour["position"], by_hour["net_lead_NQ_pp"], color=BLUE, width=0.7)
axes[1].axhline(0, color=INK_2, linewidth=0.8)
axes[1].set_ylabel("percentage points")
axes[1].set_title("Net lead of NQ over ES by hour (above 0 = NQ moves first)")
axes[2].bar(by_hour["position"], by_hour["corr_same_bar"], color="#b7d3f6", width=0.7)
axes[2].set_ylim(max(by_hour["corr_same_bar"].min() - 0.05, 0), 1)
axes[2].set_ylabel("correlation")
axes[2].set_title("Same-minute correlation by hour")
axes[2].set_xticks(by_hour["position"])
axes[2].set_xticklabels([f"{h:02d}" for h in by_hour["hour"]])
axes[2].set_xlabel("hour starting (NY). Session opens 18:00; cash market 09:30-16:00")
fig.tight_layout()
save(fig, "05_nq_es_by_hour.png")

# ---------------------------------------------------------------------------
# C) Time scale: aggregate 1-minute closes into 2, 3, 5, 10, 15, 30-minute bars
#    (last three years, so it describes today's market)
# ---------------------------------------------------------------------------
recent = px[px.index >= px.index[-1] - pd.DateOffset(years=3)]
rows = []
for k in (1, 2, 3, 5, 10, 15, 30):
    bars = recent.resample(f"{k}min", label="left", closed="left").last().dropna()
    rk = log_returns(bars, max_gap=f"{k}min")
    rth_k = rth_mask(rk.index)
    for session, mask in (("regular", rth_k), ("overnight", ~rth_k)):
        rows.append(measure(rk[mask], freq=f"{k}min", bar_minutes=k, session=session))
    tick(f"{k}-minute bars done")
by_scale = pd.DataFrame(rows)
by_scale.to_csv(config.RESULTS / "05_nq_es_by_timescale.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, session in zip(axes, ("regular", "overnight")):
    d = by_scale[by_scale["session"] == session]
    two_direction_lines(ax, d, "bar_minutes", f"{session.capitalize()} hours, last 3 years",
                        "bar size (minutes, log scale)")
    ax.plot(d["bar_minutes"], d["net_lead_NQ_pp"], color=INK_2, marker="s", markersize=4,
            label="net lead of NQ (pp)")
    ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(d["bar_minutes"])
    ax.set_xticklabels(d["bar_minutes"])
fig.suptitle("Does the lead-lag survive at longer bar sizes?", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "05_nq_es_by_timescale.png")

# ---------------------------------------------------------------------------
# D) The biggest minutes in ten years - nearly all of them are scheduled news
# ---------------------------------------------------------------------------
top = r["NQ"].abs().nlargest(20).index
moves = pd.DataFrame({
    "NQ_%": 100 * r.loc[top, "NQ"],
    "ES_%": 100 * r.loc[top, "ES"],
    "NQ_next_min_%": 100 * r["NQ"].shift(-1, freq="1min").reindex(top).to_numpy(),
    "ES_next_min_%": 100 * r["ES"].shift(-1, freq="1min").reindex(top).to_numpy(),
}, index=top)
moves.index = moves.index.strftime("%Y-%m-%d %H:%M %a")
moves.to_csv(config.RESULTS / "05_largest_moves.csv")
print("\nThe 20 largest 1-minute NQ moves (NY time):")
print(moves.round(3).to_string())
tick("finished")
