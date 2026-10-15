"""
STEP 15 - Economic significance: is any lead big enough to TRADE?

Steps 5-7 found leads that are statistically real (ES led NQ before 2020, the
futures lead the ETFs). An interviewer's next question is: "so could you make
money with it?" This script answers it with the trade an arbitrageur would make.

The trade (details in src/economics.py)
  When the leader makes one of its biggest 1-minute moves (top 1% of the month),
  buy the follower in the same direction and close one minute later (also 5).
  The gain we credit to the lead is the HEDGED one (follower minus beta x leader):
  it removes market-wide moves, so it is only the follower CATCHING UP.

  Cost: we compare with the CHEAPEST version of the trade - one round trip in the
  follower only, at a one-tick spread plus fees, no market impact. (Hedging
  would add the leader's costs too; that column is in the tables.) If the lead
  does not beat even this cost, a one-minute trader cannot profit from it.

Parts A-B: regular hours, 1-minute bars, May 2016 -> today. One estimate per
month, then the mean of the months with a 95% confidence interval (as in 06-07).

Part C: the same question at 1-SECOND speed (July 2026 -> today; needs the
1-second data). After the leader's top-0.1% seconds, buy the follower and hold
1, 2, 3 ... 10 seconds. Who could trade it, and how fast would they have to be?
One estimate per day, then the mean of the days with a 95% CI.

    python 10_economic_significance.py        (about 2 minutes)

Outputs
  figures/10_economic_significance.png, 10_economic_significance_1s.png
  results/10_monthly.csv, 10_by_period.csv, 10_threshold_robustness.csv,
          10_seconds_by_direction.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import regular_hours
from src.data import previous_tick_grid
from src.economics import big_move_catchup, round_trip_cost_bp, seconds_catchup
from src.leadlag import mean_with_ci
from src.lse import load_lse, load_seconds
from src.plots import AQUA, BLUE, INK, INK_2, ORANGE, YELLOW, save, set_style

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()

PAIRS = [("NQ", "ES", "NQ vs ES futures"), ("NQ", "QQQ", "NQ futures vs QQQ ETF"),
         ("ES", "SPY", "ES futures vs SPY ETF")]
LEADER_COLOR = {"NQ": BLUE, "ES": ORANGE, "QQQ": AQUA, "SPY": YELLOW}
PERIODS = {"2016-2019": (2016, 2019), "2020-2021": (2020, 2021), "2022-2026": (2022, 2026)}

px = regular_hours(load_lse(["NQ", "ES", "QQQ", "SPY"], how="outer"))
print(f"{len(px):,} regular-hours minutes, {px.index[0].date()} -> {px.index[-1].date()}")


def tick(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# A) Month by month: catch-up after the leader's top-1% minutes, vs cost
# ---------------------------------------------------------------------------
def monthly(q: float) -> pd.DataFrame:
    rows = []
    for a, b, label in PAIRS:
        pair = px[[a, b]].dropna()
        month = pair.index.year * 100 + pair.index.month
        for m in np.unique(month):
            sub = pair[month == m]
            if len(sub) < 2_000:
                continue
            for leader, follower in ((a, b), (b, a)):
                res = big_move_catchup(sub, leader, follower, q=q, horizons=(1, 5))
                rows.append({"pair": label, "leader": leader, "follower": follower,
                             "direction": f"{leader} moves -> trade {follower}",
                             "month": f"{m // 100}-{m % 100:02d}", "year": m // 100, **res})
        tick(f"{label}, top {100 * (1 - q):g}% moves: done")
    return pd.DataFrame(rows)


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    """Mean of the monthly numbers per direction and period, with 95% CIs."""
    out = []
    for (label, direction), g in d.groupby(["pair", "direction"], sort=False):
        for name, (y0, y1) in PERIODS.items():
            gg = g[(g["year"] >= y0) & (g["year"] <= y1)]
            row = {"pair": label, "direction": direction, "period": name, "months": len(gg),
                   "events_per_month": gg["n_events"].mean(),
                   "leader_move_bp": gg["leader_move_bp"].mean(),
                   "follower_same_minute_bp": gg["follower_same_bar_bp"].mean()}
            for h in (1, 5):
                ci = mean_with_ci(gg[f"catchup_{h}_bp"])
                row[f"catchup_{h}m_bp"] = ci["mean"]
                row[f"catchup_{h}m_ci_low"] = ci["ci_low"]
                row[f"catchup_{h}m_ci_high"] = ci["ci_high"]
                row[f"catchup_{h}m_t"] = ci["t_stat"]
            ci = mean_with_ci(gg["follower_next1_bp"])
            row["unhedged_1m_bp"], row["unhedged_1m_t"] = ci["mean"], ci["t_stat"]
            row["cost_bp"] = gg["cost_follower_bp"].mean()
            row["cost_hedged_bp"] = gg["cost_hedged_bp"].mean()
            row["catchup_1m_as_%_of_cost"] = 100 * row["catchup_1m_bp"] / row["cost_bp"]
            row["net_1m_bp"] = row["catchup_1m_bp"] - row["cost_bp"]
            out.append(row)
    return pd.DataFrame(out)


month_df = monthly(0.99)
month_df.to_csv(config.RESULTS / "10_monthly.csv", index=False)
by_period = summarise(month_df)
by_period.to_csv(config.RESULTS / "10_by_period.csv", index=False)

print("\nAfter the leader's top-1% minutes (bp per trade; mean of monthly estimates):")
print("  catch-up = hedged gain from trading the follower one minute later, BEFORE costs")
print("  unhedged = same trade without the hedge (noisier: includes moves of the whole market)")
print("  cost     = cheapest round trip in the follower (1-tick spread + fees)")
print(by_period[["direction", "period", "leader_move_bp", "follower_same_minute_bp", "catchup_1m_bp",
                 "catchup_1m_t", "catchup_5m_bp", "unhedged_1m_bp", "unhedged_1m_t", "cost_bp",
                 "cost_hedged_bp", "catchup_1m_as_%_of_cost"]].round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# B) Robustness: smaller (top 10%) and larger (top 0.1%) moves
# ---------------------------------------------------------------------------
robust = [by_period.assign(threshold="top 1%")]
for q, name in ((0.90, "top 10%"), (0.999, "top 0.1%")):
    robust.append(summarise(monthly(q)).assign(threshold=name))
robust = pd.concat(robust, ignore_index=True)
robust.to_csv(config.RESULTS / "10_threshold_robustness.csv", index=False)
print("\nRobustness - catch-up after 1 minute (bp) vs cost, by size of the leader's move:")
print(robust.pivot_table(index=["direction", "period"], columns="threshold", values="catchup_1m_bp",
                         sort=False)[["top 10%", "top 1%", "top 0.1%"]]
      .join(by_period.set_index(["direction", "period"])["cost_bp"]).round(2).to_string())

# ---------------------------------------------------------------------------
# Figure: catch-up (bars, 95% CI) vs round-trip cost (black dashes)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
x = np.arange(len(PERIODS))
w = 0.36
for ax, (a, b, label) in zip(axes, PAIRS):
    for k, (leader, follower) in enumerate(((a, b), (b, a))):
        d = by_period[(by_period["pair"] == label) & (by_period["direction"].str.startswith(leader + " "))]
        xs = x + (k - 0.5) * w
        err = np.vstack([d["catchup_1m_bp"] - d["catchup_1m_ci_low"], d["catchup_1m_ci_high"] - d["catchup_1m_bp"]])
        ax.bar(xs, d["catchup_1m_bp"], width=w * 0.9, color=LEADER_COLOR[leader],
               label=f"{leader} moves first -> trade {follower}")
        ax.errorbar(xs, d["catchup_1m_bp"], yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
        ax.hlines(d["cost_bp"], xs - w * 0.45, xs + w * 0.45, color=INK, linewidth=2.2,
                  label="cheapest round-trip cost of the traded market\n(1-tick spread + fees)" if k == 0 else None)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x, list(PERIODS))
    ax.set_title(label)
    handles, labels_ = ax.get_legend_handles_labels()
    order = [i for i, l in enumerate(labels_) if "cost" not in l] + \
            [i for i, l in enumerate(labels_) if "cost" in l]
    ax.legend([handles[i] for i in order], [labels_[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=8.5)
axes[0].set_ylabel("bp per trade (1 bp = 0.01%)")
fig.suptitle("After a top-1% one-minute move in one market, the other market's catch-up over the next "
             "minute (bars, 95% CI)\nis smaller than even the cheapest cost of trading it (black dashes)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "10_economic_significance.png")

# ---------------------------------------------------------------------------
# Plain-language summary
# ---------------------------------------------------------------------------
print("\nSummary (top-1% moves, one-minute hold):")
for _, r in by_period.iterrows():
    sig = "significant" if abs(r["catchup_1m_t"]) > 1.96 else "not significant"
    print(f"  {r['direction']:<26} {r['period']}: catch-up {r['catchup_1m_bp']:+.2f} bp ({sig}), "
          f"cost {r['cost_bp']:.2f} bp -> {r['catchup_1m_as_%_of_cost']:.0f}% of cost")
# ---------------------------------------------------------------------------
# C) 1-second speed: the same trade, held for a few seconds
# ---------------------------------------------------------------------------
if not any(config.DATA_LSE_1S.glob("*.parquet")):
    print("\nC) skipped: no 1-second data yet (run 01_download_data.py)")
else:
    prices, traded = previous_tick_grid(load_seconds(["NQ", "ES", "QQQ", "SPY"]))
    logp = np.log(prices)
    H = list(range(0, 11))
    curves, sec_rows = [], []
    for a, b, label in PAIRS:
        for leader, follower in ((a, b), (b, a)):
            ev = seconds_catchup(logp, traded, leader, follower, q=0.999, horizons=H)
            daily = ev.groupby("day").mean()
            cost = round_trip_cost_bp(follower, prices[follower].reindex(ev.index).mean())
            ci = {h: mean_with_ci(daily[f"follower_{h}"]) for h in H[1:]}
            for h in H:
                c = ci.get(h, {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0})
                curves.append({"pair": label, "leader": leader, "follower": follower, "second": h,
                               "gain_bp": c["mean"], "ci_low": c["ci_low"], "ci_high": c["ci_high"]})
            best_h = max(ci, key=lambda h: ci[h]["mean"])
            one_late = max(ci[h]["mean"] for h in H[2:]) - ci[1]["mean"]   # enter 1 s late, best exit
            sec_rows.append({"pair": label, "direction": f"{leader} moves -> trade {follower}",
                             "events": len(ev), "days": len(daily),
                             **{f"gain_{h}s_bp": ci[h]["mean"] for h in (1, 2, 3, 5, 10)},
                             "best_hold_s": best_h, "best_gain_bp": ci[best_h]["mean"],
                             "best_ci_low": ci[best_h]["ci_low"], "best_ci_high": ci[best_h]["ci_high"],
                             "hedged_gain_2s_bp": daily["catch_2"].mean(),
                             "gain_if_1s_late_bp": one_late, "cost_bp": cost,
                             "best_gain_as_%_of_cost": 100 * ci[best_h]["mean"] / cost})
        tick(f"1-second trade, {label}")
    curves = pd.DataFrame(curves)
    seconds = pd.DataFrame(sec_rows)
    seconds.to_csv(config.RESULTS / "10_seconds_by_direction.csv", index=False)
    print("\nC) 1-second speed - buy the follower right after the leader's top-0.1% seconds (bp per trade):")
    print("   gain_<h>s = follower's move over the next h seconds (mean of daily means), before costs")
    print(seconds[["direction", "events", "gain_1s_bp", "gain_2s_bp", "gain_5s_bp", "gain_10s_bp",
                   "best_gain_bp", "best_ci_low", "best_ci_high", "gain_if_1s_late_bp", "cost_bp",
                   "best_gain_as_%_of_cost"]].round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    for ax, (a, b, label) in zip(axes, PAIRS):
        for leader, follower in ((a, b), (b, a)):
            d = curves[(curves["leader"] == leader) & (curves["follower"] == follower)]
            col = LEADER_COLOR[leader]
            ax.plot(d["second"], d["gain_bp"], color=col, marker="o", markersize=3,
                    label=f"{leader} moves first -> trade {follower}")
            ax.fill_between(d["second"], d["ci_low"], d["ci_high"], color=col, alpha=0.12, linewidth=0)
            cost = seconds.loc[seconds["direction"] == f"{leader} moves -> trade {follower}", "cost_bp"].iloc[0]
            ax.axhline(cost, color=col, linewidth=1.3, linestyle="--")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(label)
        ax.set_xlabel("seconds held after the leader's move")
        handles, labels_ = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color=INK_2, linestyle="--"))
        labels_.append("cheapest cost of that trade (same colour)")
        ax.legend(handles, labels_, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=8.5)
    axes[0].set_ylabel("gain before costs, bp per trade (95% CI)")
    fig.suptitle("At 1-second speed the follower's catch-up is about the size of the cheapest cost of trading it "
                 "(dashed lines),\nand only for about 2 seconds: a game for the fastest traders", x=0.01,
                 ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, "10_economic_significance_1s.png")

print(f"\nDone in {time.time() - T0:.0f}s.")
