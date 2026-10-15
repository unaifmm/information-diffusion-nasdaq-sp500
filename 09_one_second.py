"""
STEP 14 - Zoom to 1-second data: is there a lead measured in SECONDS?

At 1-minute bars the leads we found are small or gone (05-07): news reaches
NQ, ES, QQQ and SPY within the same minute. Arbitrage today works in seconds or
less, so we look again with 1-second bars (regular hours, July 2026 -> today).

  A) Activity: in how many seconds does each market actually trade?
  B) The staleness trap. On a 1-second grid, a market that did not trade in a
     second keeps its old price, so its move shows up a second late - a FAKE
     lead. We prove it with a placebo: a copy of NQ that just trades in fewer
     seconds "lags" NQ. Then we show the fix - use only seconds in which BOTH
     markets traded - removes the fake lead from the placebo.
  C) Cross-correlation at 1-second lags (with the fix), for NQ vs ES, NQ vs QQQ,
     ES vs SPY.
  D) Lead-lag asymmetry day by day: corr(a_t, b_t+1) - corr(b_t, a_t+1), with a
     95% CI across days and the share of days with the same sign.
  E) What a big move looks like second by second: after the leader's largest
     1-second moves, how much has the follower moved after 0, 1, 2 ... seconds?

    python 09_one_second.py        (about 2 minutes)

Outputs
  figures/09_one_second_ccf.png, 09_one_second_paths.png
  results/09_activity.csv, 09_placebo.csv, 09_ccf.csv, 09_daily_asymmetry.csv, 09_paths.csv
"""
import time
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import log_returns, previous_tick_grid
from src.leadlag import _shift, fresh_cross_correlation, mean_with_ci, net_lead
from src.lse import load_seconds
from src.plots import AQUA, BLUE, INK_2, ORANGE, YELLOW, plot_ccf, save, set_style

warnings.filterwarnings("ignore", category=RuntimeWarning)
set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()
PAIRS = [("NQ", "ES"), ("NQ", "QQQ"), ("ES", "SPY")]
COLOR = {"NQ": BLUE, "ES": ORANGE, "QQQ": AQUA, "SPY": YELLOW}


def tick(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


raw = load_seconds(["NQ", "ES", "QQQ", "SPY"])
prices, traded = previous_tick_grid(raw)
logp = np.log(prices)
r = log_returns(prices, max_gap="1s")
days = sorted(set(prices.index.normalize()))
print(f"{len(days)} trading days, {days[0].date()} -> {days[-1].date()}, "
      f"{len(prices):,} seconds on the regular-hours grid")

# ---------------------------------------------------------------------------
# A) Activity
# ---------------------------------------------------------------------------
activity = pd.DataFrame({"share_of_seconds_with_a_trade_%": 100 * traded.mean(),
                         "days_with_data": traded.groupby(traded.index.normalize()).any().sum()})
activity.to_csv(config.RESULTS / "09_activity.csv")
print("\nA) Share of regular-hours seconds in which each market traded:")
print(activity.round(1).to_string())


# ---------------------------------------------------------------------------
# B) The staleness trap: placebo with no real lead
# ---------------------------------------------------------------------------
def lag1_asymmetry_by_day(ra: pd.Series, rb: pd.Series, mask: pd.Series) -> pd.DataFrame:
    """Per day: corr(a_t, b_t+1), corr(b_t, a_t+1) over seconds t in `mask`."""
    d = pd.DataFrame({"a": ra, "b": rb, "a1": _shift(ra, -1, "1s"), "b1": _shift(rb, -1, "1s")})[mask].dropna()
    g = d.groupby(d.index.normalize())
    out = pd.DataFrame({"a_then_b": g.apply(lambda x: x["a"].corr(x["b1"])),
                        "b_then_a": g.apply(lambda x: x["b"].corr(x["a1"]))})
    out["asymmetry"] = out["a_then_b"] - out["b_then_a"]
    return out


rng = np.random.default_rng(14)
placebo = []
nq_share = traded["NQ"].mean()
for target in (activity.loc["ES", "share_of_seconds_with_a_trade_%"] / 100, 0.70):
    # the copy trades only when NQ traded, and then only with probability target / NQ's share
    seen = pd.Series(rng.random(len(prices)) < target / nq_share, index=prices.index) & traded["NQ"]
    keep = seen.mean()
    copy = prices["NQ"].where(seen).groupby(prices.index.normalize()).ffill()
    rr = log_returns(pd.DataFrame({"NQ": prices["NQ"], "COPY": copy}), max_gap="1s")
    for label, mask in (("all seconds", rr.notna().all(axis=1)), ("both traded in second t", seen)):
        a = mean_with_ci(lag1_asymmetry_by_day(rr["NQ"], rr["COPY"], mask)["asymmetry"])
        placebo.append({"copy_trades_in_%_of_seconds": round(100 * keep), "sample": label,
                        "fake_asymmetry": a["mean"], "ci_low": a["ci_low"], "ci_high": a["ci_high"]})
    nl = net_lead(rr, "NQ", "COPY", freq="1s")["net_lead_NQ_pp"]
    placebo.append({"copy_trades_in_%_of_seconds": round(100 * keep), "sample": "all seconds (VAR net lead, pp)",
                    "fake_asymmetry": nl, "ci_low": np.nan, "ci_high": np.nan})
    tick(f"placebo, copy trades in {100 * keep:.0f}% of seconds")
placebo = pd.DataFrame(placebo)
placebo.to_csv(config.RESULTS / "09_placebo.csv", index=False)
print("\nB) Placebo - NQ against a copy of itself that trades in fewer seconds (the truth is NO lead):")
print(placebo.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# C) Cross-correlation at 1-second lags, both markets traded in the first second
# ---------------------------------------------------------------------------
ccfs = {}
for a, b in PAIRS:
    fresh = traded[a] & traded[b]
    ccfs[(a, b)] = fresh_cross_correlation(r[a], r[b], fresh, max_lag=5, freq="1s")
    tick(f"cross-correlation {a} vs {b}")
ccf_table = pd.concat({f"{a} vs {b}": c for (a, b), c in ccfs.items()}, names=["pair"]).reset_index()
ccf_table.to_csv(config.RESULTS / "09_ccf.csv", index=False)
print("\nC) corr(a_t, b_t+k) at 1-second lags (seconds where both traded):")
print(ccf_table.pivot(index="pair", columns="lag", values="corr").round(3).to_string())

# ---------------------------------------------------------------------------
# D) Day by day
# ---------------------------------------------------------------------------
daily, summary = [], []
for a, b in PAIRS:
    d = lag1_asymmetry_by_day(r[a], r[b], traded[a] & traded[b]).assign(pair=f"{a} vs {b}")
    daily.append(d)
    ci = mean_with_ci(d["asymmetry"])
    summary.append({"pair": f"{a} vs {b}", "corr(a_t,b_t+1)": d["a_then_b"].mean(),
                    "corr(b_t,a_t+1)": d["b_then_a"].mean(), "asymmetry": ci["mean"],
                    "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "t": ci["t_stat"], "days": ci["n"],
                    "days_a_first_%": 100 * (d["asymmetry"] > 0).mean()})
daily = pd.concat(daily)
daily.to_csv(config.RESULTS / "09_daily_asymmetry.csv")
summary = pd.DataFrame(summary)
summary.to_csv(config.RESULTS / "09_asymmetry_summary.csv", index=False)
print("\nD) Lead-lag asymmetry at 1 second, mean of daily estimates (> 0: the first market moves first):")
print(summary.round(4).to_string(index=False))
tick("daily asymmetry done")

# ---------------------------------------------------------------------------
# E) After the leader's largest 1-second moves: the path, second by second
# ---------------------------------------------------------------------------
H = np.arange(-5, 11)
paths = []
for leader, follower in PAIRS:
    fresh = traded[leader] & traded[follower] & r[[leader, follower]].notna().all(axis=1)
    rl = r.loc[fresh, leader]
    thr = rl.abs().groupby(rl.index.normalize()).transform(lambda s: s.quantile(0.999))
    ev = rl.index[(rl.abs() >= thr).to_numpy()]
    s = np.sign(rl.loc[ev]).to_numpy()
    base = ev - pd.Timedelta(seconds=1)                                  # one second before the move
    for h in H:
        row = {"leader": leader, "follower": follower, "second": h, "events": len(ev)}
        for name in (leader, follower):
            move = logp[name].reindex(ev + pd.Timedelta(seconds=int(h))).to_numpy() - logp[name].reindex(base).to_numpy()
            row[f"{'leader' if name == leader else 'follower'}_bp"] = 1e4 * np.nanmean(s * move)
        paths.append(row)
paths = pd.DataFrame(paths)
paths.to_csv(config.RESULTS / "09_paths.csv", index=False)
print("\nE) After the leader's top-0.1% 1-second moves: cumulative moves (bp) from 1 second before:")
print(paths[paths["second"].isin([-1, 0, 1, 2, 5, 10])]
      .pivot_table(index=["leader", "follower"], columns="second", values=["leader_bp", "follower_bp"], sort=False)
      .round(2).to_string())

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, (a, b) in zip(axes, PAIRS):
    plot_ccf(ax, ccfs[(a, b)], a, b, title=f"{a} vs {b}", ylim=0.12)
    ax.set_xlabel(f"lag k in seconds   (k > 0: {a} first,  k < 0: {b} first)")
fig.suptitle("At 1-second resolution the futures lead the ETFs, and NQ leads ES "
             "(seconds where both markets traded; tall bars at k = +1)", x=0.01, ha="left",
             fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "09_one_second_ccf.png")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
for ax, (leader, follower) in zip(axes, PAIRS):
    d = paths[(paths["leader"] == leader) & (paths["follower"] == follower)]
    ax.plot(d["second"], d["leader_bp"], color=COLOR[leader], marker="o", markersize=3, label=f"{leader} (leader)")
    ax.plot(d["second"], d["follower_bp"], color=COLOR[follower], marker="o", markersize=3,
            label=f"{follower} (follower)")
    ax.axvline(0, color=INK_2, linewidth=0.8, linestyle=":")
    ax.set_title(f"After {leader}'s biggest 1-second moves")
    ax.set_xlabel("seconds after the move")
    ax.set_xticks(range(-5, 11, 5))
    ax.legend(loc="lower right", fontsize=8.5)
axes[0].set_ylabel("cumulative move (bp), from 1 s before")
ax = axes[3]
for a, b in PAIRS:
    d = daily[daily["pair"] == f"{a} vs {b}"]
    ax.plot(d.index, d["asymmetry"], color=COLOR[b], linewidth=1.2, label=f"{a} vs {b}")
ax.axhline(0, color=INK_2, linewidth=0.8)
ax.set_title("Lead-lag asymmetry, day by day")
ax.set_ylabel("corr(a$_t$, b$_{t+1}$) - corr(b$_t$, a$_{t+1}$)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5, ncol=3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.suptitle("Almost all of the follower's reaction happens in the SAME second; the small rest arrives within "
             "1-2 seconds - and the futures lead on (nearly) every day", x=0.01, ha="left", fontsize=11,
             fontweight="bold")
fig.tight_layout()
save(fig, "09_one_second_paths.png")

print(f"\nDone in {time.time() - T0:.0f}s.")
