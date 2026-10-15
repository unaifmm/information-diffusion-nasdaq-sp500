"""
STEP 13 - News: what happens to information diffusion around macro releases?

Scheduled announcements are the cleanest "information shocks" there are: we
know exactly when the news arrives and how surprising it was. Four releases,
May 2016 -> today, 1-minute bars:
    CPI (08:30 New York)   payrolls (08:30)   ISM Manufacturing (10:00)   FOMC (14:00)
At 08:30 only the futures trade, so NQ vs ES is studied for all four;
futures vs ETFs only at 10:00 and 14:00.

  A) Timing check. The vendor's calendar times are an hour late for 2016-2025
     (see src/events.py). Prices settle which clock is right: where does the
     biggest move of the day around the release fall?
  B) Reaction profile: how much bigger are the minutes after a release than the
     same minutes on a normal day, and how fast does it fade?
  C) Surprise: do prices move with the SIZE and SIGN of the surprise (actual -
     consensus)? A check that the event data and timing are right, and the
     classic event-study result.
  D) Diffusion in news windows: same-minute correlation and the net lead of NQ
     over ES in the 20 minutes after each release, vs the same clock time on
     nearby normal days. 95% CIs by bootstrap over events.

    python 08_event_study.py        (about 3-4 minutes)

Outputs
  figures/08_event_study.png, 08_news_diffusion.png
  results/08_events.csv, 08_timing_check.csv, 08_reaction_profile.csv,
          08_surprise_regression.csv, 08_news_vs_normal.csv
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config
from src.data import log_returns
from src.events import load_events, stack_windows
from src.leadlag import net_lead
from src.lse import load_lse
from src.plots import AQUA, BLUE, INK, INK_2, ORANGE, YELLOW, save, set_style

set_style()
pd.set_option("display.width", 220, "display.max_columns", 30)
T0 = time.time()
RNG = np.random.default_rng(13)
KINDS = ["CPI", "Payrolls", "ISM", "FOMC"]
COLOR = {"CPI": BLUE, "Payrolls": ORANGE, "ISM": AQUA, "FOMC": YELLOW}
PERIODS = {"2016-2019": (2016, 2019), "2020-2021": (2020, 2021), "2022-2026": (2022, 2026)}
BEFORE, AFTER = 5, 20            # news window: 5 minutes before -> 20 minutes after the release


def tick(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


px = load_lse(["NQ", "ES", "QQQ", "SPY"], how="outer")
ev = load_events()
ev = ev[(ev["time"] > px.index[0]) & (ev["time"] < px.index[-1])].reset_index(drop=True)
fut = px[["NQ", "ES"]].dropna()
r = log_returns(fut, max_gap="1min")
print(f"{len(ev)} releases: " + ", ".join(f"{k} {n}" for k, n in ev["kind"].value_counts()[KINDS].items()))

event_days = set(ev["date"].dt.date)
trade_days = pd.DatetimeIndex(sorted(set(r.index.tz_localize(None).normalize())))
normal_days = trade_days[(trade_days.dayofweek < 5) & ~pd.Index(trade_days.date).isin(event_days)]


def at_clock(day, t) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {t:%H:%M}", tz=config.TZ)


# ---------------------------------------------------------------------------
# A) Timing check: official schedule vs the vendor's timestamps
# ---------------------------------------------------------------------------
absr = r["NQ"].abs()


def biggest_minute(t0: pd.Timestamp, span: int = 90) -> float:
    """Minutes between t0 and the largest |NQ return| within +/- span minutes."""
    w = absr.loc[t0 - pd.Timedelta(minutes=span): t0 + pd.Timedelta(minutes=span)].dropna()
    return (w.idxmax() - t0).total_seconds() / 60 if len(w) else np.nan


ev["peak_vs_official_min"] = [biggest_minute(t) for t in ev["time"]]
ev["peak_vs_vendor_min"] = [biggest_minute(t) for t in ev["vendor_time"]]
timing = ev.groupby("kind").apply(lambda g: pd.Series({
    "releases": len(g),
    "vendor_time_1h_late_%": 100 * ((g["vendor_time"] - g["time"]) == pd.Timedelta(hours=1)).mean(),
    "peak_at_official_time_%": 100 * (g["peak_vs_official_min"] == 0).mean(),
    "peak_at_vendor_time_%": 100 * (g["peak_vs_vendor_min"] == 0).mean()}), include_groups=False).loc[KINDS]
timing.to_csv(config.RESULTS / "08_timing_check.csv")
print("\nA) Which clock is right? Share of releases whose biggest NQ minute (+/-90 min) falls exactly on:")
print(timing.round(0).to_string())

# ---------------------------------------------------------------------------
# B) Reaction profile: |return| minute by minute vs the same minutes on normal days
# ---------------------------------------------------------------------------
offsets = np.arange(-15, 76)
normal = absr[pd.Index(absr.index.tz_localize(None).normalize()).isin(normal_days)]
baseline_by_clock = normal.groupby(normal.index.hour * 60 + normal.index.minute).mean()
profile = []
for kind in KINDS:
    times = ev.loc[ev["kind"] == kind, "time"]
    clock0 = times.iloc[0].hour * 60 + times.iloc[0].minute
    for k in offsets:
        vals = absr.reindex(times + pd.Timedelta(minutes=int(k)))
        profile.append({"kind": kind, "minute": k, "news_bp": 1e4 * vals.mean(),
                        "normal_bp": 1e4 * baseline_by_clock.get(clock0 + k, np.nan)})
profile = pd.DataFrame(profile)
profile["ratio"] = profile["news_bp"] / profile["normal_bp"]
profile.to_csv(config.RESULTS / "08_reaction_profile.csv", index=False)
print("\nB) Release minute vs a normal day at the same time (mean |NQ 1-minute return|, bp):")
p0 = profile[profile["minute"] == 0].set_index("kind")
for kind in KINDS:
    first5 = profile[(profile["kind"] == kind) & profile["minute"].between(0, 4)]
    print(f"  {kind:9} minute 0: {p0.loc[kind, 'news_bp']:5.1f} bp vs {p0.loc[kind, 'normal_bp']:4.1f} "
          f"({p0.loc[kind, 'ratio']:4.1f}x); minutes 0-4 together: {first5['ratio'].mean():.1f}x normal")
tick("timing and reaction profile done")

# ---------------------------------------------------------------------------
# C) Surprise: does the 5-minute move follow the size and sign of the surprise?
# ---------------------------------------------------------------------------
lp = np.log(px["ES"])
ev["es_move_5m_bp"] = 1e4 * (lp.reindex(ev["time"] + pd.Timedelta(minutes=4)).to_numpy()
                             - lp.reindex(ev["time"] - pd.Timedelta(minutes=1)).to_numpy())
surprise_rows = []
for kind in ["CPI", "Payrolls", "ISM"]:
    d = ev[(ev["kind"] == kind)].dropna(subset=["surprise_z", "es_move_5m_bp"])
    signed = sm.OLS(d["es_move_5m_bp"], sm.add_constant(d[["surprise_z"]])).fit(cov_type="HC1")
    size = sm.OLS(d["es_move_5m_bp"].abs(), sm.add_constant(d["surprise_z"].abs())).fit(cov_type="HC1")
    surprise_rows.append({"kind": kind, "releases": len(d),
                          "bp_per_sd_of_surprise": signed.params["surprise_z"], "t": signed.tvalues["surprise_z"],
                          "abs_move_bp_per_abs_sd": size.params.iloc[1], "t_abs": size.tvalues.iloc[1],
                          "mean_abs_move_bp": d["es_move_5m_bp"].abs().mean()})
surprise = pd.DataFrame(surprise_rows)
surprise.to_csv(config.RESULTS / "08_surprise_regression.csv", index=False)
print("\nC) ES move from 1 minute before to 4 minutes after the release, on the standardised surprise:")
print(surprise.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# D) Diffusion in news windows vs normal windows
# ---------------------------------------------------------------------------
def nearby_normal_times(times, n_each_side: int = 5) -> list:
    """Same clock time on the 5 normal days before and the 5 after each release."""
    out = []
    for t in times:
        pos = normal_days.searchsorted(t.tz_localize(None).normalize())
        for d in normal_days[max(pos - n_each_side, 0): pos + n_each_side]:
            out.append(at_clock(d, t))
    return out


def window_block(returns: pd.DataFrame, times) -> np.ndarray:
    """Windows as an array (event, bar, column) so the bootstrap can resample events fast."""
    w = stack_windows(returns, times, BEFORE, AFTER).drop(columns="k")
    return w.to_numpy().reshape(len(times), BEFORE + AFTER, w.shape[1])


def as_frame(block: np.ndarray, cols) -> pd.DataFrame:
    n, m, _ = block.shape
    base = pd.Timestamp("2000-01-03", tz=config.TZ)
    idx = (base + pd.to_timedelta(np.repeat(np.arange(n), m), unit="D")
           + pd.to_timedelta(np.tile(np.arange(m), n), unit="min"))
    return pd.DataFrame(block.reshape(n * m, -1), index=idx, columns=cols)


def diffusion_cell(returns: pd.DataFrame, a: str, b: str, times, label: dict, n_boot: int = 200) -> dict:
    cols = [a, b]
    news = window_block(returns[cols], list(times))
    normal_ = window_block(returns[cols], nearby_normal_times(times))
    after = slice(BEFORE, None)                                   # bars from the release on

    def stats(block):
        f = as_frame(block, cols)
        return net_lead(f, a, b, var_max_lags=3)[f"net_lead_{a}_pp"]

    boot = [stats(news[RNG.integers(0, len(news), len(news))]) for _ in range(n_boot)]
    flat = lambda blk: pd.DataFrame(blk[:, after].reshape(-1, 2), columns=cols).dropna()
    fn, fc = flat(news), flat(normal_)
    return {**label, "releases": len(news),
            "corr_same_minute_news": fn[a].corr(fn[b]), "corr_same_minute_normal": fc[a].corr(fc[b]),
            "net_lead_news_pp": stats(news), "ci_low": np.percentile(boot, 2.5),
            "ci_high": np.percentile(boot, 97.5), "net_lead_normal_pp": stats(normal_)}


rows = []
for name, (y0, y1) in PERIODS.items():
    sel = ev[ev["time"].dt.year.between(y0, y1)]
    rows.append(diffusion_cell(r, "NQ", "ES", sel["time"], {"pair": "NQ vs ES", "releases_in": "all four",
                                                             "period": name}))
    tick(f"NQ vs ES, all releases, {name}")
for kind in KINDS:
    rows.append(diffusion_cell(r, "NQ", "ES", ev.loc[ev["kind"] == kind, "time"],
                               {"pair": "NQ vs ES", "releases_in": kind, "period": "2016-2026"}))
    tick(f"NQ vs ES, {kind}")
rth = ev[ev["kind"].isin(["ISM", "FOMC"])]
for fut_, etf in (("NQ", "QQQ"), ("ES", "SPY")):
    rr = log_returns(px[[fut_, etf]].dropna(), max_gap="1min")
    rows.append(diffusion_cell(rr, fut_, etf, rth["time"], {"pair": f"{fut_} vs {etf}",
                                                           "releases_in": "ISM + FOMC", "period": "2016-2026"}))
    tick(f"{fut_} vs {etf}, ISM + FOMC")
news = pd.DataFrame(rows)
news.to_csv(config.RESULTS / "08_news_vs_normal.csv", index=False)
print("\nD) The 20 minutes after a release vs the same time on nearby normal days")
print("   (net lead of the first market in pp, 95% bootstrap CI over releases):")
print(news.round(3).to_string(index=False))

ev.drop(columns=["vendor_time"]).assign(vendor_time=ev["vendor_time"].astype(str), time=ev["time"].astype(str)) \
  .to_csv(config.RESULTS / "08_events.csv", index=False)

# ---------------------------------------------------------------------------
# Figure 1: reaction profile + timing check
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=False)
for ax, kind in zip(axes, KINDS):
    d = profile[profile["kind"] == kind]
    clock = ev.loc[ev["kind"] == kind, "time"].iloc[0].strftime("%H:%M")
    ax.plot(d["minute"], d["normal_bp"], color=INK_2, linewidth=1.2, linestyle="--", label="normal day, same time")
    ax.plot(d["minute"], d["news_bp"], color=COLOR[kind], linewidth=1.8, label="release day")
    ax.axvline(60, color=INK_2, linewidth=0.9, linestyle=":")
    top = d["news_bp"].max()
    note = "vendor's timestamp\n(= 09:30 cash open,\nnormal days jump too)" if clock == "08:30" else "vendor's\ntimestamp"
    ax.text(58, top * 0.97, note, fontsize=7.5, color=INK_2, va="top", ha="right")
    if kind == "FOMC":
        ax.annotate("14:30 press\nconference", xy=(32, d.loc[d["minute"] == 32, "news_bp"].iloc[0]),
                    xytext=(36, top * 0.75), fontsize=7.5, color=INK_2,
                    arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7))
    ax.set_title(f"{kind} ({clock} New York)")
    ax.set_xlabel("minutes after the official release time")
axes[0].set_ylabel("mean |1-minute NQ return| (bp)")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles[::-1], labels_[::-1], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Prices jump at the official release minute, not at the vendor's timestamp one hour later; "
             "the extra volatility fades within about 15-30 minutes", x=0.01, ha="left", fontsize=11,
             fontweight="bold")
fig.tight_layout()
save(fig, "08_event_study.png")

# ---------------------------------------------------------------------------
# Figure 2: surprise -> move (CPI), and the net lead in news vs normal windows
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
d = ev[(ev["kind"] == "CPI")].dropna(subset=["surprise_z", "es_move_5m_bp"])
ax1.scatter(d["surprise_z"], d["es_move_5m_bp"], s=18, color=BLUE, alpha=0.75, edgecolor="none")
row = surprise.set_index("kind").loc["CPI"]
xs = np.linspace(d["surprise_z"].min(), d["surprise_z"].max(), 10)
fit = sm.OLS(d["es_move_5m_bp"], sm.add_constant(d[["surprise_z"]])).fit()
ax1.plot(xs, fit.params["const"] + fit.params["surprise_z"] * xs, color=INK, linewidth=1.4)
ax1.axhline(0, color=INK_2, linewidth=0.8)
ax1.axvline(0, color=INK_2, linewidth=0.8)
ax1.set_xlabel("CPI surprise (actual - consensus), in standard deviations")
ax1.set_ylabel("ES move, 1 min before -> 4 min after (bp)")
ax1.set_title(f"Hotter-than-expected CPI -> stocks fall: {row['bp_per_sd_of_surprise']:+.0f} bp "
              f"per sd (t = {row['t']:.1f})")

d2 = news.iloc[: len(PERIODS) + len(KINDS)]
labels = [f"{r_['releases_in']}\n{r_['period']}" if r_["releases_in"] == "all four" else f"{r_['releases_in']}\n2016-2026"
          for _, r_ in d2.iterrows()]
x = np.arange(len(d2))
ax2.errorbar(x - 0.12, d2["net_lead_news_pp"], yerr=[d2["net_lead_news_pp"] - d2["ci_low"],
             d2["ci_high"] - d2["net_lead_news_pp"]], fmt="o", color=BLUE, ecolor=BLUE, capsize=3,
             label="20 minutes after a release (95% CI)")
ax2.scatter(x + 0.12, d2["net_lead_normal_pp"], marker="s", color=INK_2, s=22,
            label="same time on nearby normal days")
ax2.axhline(0, color=INK_2, linewidth=0.8)
ax2.set_xticks(x, labels, fontsize=8.5)
ax2.set_ylabel("net lead of NQ over ES (pp)")
ax2.set_title("No market leads around news at 1-minute resolution")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
fig.tight_layout()
save(fig, "08_news_diffusion.png")

print(f"\nDone in {time.time() - T0:.0f}s.")
