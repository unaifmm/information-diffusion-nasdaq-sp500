"""
STEP 3 - Validation II: do our measures survive REAL market noise?

The simple simulation (02_validate_simulation.py) has clean, well-behaved returns.
Real 1-minute futures returns are messier: bid-ask bounce, thin overnight
trading, volatility bursts, and prices that can only move in whole ticks.
Do our measures still find a lead we KNOW is there, and stay quiet when there
is none?

We take the real NQ data (all its messiness included) and build an artificial
follower from it. Three scenarios, where we know the truth:

  "3% late"       follower absorbs 97% of every NQ move in the same minute and
                  3% one minute later        -> true net lead of NQ = +3 pp
  "no lead"       same recipe, 100% / 0%     -> true net lead = 0
  "no lead, ES tick"  NO lead at all, but the follower's price can only move
                  in steps of 0.25 index points, like ES (relative to its price,
                  ES's tick is about 2-4x coarser than NQ's)  -> true lead = 0

The third scenario asks: can a coarser tick ALONE make one market look like a
follower? It answers which of our measures can be trusted on real NQ-ES data.

Needs only the NQ data:   python 03_validate_real_noise.py   (about 2-3 minutes)
Outputs: figures/03_validation_real_noise.png, results/03_validation_real_noise.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import log_returns, rth_mask
from src.leadlag import diffusion_row
from src.lse import load_lse
from src.plots import AQUA, BLUE, INK_2, ORANGE, save, set_style

set_style()
rng = np.random.default_rng(7)

nq_px = load_lse(["NQ"])["NQ"]
r_nq = log_returns(nq_px.to_frame(), max_gap="1min")["NQ"]
r_prev = r_nq.shift(1, freq="1min").reindex(r_nq.index)     # NQ one minute earlier
noise = pd.Series(rng.standard_normal(len(r_nq)) * 1e-4, index=r_nq.index)

# ES-like price: same information at the same instant, rounded to a 0.25-point grid
es_like_px = np.round(2.5 * nq_px ** 0.8 / 0.25) * 0.25
r_es_like = log_returns(pd.DataFrame({"NQ": nq_px, "F": es_like_px}), max_gap="1min")["F"]

rth = rth_mask(r_nq.index)

scenarios = {
    "3% late": (0.8 * (0.97 * r_nq + 0.03 * r_prev) + noise, 3.0),
    "no lead": (0.8 * r_nq + noise, 0.0),
    "no lead, ES tick": (r_es_like, 0.0),
}

rows = []
for scenario, (follower, truth) in scenarios.items():
    data = pd.DataFrame({"NQ": r_nq, "FOLLOWER": follower})
    for year in sorted(set(data.index.year)):
        for session, mask in (("regular", rth), ("overnight", ~rth)):
            sub = data[(data.index.year == year) & mask]
            if sub.dropna().shape[0] < 5_000:
                continue
            rows.append(diffusion_row(sub, "NQ", "FOLLOWER", freq="1min", scenario=scenario,
                                      true_lead_pp=truth, year=year, session=session))
    print(f"{scenario}: done", flush=True)

res = pd.DataFrame(rows)
res.to_csv(config.RESULTS / "03_validation_real_noise.csv", index=False)

summary = res.groupby(["scenario", "session"]).agg(
    true_lead=("true_lead_pp", "first"),
    net_lead_mean=("net_lead_NQ_pp", "mean"),
    net_lead_std=("net_lead_NQ_pp", "std"),
    granger_says_NQ_leads=("granger_p_NQ->FOLLOWER", lambda p: f"{(p < 0.01).sum()}/{len(p)} years"),
    granger_says_FOLLOWER_leads=("granger_p_FOLLOWER->NQ", lambda p: f"{(p < 0.01).sum()}/{len(p)} years"),
).round(2)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\nWhat each measure says, across 11 years (Granger counted as 'leads' when p < 0.01):")
print(summary.to_string())
print("\nRead it like this: a trustworthy measure matches the truth in every scenario."
      "\nIf Granger 'finds' NQ leading in the ES-tick scenario, where there is NO lead,"
      "\nthen Granger alone cannot be trusted on NQ vs ES, and the net lead must be used.")

fig, ax = plt.subplots(figsize=(10, 4.8))
colors = {"3% late": BLUE, "no lead": ORANGE, "no lead, ES tick": AQUA}
for (scenario, session), d in res.groupby(["scenario", "session"]):
    ax.plot(d["year"], d["net_lead_NQ_pp"], color=colors[scenario],
            linestyle="-" if session == "regular" else "--", marker="o", markersize=4,
            label=f"{scenario}, {session} hours")
for truth, color in ((3.0, BLUE), (0.0, INK_2)):
    ax.axhline(truth, color=color, linewidth=0.8, linestyle=":")
ax.set_ylabel("estimated net lead of NQ (pp)")
ax.set_xlabel("year")
ax.set_title("Validation on real NQ noise: dotted lines = the truth we built in")
ax.legend(loc="center right", fontsize=8)
save(fig, "03_validation_real_noise.png")
