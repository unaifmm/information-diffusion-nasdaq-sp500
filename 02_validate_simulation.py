"""
STEP 2 - Validation I: do our tools find a lead-lag that we KNOW exists?

Part 1 (two different markets, like NQ and ES):

We simulate two fake markets where we control the truth:
  A) FOLLOWER absorbs 80% of the news in the same minute and 20% one minute later
  B) FOLLOWER absorbs 100% instantly (no lead-lag)

If the toolkit is correct it must:
  - show a significant bar at lag +1 in A and nothing in B,
  - find LEADER -> FOLLOWER Granger causality in A only,
  - estimate ~80% "absorbed in the same bar" in A and ~100% in B.

Part 2 (two prices of the SAME asset, like NQ futures and the QQQ ETF):
  compares our net lead with the textbook price-discovery measures
  (Hasbrouck information share, Gonzalo-Granger component share, Putnins
  information leadership share) on six markets with a known answer.

No data needed:   python 02_validate_simulation.py   (about 30 seconds)
"""
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.leadlag import analyze_pair, verdict
from src.plots import plot_ccf, plot_diffusion, save, set_style
from src.simulate import simulate_pair

set_style()
scenarios = {"A: 80% now, 20% one bar later": 0.8, "B: instant (efficient)": 1.0}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
rows = []
for col, (label, a) in enumerate(scenarios.items()):
    r = simulate_pair(a=a)
    out = analyze_pair(r, "LEADER", "FOLLOWER", freq="1min", max_lag=5)
    s = out["summary"]
    rows.append({
        "scenario": label,
        "true share absorbed in same bar": a,
        "estimated share": round(s["same_bar_share_B_after_A_shock"], 3),
        "corr(LEADER_t, FOLLOWER_t+1)": round(s["corr(A_t,B_t+1)"], 3),
        "robust t": round(s["t_robust(A_t,B_t+1)"], 1),
        "Granger p LEADER->FOLLOWER": f'{s["granger_p_A->B"]:.2g}',
        "Granger p FOLLOWER->LEADER": f'{s["granger_p_B->A"]:.2g}',
        "VAR lags (BIC)": s["var_lags_bic"],
    })
    print(f"\n{label}:\n   " + verdict(s))
    plot_ccf(axes[0, col], out["ccf"], "LEADER", "FOLLOWER", title=f"Scenario {label}", ylim=0.3)
    plot_diffusion(axes[1, col], {"FOLLOWER after LEADER shock": out["diffusion"]["LEADER->FOLLOWER"]},
                   title="How fast the news reaches FOLLOWER")
    axes[1, col].set_ylim(0, 115)

fig.suptitle("Sanity check on simulated data: the toolkit recovers the lead-lag we built in",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "02_validation_simulation.png")

table = pd.DataFrame(rows).set_index("scenario").astype(object)
pd.set_option("display.width", 200, "display.max_columns", 20)
print("\n", table.T, "\n")
table.to_csv(config.RESULTS / "02_validation_simulation.csv")


# ===========================================================================
# Part 2 - same-asset pairs: net lead vs the textbook price-discovery measures
# ===========================================================================
import numpy as np

from src.data import log_returns
from src.leadlag import net_lead
from src.plots import AQUA, BLUE, INK, INK_2, ORANGE
from src.pricediscovery import vecm_price_discovery
from src.simulate import simulate_same_asset_pair

cases = [  # label, w, kappa, noise A, noise B, does A lead?
    ("A leads strongly", 0.5, 0.5, 2e-5, 2e-5, True),
    ("A leads mildly", 0.9, 0.5, 2e-5, 2e-5, True),
    ("A leads mildly, A 5x noisier", 0.9, 0.5, 1e-4, 2e-5, True),
    ("no lead", 1.0, 0.5, 2e-5, 2e-5, False),
    ("no lead, B 5x noisier", 1.0, 0.5, 2e-5, 1e-4, False),
    ("no lead, A 5x noisier", 1.0, 0.5, 1e-4, 2e-5, False),
]
rows = []
for label, w, kappa, na, nb, a_leads in cases:
    logp = simulate_same_asset_pair(w, kappa, na, nb)
    nl = net_lead(log_returns(np.exp(logp), max_gap="1min"), "A", "B")
    pd_ = vecm_price_discovery(logp, lags=5)
    rows.append({"case": label, "truth": "A leads" if a_leads else "no lead",
                 "net_lead_A_pp": nl["net_lead_A_pp"], "CS_A": pd_["CS_A"],
                 "IS_low_A": pd_["IS_low_A"], "IS_high_A": pd_["IS_high_A"], "ILS_A": pd_["ILS_A"],
                 "corr_same_minute": nl["corr_same_bar"]})
pdv = pd.DataFrame(rows)
pdv.to_csv(config.RESULTS / "02_validation_price_discovery.csv", index=False)
print("\nPart 2 - same asset, 1-minute bars. Should say: 'A leads' -> net lead > 0 and shares > 0.5;"
      "\n'no lead' -> net lead ~ 0 and shares ~ 0.5.")
print(pdv.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
ypos = np.arange(len(pdv))[::-1]
colors = [BLUE if t == "A leads" else INK_2 for t in pdv["truth"]]
axes[0].barh(ypos, pdv["net_lead_A_pp"], color=colors, height=0.6)
axes[0].axvline(0, color=INK_2, linewidth=0.8)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([f"{c}  (truth: {t})" for c, t in zip(pdv["case"], pdv["truth"])], fontsize=8)
axes[0].set_xlabel("net lead of A (pp)")
axes[0].set_title("Our net lead: > 0 exactly when A leads")
for y, (lo, hi, cs, ils) in zip(ypos, pdv[["IS_low_A", "IS_high_A", "CS_A", "ILS_A"]].to_numpy()):
    axes[1].plot([lo, hi], [y, y], color=AQUA, linewidth=6, solid_capstyle="butt",
                 label="Hasbrouck IS range" if y == ypos[0] else None)
    axes[1].plot(np.clip(cs, -0.2, 1.2), y, "o", color=ORANGE, markersize=6,
                 label="component share (CS)" if y == ypos[0] else None)
    axes[1].plot(ils, y, "D", color=INK, markersize=5,
                 label="leadership share (ILS)" if y == ypos[0] else None)
axes[1].axvline(0.5, color=INK_2, linewidth=0.8, linestyle="--")
axes[1].set_xlim(-0.25, 1.25)
axes[1].set_xlabel("share of price discovery for A (0.5 = no leader)")
axes[1].set_title("Textbook measures at 1-minute bars: wide or misleading")
axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8)
fig.suptitle("Validation, same-asset pairs: which measure recovers the truth?", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "02_validation_price_discovery.png")
