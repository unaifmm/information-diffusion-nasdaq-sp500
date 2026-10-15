"""1-second tools: previous-tick grid, the staleness trap and the hedged catch-up."""
import numpy as np
import pandas as pd

from src import config
from src.data import log_returns, previous_tick_grid
from src.economics import seconds_catchup
from src.leadlag import fresh_cross_correlation


def day_index(n=23_400, day="2026-09-14"):
    return pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=config.TZ)


def test_previous_tick_grid_fills_within_the_day_only():
    raw = pd.DataFrame({"A": [100.0, 101.0, 200.0]},
                       index=pd.DatetimeIndex(["2026-09-14 09:30:00", "2026-09-14 09:30:02",
                                               "2026-09-15 09:30:01"]).tz_localize(config.TZ))
    p, traded = previous_tick_grid(raw)
    assert len(p) == 2 * 23_400
    assert p.loc["2026-09-14 09:30:01", "A"] == 100.0                        # stale price carried
    assert traded.loc["2026-09-14 09:30:01", "A"] == False                   # noqa: E712
    assert np.isnan(p.loc["2026-09-15 09:30:00", "A"])                       # not carried overnight


def test_fresh_seconds_remove_the_fake_lead_of_a_stale_copy():
    rng = np.random.default_rng(5)
    idx = day_index()
    a = pd.Series(100 * np.exp(np.cumsum(rng.standard_normal(len(idx)) * 1e-4)), index=idx)
    seen = pd.Series(rng.random(len(idx)) < 0.7, index=idx)                 # copy trades 70% of seconds
    copy = a.where(seen).ffill()
    r = log_returns(pd.DataFrame({"A": a, "COPY": copy}), max_gap="1s")
    naive = fresh_cross_correlation(r["A"], r["COPY"], pd.Series(True, index=idx), max_lag=1)
    fresh = fresh_cross_correlation(r["A"], r["COPY"], seen, max_lag=1)
    assert naive.loc[1, "corr"] - naive.loc[-1, "corr"] > 0.15               # stale copy looks like a follower
    assert abs(fresh.loc[1, "corr"] - fresh.loc[-1, "corr"]) < 0.02          # the fix removes it


def test_seconds_catchup_measures_a_planted_one_second_lead():
    rng = np.random.default_rng(6)
    idx = day_index()
    e = rng.standard_normal(len(idx)) * 1e-4
    e_lag = np.r_[0.0, e[:-1]]
    logp = pd.DataFrame({"L": np.log(100) + np.cumsum(e),
                         "F": np.log(50) + np.cumsum(0.6 * e + 0.4 * e_lag)}, index=idx)
    traded = pd.DataFrame(True, index=idx, columns=["L", "F"])
    ev = seconds_catchup(logp, traded, "L", "F", q=0.99, horizons=[1, 2])
    # F makes 40% of its move one second late -> next-second gain = 0.4 x the leader's move
    assert np.isclose(ev["follower_1"].mean(), 0.4 * ev["leader_move_bp"].mean(), rtol=0.1)
    assert np.isclose(ev["follower_2"].mean(), ev["follower_1"].mean(), rtol=0.15)   # nothing after 1 s
