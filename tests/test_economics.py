"""Economic significance: trading costs and the hedged catch-up trade."""
import numpy as np
import pandas as pd

from src.economics import big_move_catchup, round_trip_cost_bp


def test_round_trip_cost():
    # ES at 5,000: one tick (0.25) + $2 fee per side on a $50-per-point contract
    expected = (0.25 + 2 * 2.0 / 50) / 5000 * 1e4
    assert np.isclose(round_trip_cost_bp("ES", 5000.0), expected)
    assert round_trip_cost_bp("ES", 10_000.0) < round_trip_cost_bp("ES", 5000.0)   # cheaper as price rises


def prices(late, n=100_000, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) * 5e-4
    e_lag = np.r_[0.0, e[:-1]]
    idx = pd.date_range("2026-01-05", periods=n, freq="min", tz="America/New_York")
    return pd.DataFrame({"ES": 5000 * np.exp(np.cumsum(e)),
                         "NQ": 20000 * np.exp(np.cumsum((1 - late) * e + late * e_lag))}, index=idx)


def test_catchup_measures_the_planted_lead():
    res = big_move_catchup(prices(late=0.3), "ES", "NQ", q=0.99)
    # NQ makes 30% of its move one minute late -> it catches up 0.3 x the ES move
    assert np.isclose(res["catchup_1_bp"], 0.3 * res["leader_move_bp"], rtol=0.05)
    assert np.isclose(res["follower_same_bar_bp"], 0.7 * res["leader_move_bp"], rtol=0.05)


def test_no_lead_means_no_catchup():
    res = big_move_catchup(prices(late=0.0), "ES", "NQ", q=0.99)
    assert abs(res["catchup_1_bp"]) < 1e-6
    assert res["cost_hedged_bp"] > res["cost_follower_bp"] > 0
