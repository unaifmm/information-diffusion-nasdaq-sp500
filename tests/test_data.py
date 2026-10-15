"""Basic building blocks: trading hours, returns, daily closes."""
import numpy as np
import pandas as pd

from src.data import daily_close, log_returns, rth_mask


def ny(*stamps):
    return pd.DatetimeIndex(stamps).tz_localize("America/New_York")


def test_regular_hours_boundaries():
    idx = ny("2026-10-12 09:29", "2026-10-12 09:30", "2026-10-12 15:59",
             "2026-10-12 16:00", "2026-10-10 10:00")            # the last one is a Saturday
    assert rth_mask(idx).tolist() == [False, True, True, False, False]


def test_returns_are_nan_across_gaps():
    idx = ny("2026-10-12 09:30", "2026-10-12 09:31", "2026-10-12 09:35")
    px = pd.DataFrame({"A": [100.0, 101.0, 102.0]}, index=idx)
    r = log_returns(px, max_gap="1min")["A"]
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], np.log(101 / 100))
    assert np.isnan(r.iloc[2])                                   # 4-minute gap: no return


def test_log_returns_add_up():
    idx = pd.date_range("2026-10-12 09:30", periods=6, freq="min", tz="America/New_York")
    px = pd.DataFrame({"A": [100, 101, 99, 102, 103, 104.5]}, index=idx, dtype=float)
    assert np.isclose(log_returns(px)["A"].sum(), np.log(104.5 / 100))


def test_daily_close_is_last_regular_bar():
    idx = ny("2026-10-12 15:58", "2026-10-12 15:59", "2026-10-12 16:30", "2026-10-13 15:59")
    px = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    d = daily_close(px)["A"]
    assert d.tolist() == [2.0, 4.0]                              # 16:30 is after the close
    assert d.index.tz is None
