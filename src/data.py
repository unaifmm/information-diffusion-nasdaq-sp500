"""
Small data helpers shared by every analysis script: trading-session masks,
returns, and daily closes built from 1-minute bars.

(Loading the raw files lives in src/lse.py.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for bars inside the US regular cash session (09:30-16:00 NY), weekdays."""
    minute = index.hour * 60 + index.minute
    return (minute >= 9 * 60 + 30) & (minute < 16 * 60) & (index.dayofweek < 5)


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-session bars."""
    return df[rth_mask(df.index)]


def log_returns(prices: pd.DataFrame, max_gap: str | pd.Timedelta | None = None) -> pd.DataFrame:
    """Log return r_t = ln(P_t / P_{t-1}).

    Why log returns?  They add up over time (the 5-minute return is the sum of
    five 1-minute returns) and are roughly symmetric, which suits statistics.

    max_gap: if two consecutive rows are further apart than this (overnight,
    weekend, the futures' daily 17:00-18:00 break, a missing bar), the return
    is set to NaN - it would mix many hours of news into one "bar".
    """
    r = np.log(prices).diff()
    if max_gap is not None:
        gap = prices.index.to_series().diff()
        r[gap > pd.Timedelta(max_gap)] = np.nan
    return r


def daily_close(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily close = last regular-session 1-minute close of each day (normally
    the 15:59 bar, 12:59 on half-days). Index = calendar date (no time zone)."""
    rth = regular_hours(prices)
    daily = rth.groupby(rth.index.normalize()).last()
    daily.index = daily.index.tz_localize(None)
    daily.index.name = "date"
    return daily


def previous_tick_grid(raw: pd.DataFrame, start: str = "09:30:00", end: str = "15:59:59",
                       freq: str = "1s") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put several instruments on one common clock (every second of the regular
    session) using PREVIOUS-TICK sampling: in a second with no trade, the price
    is the last one traded. Standard for high-frequency lead-lag work.

    Returns (prices, traded): traded is True where the instrument actually
    traded in that second. A second with no trade has a zero return - the price
    is STALE, which can make a less active market look like a follower, so
    always check how often each market trades (Epps effect)."""
    prices, traded = [], []
    for day in sorted(set(raw.index.normalize())):
        grid = pd.date_range(f"{day.date()} {start}", f"{day.date()} {end}", freq=freq, tz=raw.index.tz)
        x = raw.reindex(grid)
        traded.append(x.notna())
        prices.append(x.ffill())                    # within the day only: never carries over a night
    return pd.concat(prices), pd.concat(traded)
