"""Macro-release calendar: number parsing, official times, event windows."""
import numpy as np
import pandas as pd

from src import config
from src.events import load_events, parse_number, stack_windows


def test_parse_number():
    assert parse_number("0.3%") == 0.3 and parse_number("-23K") == -23
    assert parse_number("1,234.5K") == 1234.5 and np.isnan(parse_number(None))


def test_official_time_replaces_the_vendor_time(tmp_path):
    rows = [  # the vendor's back-filled history is an hour late; the live 2026 entry is right
        {"date": "2024-01-11", "datetime": "2024-01-11T14:30:00Z", "event": "Inflation Rate MoM DEC",
         "actual": "0.3%", "consensus": "0.2%"},
        {"date": "2026-07-14", "datetime": "2026-07-14T12:30:00Z", "event": "Inflation Rate MoM JUN",
         "actual": "-0.4%", "consensus": "-0.1%"},
        {"date": "2020-03-15", "datetime": "2020-03-15T21:00:00Z", "event": "Fed Interest Rate Decision",
         "actual": "0.25%", "consensus": "1.25%"},      # unscheduled Sunday cut: excluded
        {"date": "2026-10-14", "datetime": "2026-10-14T12:30:00Z", "event": "Inflation Rate MoM SEP",
         "actual": None, "consensus": "0.6%"},          # not released yet: excluded
    ]
    path = tmp_path / "cal.parquet"
    pd.DataFrame(rows).to_parquet(path)
    ev = load_events(path=path)
    assert len(ev) == 2 and set(ev["kind"]) == {"CPI"}
    assert (ev["time"].dt.strftime("%H:%M") == "08:30").all()               # New York time
    assert np.allclose(ev["surprise"], [0.1, -0.3])


def test_stack_windows_keeps_events_apart():
    idx = pd.date_range("2026-01-05 08:00", "2026-01-05 09:00", freq="min", tz=config.TZ)
    r = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
    t0 = pd.Timestamp("2026-01-05 08:30", tz=config.TZ)
    w = stack_windows(r, [t0, t0], before=2, after=3)                      # the same event twice
    assert len(w) == 10 and w.index.is_unique
    assert w.loc[w["k"] == 0, "A"].tolist() == [30.0, 30.0]
    assert (w.index[5] - w.index[4]) > pd.Timedelta("1h")                  # a gap between windows
