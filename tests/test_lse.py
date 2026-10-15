"""Downloader logic, tested with a FAKE data vendor (no internet, no API key)."""
import numpy as np
import pandas as pd
import pytest

import src.lse as L
from src import config


def test_normalise_handles_ts_duplicates_and_missing_volume():
    raw = pd.DataFrame({"ts": ["2026-01-05T14:31:00Z", "2026-01-05T14:30:00Z", "2026-01-05T14:31:00Z"],
                        "open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0]})
    df = L.normalise(raw)
    assert list(df.columns) == L.COLS
    assert df["timestamp"].is_monotonic_increasing and str(df["timestamp"].dt.tz) == "UTC"
    assert df["close"].tolist() == [2.0, 3.0]          # duplicate minute: the newer copy wins
    assert (df["volume"] == 0).all()


def test_resolve_dataset_ignores_options():
    catalog = [{"dataset": "options", "symbol": "SPY"}, {"dataset": "etf", "symbol": "SPY"}]
    assert L.resolve_dataset(catalog, "SPY") == "etf"
    assert L.resolve_dataset(catalog, "XYZ") is None


@pytest.mark.parametrize("day, open_utc", [("2026-01-15", "14:30"), ("2026-07-15", "13:30")])
def test_one_second_windows_cover_the_session(day, open_utc):
    w = L._rth_windows_utc(pd.Timestamp(day))
    assert w[0][0] == pd.Timestamp(f"{day} {open_utc}", tz="UTC")       # winter vs summer time
    assert w[-1][1] - w[0][0] == pd.Timedelta(hours=6, minutes=30)
    assert all(b == a2 for (_, b), (a2, _) in zip(w[:-1], w[1:]))         # no holes, no overlaps
    assert all(b - a <= pd.Timedelta(minutes=80) for a, b in w)          # < 5,000 bars each


def test_parquet_safe_mixed_columns(tmp_path):
    df = pd.DataFrame({"actual": [3.1, "3.1%", None], "meta": [{"u": "%"}, None, None], "n": [1, 2, 3]})
    safe = L._parquet_safe(df)
    safe.to_parquet(tmp_path / "x.parquet")                              # would fail without it
    assert safe["actual"].tolist()[:2] == ["3.1", "3.1%"] and pd.isna(safe["actual"].iloc[2])
    assert safe["meta"].iloc[0] == '{"u": "%"}' and safe["n"].dtype == np.int64


# ---------------------------------------------------------------------------
# 1-second downloader against a fake vendor
# ---------------------------------------------------------------------------
class FakeVendor:
    """Serves 1-second bars. Only accepts "YYYY-MM-DD HH:MM:SS" timestamps (like a
    picky server); with ignore_time=True it ignores the time of day instead."""

    def __init__(self, ignore_time=False):
        self.ignore_time, self.calls = ignore_time, 0

    def candles(self, symbol, tf, start=None, end=None, limit=5000, dataset=None):
        self.calls += 1
        if self.ignore_time:
            start, end = start[:10], start[:10] + " 23:59:59"
        elif "T" in start:
            raise L_error(422)
        idx = pd.date_range(start, end, freq="s", tz="UTC", inclusive="left")[:limit]
        return [{"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "open": 1.0, "high": 1.0,
                 "low": 1.0, "close": 1.0, "volume": 1.0} for t in idx]


def L_error(status):
    exc = Exception(f"[{status}] bad filter")
    exc.status = status
    return exc


@pytest.fixture
def fake_disk(tmp_path, monkeypatch):
    """1-minute files on a temporary disk: 14-18 Sep 2026, with Wednesday missing
    (a 'holiday'), so the 1-second downloader should fetch exactly 4 days."""
    monkeypatch.setattr(config, "DATA_LSE", tmp_path / "lse")
    monkeypatch.setattr(config, "DATA_LSE_1S", tmp_path / "lse_1s")
    config.DATA_LSE.mkdir()
    config.DATA_LSE_1S.mkdir()
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    days = [d for d in pd.bdate_range("2026-09-14", "2026-09-18") if d.day != 16]
    stamps = pd.DatetimeIndex(np.concatenate([
        pd.date_range(f"{d.date()} 09:30", f"{d.date()} 15:59", freq="min", tz=config.TZ).tz_convert("UTC").values
        for d in days])).tz_localize("UTC")
    pd.DataFrame({"timestamp": stamps, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}) \
        .to_parquet(config.DATA_LSE / "NQ.F_2026.parquet", index=False)
    return days


def test_update_seconds_downloads_each_trading_day_once(fake_disk):
    vendor, state = FakeVendor(), {}
    L.update_seconds(vendor, "NQ", "NQ.F", "futures", start="2026-09-14", state=state)
    files = sorted(p.name for p in config.DATA_LSE_1S.glob("*.parquet"))
    assert files == [f"NQ.F_{d.date()}.parquet" for d in fake_disk]      # holiday skipped
    assert state["time_format"] == "%Y-%m-%d %H:%M:%S"                  # fell back to a format that works
    day = pd.read_parquet(config.DATA_LSE_1S / files[0])
    assert len(day) == 6 * 3600 + 30 * 60                                 # 09:30:00 -> 15:59:59

    calls = vendor.calls
    L.update_seconds(vendor, "NQ", "NQ.F", "futures", start="2026-09-14", state=state)
    assert vendor.calls == calls                                          # re-run: nothing to fetch


def test_update_seconds_stops_if_server_ignores_time(fake_disk):
    state = {}
    L.update_seconds(FakeVendor(ignore_time=True), "NQ", "NQ.F", "futures", start="2026-09-14", state=state)
    assert state.get("time_filter_ignored") is True
    assert not list(config.DATA_LSE_1S.glob("*.parquet"))
