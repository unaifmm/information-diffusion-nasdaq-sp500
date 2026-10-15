"""
Central configuration: instruments, dates and folder paths.

Keeping every "setting" in one file is a professional habit: change it HERE
and every script picks it up.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders (relative to this file, so the project works on any computer)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_LSE = ROOT / "data" / "raw" / "lse"          # one parquet file per instrument-year
LSE_CATALOG = ROOT / "data" / "raw" / "lse_catalog.csv"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

for _folder in (DATA_LSE, FIGURES, RESULTS):
    _folder.mkdir(parents=True, exist_ok=True)

# All timestamps are converted to New York time (where US markets live).
TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Data source: London Strategic Edge (LSE), 1-minute bars, 2016 -> today.
# One source for everything = consistent timestamps, cleaning and quality.
# ---------------------------------------------------------------------------
LSE_START = "2016-05-29"          # first NQ.F bar in the LSE vault
LSE_SYMBOLS = {                   # our name -> LSE symbol
    "NQ": "NQ.F",    # E-mini Nasdaq-100 futures (CME)
    "ES": "ES.F",    # E-mini S&P 500 futures (CME)
    "QQQ": "QQQ",    # Invesco QQQ ETF (tracks the Nasdaq-100)
    "SPY": "SPY",    # SPDR S&P 500 ETF
}

# Contract specifications (CME): dollar value of one index point, and tick size
POINT_VALUE = {"NQ": 20.0, "ES": 50.0}
TICK_SIZE = {"NQ": 0.25, "ES": 0.25, "QQQ": 0.01, "SPY": 0.01}

# US regular cash session (New York time)
RTH_START, RTH_END = "09:30", "15:59"

# ---------------------------------------------------------------------------
# Extra data for steps 13-14 (news-event study, 1-second zoom)
# ---------------------------------------------------------------------------
ECON_CALENDAR = ROOT / "data" / "raw" / "economic_calendar_us.parquet"   # US macro releases
DATA_LSE_1S = ROOT / "data" / "raw" / "lse_1s"     # 1-second bars, one file per instrument-day
DATA_LSE_1S.mkdir(parents=True, exist_ok=True)
SECONDS_START = "2026-07-01"   # 1-second history: LSE's 1-second ETF data starts in late April 2026

# ---------------------------------------------------------------------------
# Trading-cost assumptions for step 15 (economic significance).
# Deliberately FAVOURABLE to the trader: the bid-ask spread is assumed to be one
# tick (its minimum - it is wider in fast markets) and there is no market impact.
# If a lead does not pay even under these costs, it does not pay in practice.
# ---------------------------------------------------------------------------
MULTIPLIER = {**POINT_VALUE, "QQQ": 1.0, "SPY": 1.0}   # $ per index point / per share
FEE_PER_SIDE = {"NQ": 2.0, "ES": 2.0,                  # $ per contract per side, all-in
                "QQQ": 0.0035, "SPY": 0.0035}          # $ per share per side (exchange + clearing + broker)
