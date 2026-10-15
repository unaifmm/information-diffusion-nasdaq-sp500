"""
STEP 1 - Download 1-minute history (2016 -> today) from London Strategic Edge.

Run on YOUR computer (needs the LSE key you saved with `lse auth`):

    python 01_download_data.py

What it does:
  1. saves the full LSE catalog to data/raw/lse_catalog.csv (so we can see
     everything that is available);
  2. 1-minute bars 2016 -> today for NQ, ES, QQQ, SPY (finished years are skipped);
  3. the US economic calendar (CPI, payrolls, FOMC, ...) for the event study;
  4. 1-second bars (regular hours) from SECONDS_START (src/config.py) to yesterday,
     one file per trading day - the first run takes about 30-40 minutes.

Safe to stop and re-run at any time: finished years are cached in data/raw/lse/.
A full first run can take 20-40 minutes if the bulk-export quota runs out and
the script has to page through the data.
"""
import sys

import pandas as pd

from src import config
from src.lse import connect, download_calendar, resolve_dataset, update_seconds, update_symbol

sys.stdout.reconfigure(errors="replace")   # never crash on an odd character

client = connect()

# 1) Catalog ---------------------------------------------------------------
catalog = client.datasets()
cat = pd.DataFrame(catalog)
cat.to_csv(config.LSE_CATALOG, index=False)
print(f"Catalog: {len(cat):,} instruments saved to data/raw/lse_catalog.csv")
if "dataset" in cat:
    print(cat["dataset"].value_counts().to_string(), "\n")

futures = cat[cat["dataset"] == "futures"] if "dataset" in cat else cat.iloc[0:0]
print("Futures available:")
print(futures.drop(columns=[c for c in ("ticks",) if c in futures]).to_string(index=False), "\n")

# 2) + 3) Download ------------------------------------------------------------
state = {}
for name, symbol in config.LSE_SYMBOLS.items():
    dataset = resolve_dataset(catalog, symbol)
    if dataset is None:
        print(f"!! {name}: symbol '{symbol}' is not in the LSE catalog - skipped. "
              f"Look in data/raw/lse_catalog.csv for the right symbol and tell Claude.")
        continue
    print(f"\n=== {name} ({symbol}, dataset '{dataset}') ===")
    update_symbol(client, name, symbol, dataset, state=state)

# 3) Economic calendar ---------------------------------------------------------
print("\n=== US economic calendar ===")
try:
    download_calendar(client)
except Exception as exc:
    print(f"!! calendar download failed: {exc}  (tell Claude; everything else is unaffected)")

# 4) 1-second bars ----------------------------------------------------------------
print(f"\n=== 1-second bars, regular hours, from {config.SECONDS_START} ===")
for name, symbol in config.LSE_SYMBOLS.items():
    dataset = resolve_dataset(catalog, symbol)
    if dataset is None:
        continue
    update_seconds(client, name, symbol, dataset, state=state)
    if state.get("time_filter_ignored"):
        break

print("\nDone. Files are in", config.DATA_LSE.parent)
