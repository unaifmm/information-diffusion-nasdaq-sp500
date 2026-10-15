"""
Re-run the whole project with one command, in the right order.

    python run_all.py              # tests + every analysis script, on the data you have
    python run_all.py --download   # first update the data from London Strategic Edge (01)
    python run_all.py --from 05    # start at script 05 (e.g. after changing it)

It stops at the first error, so a broken step is never hidden by later output,
and prints how long each step took. All charts go to figures/, all tables to
results/. Scripts whose data is not downloaded yet are skipped with a note.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

# (script, what it does, data it needs: None = only the 1-minute files)
STEPS = [
    ("02_validate_simulation.py", "validate the methods on simulated data", None),
    ("03_validate_real_noise.py", "validate the net lead on real NQ noise", None),
    ("04_market_anatomy.py", "data coverage, prices, intraday profile, liquidity", None),
    ("05_nq_vs_es.py", "NQ vs ES by year, hour and time scale", None),
    ("06_nq_es_robustness.py", "confidence intervals + liquidity mechanism", None),
    ("07_futures_vs_etf.py", "futures vs ETF price discovery", None),
    ("08_event_study.py", "news events: CPI, payrolls, FOMC", config.ECON_CALENDAR),
    ("09_one_second.py", "1-second zoom", config.DATA_LSE_1S),
    ("10_economic_significance.py", "is the lead big enough to trade?", None),
]


def has_data(path) -> bool:
    if path is None:
        return any(config.DATA_LSE.glob("*.parquet"))
    return path.exists() and (path.is_file() or any(path.glob("*.parquet")))


def run(cmd: list[str], label: str, resume: str | None = None) -> float:
    print(f"\n{'=' * 78}\n>>> {label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        hint = f" Fix it, then continue with:  python run_all.py --from {resume}" if resume else ""
        print(f"\n!! '{label}' failed (exit code {result.returncode}).{hint}")
        sys.exit(result.returncode)
    return time.time() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="update the data first (01_download_data.py)")
    parser.add_argument("--from", dest="start", default="02", help="first script number to run, e.g. 05")
    parser.add_argument("--no-tests", action="store_true", help="skip the unit tests")
    args = parser.parse_args()
    args.start = args.start.zfill(2)

    timings = []
    if args.download:
        timings.append(("01_download_data.py", run([sys.executable, "01_download_data.py"], "01 download data")))
    if not args.no_tests:
        timings.append(("unit tests", run([sys.executable, "-m", "pytest"], "unit tests (pytest)")))

    for script, what, needs in STEPS:
        if script[:2] < args.start:
            continue
        if not (ROOT / script).exists():
            continue                                      # step not built yet
        if not has_data(needs):
            print(f"\n-- skipping {script}: its data is not downloaded yet (run with --download)")
            continue
        timings.append((script, run([sys.executable, script], f"{script[:2]} {what}", resume=script[:2])))

    print(f"\n{'=' * 78}\nAll done.")
    for name, secs in timings:
        print(f"  {name:<32} {secs:6.0f} s")
    print(f"  {'total':<32} {sum(s for _, s in timings):6.0f} s")
    print("Charts: figures/    Tables: results/")


if __name__ == "__main__":
    main()
