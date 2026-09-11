"""Live terminal dashboard for the scraper. Run it in a second terminal.

Usage:
    python src/scraper/scrape_dashboard.py
    python src/scraper/scrape_dashboard.py --watch   # auto-refresh every 5s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROC_DIR, RAW_DIR

STATUS_FILE = PROC_DIR / "scrape_status.json"
OUTPUT_CSV  = PROC_DIR / "historical_results.csv"


SYMBOLS = {"done": "✓", "failed": "✗", "pending": "·", "no_data": "—"}
COLORS  = {
    "done":    "\033[92m",   # green
    "failed":  "\033[91m",   # red
    "pending": "\033[93m",   # yellow
    "no_data": "\033[90m",   # gray
    "reset":   "\033[0m",
}


def load_status() -> dict:
    if not STATUS_FILE.exists():
        return {}
    try:
        return json.loads(STATUS_FILE.read_text())
    except Exception:
        return {}


def count_parquets() -> int:
    return len(list(RAW_DIR.rglob("*.parquet")))


def csv_rows() -> int:
    if not OUTPUT_CSV.exists():
        return 0
    try:
        with open(OUTPUT_CSV) as f:
            return sum(1 for _ in f) - 1   # subtract header
    except Exception:
        return 0


def render_dashboard() -> None:
    data = load_status()
    if not data:
        print("No scrape status found. Start the scraper first.")
        return

    # Group by year
    by_year: dict[int, list] = {}
    for race in data.values():
        y = race["year"]
        by_year.setdefault(y, []).append(race)

    total = len(data)
    counts = {"done": 0, "failed": 0, "pending": 0, "no_data": 0}
    for r in data.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    # Header
    print("\n" + "═" * 64)
    print("  F1 SCRAPER DASHBOARD")
    print("═" * 64)
    print(f"  Status file:  {STATUS_FILE}")
    print(f"  Parquet files:{count_parquets():>5}")
    print(f"  CSV rows:     {csv_rows():>5}")
    print()
    print(f"  Total races:  {total}")
    for status, symbol in SYMBOLS.items():
        c = counts.get(status, 0)
        col = COLORS.get(status, "")
        rst = COLORS["reset"]
        print(f"  {col}{symbol} {status:<10}{rst} {c:>3}  ({100*c/total:.0f}%)" if total else "")

    # Per-year grid
    print()
    for year in sorted(by_year.keys()):
        races = sorted(by_year[year], key=lambda r: r["round_num"])
        row_str = f"  {year}  "
        for r in races:
            s = r["status"]
            col = COLORS.get(s, "")
            rst = COLORS["reset"]
            row_str += f"{col}{SYMBOLS.get(s, '?')}{rst}"
        done_count = sum(1 for r in races if r["status"] == "done")
        row_str += f"  {done_count}/{len(races)}"
        print(row_str)

    print()

    # Recent failures
    failures = [r for r in data.values() if r["status"] == "failed"]
    if failures:
        print("  Recent failures:")
        for r in sorted(failures, key=lambda x: x.get("timestamp",""))[-5:]:
            err = r.get("error", "")[:55]
            print(f"    {r['year']} {r['gp']:<25}  {err}")

    print("═" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true",
                        help="Auto-refresh every 5 seconds")
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")  # clear screen
                render_dashboard()
                print(f"\n  (refreshing every {args.interval}s, Ctrl+C to exit)")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        render_dashboard()
