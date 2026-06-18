"""
src/scraper/historical_scraper.py
──────────────────────────────────
Bulk-downloads F1 race weekends from 2018 → present using FastF1.

Features:
  - Resume support: skips already-scraped races (checks .parquet on disk)
  - Rate limiting: respects FastF1 cache and adds polite delays
  - Retry logic: exponential backoff on transient errors
  - Validation: schema + sanity checks on every scraped row
  - Progress tracking: rich progress table + persistent status log
  - Incremental writes: saves each race immediately (no data loss on crash)

Output: data/processed/historical_results.csv  (append-mode, deduped)
        data/raw/{year}/{gp_slug}/              (per-race parquets)

Usage:
    python src/scraper/historical_scraper.py
    python src/scraper/historical_scraper.py --years 2023 2024
    python src/scraper/historical_scraper.py --years 2024 --gp Bahrain
    python src/scraper/historical_scraper.py --resume          # skip done races
    python src/scraper/historical_scraper.py --validate-only   # check existing data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import FASTF1_CACHE, RAW_DIR, PROC_DIR

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Setup FastF1 cache (must happen before any fastf1 import that triggers network)
# ─────────────────────────────────────────────────────────────────────────────
import fastf1

Path(FASTF1_CACHE).mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SCRAPE_YEARS        = list(range(2018, 2025))   # inclusive
STATUS_FILE         = PROC_DIR / "scrape_status.json"
OUTPUT_CSV          = PROC_DIR / "historical_results.csv"
RETRY_MAX           = 5
RETRY_BASE_DELAY    = 2.0   # seconds (doubles on each retry)
RATE_LIMIT_MIN_DELAY = 600   # seconds (10 minutes)
RATE_LIMIT_MAX_DELAY = 900   # seconds (15 minutes)
INTER_RACE_DELAY    = 10.0   # polite pause between races
INTER_SESSION_DELAY = 4.0    # pause between sessions within a race

# Minimum valid rows per race (catches partial scrapes)
MIN_DRIVERS_PER_RACE = 15

# ─────────────────────────────────────────────────────────────────────────────
# Status tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RaceStatus:
    year:      int
    gp:        str
    round_num: int
    status:    str       # "pending" | "done" | "failed" | "no_data"
    rows:      int  = 0
    error:     str  = ""
    timestamp: str  = ""

    def mark_done(self, rows: int) -> None:
        self.status    = "done"
        self.rows      = rows
        self.timestamp = datetime.utcnow().isoformat()

    def mark_failed(self, error: str) -> None:
        self.status    = "failed"
        self.error     = str(error)[:200]
        self.timestamp = datetime.utcnow().isoformat()

    def mark_no_data(self) -> None:
        self.status    = "no_data"
        self.timestamp = datetime.utcnow().isoformat()


class ScrapeTracker:
    """Persists scrape progress to disk so runs can be resumed."""

    def __init__(self, path: Path = STATUS_FILE):
        self.path = path
        self._data: dict[str, RaceStatus] = {}
        self._load()

    def _key(self, year: int, gp: str) -> str:
        return f"{year}_{gp.replace(' ', '_')}"

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._data = {k: RaceStatus(**v) for k, v in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._data.items()}, indent=2)
        )

    def get(self, year: int, gp: str) -> Optional[RaceStatus]:
        return self._data.get(self._key(year, gp))

    def set(self, status: RaceStatus) -> None:
        self._data[self._key(status.year, status.gp)] = status
        self.save()

    def is_done(self, year: int, gp: str) -> bool:
        s = self.get(year, gp)
        return s is not None and s.status == "done"

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for s in self._data.values():
            counts[s.status] = counts.get(s.status, 0) + 1
        return counts


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    """Detect HTTP 429 / rate-limit errors from FastF1 / underlying requests."""
    msg = str(exc).lower()
    return any(x in msg for x in [
        "429", "rate limit", "too many requests", "ratelimit",
        "rate limit exceeded", "500calls/h", "calls/h",
        "500apicall/hr", "500 api call", "api:500calls/h", "500 calls/h", "api call/hr",
        # FastF1 can surface quota exhaustion as a transient 'No race data' error.
        "no race data for 2022"
    ])


def _rate_limit_pause(wait_seconds: int = 600) -> None:
    """Pause with a live countdown. Default = 10 minutes."""
    print()
    log.warning("=" * 60)
    log.warning("RATE LIMITED - pausing for %d minutes before continuing",
                wait_seconds // 60)
    log.warning("=" * 60)
    for remaining in range(wait_seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"\r  Resuming in {mins:02d}:{secs:02d} ...", end="", flush=True)
        time.sleep(1)
    print("\r  Resuming now...                    ")
    log.info("Pause complete - continuing scrape.")


def with_retry(fn, *args, max_retries=RETRY_MAX, base_delay=RETRY_BASE_DELAY, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff.
    On rate limit errors, waits 10-15 minutes with a live countdown then retries."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if attempt == max_retries:
                raise
            if _is_rate_limit(exc):
                wait_seconds = int(np.random.uniform(RATE_LIMIT_MIN_DELAY, RATE_LIMIT_MAX_DELAY + 1))
                _rate_limit_pause(wait_seconds=wait_seconds)
            else:
                jitter = np.random.uniform(0, 5)
                delay  = base_delay * (2 ** (attempt - 1)) + jitter
                log.warning("Attempt %d/%d failed: %s - retrying in %.0fs",
                            attempt, max_retries, exc, delay)
                time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# Per-session extractors (same logic as fastf1_loader but standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_td_to_s(val) -> Optional[float]:
    """Convert timedelta / float / NaT to seconds, or None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return val.total_seconds()
    except AttributeError:
        try:
            return float(val)
        except Exception:
            return None


def extract_quali(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract Q1/Q2/Q3 times and gaps per driver."""
    laps = session.laps
    records = []

    for driver in laps["Driver"].unique():
        drv_laps = laps.pick_driver(driver)
        row: dict = {"driver_code": driver}

        for seg, part_num in [("Q1", 1), ("Q2", 2), ("Q3", 3)]:
            if "SessionPart" in drv_laps.columns:
                part_laps = drv_laps[drv_laps["SessionPart"] == part_num]
            else:
                # Fallback: use all laps and approximate
                part_laps = drv_laps

            best = part_laps["LapTime"].dropna().min()
            row[f"{seg.lower()}_time_s"] = _safe_td_to_s(best)

        records.append(row)

    df = pd.DataFrame(records)

    for seg in ["q1", "q2", "q3"]:
        col = f"{seg}_time_s"
        pole = df[col].dropna().min()
        df[f"{seg}_gap_to_pole"] = df[col] - pole if pole else np.nan

    # Official grid positions
    results = session.results.reset_index()
    if "GridPosition" in results.columns:
        grid = results[["Abbreviation", "GridPosition"]].rename(
            columns={"Abbreviation": "driver_code", "GridPosition": "grid_position"}
        )
        df = df.merge(grid, on="driver_code", how="left")
    elif "Position" in results.columns:
        grid = results[["Abbreviation", "Position"]].rename(
            columns={"Abbreviation": "driver_code", "Position": "grid_position"}
        )
        df = df.merge(grid, on="driver_code", how="left")

    return df


def extract_practice(session: fastf1.core.Session, label: str) -> pd.DataFrame:
    """Extract best lap and long-run pace from a practice session."""
    try:
        laps = session.laps.pick_quicklaps(threshold=1.07)
    except Exception:
        laps = session.laps

    records = []
    for driver in laps["Driver"].unique():
        drv = laps.pick_driver(driver)
        if drv.empty:
            continue

        best = drv["LapTime"].dropna().min()

        long_run_times = []
        for _, stint in drv.groupby("Stint"):
            timed = stint["LapTime"].dropna()
            if len(timed) >= 5:
                long_run_times.extend(timed.dt.total_seconds().tolist())

        records.append({
            "driver_code": driver,
            f"{label}_best_lap_s":     _safe_td_to_s(best),
            f"{label}_long_run_pace_s":float(np.median(long_run_times)) if long_run_times else np.nan,
        })

    return pd.DataFrame(records)


def extract_tire_deg(session: fastf1.core.Session) -> pd.DataFrame:
    """Fit degradation slope per compound per driver."""
    try:
        laps = session.laps.pick_quicklaps(threshold=1.10)
    except Exception:
        laps = session.laps

    records = []
    for driver in laps["Driver"].unique():
        drv  = laps.pick_driver(driver)
        row  = {"driver_code": driver}

        for compound in ["SOFT", "MEDIUM", "HARD"]:
            comp_laps = drv[drv["Compound"] == compound]
            slopes = []
            for _, stint in comp_laps.groupby("Stint"):
                times = stint["LapTime"].dropna().dt.total_seconds().values
                if len(times) >= 3:
                    slope, _ = np.polyfit(np.arange(len(times)), times, 1)
                    slopes.append(float(slope))
            row[f"deg_slope_{compound.lower()}"] = float(np.mean(slopes)) if slopes else np.nan

        records.append(row)

    return pd.DataFrame(records)


def extract_race_results(session: fastf1.core.Session) -> pd.DataFrame:
    """Extract official race finishing order."""
    results = session.results.reset_index()
    col_map = {
        "Abbreviation":  "driver_code",
        "FullName":      "driver_name",
        "TeamName":      "team_name",
        "Position":      "finish_position",
        "GridPosition":  "grid_position_race",
        "Status":        "status",
        "Points":        "points",
        "Time":          "race_time",
        "DriverNumber":  "driver_number",
    }
    df = results[[c for c in col_map if c in results.columns]].rename(columns=col_map)
    df["classified"] = df["status"].apply(
        lambda s: str(s).startswith("Finished") or "Lap" in str(s)
    )
    df["dnf"] = (~df["classified"]).astype(int)
    df["finish_position_top3"]  = (df["finish_position"] <= 3).astype(int)
    df["finish_position_top10"] = (df["finish_position"] <= 10).astype(int)
    return df


def extract_weather_summary(session: fastf1.core.Session) -> dict:
    w = session.weather_data
    if w is None or w.empty:
        return {}
    out = {}
    for col, key in [
        ("AirTemp",    "air_temp_c"),
        ("TrackTemp",  "track_temp_c"),
        ("Humidity",   "humidity_pct"),
        ("WindSpeed",  "wind_speed_ms"),
        ("Rainfall",   "rainfall"),
    ]:
        if col in w.columns:
            if col == "Rainfall":
                out[key] = int(w[col].any())
            else:
                out[key] = round(float(w[col].mean()), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build one race's feature row
# ─────────────────────────────────────────────────────────────────────────────

def scrape_race_weekend(year: int, gp: str | int, round_num: int) -> pd.DataFrame:
    """
    Loads all sessions for one race weekend and returns a merged
    per-driver DataFrame ready for the training pipeline.

    Raises an exception if critical data (qualifying or race) is unavailable.
    """
    log.info("  Scraping %d Round %d: %s", year, round_num, gp)

    def load(session_name: str) -> Optional[fastf1.core.Session]:
        try:
            s = fastf1.get_session(year, gp, session_name)
            s.load(laps=True, telemetry=False, weather=True, messages=False)
            return s
        except Exception as exc:
            log.debug("    Session %s unavailable: %s", session_name, exc)
            return None

    # ── Load sessions ──────────────────────────────────────────────────────
    race_session = with_retry(load, "R")
    if race_session is None:
        raise ValueError(f"No race data for {year} {gp}")

    quali_session = with_retry(load, "Q")
    fp2_session   = with_retry(load, "FP2")
    fp1_session   = with_retry(load, "FP1")
    fp3_session   = with_retry(load, "FP3")

    # ── Extract per-driver data ────────────────────────────────────────────
    dfs = []

    # Race results (always required)
    race_df = extract_race_results(race_session)
    dfs.append(race_df)

    # Qualifying times
    if quali_session is not None:
        time.sleep(INTER_SESSION_DELAY)
        try:
            q_df = extract_quali(quali_session)
            dfs.append(q_df)
            weather = extract_weather_summary(quali_session)
        except Exception as e:
            log.warning("    Quali extraction failed: %s", e)
            weather = {}
    else:
        weather = extract_weather_summary(race_session)

    # Practice sessions
    for sess, label in [(fp1_session, "fp1"), (fp2_session, "fp2"), (fp3_session, "fp3")]:
        if sess is not None:
            time.sleep(INTER_SESSION_DELAY)
            try:
                dfs.append(extract_practice(sess, label))
            except Exception as e:
                log.debug("    Practice %s failed: %s", label, e)

    # Tire degradation — prefer FP2
    for sess in [fp2_session, fp3_session, fp1_session]:
        if sess is not None:
            time.sleep(INTER_SESSION_DELAY)
            try:
                dfs.append(extract_tire_deg(sess))
                break
            except Exception as e:
                log.debug("    Tire deg extraction failed: %s", e)

    # ── Merge on driver_code ───────────────────────────────────────────────
    merged = dfs[0]
    for df in dfs[1:]:
        if df is not None and not df.empty and "driver_code" in df.columns:
            merged = merged.merge(df, on="driver_code", how="left")

    # ── Add context columns ────────────────────────────────────────────────
    merged["year"]      = year
    merged["round"]     = round_num
    merged["gp"]        = str(gp)
    merged["circuit_id"] = _slugify(str(gp))

    for k, v in weather.items():
        merged[k] = v

    return merged


def _slugify(s: str) -> str:
    return s.lower().replace(" ", "_").replace("-", "_")


# ─────────────────────────────────────────────────────────────────────────────
# Season schedule fetcher
# ─────────────────────────────────────────────────────────────────────────────

def get_season_schedule(year: int) -> list[dict]:
    """
    Returns list of {round_num, gp_name} for all races in a season.
    Uses FastF1's ergast integration as the source of truth.
    """
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        races = []
        for _, row in schedule.iterrows():
            # Skip pre-season tests
            if row.get("EventFormat", "") == "testing":
                continue
            races.append({
                "round": int(row["RoundNumber"]),
                "gp":    str(row["EventName"]),
            })
        return races
    except Exception as exc:
        log.error("Could not fetch %d schedule: %s", year, exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_race_parquet(df: pd.DataFrame, year: int, gp: str) -> Path:
    """Save per-race DataFrame to a parquet file under data/raw/."""
    slug = _slugify(gp)
    path = RAW_DIR / str(year) / f"{slug}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def append_to_master_csv(df: pd.DataFrame) -> None:
    """Append race rows to the master CSV, deduplicating on (year, gp, driver_code)."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["year", "gp", "driver_code"], keep="last"
        )
    else:
        combined = df
    combined.to_csv(OUTPUT_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS = [
    "driver_code", "team_name", "finish_position", "dnf",
    "year", "gp", "round",
]

NUMERIC_RANGE_CHECKS = {
    "finish_position":    (1, 30),
    "grid_position":      (0, 30),   # 0 = pit lane start
    "q1_time_s":          (60, 200),
    "q3_time_s":          (60, 200),
    "fp2_best_lap_s":     (60, 200),
    "deg_slope_medium":   (-0.1, 2.0),
    "air_temp_c":         (0, 60),
    "track_temp_c":       (10, 80),
}


def validate_race_df(df: pd.DataFrame, year: int, gp: str) -> list[str]:
    """Returns a list of validation warnings (empty = all good)."""
    warnings = []

    # Row count
    if len(df) < MIN_DRIVERS_PER_RACE:
        warnings.append(f"Only {len(df)} drivers (expected ≥ {MIN_DRIVERS_PER_RACE})")

    # Required columns
    for col in REQUIRED_COLS:
        if col not in df.columns:
            warnings.append(f"Missing column: {col}")

    # Numeric ranges
    for col, (lo, hi) in NUMERIC_RANGE_CHECKS.items():
        if col in df.columns:
            out_of_range = df[col].dropna()
            out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
            if not out_of_range.empty:
                warnings.append(
                    f"{col} has {len(out_of_range)} out-of-range values: "
                    f"{out_of_range.values[:3]}"
                )

    # Finish position uniqueness
    if "finish_position" in df.columns:
        dupes = df["finish_position"].dropna()
        if dupes.duplicated().any():
            warnings.append("Duplicate finish positions detected")

    # DNF consistency
    if "dnf" in df.columns and "classified" in df.columns:
        inconsistent = df[df["dnf"] == 1][df["classified"] == 1]
        if not inconsistent.empty:
            warnings.append(
                f"{len(inconsistent)} drivers marked both DNF and classified"
            )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Main scraper
# ─────────────────────────────────────────────────────────────────────────────

def run_scraper(
    years: Optional[list[int]] = None,
    gp_filter: Optional[str] = None,
    resume: bool = True,
    validate_only: bool = False,
    dry_run: bool = False,
    inter_race_delay: float = INTER_RACE_DELAY,
) -> None:
    years = years or SCRAPE_YEARS
    tracker = ScrapeTracker()

    log.info("F1 Historical Scraper")
    log.info("Years: %s | Resume: %s | Validate-only: %s", years, resume, validate_only)

    if validate_only:
        _run_validation_pass(tracker)
        return

    total_races = 0
    total_rows  = 0
    failed      = []

    for year in years:
        log.info("\n-- Season %d --", year)
        schedule = with_retry(get_season_schedule, year)

        if not schedule:
            log.warning("  No schedule found for %d - skipping", year)
            continue

        for race_info in tqdm(schedule, desc=f"{year}", unit="race"):
            gp        = race_info["gp"]
            round_num = race_info["round"]

            # GP filter
            if gp_filter and gp_filter.lower() not in gp.lower():
                continue

            # Resume: skip already-done races
            if resume and tracker.is_done(year, gp):
                log.debug("  Skipping %s %s (already done)", year, gp)
                continue

            status = RaceStatus(year=year, gp=gp, round_num=round_num, status="pending")
            tracker.set(status)

            if dry_run:
                log.info("  [DRY RUN] Would scrape: %d %s", year, gp)
                continue

            try:
                df = with_retry(
                    scrape_race_weekend, year, gp, round_num,
                    max_retries=RETRY_MAX,
                )

                # Validate
                warnings = validate_race_df(df, year, gp)
                for w in warnings:
                    log.warning("  Validation [%d %s]: %s", year, gp, w)

                # Save
                save_race_parquet(df, year, gp)
                append_to_master_csv(df)

                status.mark_done(len(df))
                tracker.set(status)

                total_races += 1
                total_rows  += len(df)
                log.info("  [OK] %d %s - %d drivers", year, gp, len(df))

            except Exception as exc:
                log.error("  [FAIL] %d %s - %s", year, gp, exc)
                status.mark_failed(str(exc))
                tracker.set(status)
                failed.append((year, gp, str(exc)))

            jitter = np.random.uniform(0, 5)
            time.sleep(inter_race_delay + jitter)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Scrape complete")
    print(f"  Races saved:  {total_races}")
    print(f"  Driver rows:  {total_rows}")
    print(f"  Failed races: {len(failed)}")
    print(f"  Output CSV:   {OUTPUT_CSV}")
    print("=" * 60)

    if failed:
        print("\nFailed races:")
        for year, gp, err in failed:
            print(f"  {year} {gp}: {err[:80]}")

    # Print status summary
    print("\nTracker summary:", tracker.summary())


def _run_validation_pass(tracker: ScrapeTracker) -> None:
    """Scan all saved parquets and report validation issues."""
    parquet_paths = sorted(RAW_DIR.rglob("*.parquet"))
    if not parquet_paths:
        log.info("No parquet files found in %s", RAW_DIR)
        return

    issues: list[tuple[str, list[str]]] = []
    for path in tqdm(parquet_paths, desc="Validating"):
        try:
            df = pd.read_parquet(path)
            year = int(path.parent.name)
            gp   = path.stem.replace("_", " ")
            warns = validate_race_df(df, year, gp)
            if warns:
                issues.append((str(path), warns))
        except Exception as exc:
            issues.append((str(path), [f"Could not load: {exc}"]))

    if issues:
        print(f"\n{len(issues)} files with issues:")
        for p, warns in issues:
            print(f"\n  {p}:")
            for w in warns:
                print(f"    - {w}")
    else:
        print(f"\nAll {len(parquet_paths)} files passed validation [OK]")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(PROC_DIR / "scrape.log"),
        ],
    )
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="F1 Historical Data Scraper")
    parser.add_argument(
        "--years", type=int, nargs="+", default=None,
        help="Seasons to scrape (e.g. --years 2022 2023 2024)"
    )
    parser.add_argument(
        "--gp", type=str, default=None,
        help="Filter to a specific GP (substring match, e.g. --gp Bahrain)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-scrape all races even if already done"
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Only validate existing parquet files, no new scraping"
    )
    parser.add_argument(
        "--delay", type=float, default=None,
        help="Override inter-race delay in seconds (default 10). Use 30+ if getting rate limited)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be scraped without actually doing it"
    )

    args = parser.parse_args()

    inter_race_delay = args.delay if args.delay is not None else INTER_RACE_DELAY
    if args.delay is not None:
        log.info("Inter-race delay set to %.0fs", inter_race_delay)

    run_scraper(
        years=args.years,
        gp_filter=args.gp,
        resume=not args.no_resume,
        validate_only=args.validate_only,
        dry_run=args.dry_run,
        inter_race_delay=inter_race_delay,
    )
