"""
src/ingestion/fastf1_loader.py
─────────────────────────────
Loads qualifying, practice, and race data via the FastF1 library.
Caches to disk so repeated calls are instant.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd
import numpy as np

from config.settings import FASTF1_CACHE

log = logging.getLogger(__name__)

# ── Enable FastF1 cache ────────────────────────────────────────────────────────
Path(FASTF1_CACHE).mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(FASTF1_CACHE)


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_session(year: int, gp: str | int, session: str) -> fastf1.core.Session:
    """Load a FastF1 session with laps + telemetry + weather."""
    log.info("Loading %s | %s | %s", year, gp, session)
    s = fastf1.get_session(year, gp, session)
    s.load(laps=True, telemetry=True, weather=True, messages=True)
    return s


def load_weekend(year: int, gp: str | int) -> dict[str, fastf1.core.Session]:
    """Load FP1, FP2, FP3 (or FP2 only for sprint), Q, and R for a full weekend."""
    sessions: dict[str, fastf1.core.Session] = {}
    for label in ["FP1", "FP2", "FP3", "Q", "R"]:
        try:
            sessions[label] = load_session(year, gp, label)
        except Exception as exc:
            log.info("Session %s not available: %s", label, exc)

    # Sprint weekends often skip FP3 and add SQ/S.
    if "FP3" not in sessions:
        for label in ["SQ", "S"]:
            try:
                sessions[label] = load_session(year, gp, label)
            except Exception as exc:
                log.info("Sprint session %s not available: %s", label, exc)

    log.info("Loaded sessions for %s %s: %s", year, gp, sorted(sessions.keys()))
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Qualifying data
# ─────────────────────────────────────────────────────────────────────────────

def extract_qualifying_times(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Returns a DataFrame with Q1/Q2/Q3 best lap times and gaps to pole
    for each driver.

    Columns:
        driver_code, grid_position,
        q1_time_s, q2_time_s, q3_time_s,
        q1_gap_to_pole, q2_gap_to_pole, q3_gap_to_pole
    """
    laps = session.laps
    records = []

    for driver in laps["Driver"].dropna().unique():
        drv_laps = laps.pick_driver(driver)
        row: dict = {"driver_code": driver}

        for seg in ["Q1", "Q2", "Q3"]:
            # Safer: filter by SessionPart column when available
            if "SessionPart" in drv_laps.columns:
                part_map = {"Q1": 1, "Q2": 2, "Q3": 3}
                part_laps = drv_laps[drv_laps["SessionPart"] == part_map[seg]]
            else:
                part_laps = drv_laps

            best = part_laps["LapTime"].dropna().min()
            row[f"{seg.lower()}_time_s"] = (
                best.total_seconds() if pd.notna(best) else np.nan
            )

        records.append(row)

    df = pd.DataFrame(records)

    # Compute gaps to pole for each segment
    for seg in ["q1", "q2", "q3"]:
        col = f"{seg}_time_s"
        pole = df[col].min()
        df[f"{seg}_gap_to_pole"] = df[col] - pole

    # Add official grid positions from the results
    results = session.results.reset_index()
    grid_col = None
    for candidate in ["GridPosition", "Position"]:
        if candidate in results.columns:
            grid_col = candidate
            break

    if grid_col is not None and "Abbreviation" in results.columns:
        grid = results[["Abbreviation", grid_col]].rename(
            columns={"Abbreviation": "driver_code", grid_col: "grid_position"}
        )
        grid["grid_position"] = pd.to_numeric(grid["grid_position"], errors="coerce")
        df = df.merge(grid, on="driver_code", how="left")
    else:
        df["grid_position"] = np.nan

    if df["grid_position"].isna().any():
        fallback = df["q3_time_s"].copy()
        if fallback.isna().all():
            fallback = df["q2_time_s"].copy()
        if fallback.isna().all():
            fallback = df["q1_time_s"].copy()
        df["grid_position"] = df["grid_position"].fillna(
            fallback.rank(method="first", ascending=True)
        )

    return df.sort_values("grid_position").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Practice pace
# ─────────────────────────────────────────────────────────────────────────────

def extract_practice_pace(session: fastf1.core.Session, session_label: str) -> pd.DataFrame:
    """
    Returns per-driver pace summary from a practice session:
        - best_lap_s          : single-lap outright pace
        - long_run_pace_s     : median lap time over stints > 5 laps
        - fuel_corrected_s    : estimated fuel-corrected pace (0.08s/lap fuel correction)
    """
    laps = session.laps.pick_quicklaps(threshold=1.07)  # remove outliers
    records = []

    for driver in laps["Driver"].unique():
        drv = laps.pick_driver(driver)
        if drv.empty:
            continue

        best = drv["LapTime"].dropna().min()

        # Long run: stints with >= 5 timed laps
        long_run_times = []
        for _, stint in drv.groupby("Stint"):
            timed = stint["LapTime"].dropna()
            if len(timed) >= 5:
                long_run_times.extend(timed.dt.total_seconds().tolist())

        long_run_pace = np.median(long_run_times) if long_run_times else np.nan

        # Crude fuel correction: assume 0.08s / lap of fuel weight improvement
        # Each lap burns ~1.5 kg; F1 car starts with ~110 kg
        lap_number_avg = drv["LapNumber"].mean()
        fuel_correction = lap_number_avg * 1.5 * 0.08 / 110 * 60  # rough
        fuel_corrected = (
            best.total_seconds() - fuel_correction if pd.notna(best) else np.nan
        )

        records.append(
            {
                "driver_code": driver,
                f"{session_label.lower()}_best_lap_s": (
                    best.total_seconds() if pd.notna(best) else np.nan
                ),
                f"{session_label.lower()}_long_run_pace_s": long_run_pace,
                f"{session_label.lower()}_fuel_corrected_s": fuel_corrected,
            }
        )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Tire degradation
# ─────────────────────────────────────────────────────────────────────────────

def extract_tire_degradation(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Fits a linear slope to lap times within each compound + driver stint.
    Returns mean degradation rate (seconds per lap) per compound per driver.

    Columns: driver_code, deg_slope_soft, deg_slope_medium, deg_slope_hard
    """
    laps = session.laps.pick_quicklaps(threshold=1.10)
    records = {}

    for driver in laps["Driver"].unique():
        drv = laps.pick_driver(driver)
        row: dict = {"driver_code": driver}

        for compound in ["SOFT", "MEDIUM", "HARD"]:
            comp_laps = drv[drv["Compound"] == compound].copy()
            slopes = []

            for _, stint in comp_laps.groupby("Stint"):
                times = stint["LapTime"].dropna().dt.total_seconds().values
                if len(times) < 3:
                    continue
                x = np.arange(len(times))
                slope, _ = np.polyfit(x, times, 1)
                slopes.append(slope)

            key = f"deg_slope_{compound.lower()}"
            row[key] = float(np.mean(slopes)) if slopes else np.nan

        records[driver] = row

    df = pd.DataFrame(list(records.values()))

    # Crossover lap estimation: when does compound A become slower than B?
    # crossover_lap = (offset_B - offset_A) / (slope_A - slope_B)
    def crossover(df: pd.DataFrame, base_pace_a: float, base_pace_b: float,
                  slope_col_a: str, slope_col_b: str, result_col: str) -> pd.DataFrame:
        slope_diff = df[slope_col_a] - df[slope_col_b]
        pace_diff  = base_pace_b - base_pace_a
        df[result_col] = np.where(slope_diff > 0, pace_diff / slope_diff, np.nan)
        return df

    # Approximate base offsets: SOFT ~0.8s faster than MEDIUM, MEDIUM ~0.5s faster than HARD
    df = crossover(df, 0.0, 0.8, "deg_slope_soft", "deg_slope_medium", "crossover_lap_s_m")
    df = crossover(df, 0.0, 0.5, "deg_slope_medium", "deg_slope_hard", "crossover_lap_m_h")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Weather
# ─────────────────────────────────────────────────────────────────────────────

def extract_weather(session: fastf1.core.Session) -> dict:
    """Returns mean weather conditions from a session as a flat dict."""
    w = session.weather_data
    if w is None or w.empty:
        return {}
    return {
        "air_temp_c":    round(w["AirTemp"].mean(), 1),
        "track_temp_c":  round(w["TrackTemp"].mean(), 1),
        "humidity_pct":  round(w["Humidity"].mean(), 1),
        "wind_speed_kph":round(w["WindSpeed"].mean() * 3.6, 1),
        "rainfall":      bool(w["Rainfall"].any()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Race results (for building training data)
# ─────────────────────────────────────────────────────────────────────────────

def extract_race_results(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Returns official race results with timing and status.

    Columns: driver_code, finish_position, classified, dnf, total_time_s
    """
    results = session.results.reset_index()
    col_map = {
        "Abbreviation": "driver_code",
        "Position":     "finish_position",
        "Status":       "status",
        "Time":         "race_time",
        "Points":       "points",
    }
    df = results[[c for c in col_map if c in results.columns]].rename(columns=col_map)
    df["classified"] = df["status"].apply(
        lambda s: "Finished" in str(s) or "Lapped" in str(s)
    )
    df["dnf"] = ~df["classified"]
    df["finish_position_top3"]  = df["finish_position"] <= 3
    df["finish_position_top10"] = df["finish_position"] <= 10
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build a full feature row for a single race weekend
# ─────────────────────────────────────────────────────────────────────────────

def build_weekend_features(year: int, gp: str | int) -> pd.DataFrame:
    """
    High-level function: loads all sessions for a weekend and returns
    a merged per-driver feature DataFrame ready for the ML pipeline.
    """
    sessions = load_weekend(year, gp)
    dfs = []

    # Qualifying features
    if "Q" in sessions:
        dfs.append(extract_qualifying_times(sessions["Q"]))

    # Practice pace per session
    for label in ["FP1", "FP2", "FP3", "S"]:
        if label in sessions:
            dfs.append(extract_practice_pace(sessions[label], label))

    # Tire degradation — prefer FP2 long-run data
    for label in ["FP2", "FP3", "S", "FP1"]:
        if label in sessions:
            deg_df = extract_tire_degradation(sessions[label])
            dfs.append(deg_df)
            break

    # Weather from qualifying (representative of race conditions)
    weather = {}
    if "Q" in sessions:
        weather = extract_weather(sessions["Q"])
    elif "FP2" in sessions:
        weather = extract_weather(sessions["FP2"])

    # Merge all driver-level DataFrames on driver_code
    if not dfs:
        raise ValueError(f"No usable sessions found for {year} {gp}")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="driver_code", how="outer")

    # Broadcast weather (circuit-level) to all drivers
    for k, v in weather.items():
        merged[k] = v

    # Attach core metadata (team/driver/circuit) used by feature engineering.
    meta_session = sessions.get("Q") or sessions.get("R") or next(iter(sessions.values()))
    meta = pd.DataFrame({"driver_code": merged["driver_code"].astype(str)})
    results = getattr(meta_session, "results", None)
    if results is not None and not results.empty:
        res = results.reset_index()
        if "Abbreviation" in res.columns:
            meta_map = pd.DataFrame({"driver_code": res["Abbreviation"].astype(str)})
            if "TeamName" in res.columns:
                meta_map["team_name"] = res["TeamName"].astype(str)
            elif "Team" in res.columns:
                meta_map["team_name"] = res["Team"].astype(str)
            if "FullName" in res.columns:
                meta_map["driver_name"] = res["FullName"].astype(str)
            elif "BroadcastName" in res.columns:
                meta_map["driver_name"] = res["BroadcastName"].astype(str)
            meta = meta.merge(meta_map, on="driver_code", how="left")

    merged = merged.merge(meta.drop_duplicates(subset=["driver_code"]), on="driver_code", how="left")
    if "team_name" not in merged.columns:
        merged["team_name"] = "UNKNOWN"
    else:
        merged["team_name"] = merged["team_name"].fillna("UNKNOWN")

    gp_text = str(gp)
    circuit_id = gp_text.lower().replace(" grand prix", "").replace(" ", "_")
    merged["circuit_id"] = circuit_id
    merged["year"] = year
    merged["gp"] = gp_text

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s - %(message)s")
    # Quick smoke test — 2024 Bahrain GP qualifying
    df = build_weekend_features(2024, "Bahrain")
    print(df.head())
    print(df.columns.tolist())
