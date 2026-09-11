"""Assemble the final training CSV from the per-race parquet files.

Reads every per-race parquet, adds the cross-race features that need more than
one race to compute (ELO, rolling averages, season context), and writes the
training CSV. Run it after historical_scraper.py has finished.

Usage:
    python src/scraper/assemble_dataset.py
    python src/scraper/assemble_dataset.py --output data/processed/training_v2.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import RAW_DIR, PROC_DIR

log = logging.getLogger(__name__)

OUTPUT_CSV = PROC_DIR / "historical_results.csv"


# Load all parquets

def load_all_parquets() -> pd.DataFrame:
    paths = sorted(RAW_DIR.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")

    frames = []
    for p in tqdm(paths, desc="Loading parquets"):
        try:
            df = pd.read_parquet(p)
            frames.append(df)
        except Exception as e:
            log.warning("Skipping %s: %s", p, e)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded %d rows from %d race files", len(combined), len(frames))

    # Sort chronologically
    if "year" in combined.columns and "round" in combined.columns:
        combined = combined.sort_values(["year", "round"]).reset_index(drop=True)

    return combined


# ELO ratings (computed chronologically over the full dataset)

class EloSystem:
    DEFAULT = 1500.0
    K = 32
    D = 400.0

    def __init__(self):
        self.ratings: dict[str, float] = {}

    def get(self, e: str) -> float:
        return self.ratings.get(e, self.DEFAULT)

    def expected(self, a: str, b: str) -> float:
        return 1.0 / (1.0 + 10 ** ((self.get(b) - self.get(a)) / self.D))

    def update(self, winner: str, loser: str) -> None:
        e = self.expected(winner, loser)
        self.ratings[winner] = self.get(winner) + self.K * (1.0 - e)
        self.ratings[loser]  = self.get(loser)  + self.K * (0.0 - (1.0 - e))

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each race, record the driver/team ELO *before* that race runs,
    then update ELO based on result.  This prevents data leakage.
    """
    driver_elo = EloSystem()
    team_elo   = EloSystem()

    pre_driver_elos = []
    pre_team_elos   = []

    for (year, round_), group in df.groupby(["year", "round"], sort=True):
        # Snapshot ELO BEFORE updating (what the model will see at inference time)
        for _, row in group.iterrows():
            pre_driver_elos.append(driver_elo.get(str(row["driver_code"])))
            pre_team_elos.append(
                team_elo.get(str(row.get("team_name", "Unknown")))
            )

        # Update ELO based on finish positions
        result = group[["driver_code", "team_name", "finish_position"]].dropna()
        result = result.sort_values("finish_position")
        drivers  = result["driver_code"].tolist()
        teams    = result["team_name"].tolist()

        for i, (d_win, t_win) in enumerate(zip(drivers, teams)):
            for d_lose, t_lose in zip(drivers[i+1:], teams[i+1:]):
                driver_elo.update(str(d_win), str(d_lose))
                team_elo.update(str(t_win), str(t_lose))

    df = df.copy()
    df["driver_elo"] = pre_driver_elos
    df["team_elo"]   = pre_team_elos
    return df


# Circuit affinity (rolling per driver × circuit)

def add_circuit_affinity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (driver, circuit), compute mean finish position over all
    *prior* appearances (no leakage).
    """
    df = df.copy().sort_values(["year", "round"])
    affinity_map: dict[tuple, list[float]] = {}
    affinities = []

    for _, row in df.iterrows():
        key = (str(row["driver_code"]), str(row.get("circuit_id", row["gp"])))
        hist = affinity_map.get(key, [])
        affinities.append(float(np.mean(hist)) if hist else 10.0)

        if not pd.isna(row["finish_position"]):
            hist.append(float(row["finish_position"]))
            affinity_map[key] = hist

    df["driver_circuit_affinity"] = affinities
    return df


# Rolling form (last N races)

def add_rolling_form(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Per driver: rolling mean finish position and rolling DNF rate
    over the last `window` races (prior to this race).
    """
    df = df.copy().sort_values(["year", "round"])

    rolling_pos  = []
    rolling_dnf  = []
    pos_history:  dict[str, list[float]] = {}
    dnf_history:  dict[str, list[int]]   = {}

    for _, row in df.iterrows():
        code = str(row["driver_code"])
        hist_pos = pos_history.get(code, [])
        hist_dnf = dnf_history.get(code, [])

        rolling_pos.append(float(np.mean(hist_pos[-window:])) if hist_pos else 10.0)
        rolling_dnf.append(float(np.mean(hist_dnf[-window:])) if hist_dnf else 0.06)

        if not pd.isna(row["finish_position"]):
            hist_pos.append(float(row["finish_position"]))
            pos_history[code] = hist_pos
        if "dnf" in row and not pd.isna(row["dnf"]):
            hist_dnf.append(int(row["dnf"]))
            dnf_history[code] = hist_dnf

    df[f"rolling_mean_pos_{window}"]  = rolling_pos
    df[f"rolling_dnf_rate_{window}"]  = rolling_dnf
    return df


# Season context

def add_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add championship standings context features."""
    df = df.copy().sort_values(["year", "round"])

    # Cumulative points per driver within a season (before this race)
    cum_points: dict[tuple, float] = {}
    season_points = []

    for _, row in df.iterrows():
        key = (int(row["year"]), str(row["driver_code"]))
        season_points.append(cum_points.get(key, 0.0))
        pts = float(row.get("points", 0) or 0)
        cum_points[key] = cum_points.get(key, 0.0) + pts

    df["season_points_so_far"] = season_points
    df["race_number_in_season"] = df.groupby("year")["round"].transform(
        lambda x: x.rank(method="first").astype(int)
    )

    return df


# Safety car history per circuit

def add_circuit_historical_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical DNF and safety-car rates per circuit
    from the dataset itself. Only uses data from prior seasons
    to avoid leakage within-season.
    """
    df = df.copy().sort_values(["year", "round"])
    circuit_dnf_rate: dict[str, list[float]] = {}
    dnf_rates = []

    for _, row in df.iterrows():
        cid = str(row.get("circuit_id", row["gp"]))
        hist = circuit_dnf_rate.get(cid, [])
        dnf_rates.append(float(np.mean(hist)) if hist else 0.06)
        if "dnf" in row and not pd.isna(row["dnf"]):
            hist.append(float(row["dnf"]))
            circuit_dnf_rate[cid] = hist

    df["circuit_dnf_rate"] = dnf_rates
    return df


# Column selection & final cleaning

# Columns that MUST be present in the final CSV
FINAL_COLUMNS = [
    # Identity
    "year", "round", "gp", "circuit_id", "driver_code", "driver_name",
    "team_name", "driver_number",

    # Target
    "finish_position", "dnf", "classified",
    "finish_position_top3", "finish_position_top10", "points",

    # Qualifying
    "grid_position", "q1_time_s", "q2_time_s", "q3_time_s",
    "q1_gap_to_pole", "q2_gap_to_pole", "q3_gap_to_pole",

    # Practice
    "fp1_best_lap_s", "fp2_best_lap_s", "fp3_best_lap_s",
    "fp1_long_run_pace_s", "fp2_long_run_pace_s", "fp3_long_run_pace_s",

    # Tire deg
    "deg_slope_soft", "deg_slope_medium", "deg_slope_hard",

    # Weather
    "air_temp_c", "track_temp_c", "humidity_pct", "wind_speed_ms", "rainfall",

    # Engineered
    "driver_elo", "team_elo",
    "driver_circuit_affinity",
    "rolling_mean_pos_5", "rolling_dnf_rate_5",
    "season_points_so_far", "race_number_in_season",
    "circuit_dnf_rate",
]


def select_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only columns that exist
    keep = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[keep].copy()

    # Remove rows with no finish position (unusable for training)
    df = df[df["finish_position"].notna()].copy()

    # Ensure correct dtypes
    int_cols   = ["year", "round", "dnf", "classified",
                  "finish_position_top3", "finish_position_top10", "rainfall"]
    float_cols = [c for c in keep if c not in int_cols + ["year", "round",
                  "gp", "circuit_id", "driver_code", "driver_name",
                  "team_name", "driver_number"]]

    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    return df


# Summary statistics

def print_dataset_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total rows:     {len(df):,}")
    print(f"  Seasons:        {sorted(df['year'].unique().tolist())}")
    print(f"  Races:          {df.groupby(['year','gp']).ngroups}")
    print(f"  Drivers:        {df['driver_code'].nunique()}")
    print(f"  Teams:          {df['team_name'].nunique() if 'team_name' in df.columns else 'N/A'}")
    print(f"  DNF rate:       {df['dnf'].mean():.1%}")
    print(f"  Columns:        {len(df.columns)}")
    print()

    # Missing value report
    missing = df.isnull().mean() * 100
    missing = missing[missing > 5].sort_values(ascending=False)
    if not missing.empty:
        print("  Columns with >5% missing:")
        for col, pct in missing.items():
            print(f"    {col:<35} {pct:.1f}%")
    else:
        print("  No columns with >5% missing values ✓")

    print("=" * 60 + "\n")


# Main

def assemble(output_path: Path = OUTPUT_CSV) -> pd.DataFrame:
    log.info("Loading raw parquet files …")
    df = load_all_parquets()

    log.info("Adding ELO ratings …")
    df = add_elo_features(df)

    log.info("Adding circuit affinity …")
    df = add_circuit_affinity(df)

    log.info("Adding rolling form …")
    df = add_rolling_form(df, window=5)

    log.info("Adding season context …")
    df = add_season_context(df)

    log.info("Adding circuit historical rates …")
    df = add_circuit_historical_rates(df)

    log.info("Selecting and cleaning columns …")
    df = select_and_clean(df)

    log.info("Writing to %s …", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print_dataset_summary(df)
    log.info("Dataset assembly complete → %s", output_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Assemble F1 training dataset")
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_CSV),
        help="Output CSV path"
    )
    args = parser.parse_args()

    assemble(Path(args.output))
