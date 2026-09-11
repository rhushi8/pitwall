"""Turn raw ingestion frames into model-ready features.

Computes driver and team ELO, adds circuit-level history, normalises and
encodes the categoricals, builds interaction terms like pace x tire-deg, and
fills missing values with sensible defaults.
"""
from __future__ import annotations

import logging
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

log = logging.getLogger(__name__)
_ELO_CACHE: dict[tuple[int, int, Optional[tuple[int, str]]], tuple[EloRating, EloRating]] = {}

# ELO rating system

class EloRating:
    """
    Simple K-factor ELO for F1 drivers/teams.
    Higher ELO = historically stronger performer.
    """

    DEFAULT_ELO = 1500.0
    K           = 32          # update speed, tune per season
    D           = 400.0       # logistic scale

    def __init__(self, initial_ratings: Optional[dict[str, float]] = None):
        self.ratings: dict[str, float] = dict(initial_ratings or {})

    def copy(self) -> "EloRating":
        return EloRating(self.ratings)

    def get(self, entity: str) -> float:
        return self.ratings.get(entity, self.DEFAULT_ELO)

    def expected_score(self, a: str, b: str) -> float:
        """P(A beats B)."""
        return 1.0 / (1.0 + 10 ** ((self.get(b) - self.get(a)) / self.D))

    def update(self, winner: str, loser: str) -> None:
        exp = self.expected_score(winner, loser)
        self.ratings[winner] = self.get(winner) + self.K * (1.0 - exp)
        self.ratings[loser]  = self.get(loser)  + self.K * (0.0 - (1 - exp))

    def update_from_race(self, results_df: pd.DataFrame,
                          entity_col: str = "driver_code",
                          position_col: str = "finish_position") -> None:
        """Update ELO from a full race result (pairwise comparisons)."""
        df = results_df[[entity_col, position_col]].dropna()
        df = df.sort_values(position_col)
        entities = df[entity_col].tolist()
        # Each driver beats all drivers behind them
        for i, winner in enumerate(entities):
            for loser in entities[i + 1:]:
                self.update(winner, loser)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(self.ratings.items()), columns=["entity", "elo"]
        ).sort_values("elo", ascending=False).reset_index(drop=True)


def build_elo_ratings(
    historical_results: pd.DataFrame,
    exclude_race: Optional[tuple[int, str]] = None,
) -> tuple[EloRating, EloRating]:
    """
    Build driver and team ELO ratings from a historical results DataFrame.

    Expected columns: year, gp, driver_code, team_name, finish_position
    Returns: (driver_elo, team_elo)
    """
    if historical_results.empty:
        return EloRating(), EloRating()

    key = (
        len(historical_results),
        int(pd.util.hash_pandas_object(historical_results[["year", "gp", "driver_code", "team_name", "finish_position"]], index=False).sum()),
        exclude_race,
    )
    if key in _ELO_CACHE:
        cached_driver, cached_team = _ELO_CACHE[key]
        return cached_driver.copy(), cached_team.copy()

    driver_elo = EloRating()
    team_elo = EloRating()

    data = historical_results.copy()
    if exclude_race is not None and {"year", "gp"}.issubset(data.columns):
        year, gp = exclude_race
        data = data[(data["year"] != year) | (data["gp"] != gp)]

    for (_, _), group in data.groupby(["year", "gp"]):
        driver_elo.update_from_race(group, entity_col="driver_code")
        team_elo.update_from_race(group, entity_col="team_name")

    _ELO_CACHE[key] = (driver_elo.copy(), team_elo.copy())
    return driver_elo, team_elo


def _sorted_races(historical_results: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "gp"]
    if "round" in historical_results.columns:
        cols.append("round")
    race_df = historical_results[cols].drop_duplicates()
    if "round" in race_df.columns:
        return race_df.sort_values(["year", "round", "gp"]).reset_index(drop=True)
    return race_df.sort_values(["year", "gp"]).reset_index(drop=True)


def _add_pre_race_elo(
    df: pd.DataFrame,
    historical_results: pd.DataFrame,
) -> tuple[pd.DataFrame, EloRating, EloRating]:
    """Attach pre-race ELO values so race outcomes do not leak into features."""
    if historical_results.empty:
        return df, EloRating(), EloRating()

    driver_elo = EloRating()
    team_elo = EloRating()
    race_snapshots: list[pd.DataFrame] = []
    ordered_races = _sorted_races(historical_results)

    for _, race in ordered_races.iterrows():
        race_year = int(race["year"])
        race_gp = str(race["gp"])
        race_df = historical_results[
            (historical_results["year"] == race_year) &
            (historical_results["gp"] == race_gp)
        ].copy()
        if race_df.empty:
            continue

        race_df["driver_elo"] = race_df["driver_code"].map(driver_elo.ratings).fillna(EloRating.DEFAULT_ELO)
        race_df["team_elo"] = race_df["team_name"].map(team_elo.ratings).fillna(EloRating.DEFAULT_ELO)
        race_snapshots.append(race_df[["year", "gp", "driver_code", "team_name", "driver_elo", "team_elo"]])

        driver_elo.update_from_race(race_df, entity_col="driver_code")
        team_elo.update_from_race(race_df, entity_col="team_name")

    elo_snapshot = (
        pd.concat(race_snapshots, ignore_index=True)
        if race_snapshots
        else pd.DataFrame(columns=["year", "gp", "driver_code", "team_name", "driver_elo", "team_elo"])
    )
    # Input data may already carry ELO columns; drop them so merge creates
    # canonical names instead of driver_elo_x/driver_elo_y.
    base_df = df.drop(columns=["driver_elo", "team_elo"], errors="ignore")
    merged = base_df.merge(elo_snapshot, on=["year", "gp", "driver_code", "team_name"], how="left")
    merged["driver_elo"] = merged["driver_elo"].fillna(
        merged["driver_code"].map(driver_elo.ratings).fillna(EloRating.DEFAULT_ELO)
    )
    merged["team_elo"] = merged["team_elo"].fillna(
        merged["team_name"].map(team_elo.ratings).fillna(EloRating.DEFAULT_ELO)
    )
    return merged, driver_elo, team_elo


def _add_pre_race_form_features(df: pd.DataFrame, historical_results: pd.DataFrame) -> pd.DataFrame:
    """Attach recent form features using races strictly before the target race."""
    out = df.copy()
    if historical_results.empty:
        return out

    required = {"year", "gp", "driver_code", "finish_position"}
    if not required.issubset(historical_results.columns):
        return out

    race_year = int(out["year"].iloc[0]) if "year" in out.columns else None
    race_gp = str(out["gp"].iloc[0]) if "gp" in out.columns else None

    hist = historical_results.copy()
    if race_year is not None and race_gp is not None and {"year", "gp"}.issubset(hist.columns):
        hist = hist[(hist["year"] < race_year) | ((hist["year"] == race_year) & (hist["gp"] != race_gp))]

    driver_form_mean = (
        hist.sort_values(["year", "gp"]).groupby("driver_code")["finish_position"]
        .apply(lambda s: s.tail(5).mean())
    )
    out["rolling_mean_pos_5"] = out.get("rolling_mean_pos_5", np.nan)
    out["rolling_mean_pos_5"] = out["rolling_mean_pos_5"].fillna(out["driver_code"].map(driver_form_mean))

    if "dnf" in hist.columns:
        dnf_series = hist["dnf"].astype(float)
    elif "classified" in hist.columns:
        dnf_series = 1.0 - hist["classified"].astype(float)
    else:
        dnf_series = pd.Series(0.0, index=hist.index)
    dnf_tmp = hist.assign(_dnf=dnf_series)
    driver_dnf_rate = (
        dnf_tmp.sort_values(["year", "gp"]).groupby("driver_code")["_dnf"]
        .apply(lambda s: s.tail(5).mean())
    )
    out["rolling_dnf_rate_5"] = out.get("rolling_dnf_rate_5", np.nan)
    out["rolling_dnf_rate_5"] = out["rolling_dnf_rate_5"].fillna(out["driver_code"].map(driver_dnf_rate))

    if "points" in hist.columns:
        season_points = hist.groupby(["year", "driver_code"])["points"].sum()
        out["season_points_so_far"] = out.get("season_points_so_far", np.nan)
        season_points_vals = pd.Series(
            [season_points.get((race_year, code), 0.0) for code in out["driver_code"]],
            index=out.index,
            dtype=float,
        )
        out["season_points_so_far"] = out["season_points_so_far"].fillna(season_points_vals)

    if "gp" in hist.columns:
        race_counts = hist[hist["year"] == race_year].groupby("gp").ngroups if race_year is not None else 0
        out["race_number_in_season"] = out.get("race_number_in_season", np.nan)
        out["race_number_in_season"] = out["race_number_in_season"].fillna(float(race_counts + 1))

    if "circuit_id" in hist.columns:
        dnf_col = "dnf" if "dnf" in hist.columns else None
        if dnf_col is not None:
            circuit_dnf = hist.groupby("circuit_id")[dnf_col].mean()
            circuit_id = out["circuit_id"].iloc[0] if "circuit_id" in out.columns else None
            out["circuit_dnf_rate"] = out.get("circuit_dnf_rate", np.nan)
            if circuit_id is not None:
                out["circuit_dnf_rate"] = out["circuit_dnf_rate"].fillna(float(circuit_dnf.get(circuit_id, 0.06)))

    return out


# Circuit affinity

def compute_circuit_affinity(
    historical_results: pd.DataFrame,
    circuit_col: str = "circuit_id",
) -> pd.DataFrame:
    """
    For each (driver, circuit) pair, compute mean and std finish position
    over historical races.  Returns a DataFrame with columns:
        driver_code, circuit_id, circuit_affinity_mean, circuit_affinity_std
    """
    aff = (
        historical_results
        .groupby(["driver_code", circuit_col])["finish_position"]
        .agg(circuit_affinity_mean="mean", circuit_affinity_std="std")
        .reset_index()
    )
    return aff


# Track evolution coefficient

def compute_track_evolution(
    practice_laps_df: pd.DataFrame,
    session_order: list[str] = ["FP1", "FP2", "FP3", "Q"],
) -> float:
    """
    Estimate how much the track 'rubbered in' across the weekend.
    Returns a coefficient: pace improvement (seconds) from FP1 to Q.
    Positive = track improved = faster over weekend.
    """
    session_bests = {}
    for sess in session_order:
        col = f"{sess.lower()}_best_lap_s"
        if col in practice_laps_df.columns:
            session_bests[sess] = practice_laps_df[col].dropna().min()

    if "FP1" in session_bests and "Q" in session_bests:
        return round(session_bests["FP1"] - session_bests["Q"], 3)
    return 0.0


# Interaction features

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived interaction features:
    - pace_deg_product: long-run pace × tire deg slope (punishes cars with
      both high deg and poor pace)
    - qual_practice_delta: qualifying gap minus best practice lap delta
      (positive = over-performed in qualifying)
    - wet_flag × wet_skill: interaction term for wet races
    """
    df = df.copy()

    # Pace × degradation interaction (use FP2 long run pace as proxy)
    if "fp2_long_run_pace_s" in df and "deg_slope_medium" in df:
        df["pace_deg_product"] = df["fp2_long_run_pace_s"] * df["deg_slope_medium"]

    # Qualification over/under performance vs FP2 pace
    if "q3_time_s" in df and "fp2_best_lap_s" in df:
        df["qual_practice_delta"] = df["q3_time_s"] - df["fp2_best_lap_s"]

    # Rolling average of last 3 races (requires historical context)
    # added during dataset construction, not here

    return df


# Imputation

IMPUTATION_DEFAULTS = {
    # If driver didn't make Q2/Q3, impute with field median + penalty
    "q2_time_s":        None,   # will use column median + 0.5 s
    "q3_time_s":        None,
    "q2_gap_to_pole":   2.0,
    "q3_gap_to_pole":   3.0,

    # Practice availability
    "fp1_best_lap_s":   None,
    "fp2_best_lap_s":   None,
    "fp3_best_lap_s":   None,
    "fp2_long_run_pace_s": None,

    # Tire deg: assume moderate deg if no data
    "deg_slope_soft":   0.15,
    "deg_slope_medium": 0.08,
    "deg_slope_hard":   0.04,
    "crossover_lap_s_m":10.0,
    "crossover_lap_m_h":20.0,

    # ELO defaults
    "driver_elo":       1500.0,
    "team_elo":         1500.0,
    "driver_circuit_affinity": 10.0,
    "rolling_mean_pos_5": 10.0,
    "rolling_dnf_rate_5": 0.08,
    "season_points_so_far": 0.0,
    "race_number_in_season": 1.0,
    "circuit_dnf_rate": 0.06,

    # Weather
    "air_temp_c":       25.0,
    "track_temp_c":     40.0,
    "humidity_pct":     50.0,
    "wind_speed_kph":   10.0,
    "track_evolution_coeff": 1.0,
}


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with domain-aware defaults."""
    df = df.copy()
    imputation_log: list[tuple[str, int, str]] = []
    for col, default in IMPUTATION_DEFAULTS.items():
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if default is None:
            # Use median + small penalty
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
            penalty = 0.3
            df[col] = df[col].fillna(median + penalty)
            imputation_log.append((col, n_missing, f"median+{penalty}"))
        else:
            df[col] = df[col].fillna(default)
            imputation_log.append((col, n_missing, str(default)))
    if imputation_log:
        for col, count, strategy in imputation_log:
            log.info("Imputed %d missing values in %s using %s", count, col, strategy)
    return df


# Encoding

def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: Optional[list[str]] = None,
    fit: bool = True,
    encoders: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns (encoded_df, encoders_dict) so encoders can be reused at inference.
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    df = df.copy()
    encoders = encoders or {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("UNKNOWN"))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                df[col] = 0
            else:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    return df, encoders


# Scaling

def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    fit: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale numeric feature columns.
    Returns (scaled_df, fitted_scaler).
    """
    scaler = scaler or StandardScaler()
    df = df.copy()
    present = [c for c in feature_cols if c in df.columns]

    if fit:
        if not present:
            return df, scaler
        df[present] = scaler.fit_transform(df[present])
    else:
        # Keep inference schema aligned with training-time scaler inputs.
        expected = list(getattr(scaler, "feature_names_in_", present))
        if not expected:
            return df, scaler
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0
        df[expected] = scaler.transform(df[expected])

    return df, scaler


class FeatureMatrixResult(NamedTuple):
    X: pd.DataFrame
    artifacts: dict[str, Any]


def _validate_dataframe(df: pd.DataFrame, required: list[str], name: str) -> None:
    if df.empty:
        raise ValueError(f"{name} cannot be empty")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


# Full pipeline

def build_feature_matrix(
    raw_df: pd.DataFrame,
    historical_results: Optional[pd.DataFrame] = None,
    fit: bool = True,
    artifacts: Optional[dict] = None,
    feature_cols: Optional[list[str]] = None,
) -> FeatureMatrixResult:
    """
    End-to-end feature construction from a raw weekend DataFrame.

    Parameters
    ----------
    raw_df             : output of fastf1_loader.build_weekend_features()
    historical_results : all historical race results (for ELO, affinity)
    fit                : True during training, False at inference
    artifacts          : dict of {encoders, scaler, driver_elo, team_elo}
                         passed in at inference time

    Returns
    -------
    (X_df, artifacts)  : model-ready feature matrix and fitted artifacts
    """
    artifacts = artifacts or {}
    df = raw_df.copy()
    _validate_dataframe(df, ["driver_code"], "raw_df")
    if "team_name" not in df.columns:
        df["team_name"] = "UNKNOWN"

    race_cols_present = {"year", "gp"}.issubset(df.columns)
    if not race_cols_present:
        df["year"] = -1
        df["gp"] = "unknown"

    # 1. ELO ratings
    if historical_results is not None:
        _validate_dataframe(
            historical_results,
            ["year", "gp", "driver_code", "team_name", "finish_position"],
            "historical_results",
        )
        if fit and {"year", "gp", "driver_code", "team_name"}.issubset(df.columns):
            df, driver_elo, team_elo = _add_pre_race_elo(df, historical_results)
        else:
            exclude_race: Optional[tuple[int, str]] = None
            if race_cols_present and df[["year", "gp"]].drop_duplicates().shape[0] == 1:
                race_row = df[["year", "gp"]].iloc[0]
                exclude_race = (int(race_row["year"]), str(race_row["gp"]))
            driver_elo, team_elo = build_elo_ratings(historical_results, exclude_race=exclude_race)
            df["driver_elo"] = df["driver_code"].map(driver_elo.ratings).fillna(1500.0)
            df["team_elo"] = df["team_name"].map(team_elo.ratings).fillna(1500.0)

        artifacts["driver_elo"] = driver_elo
        artifacts["team_elo"]   = team_elo

        # Circuit affinity
        circuit_id = df["circuit_id"].iloc[0] if "circuit_id" in df.columns else "unknown"
        aff = compute_circuit_affinity(historical_results)
        aff_at_circuit = aff[aff["circuit_id"] == circuit_id][
            ["driver_code", "circuit_affinity_mean"]
        ]
        df = df.merge(aff_at_circuit, on="driver_code", how="left")
        df["driver_circuit_affinity"] = df.get(
            "circuit_affinity_mean", pd.Series(dtype=float)
        ).fillna(10.0)

        if not fit:
            df = _add_pre_race_form_features(df, historical_results)

    # 2. Track evolution
    df["track_evolution_coeff"] = compute_track_evolution(df)

    # 3. Interaction features
    df = add_interaction_features(df)

    # Guard against divide-by-zero artifacts from interaction features.
    inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
    inf_count = int(inf_mask.sum().sum()) if hasattr(inf_mask, "sum") else 0
    if inf_count:
        log.warning("Replacing %d infinite feature values with NaN before imputation", inf_count)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4. Impute
    df = impute_features(df)

    # 5. Encode
    cat_cols = ["driver_code", "team_name", "circuit_id", "first_compound"]
    df, encoders = encode_categoricals(
        df, cat_cols=cat_cols, fit=fit, encoders=artifacts.get("encoders")
    )
    artifacts["encoders"] = encoders

    # 6. Scale
    if feature_cols is None:
        from config.settings import ALL_FEATURES
        feature_cols = list(ALL_FEATURES)
    numeric_features = [c for c in feature_cols if c in df.columns]
    df, scaler = scale_features(
        df, feature_cols=numeric_features, fit=fit, scaler=artifacts.get("scaler")
    )
    artifacts["scaler"] = scaler

    return FeatureMatrixResult(X=df, artifacts=artifacts)
