"""
src/predict.py
──────────────
End-to-end prediction pipeline.

Usage:
    python src/predict.py --year 2024 --gp Bahrain --sims 10000

Steps:
  1. Load race weekend data (FastF1 + OpenF1)
  2. Engineer features
  3. Load or train ensemble model
  4. Build DriverProfile objects from model predictions
  5. Run Monte Carlo simulation
  6. Print and save results
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    MODEL_DIR,
    PROC_DIR,
    MC_SIMULATIONS,
    DNF_BASE_PROB_PER_RACE,
    CALIBRATION_PARAMS,
)
from src.ingestion.fastf1_loader import build_weekend_features
from src.ingestion.openf1_client import enrich_with_openf1
from src.features.engineer import build_feature_matrix
from src.logging_config import setup_logging
from src.models.ensemble import F1StackingEnsemble
from src.simulation.monte_carlo import (
    DriverProfile, CircuitProfile, run_monte_carlo, optimise_strategy,
    STRATEGY_OPTIONS,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Default circuit profiles (add more as needed)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CIRCUIT_PROFILES: dict[str, CircuitProfile] = {
    "bahrain": CircuitProfile("Bahrain International Circuit", 57, safety_car_rate=0.05, overtaking_factor=1.2),
    "jeddah": CircuitProfile("Jeddah Corniche Circuit", 50, safety_car_rate=0.09, overtaking_factor=0.7),
    "suzuka": CircuitProfile("Suzuka Circuit", 53, safety_car_rate=0.06, overtaking_factor=0.8),
    "monaco": CircuitProfile("Circuit de Monaco", 78, safety_car_rate=0.12, overtaking_factor=0.3),
    "spa": CircuitProfile("Circuit de Spa-Francorchamps", 44, safety_car_rate=0.08, overtaking_factor=1.3, weather_rain_prob=0.30),
    "monza": CircuitProfile("Autodromo Nazionale Monza", 53, safety_car_rate=0.07, overtaking_factor=1.5),
    "silverstone": CircuitProfile("Silverstone Circuit", 52, safety_car_rate=0.07, overtaking_factor=1.1, weather_rain_prob=0.20),
    "cota": CircuitProfile("Circuit of the Americas", 56, safety_car_rate=0.06, overtaking_factor=1.2),
}


def load_circuit_profiles(config_path: Path | None = None) -> dict[str, CircuitProfile]:
    config_path = config_path or Path(__file__).parent.parent / "config" / "circuits.json"
    if not config_path.exists():
        return DEFAULT_CIRCUIT_PROFILES

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to load circuit profile config %s: %s", config_path, exc)
        return DEFAULT_CIRCUIT_PROFILES

    profiles: dict[str, CircuitProfile] = {}
    for key, cfg in raw.items():
        profiles[key] = CircuitProfile(
            name=cfg.get("name", key),
            total_laps=int(cfg.get("total_laps", 55)),
            safety_car_rate=float(cfg.get("safety_car_rate", 0.06)),
            overtaking_factor=float(cfg.get("overtaking_factor", 1.0)),
            weather_rain_prob=float(cfg.get("weather_rain_prob", 0.0)),
        )
    return profiles or DEFAULT_CIRCUIT_PROFILES


CIRCUIT_PROFILES = load_circuit_profiles()


def get_circuit_profile(gp_name: str) -> CircuitProfile:
    key = gp_name.lower().replace(" ", "")
    for k, v in CIRCUIT_PROFILES.items():
        if k in key or key in k:
            return v
    log.warning("No circuit profile for '%s' — using generic defaults", gp_name)
    return CircuitProfile(gp_name, total_laps=55, safety_car_rate=0.06)


def select_strategy(row: pd.Series, circuit: CircuitProfile) -> list[tuple[str, int]]:
    deg = float(row.get("deg_slope_medium", 0.08))
    qual_practice_delta = float(row.get("qual_practice_delta", 0.0))
    overtaking = float(circuit.overtaking_factor)

    if deg < 0.05 and qual_practice_delta < 0.2 and overtaking > 1.15:
        return STRATEGY_OPTIONS["1-stop: S→H"]
    if deg < 0.10:
        return STRATEGY_OPTIONS["2-stop: S→M→H"]
    return STRATEGY_OPTIONS["2-stop: S→H→M"]


def sanitize_driver_rows(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one valid row per driver before feature engineering.

    FastF1/OpenF1 joins can occasionally create duplicate rows or malformed
    driver identifiers. This sanitization keeps the best qualifying/grid row
    for each driver and prevents inflated race fields from distorting rankings.
    """
    df = feature_df.copy()
    if df.empty:
        return df

    code_col = "driver_code" if "driver_code" in df.columns else None
    if code_col is None:
        return df

    codes = df[code_col].astype(str).str.upper().str.strip()
    valid_mask = (codes.str.len() == 3) & codes.str.isalpha()
    dropped_invalid = int((~valid_mask).sum())
    if dropped_invalid:
        log.warning("Dropping %d rows with invalid driver codes", dropped_invalid)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return df

    df[code_col] = df[code_col].astype(str).str.upper().str.strip()

    # Prefer rows with strongest quali evidence, then lower grid position.
    sort_cols: list[str] = []
    for col in ["q3_gap_to_pole", "q2_gap_to_pole", "q1_gap_to_pole", "grid_position"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            sort_cols.append(col)

    if sort_cols:
        for col in sort_cols:
            df[col] = df[col].fillna(np.inf)
        df = df.sort_values(sort_cols, ascending=True)

    before = len(df)
    df = df.drop_duplicates(subset=[code_col], keep="first")
    dropped_dupes = before - len(df)
    if dropped_dupes:
        log.warning("Dropped %d duplicate driver rows", dropped_dupes)

    # F1 race field should be at most 20; clip to best-qualified entries.
    if len(df) > 20:
        log.warning("Detected %d driver rows; clipping to top 20 by qualifying/grid", len(df))
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True)
        df = df.head(20).copy()

    return df.reset_index(drop=True)


def _qualifying_rank_from_features(df: pd.DataFrame) -> pd.Series:
    """Best-effort qualifying rank using grid first, then qualifying gaps."""
    if "grid_position" in df.columns and not df["grid_position"].isna().all():
        return pd.to_numeric(df["grid_position"], errors="coerce")

    for col in ["q3_gap_to_pole", "q2_gap_to_pole", "q1_gap_to_pole"]:
        if col in df.columns and not df[col].isna().all():
            return df[col].rank(method="first", ascending=True)

    return pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float)


def calibrate_position_predictions(
    feature_df: pd.DataFrame,
    preds: dict[str, np.ndarray],
    model_weight: float | None = None,
) -> dict[str, np.ndarray]:
    """Blend model position prediction with qualifying prior for realism."""
    if model_weight is None:
        model_weight = float(CALIBRATION_PARAMS.get("model_position_weight", 0.50))
    out = dict(preds)
    model_pos = np.asarray(out.get("position_pred", np.full(len(feature_df), 10.0)), dtype=float)
    qual_rank = _qualifying_rank_from_features(feature_df).to_numpy(dtype=float)
    blended = model_weight * model_pos + (1.0 - model_weight) * qual_rank
    out["position_pred"] = np.clip(blended, 1.0, 20.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build DriverProfile list from model predictions
# ─────────────────────────────────────────────────────────────────────────────

def build_driver_profiles(
    feature_df: pd.DataFrame,
    model_preds: dict[str, np.ndarray],
    circuit: CircuitProfile,
) -> list[DriverProfile]:
    """Convert feature matrix + model predictions into DriverProfile objects."""
    def _safe_float(value: object, default: float) -> float:
        try:
            out = float(value)
            return out if np.isfinite(out) else default
        except (TypeError, ValueError):
            return default

    profiles = []
    df_local = feature_df.copy()

    # If grid positions are missing, approximate them from qualifying gaps.
    if "grid_position" not in df_local.columns or df_local["grid_position"].isna().all():
        qual_col = None
        for c in ["q3_gap_to_pole", "q2_gap_to_pole", "q1_gap_to_pole"]:
            if c in df_local.columns and not df_local[c].isna().all():
                qual_col = c
                break
        if qual_col is not None:
            order = df_local[qual_col].rank(method="first", ascending=True)
            df_local["grid_position"] = order.astype(int)
        else:
            df_local["grid_position"] = np.arange(1, len(df_local) + 1)

    pred_positions = np.asarray(model_preds.get("position_pred", np.full(len(df_local), 10.0)), dtype=float)

    for i, row in df_local.iterrows():
        code         = str(row.get("driver_code_raw", row.get("driver_code", f"D{i:02d}")))
        grid_pos_raw = _safe_float(row.get("grid_position", i + 1), float(i + 1))
        grid_pos     = int(np.clip(round(grid_pos_raw), 1, 20))
        base_ref     = _safe_float(
            row.get("fp2_fuel_corrected_s", row.get("fp2_best_lap_s", 90.0)),
            90.0,
        )
        pred_pos     = _safe_float(pred_positions[i], 10.0)
        # Strongly anchor simulation pace to model rank and grid so results
        # remain plausible while still preserving stochastic race variation.
        model_pace_offset = (pred_pos - 10.0) * float(
            CALIBRATION_PARAMS.get("model_pace_offset_per_pos", 0.18)
        )
        grid_pace_offset = (grid_pos - 10.0) * float(
            CALIBRATION_PARAMS.get("grid_pace_offset_per_pos", 0.05)
        )
        base_pace = base_ref + model_pace_offset + grid_pace_offset
        deg_slope    = _safe_float(row.get("deg_slope_medium", 0.08), 0.08)
        dnf_prob_raw = _safe_float(model_preds.get("dnf_prob", np.full(len(df_local), 0.06))[i], 0.06)
        dnf_prob     = float(np.clip(dnf_prob_raw, 0.02, 0.18))

        strategy = select_strategy(row, circuit)

        profiles.append(
            DriverProfile(
                code=code,
                grid_position=grid_pos,
                base_pace_s=base_pace,
                pace_std=min(0.18, max(0.05, 0.06 + abs(deg_slope) * 0.35)),
                deg_slope=deg_slope,
                dnf_prob=dnf_prob,
                team_pit_time=_safe_float(row.get("pit_time_mean_s", 22.0), 22.0),
                wet_skill=_safe_float(row.get("driver_wet_skill", 0.0), 0.0),
                strategy=strategy,
            )
        )

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_race(
    year: int,
    gp: str,
    n_simulations: int = MC_SIMULATIONS,
    model_path: Path | None = None,
    historical_csv: Path | None = None,
    save_results: bool = True,
) -> dict:
    """
    Full prediction pipeline. Returns a results dict.
    """
    log.info("═" * 60)
    log.info("F1 RACE PREDICTOR  |  %d %s  |  %d simulations", year, gp, n_simulations)
    log.info("═" * 60)

    # 1. Data ingestion
    log.info("[1/5] Ingesting race weekend data …")
    feature_df = build_weekend_features(year, gp)
    feature_df = enrich_with_openf1(feature_df, year, gp)
    feature_df = sanitize_driver_rows(feature_df)

    # 2. Feature engineering
    log.info("[2/5] Engineering features …")
    historical = None
    default_historical = PROC_DIR / "historical_results.csv"
    if historical_csv is None and default_historical.exists():
        historical_csv = default_historical
        log.info("Using default historical CSV: %s", historical_csv)
    if historical_csv and Path(historical_csv).exists():
        historical = pd.read_csv(historical_csv)
    elif historical_csv:
        log.warning("Historical CSV not found at %s; proceeding without historical features", historical_csv)

    inference_artifacts: dict = {}

    # 3. Model predictions
    log.info("[3/5] Running ensemble model …")
    model_path = model_path or MODEL_DIR / "ensemble.pkl"
    if model_path.exists():
        ensemble = F1StackingEnsemble.load(model_path)
        inference_artifacts = getattr(ensemble, "_artifacts", {})
        feature_result = build_feature_matrix(
            feature_df,
            historical_results=historical,
            fit=False,
            artifacts=inference_artifacts,
            feature_cols=getattr(ensemble, "feature_cols", None),
        )
        feature_matrix = feature_result.X
        preds = ensemble.predict(feature_matrix)
        preds = calibrate_position_predictions(feature_df, preds)
    else:
        log.warning("No trained model found at %s — using heuristic preds", model_path)
        feature_result = build_feature_matrix(
            feature_df,
            historical_results=historical,
            fit=True,
            artifacts=None,
        )
        feature_matrix = feature_result.X
        # Heuristic fallback: rank by qualifying time
        preds = {
            "position_pred": feature_df.get("grid_position", pd.Series(range(1, len(feature_df)+1))).values.astype(float),
            "dnf_prob":      np.full(len(feature_df), DNF_BASE_PROB_PER_RACE),
            "safety_car_prob": np.full(len(feature_df), 0.15),
        }

    # Save raw driver codes before encoding overwrites them
    if "driver_code_raw" not in feature_df.columns:
        feature_df["driver_code_raw"] = feature_df["driver_code"]
    feature_matrix["driver_code_raw"] = feature_df["driver_code_raw"].values

    # 4. Monte Carlo simulation
    log.info("[4/5] Running Monte Carlo simulation (%d runs) …", n_simulations)
    circuit = get_circuit_profile(gp)
    driver_profiles = build_driver_profiles(feature_df, preds, circuit)
    mc_results = run_monte_carlo(
        driver_profiles, circuit,
        n_simulations=n_simulations,
        show_progress=True,
    )

    # 5. Output
    log.info("[5/5] Compiling results …")
    summary_df = mc_results.summary()
    mc_results.print_summary()

    # Strategy analysis for top-3 predicted drivers
    print("\n── STRATEGY ANALYSIS (top 3 drivers) ──\n")
    top3 = summary_df.head(3)["driver_code"].tolist()
    strategy_results = {}
    for code in top3:
        drv = next((d for d in driver_profiles if d.code == code), None)
        if drv:
            strat_df = optimise_strategy(drv, circuit, n_sims=2_000)
            strategy_results[code] = strat_df
            print(f"{code}:")
            print(strat_df.to_string(index=False))
            print()

    # Save to disk
    if save_results:
        out_dir = PROC_DIR / f"{year}_{gp.replace(' ', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_dir / "predictions.csv", index=False)
        for code, df in strategy_results.items():
            df.to_csv(out_dir / f"strategy_{code}.csv", index=False)
        log.info("Results saved to %s", out_dir)

    return {
        "summary":           summary_df,
        "strategy_analysis": strategy_results,
        "mc_results":        mc_results,
        "driver_profiles":   driver_profiles,
        "circuit":           circuit,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    setup_logging(log_file=PROC_DIR / "f1_predictor.log")

    parser = argparse.ArgumentParser(description="F1 Race Prediction Pipeline")
    parser.add_argument("--year",  type=int, default=2024)
    parser.add_argument("--gp",    type=str, default="Bahrain")
    parser.add_argument("--sims",  type=int, default=MC_SIMULATIONS)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--historical", type=str, default=None,
                        help="Path to historical results CSV for training")
    parser.add_argument("--no-save", action="store_true")

    args = parser.parse_args()

    predict_race(
        year=args.year,
        gp=args.gp,
        n_simulations=args.sims,
        model_path=Path(args.model) if args.model else None,
        historical_csv=Path(args.historical) if args.historical else None,
        save_results=not args.no_save,
    )
