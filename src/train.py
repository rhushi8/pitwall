"""
src/train.py
────────────
Train the stacking ensemble on historical race data.

Usage:
    python src/train.py --csv data/processed/historical_results.csv

Expected CSV columns (minimum):
    year, gp, circuit_id, driver_code, team_name,
    grid_position, finish_position, dnf,
    q1_time_s, q2_time_s, q3_time_s,
    fp2_best_lap_s, fp2_long_run_pace_s,
    deg_slope_soft, deg_slope_medium, deg_slope_hard,
    air_temp_c, track_temp_c, humidity_pct
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import MODEL_DIR
from src.features.engineer import build_feature_matrix
from src.logging_config import setup_logging
from src.models.ensemble import F1StackingEnsemble

log = logging.getLogger(__name__)


def load_historical(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows × %d cols from %s", *df.shape, csv_path)
    required = ["year", "gp", "driver_code", "finish_position"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in required:
        null_count = int(df[col].isna().sum())
        if null_count:
            log.warning("%s has %d null values", col, null_count)

    if not pd.api.types.is_numeric_dtype(df["finish_position"]):
        df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")

    invalid_finish = (df["finish_position"] < 1) | (df["finish_position"] > 30)
    invalid_count = int(invalid_finish.fillna(False).sum())
    if invalid_count:
        log.warning("Dropping %d rows with invalid finish_position", invalid_count)
        df = df[~invalid_finish.fillna(False)].copy()

    dup_keys = ["year", "gp", "driver_code"]
    if set(dup_keys).issubset(df.columns):
        dup_count = int(df.duplicated(subset=dup_keys).sum())
        if dup_count:
            log.warning("Dropping %d duplicate race-driver rows", dup_count)
            df = df.drop_duplicates(subset=dup_keys, keep="last")

    race_count = df[["year", "gp"]].drop_duplicates().shape[0]
    if race_count < 5:
        raise ValueError(f"Need at least 5 races for training, found {race_count}")

    log.info("Validated historical data: %d races, %d drivers", race_count, df["driver_code"].nunique())
    return df


def train(csv_path: Path, save: bool = True) -> F1StackingEnsemble:
    df = load_historical(csv_path)

    # Feature engineering
    log.info("Building feature matrix …")
    feature_result = build_feature_matrix(
        df,
        historical_results=df,
        fit=True,
    )
    X, artifacts = feature_result

    y_position = df["finish_position"].astype(float)
    y_dnf      = df["dnf"].astype(int) if "dnf" in df.columns else None
    y_sc       = df.get("safety_car_affected", None)

    # Train / validation split (group by race to avoid leakage)
    races = df["year"].astype(str) + "_" + df["gp"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y_position, groups=races))

    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_vl = y_position.iloc[train_idx], y_position.iloc[val_idx]
    y_dnf_tr   = y_dnf.iloc[train_idx] if y_dnf is not None else None
    y_sc_tr    = y_sc.iloc[train_idx]  if y_sc  is not None else None

    log.info("Training set: %d rows | Validation: %d rows", len(X_tr), len(X_vl))

    # Train ensemble
    ensemble = F1StackingEnsemble(n_splits=5)
    ensemble.fit(X_tr, y_tr, y_dnf=y_dnf_tr, y_sc=y_sc_tr)

    # Evaluate on held-out races
    preds = ensemble.predict(X_vl)
    mae   = mean_absolute_error(y_vl, preds["position_pred"])
    log.info("Validation MAE (position): %.3f positions", mae)

    # Feature importance
    fi = ensemble.feature_importance()
    if not fi.empty:
        print("\nTop 10 features:")
        print(fi.head(10).to_string(index=False))

    if save:
        ensemble._artifacts = artifacts
        path = ensemble.save()
        log.info("Model saved to %s", path)

    return ensemble


if __name__ == "__main__":
    setup_logging(log_file=MODEL_DIR.parent / "data" / "processed" / "f1_predictor.log")
    parser = argparse.ArgumentParser(description="Train F1 prediction ensemble")
    parser.add_argument("--csv",    type=str, required=True, help="Path to historical results CSV")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    train(Path(args.csv), save=not args.no_save)
