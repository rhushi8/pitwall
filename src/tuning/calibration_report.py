"""
Multi-season calibration analysis and reporting.

Usage:
    python src/tuning/calibration_report.py
    python src/tuning/calibration_report.py --years 2021 2022 2023 2024
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import PROC_DIR, CALIBRATION_PARAMS


def generate_report(years: list[int]) -> None:
    """Generate multi-season calibration analysis report."""
    print("\n" + "=" * 80)
    print("F1 PREDICTOR — CALIBRATION ANALYSIS REPORT".center(80))
    print("=" * 80 + "\n")

    sweep_dfs: dict[int, pd.DataFrame] = {}
    for year in years:
        csv_path = PROC_DIR / f"walk_forward_{year}_calibration_sweep.csv"
        if csv_path.exists():
            sweep_dfs[year] = pd.read_csv(csv_path)
            print(f"✓ Loaded {year} sweep ({len(sweep_dfs[year])} weights)")
        else:
            print(f"✗ No sweep found for {year}")

    if not sweep_dfs:
        print("\nNo calibration sweeps found. Run backtests first:")
        print("  python src/tuning/walk_forward_backtest.py --year 2021 --optimize-calibration")
        return

    # Merge all sweeps
    merged = None
    for year, df in sweep_dfs.items():
        df_copy = df[["weight", "mae", "rounds"]].copy()
        df_copy.columns = [
            "weight",
            f"mae_{year}",
            f"rounds_{year}",
        ]
        if merged is None:
            merged = df_copy
        else:
            merged = merged.merge(df_copy, on="weight")

    # Compute weighted aggregate MAE
    round_cols = [c for c in merged.columns if c.startswith("rounds_")]

    merged["total_rounds"] = merged[round_cols].sum(axis=1)
    merged["weighted_mae"] = (
        sum(
            merged[f"mae_{year}"] * merged[f"rounds_{year}"]
            for year in sweep_dfs.keys()
        )
        / merged["total_rounds"]
    )

    # Sort by weighted MAE
    merged = merged.sort_values("weighted_mae")

    # Print per-year summary table
    print("\nPER-YEAR BEST WEIGHTS:")
    print("-" * 80)
    for year in sorted(sweep_dfs.keys()):
        best = sweep_dfs[year].sort_values("mae").iloc[0]
        print(
            f"  {year}: weight={best['weight']:>4.2f}  MAE={best['mae']:>6.3f}  "
            f"({int(best['rounds'])} rounds)"
        )

    # Print global optimization
    print("\n" + "-" * 80)
    print("GLOBAL OPTIMIZATION (all years weighted):")
    print("-" * 80)
    best_global = merged.iloc[0]
    current_weight = float(CALIBRATION_PARAMS.get("model_position_weight", 0.50))
    current_row = merged[np.isclose(merged["weight"], current_weight)]

    print(f"\n  Current default:    {current_weight:>4.2f}")
    print(f"  Best global weight: {best_global['weight']:>4.2f}")

    if not current_row.empty:
        current_mae = float(current_row["weighted_mae"].iloc[0])
        improvement_pct = 100 * (current_mae - best_global["weighted_mae"]) / current_mae
        print(f"\n  Current MAE:        {current_mae:>6.3f}")
        print(f"  Best MAE:           {best_global['weighted_mae']:>6.3f}")
        print(f"  Improvement:        {improvement_pct:>6.1f}%")

    # Top 5 weights
    print("\n  Top 5 weights by aggregate MAE:")
    top5 = merged[["weight", "weighted_mae"]].head(5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"    {i}. weight={row['weight']:>4.2f}  MAE={row['weighted_mae']:>6.3f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION:".ljust(40))
    if best_global["weight"] == current_weight:
        print(f"✓ Current default {current_weight:.2f} is already optimal globally.")
    else:
        print(
            f"→ Consider updating default to {best_global['weight']:.2f} for {improvement_pct:.1f}% MAE improvement."
        )
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-season calibration analysis")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024],
        help="Years to include in analysis (default: 2022 2023 2024)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_report(args.years)
