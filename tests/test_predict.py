import numpy as np
import pandas as pd

from src import predict
from src.simulation.monte_carlo import CircuitProfile, STRATEGY_OPTIONS


def test_circuits_json_is_loaded_not_hardcoded_defaults():
    """Regression for the predict.py path bug: the committed config/circuits.json
    must actually be read. suzuka.weather_rain_prob is 0.12 in the JSON but
    defaults to 0.0 in the hardcoded fallback profiles."""
    profiles = predict.load_circuit_profiles()
    assert {"bahrain", "jeddah", "suzuka", "monaco"} <= set(profiles)
    assert profiles["suzuka"].weather_rain_prob == 0.12


def test_get_circuit_profile_matches_known_circuit():
    prof = predict.get_circuit_profile("Bahrain")
    assert prof.total_laps > 0


def test_get_circuit_profile_unknown_falls_back():
    prof = predict.get_circuit_profile("Nonexistent GP")
    assert isinstance(prof, CircuitProfile)
    assert prof.total_laps > 0


def test_select_strategy_returns_valid_option():
    circuit = predict.get_circuit_profile("Bahrain")
    row = pd.Series({"deg_slope_medium": 0.08, "qual_practice_delta": 0.0})
    strat = predict.select_strategy(row, circuit)
    assert strat in STRATEGY_OPTIONS.values()


def test_calibrate_position_predictions_blends_into_range():
    feature_df = pd.DataFrame({"grid_position": [1, 5, 10, 20]})
    preds = {"position_pred": np.array([3.0, 3.0, 3.0, 3.0])}
    out = predict.calibrate_position_predictions(feature_df, preds, model_weight=0.5)
    assert out["position_pred"].min() >= 1.0
    assert out["position_pred"].max() <= 20.0
    assert len(out["position_pred"]) == 4


def test_sanitize_driver_rows_dedupes_and_validates():
    df = pd.DataFrame(
        {
            "driver_code": ["VER", "VER", "bad1", "ham"],
            "grid_position": [1, 1, 5, 3],
        }
    )
    out = predict.sanitize_driver_rows(df)
    codes = set(out["driver_code"])
    assert "VER" in codes and "HAM" in codes  # uppercased
    assert "BAD1" not in codes                 # non-alpha code dropped
    assert len(out) == len(out.drop_duplicates("driver_code"))
