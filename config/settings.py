"""Central configuration for the predictor."""
import importlib.util
import warnings
from pathlib import Path

# Paths
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
CACHE_DIR  = DATA_DIR / "cache"
MODEL_DIR  = ROOT_DIR / "models"

for d in [RAW_DIR, PROC_DIR, CACHE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# FastF1
FASTF1_CACHE = str(CACHE_DIR / "fastf1")

# OpenF1
OPENF1_BASE_URL = "https://api.openf1.org/v1"

# Race weekend sessions
SESSION_TYPES = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q":   "Qualifying",
    "R":   "Race",
    "S":   "Sprint",
    "SQ":  "Sprint Qualifying",
}

# Compounds
TIRE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
DRY_COMPOUNDS  = ["SOFT", "MEDIUM", "HARD"]

# Feature columns
QUALIFYING_FEATURES = [
    "q1_time_s", "q2_time_s", "q3_time_s",
    "q1_gap_to_pole", "q2_gap_to_pole", "q3_gap_to_pole",
    "grid_position",
]

PACE_FEATURES = [
    "fp1_best_lap_s", "fp2_best_lap_s", "fp3_best_lap_s",
    "fp2_long_run_pace_s",  # median lap time in long runs > 5 laps
    "fp2_fuel_corrected_s",
]

TIRE_DEG_FEATURES = [
    "deg_slope_soft",      # lap-time increase per lap on softs
    "deg_slope_medium",
    "deg_slope_hard",
    "crossover_lap_s_m",   # optimal crossover lap soft → medium
    "crossover_lap_m_h",
]

CONTEXTUAL_FEATURES = [
    "circuit_id",
    "air_temp_c",
    "track_temp_c",
    "humidity_pct",
    "wind_speed_kph",
    "track_evolution_coeff",  # synthetic: improves through weekend
    "circuit_dnf_rate",       # historical DNF rate for this circuit
    "race_number_in_season",
]

DRIVER_FEATURES = [
    "driver_elo",
    "team_elo",
    "driver_circuit_affinity",   # mean historical finish pos at this circuit
    "rolling_mean_pos_5",
    "rolling_dnf_rate_5",
    "season_points_so_far",
]

ALL_FEATURES = (
    QUALIFYING_FEATURES
    + PACE_FEATURES
    + TIRE_DEG_FEATURES
    + CONTEXTUAL_FEATURES
    + DRIVER_FEATURES
)

TARGET_COLS = ["finish_position", "finish_position_top3", "finish_position_top10"]

# Monte Carlo
MC_SIMULATIONS     = 10_000
MC_RANDOM_SEED     = 42

# Incident probabilities (per lap, per driver)
SAFETY_CAR_PROB_PER_LAP = 0.04   # ~4 % chance per lap averaged across circuits
DNF_BASE_PROB_PER_RACE  = 0.06   # 6 % baseline mechanical DNF

# Pit-stop time loss (seconds)
PIT_TIME_MEAN = 22.0
PIT_TIME_STD  = 1.2

# Simulation tuning parameters used by src/simulation/monte_carlo.py
SIMULATION_PARAMS = {
    "compound_pace_offsets": {
        "SOFT": 0.0,
        "MEDIUM": 0.6,
        "HARD": 1.1,
        "INTERMEDIATE": 2.0,
        "WET": 3.5,
    },
    "compound_deg_multipliers": {
        "SOFT": 2.0,
        "MEDIUM": 1.0,
        "HARD": 0.5,
        "INTERMEDIATE": 0.8,
        "WET": 0.6,
    },
    "pit_time_bounds": {
        "min_sec": 18.0,
        "max_sec": 35.0,
    },
    "safety_car_duration_laps": (3, 5),
    "safety_car_lap_time": {
        "base_s": 80.0,
        "std_s": 0.5,
    },
    "lap_time_min_s": 60.0,
}

# Prediction calibration (post-model, pre-simulation)
CALIBRATION_PARAMS = {
    # Blend between model predicted position and qualifying prior.
    # 1.0 = model-only, 0.0 = qualifying-only.
    "model_position_weight": 0.10,
    # Convert rank deltas to lap-time offsets in simulation profile building.
    "model_pace_offset_per_pos": 0.18,
    "grid_pace_offset_per_pos": 0.05,
}

# Model hyper-parameters (sensible defaults; tune via Optuna)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "objective": "regression",
    "random_state": 42,
    "verbose": -1,
}

NN_PARAMS = {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "epochs": 200,
    "batch_size": 64,
    "patience": 20,   # early stopping
}


def validate_config() -> None:
    """Validate critical configuration values and required packages."""
    errors: list[str] = []

    for name, path in {
        "DATA_DIR": DATA_DIR,
        "RAW_DIR": RAW_DIR,
        "PROC_DIR": PROC_DIR,
        "CACHE_DIR": CACHE_DIR,
        "MODEL_DIR": MODEL_DIR,
    }.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"{name} is not writable: {exc}")

    if MC_RANDOM_SEED < 0:
        errors.append("MC_RANDOM_SEED must be >= 0")

    if MC_SIMULATIONS <= 0:
        errors.append("MC_SIMULATIONS must be > 0")

    if importlib.util.find_spec("fastf1") is None:
        errors.append("Missing required dependency: fastf1")

    if importlib.util.find_spec("torch") is None:
        warnings.warn("PyTorch not installed - TireDegNN will use LightGBM fallback", RuntimeWarning)

    if errors:
        raise RuntimeError("Configuration errors:\n- " + "\n- ".join(errors))


validate_config()
