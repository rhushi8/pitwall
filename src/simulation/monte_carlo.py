"""
src/simulation/monte_carlo.py
──────────────────────────────
Monte Carlo race simulator.

Each simulation run:
  1. Samples lap-time noise for each driver based on predicted pace + std dev
  2. Applies tire degradation model per stint
  3. Samples pit-stop timing and execution errors
  4. Samples safety-car deployments and their duration
  5. Samples DNFs from per-driver mechanical failure probabilities
  6. Rolls up to a final finishing order

Running 10,000 simulations produces probability distributions for:
  - Win probability
  - Podium (top-3) probability
  - Points (top-10) probability
  - Expected finish position ± CI
  - Optimal race strategy (1-stop vs 2-stop)
"""
from __future__ import annotations

import math
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.settings import (
    MC_SIMULATIONS, MC_RANDOM_SEED,
    SAFETY_CAR_PROB_PER_LAP, SIMULATION_PARAMS,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DriverProfile:
    """All per-driver inputs needed by the simulator."""
    code:              str
    grid_position:     int
    base_pace_s:       float          # predicted race pace (sec/lap)
    pace_std:          float = 0.35   # lap-time noise std dev (seconds)
    deg_slope:         float = 0.08   # seconds degradation per lap (on primary compound)
    dnf_prob:          float = 0.06   # probability of mechanical DNF this race
    safety_car_prob:   float = 0.15   # probability of being caught by SC
    team_pit_time:     float = 22.0   # mean pit-stop time loss (seconds)
    team_pit_std:      float = 1.2    # pit-stop execution noise
    wet_skill:         float = 0.0    # pace delta in wet conditions (negative = faster)

    # Strategy options (list of stints, each is compound + planned stint length)
    strategy: list[tuple[str, int]] = field(
        default_factory=lambda: [("SOFT", 20), ("MEDIUM", 35)]
    )


@dataclass
class CircuitProfile:
    """Circuit-level parameters for a specific race."""
    name:               str
    total_laps:         int
    safety_car_rate:    float = SAFETY_CAR_PROB_PER_LAP
    overtaking_factor:  float = 1.0   # 1.0 = average; >1 = easy to overtake
    weather_rain_prob:  float = 0.0   # probability of rain during race


# ─────────────────────────────────────────────────────────────────────────────
# Strategy utilities
# ─────────────────────────────────────────────────────────────────────────────

COMPOUND_OFFSETS = {
    **SIMULATION_PARAMS["compound_pace_offsets"],
}

COMPOUND_DEG_MULTIPLIERS = {
    **SIMULATION_PARAMS["compound_deg_multipliers"],
}

PIT_TIME_MIN = float(SIMULATION_PARAMS["pit_time_bounds"]["min_sec"])
PIT_TIME_MAX = float(SIMULATION_PARAMS["pit_time_bounds"]["max_sec"])
SAFETY_CAR_DURATION_MIN, SAFETY_CAR_DURATION_MAX = SIMULATION_PARAMS["safety_car_duration_laps"]
SAFETY_CAR_DURATION_MAX_EXCL = int(SAFETY_CAR_DURATION_MAX) + 1
SC_LAP_BASE = float(SIMULATION_PARAMS["safety_car_lap_time"]["base_s"])
SC_LAP_STD = float(SIMULATION_PARAMS["safety_car_lap_time"]["std_s"])
LAP_TIME_MIN = float(SIMULATION_PARAMS["lap_time_min_s"])


def compound_pace_adjustment(compound: str) -> float:
    return COMPOUND_OFFSETS.get(compound, 0.0)


def effective_deg_slope(base_slope: float, compound: str) -> float:
    return base_slope * COMPOUND_DEG_MULTIPLIERS.get(compound, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────────────────────────────────────

def simulate_race(
    drivers: list[DriverProfile],
    circuit: CircuitProfile,
    rng: np.random.Generator,
    rain: bool = False,
) -> pd.DataFrame:
    """
    Simulate one complete race and return per-driver outcomes.

    Simulation outline:
    1. Sample DNF outcomes for each driver from per-driver race DNF probability.
    2. Expand each planned strategy to match circuit total lap count.
    3. Sample safety-car windows (3-5 lap periods by default).
    4. Simulate lap time per stint/lap from:
       base_pace + compound_offset + degradation + gaussian noise.
    5. Override to safety-car pace during SC laps.
    6. Add pit stop losses with team mean/std and physical bounds.
    7. Rank by laps completed, then race time; assign finish positions.

    Returns DataFrame columns:
    driver_code, total_time_s, dnf, laps_completed, pit_stops, finish_position
    """
    total_laps = circuit.total_laps
    results = []

    for drv in drivers:
        if drv.dnf_prob > 0 and rng.random() < drv.dnf_prob:
            # DNF: random lap
            dnf_lap = int(rng.integers(1, total_laps))
            results.append(
                {
                    "driver_code":    drv.code,
                    "total_time_s":   1e9,   # large sentinel — sorts to back
                    "dnf":            True,
                    "laps_completed": dnf_lap,
                    "pit_stops":      0,
                }
            )
            continue

        total_time = 0.0
        laps_run   = 0
        pit_stops  = 0

        # --- Build stint schedule ---
        # Expand strategy: fill to total_laps, last stint runs to end
        strategy = list(drv.strategy)
        planned_laps = sum(s for _, s in strategy)
        if planned_laps < total_laps:
            # Extend last stint
            last_compound, last_laps = strategy[-1]
            strategy[-1] = (last_compound, last_laps + (total_laps - planned_laps))

        # --- Safety car: sample SC laps ---
        sc_laps = set()
        for lap in range(1, total_laps + 1):
            if rng.random() < circuit.safety_car_rate:
                duration = int(rng.integers(int(SAFETY_CAR_DURATION_MIN), SAFETY_CAR_DURATION_MAX_EXCL))
                for sc_lap in range(lap, min(lap + duration, total_laps + 1)):
                    sc_laps.add(sc_lap)

        # --- Simulate lap times stint by stint ---
        lap_counter = 0
        for s_idx, (compound, stint_len) in enumerate(strategy):
            # Pit stop at start of each stint (except first)
            if s_idx > 0:
                pit_time = rng.normal(drv.team_pit_time, drv.team_pit_std)
                pit_time = float(np.clip(pit_time, PIT_TIME_MIN, PIT_TIME_MAX))
                total_time += pit_time
                pit_stops  += 1

            for stint_lap in range(stint_len):
                lap_counter += 1
                if lap_counter > total_laps:
                    break

                # Base lap time
                base   = drv.base_pace_s
                noise  = rng.normal(0.0, drv.pace_std)
                deg    = effective_deg_slope(drv.deg_slope, compound) * stint_lap
                offset = compound_pace_adjustment(compound)
                wet_adj = drv.wet_skill if (rain or lap_counter in sc_laps) else 0.0

                lap_time = base + offset + deg + noise + wet_adj

                # Safety car: pace drops to ~80s/lap during SC (bunching)
                if lap_counter in sc_laps:
                    lap_time = SC_LAP_BASE + rng.normal(0.0, SC_LAP_STD)

                total_time += max(lap_time, LAP_TIME_MIN)
                laps_run   += 1

        results.append(
            {
                "driver_code":    drv.code,
                "total_time_s":   total_time,
                "dnf":            False,
                "laps_completed": laps_run,
                "pit_stops":      pit_stops,
            }
        )

    # --- Assign positions ---
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["laps_completed", "total_time_s"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df["finish_position"] = df.index + 1

    # DNF drivers get positions 21+
    classified_mask = ~df["dnf"]
    dnf_mask        = df["dnf"]
    df.loc[classified_mask, "finish_position"] = range(1, classified_mask.sum() + 1)
    df.loc[dnf_mask,        "finish_position"] = range(
        classified_mask.sum() + 1,
        classified_mask.sum() + 1 + dnf_mask.sum(),
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Full Monte Carlo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    drivers: list[DriverProfile],
    circuit: CircuitProfile,
    n_simulations: int = MC_SIMULATIONS,
    seed: int = MC_RANDOM_SEED,
    show_progress: bool = True,
    n_workers: Optional[int] = None,
) -> "MonteCarloResults":
    """
    Run `n_simulations` race simulations and aggregate results.
    Returns a MonteCarloResults object with probability tables.
    """
    driver_codes = [d.code for d in drivers]

    # Accumulators: driver → list of finish positions across sims
    finish_positions: dict[str, list[int]] = {d: [] for d in driver_codes}
    dnf_counts:       dict[str, int]       = {d: 0  for d in driver_codes}

    worker_count = int(n_workers or 1)
    if worker_count <= 1:
        rng = np.random.default_rng(seed)
        iterator = tqdm(range(n_simulations), desc="Simulating races") if show_progress else range(n_simulations)
        rain_prob = circuit.weather_rain_prob
        for _ in iterator:
            rain = rng.random() < rain_prob
            race_df = simulate_race(drivers, circuit, rng, rain=rain)
            for _, row in race_df.iterrows():
                finish_positions[row["driver_code"]].append(row["finish_position"])
                if row["dnf"]:
                    dnf_counts[row["driver_code"]] += 1
    else:
        batch_size = max(100, n_simulations // worker_count)
        n_batches = int(math.ceil(n_simulations / batch_size))
        work_items = []
        remaining = n_simulations
        for i in range(n_batches):
            n_batch = min(batch_size, remaining)
            remaining -= n_batch
            work_items.append((drivers, circuit, n_batch, seed + i))

        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            result_iter = ex.map(_run_batch, work_items)
            if show_progress:
                result_iter = tqdm(result_iter, total=n_batches, desc="Simulating races (parallel)")

            for batch_positions, batch_dnfs in result_iter:
                for drv, pos in batch_positions.items():
                    finish_positions[drv].extend(pos)
                for drv, cnt in batch_dnfs.items():
                    dnf_counts[drv] += cnt

    return MonteCarloResults(
        driver_codes=driver_codes,
        finish_positions=finish_positions,
        dnf_counts=dnf_counts,
        n_simulations=n_simulations,
    )


def _run_batch(args: tuple[list[DriverProfile], CircuitProfile, int, int]) -> tuple[dict[str, list[int]], dict[str, int]]:
    drivers, circuit, n_batch, seed = args
    rng = np.random.default_rng(seed)
    positions: dict[str, list[int]] = {d.code: [] for d in drivers}
    dnf_counts: dict[str, int] = {d.code: 0 for d in drivers}
    rain_prob = circuit.weather_rain_prob

    for _ in range(n_batch):
        rain = rng.random() < rain_prob
        race_df = simulate_race(drivers, circuit, rng, rain=rain)
        for _, row in race_df.iterrows():
            positions[row["driver_code"]].append(int(row["finish_position"]))
            if bool(row["dnf"]):
                dnf_counts[row["driver_code"]] += 1

    return positions, dnf_counts


# ─────────────────────────────────────────────────────────────────────────────
# Results container
# ─────────────────────────────────────────────────────────────────────────────

class MonteCarloResults:
    """
    Aggregates Monte Carlo simulation results into probability tables.
    """

    def __init__(
        self,
        driver_codes:     list[str],
        finish_positions: dict[str, list[int]],
        dnf_counts:       dict[str, int],
        n_simulations:    int,
    ):
        self.driver_codes  = driver_codes
        self._positions    = finish_positions
        self._dnf_counts   = dnf_counts
        self.n_simulations = n_simulations
        self._summary: Optional[pd.DataFrame] = None

    # ── Probability helpers ───────────────────────────────────────────────────

    def win_prob(self, driver: str) -> float:
        return np.mean(np.array(self._positions[driver]) == 1)

    def podium_prob(self, driver: str) -> float:
        return np.mean(np.array(self._positions[driver]) <= 3)

    def points_prob(self, driver: str) -> float:
        return np.mean(np.array(self._positions[driver]) <= 10)

    def expected_position(self, driver: str) -> float:
        return float(np.mean(self._positions[driver]))

    def position_ci(self, driver: str, pct: float = 90.0) -> tuple[float, float]:
        lo = (100 - pct) / 2
        arr = self._positions[driver]
        return float(np.percentile(arr, lo)), float(np.percentile(arr, 100 - lo))

    def dnf_prob(self, driver: str) -> float:
        return self._dnf_counts[driver] / self.n_simulations

    # ── Summary table ─────────────────────────────────────────────────────────

    def summary(self, sort_by: str = "win_prob") -> pd.DataFrame:
        if self._summary is not None and sort_by == "win_prob":
            return self._summary

        rows = []
        for drv in self.driver_codes:
            lo, hi = self.position_ci(drv)
            rows.append(
                {
                    "driver_code":      drv,
                    "win_prob":         round(self.win_prob(drv) * 100, 1),
                    "podium_prob":      round(self.podium_prob(drv) * 100, 1),
                    "points_prob":      round(self.points_prob(drv) * 100, 1),
                    "expected_position":round(self.expected_position(drv), 1),
                    "position_ci_lo":   round(lo, 1),
                    "position_ci_hi":   round(hi, 1),
                    "dnf_prob":         round(self.dnf_prob(drv) * 100, 1),
                }
            )

        df = pd.DataFrame(rows).sort_values(sort_by, ascending=(sort_by == "expected_position"))
        self._summary = df
        return df

    def print_summary(self) -> None:
        df = self.summary()
        print("\n" + "=" * 80)
        print(f"{'F1 RACE PREDICTION':^80}")
        print(f"{'Monte Carlo Results':^80}  ({self.n_simulations:,} simulations)")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy optimizer (brute-force over pit window options)
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_OPTIONS = {
    "1-stop: S→M":    [("SOFT", 20), ("MEDIUM", 36)],
    "1-stop: M→H":    [("MEDIUM", 25), ("HARD", 31)],
    "2-stop: S→M→H":  [("SOFT", 15), ("MEDIUM", 20), ("HARD", 21)],
    "2-stop: S→H→M":  [("SOFT", 15), ("HARD", 22), ("MEDIUM", 19)],
    "1-stop: S→H":    [("SOFT", 18), ("HARD", 38)],
}


def optimise_strategy(
    driver: DriverProfile,
    circuit: CircuitProfile,
    n_sims: int = 2_000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    For a single driver, test all strategy options in isolation and rank by
    expected finish position (lower = better).
    """
    rows = []
    for name, stints in STRATEGY_OPTIONS.items():
        # Clone driver with this strategy
        test_driver = DriverProfile(
            code=driver.code,
            grid_position=driver.grid_position,
            base_pace_s=driver.base_pace_s,
            pace_std=driver.pace_std,
            deg_slope=driver.deg_slope,
            dnf_prob=driver.dnf_prob,
            team_pit_time=driver.team_pit_time,
            team_pit_std=driver.team_pit_std,
            strategy=stints,
        )
        # Simulate against a 'ghost' field of 19 generic cars
        ghost_drivers = [test_driver] + [
            DriverProfile(
                code=f"G{i:02d}",
                grid_position=i + 1,
                base_pace_s=driver.base_pace_s + (i * 0.08),
                strategy=stints,
            )
            for i in range(1, 20)
        ]
        results = run_monte_carlo(
            ghost_drivers, circuit,
            n_simulations=n_sims,
            seed=seed,
            show_progress=False,
        )
        rows.append(
            {
                "strategy":           name,
                "expected_position":  round(results.expected_position(driver.code), 2),
                "win_prob_pct":       round(results.win_prob(driver.code) * 100, 1),
                "podium_prob_pct":    round(results.podium_prob(driver.code) * 100, 1),
            }
        )

    return pd.DataFrame(rows).sort_values("expected_position").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s - %(message)s")

    circuit = CircuitProfile(
        name="Bahrain International Circuit",
        total_laps=57,
        safety_car_rate=0.05,
        overtaking_factor=1.2,
        weather_rain_prob=0.02,
    )

    drivers = [
        DriverProfile("VER", 1, 92.5, deg_slope=0.07, dnf_prob=0.04,
                      strategy=[("SOFT", 18), ("MEDIUM", 22), ("HARD", 17)]),
        DriverProfile("LEC", 2, 92.8, deg_slope=0.08, dnf_prob=0.05,
                      strategy=[("SOFT", 20), ("MEDIUM", 37)]),
        DriverProfile("HAM", 3, 93.0, deg_slope=0.09, dnf_prob=0.05,
                      strategy=[("MEDIUM", 25), ("HARD", 32)]),
        DriverProfile("SAI", 4, 93.1, deg_slope=0.08, dnf_prob=0.05,
                      strategy=[("SOFT", 18), ("MEDIUM", 39)]),
        DriverProfile("NOR", 5, 93.3, deg_slope=0.10, dnf_prob=0.06,
                      strategy=[("SOFT", 22), ("MEDIUM", 35)]),
    ]

    results = run_monte_carlo(drivers, circuit, n_simulations=5_000)
    results.print_summary()

    print("\nOptimal strategy for VER:")
    print(optimise_strategy(drivers[0], circuit, n_sims=1_000))
