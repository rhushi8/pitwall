import numpy as np

from src.simulation.monte_carlo import (
    DriverProfile,
    CircuitProfile,
    run_monte_carlo,
    simulate_race,
    optimise_strategy,
    STRATEGY_OPTIONS,
)


def _drivers(n=5):
    return [
        DriverProfile(code=c, grid_position=i + 1, base_pace_s=90.0 + i * 0.2, dnf_prob=0.05)
        for i, c in enumerate(["VER", "LEC", "HAM", "NOR", "RUS"][:n])
    ]


def _circuit():
    return CircuitProfile("Test Circuit", total_laps=40, safety_car_rate=0.05)


def test_simulate_race_assigns_unique_positions():
    rng = np.random.default_rng(0)
    df = simulate_race(_drivers(5), _circuit(), rng)
    assert len(df) == 5
    assert sorted(df["finish_position"]) == [1, 2, 3, 4, 5]


def test_monte_carlo_is_deterministic_with_seed():
    a = run_monte_carlo(_drivers(), _circuit(), n_simulations=200, seed=42, show_progress=False)
    b = run_monte_carlo(_drivers(), _circuit(), n_simulations=200, seed=42, show_progress=False)
    # Same seed -> identical aggregate results.
    assert a.summary().equals(b.summary())


def test_win_probabilities_sum_to_one():
    res = run_monte_carlo(_drivers(), _circuit(), n_simulations=300, seed=1, show_progress=False)
    total = sum(res.win_prob(d) for d in res.driver_codes)
    assert abs(total - 1.0) < 1e-9  # exactly one P1 per simulation


def test_probabilities_in_valid_range():
    res = run_monte_carlo(_drivers(), _circuit(), n_simulations=300, seed=2, show_progress=False)
    for d in res.driver_codes:
        assert 0.0 <= res.win_prob(d) <= 1.0
        assert 0.0 <= res.podium_prob(d) <= 1.0
        assert 1.0 <= res.expected_position(d) <= 20.0


def test_optimise_strategy_returns_all_options():
    res = optimise_strategy(_drivers(1)[0], _circuit(), n_sims=50)
    assert set(res["strategy"]) == set(STRATEGY_OPTIONS)
