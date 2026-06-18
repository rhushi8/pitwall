# F1 Race Predictor

Forecast a Formula 1 race **before it happens** — not as a single predicted finishing order, but as win / podium / points **probabilities** with confidence intervals, produced by a machine-learning ensemble feeding a **10,000-run Monte Carlo simulation**.

## Problem Statement
A Formula 1 race outcome is inherently stochastic — pace, tire degradation, pit-stop timing, safety cars, and DNFs all introduce randomness — so a single predicted finishing order misrepresents the result by hiding that uncertainty. Models that look accurate in-sample also tend to fail on unseen races. The problem this project addresses: forecast race outcomes as calibrated probabilities with explicit uncertainty, validated strictly on races outside the training window.

## Key features
- **Multi-source data** — qualifying, practice telemetry, tire data, and weather from **FastF1**; stints, pit stops, and live timing from **OpenF1**.
- **Engineered features** — driver & team **ELO ratings**, circuit affinity, tire-degradation slope, fuel-corrected pace, and interaction terms.
- **Stacking ensemble** — XGBoost (finish position) + LightGBM (race pace) + a PyTorch neural net (tire degradation) + logistic regression (DNF / safety-car), combined by a **Ridge meta-learner on out-of-fold predictions**.
- **Monte Carlo simulation** — 10,000 race runs sampling lap-time noise, tire degradation, pit-stop variance, safety cars, DNFs, and weather.
- **Probabilistic output** — win / podium / points probabilities, expected finish + 90% confidence interval, DNF probability, and a strategy recommendation.
- **Honest validation** — strict **walk-forward backtesting** and calibration across 2020–2024, with no future-data leakage.
- **Interactive dashboard** built with Dash.

## How it works
```
FastF1 + OpenF1 ingestion → feature engineering → stacking ensemble → 10k Monte Carlo simulations → probabilities + strategy → Dash dashboard
```

## Tech stack
Python · FastF1 · OpenF1 · XGBoost · LightGBM · PyTorch · scikit-learn · Dash · Plotly

## Quickstart
```bash
python -m venv venv
venv\Scripts\activate               # Windows
pip install -r requirements.txt

# Predict a race (heuristic mode — no training required)
python src/predict.py --year 2024 --gp Bahrain --sims 10000

# Train on historical data, then predict with the trained model
python src/train.py --csv data/processed/historical_results.csv
python src/predict.py --year 2024 --gp Bahrain --sims 10000 --model models/ensemble.pkl
```

## Validation
```bash
# Strict walk-forward backtest + calibration sweep
python src/tuning/walk_forward_backtest.py --year 2024 --optimize-calibration --lock-best
# One-command multi-year evaluation
python src/tuning/run_full_evaluation.py --years 2020 2021 2022 2023 2024
```

## Tests
```bash
pip install -r requirements-dev.txt
pytest                # unit tests: Monte Carlo, ensemble save/load, prediction helpers
```
Tests are offline (no FastF1/network) and use small synthetic data.

## Project structure
- `src/ingestion/` — FastF1 & OpenF1 data loaders
- `src/features/` — feature engineering (ELO, circuit affinity, tire degradation, pace)
- `src/models/` — stacking ensemble
- `src/simulation/` — Monte Carlo engine + strategy optimizer
- `src/tuning/` — walk-forward backtesting & calibration
- `src/dashboard/` — Dash app
- `models/` — trained ensemble artifacts (`.pkl`)
- `data/processed/` — sample predictions & strategy outputs
- `scrape_and_build.py` — rebuilds the historical dataset

## Notes
- Raw FastF1 data isn't committed — regenerate it with `python scrape_and_build.py`. Processed sample outputs are included so results are visible without a full run.
- Research / portfolio project.
