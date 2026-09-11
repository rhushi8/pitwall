# F1 Race Predictor

Forecast a Formula 1 race before it happens, as win, podium and points
probabilities with confidence intervals, from a 10,000-run Monte Carlo
simulation whose inputs combine a machine-learning ensemble with a calibrated
qualifying-position prior.

The honest framing, up front: the strength here is the methodology, walk-forward
validation and Monte Carlo uncertainty, and not raw predictive accuracy. The
backtest says qualifying order is a very strong baseline that the ensemble does
not beat on its own, so the shipped prediction blends the two at a weight the
backtest picked. The numbers are in [Model performance](#model-performance).

## Why probabilities and not a finishing order

A race outcome is stochastic. Pace, tire degradation, pit timing, safety cars
and DNFs all inject randomness, so a single predicted finishing order hides the
very thing you wanted to know. Models that look sharp in-sample also tend to
fall over on races they have never seen. This one forecasts calibrated
probabilities instead, validated strictly outside the training window.

## What it does

Data comes from two sources: qualifying, practice telemetry, tire data and
weather from FastF1, and stints, pit stops and live timing from OpenF1.

Features are engineered from that: driver and team ELO ratings, circuit
affinity, tire-degradation slope, fuel-corrected pace, and interaction terms.

Three base learners regress finish position over those features, XGBoost,
LightGBM and a small PyTorch MLP, plus logistic regression for DNF and safety
car, combined by a Ridge meta-learner on out-of-fold predictions. Pace and tire
degradation are features here, not separate model targets.

The ensemble's position estimate is then blended with a qualifying-position
prior at a weight the walk-forward backtest chooses, defaulting to 0.10,
because grid order is a strong baseline and pretending otherwise would be
dishonest.

The simulation runs 10,000 races, sampling lap-time noise, tire degradation,
pit-stop variance, safety cars, DNFs and weather.

Output is win, podium and points probabilities, an expected finish with a 90%
confidence interval, a DNF probability and a strategy recommendation, rendered
in a Dash dashboard.

Validation is walk-forward across 2020 to 2024, with no future data leaking
backwards.

## How it works

```
FastF1 + OpenF1 → feature engineering → stacking ensemble
                → 10k Monte Carlo runs → probabilities and strategy → dashboard
```

## Stack

Python, FastF1, OpenF1, XGBoost, LightGBM, PyTorch, scikit-learn, Dash, Plotly.

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# heuristic mode, no training needed
python src/predict.py --year 2024 --gp Bahrain --sims 10000

# train first, then predict with the trained model
python src/train.py --csv data/processed/historical_results.csv
python src/predict.py --year 2024 --gp Bahrain --sims 10000 --model models/ensemble.pkl
```

## Validation

```bash
python src/tuning/walk_forward_backtest.py --year 2024 --optimize-calibration --lock-best
python src/tuning/run_full_evaluation.py --years 2020 2021 2022 2023 2024
```

## Model performance

From the strict 2024 walk-forward calibration sweep
(`data/processed/walk_forward_2024_calibration_sweep.csv`), mean absolute error
of predicted against actual finishing position, by blend weight:

| blend weight | what it means | MAE |
|---|---|---|
| 0.0 | qualifying order only, no model | **2.75** |
| 0.10 | shipped default | 2.77 |
| 1.0 | ML ensemble only | 4.70 |

MAE rises steadily as the model's weight goes up. The ensemble does not beat a
pure qualifying-order baseline out of sample, which is why the shipped blend
leans about 90% on qualifying.

That is a real result and it is in the README on purpose. A model that loses to
its own baseline is worth knowing about, and finding that out is what the
walk-forward backtest is for. What the project actually demonstrates is the
validation discipline and the probabilistic layer on top, not state-of-the-art
accuracy.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Unit tests cover the Monte Carlo engine, ensemble save and load, and the
prediction helpers. They run offline on small synthetic data, so no FastF1 and
no network.

## Layout

```
src/ingestion/    FastF1 and OpenF1 loaders
src/features/     ELO, circuit affinity, tire degradation, pace
src/models/       stacking ensemble
src/simulation/   Monte Carlo engine and strategy optimizer
src/tuning/       walk-forward backtesting and calibration
src/dashboard/    Dash app
models/           trained ensemble artifacts
data/processed/   sample predictions and strategy outputs
scrape_and_build.py   rebuilds the historical dataset
```

## Notes

Raw FastF1 data isn't committed. Regenerate it with `python scrape_and_build.py`.
Processed sample outputs are included so the results are visible without a full
run.
