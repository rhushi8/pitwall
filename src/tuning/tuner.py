"""
src/tuning/tuner.py
────────────────────
Optuna-powered hyperparameter search for all models in the ensemble.

Tunes independently:
  • XGBoost  (finish-position regressor)
  • LightGBM (pace regressor)
  • Neural net hidden-layer / dropout / lr
  • Ridge meta-learner alpha

Then runs a joint stacking cross-validation with the best params found
and saves the final tuned model.

Usage:
    python src/tuning/tuner.py --csv data/processed/historical_results.csv
    python src/tuning/tuner.py --csv data/processed/historical_results.csv \
        --trials 150 --model xgb          # tune one model only
    python src/tuning/tuner.py --load-study studies/xgb_study.pkl  # resume
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import MODEL_DIR, ALL_FEATURES
from src.features.engineer import build_feature_matrix
from src.models.ensemble import F1StackingEnsemble, PositionXGB, PaceLGBM

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

STUDIES_DIR = MODEL_DIR / "studies"
STUDIES_DIR.mkdir(parents=True, exist_ok=True)

N_CV_SPLITS = 5
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    X, _ = build_feature_matrix(df, historical_results=df, fit=True)
    feature_cols = [c for c in ALL_FEATURES if c in X.columns]
    X = X[feature_cols].fillna(0.0)

    y_pos = df["finish_position"].astype(float)
    y_dnf = df["dnf"].astype(int) if "dnf" in df.columns else pd.Series(
        np.zeros(len(df)), dtype=int
    )
    groups = (df["year"].astype(str) + "_" + df["gp"]).values
    return X, y_pos, y_dnf, groups


def train_val_split(X, y, groups, test_size=0.15):
    splitter = GroupShuffleSplit(1, test_size=test_size, random_state=RANDOM_SEED)
    tr, vl = next(splitter.split(X, y, groups=groups))
    return (X.iloc[tr], X.iloc[vl],
            y.iloc[tr], y.iloc[vl],
            groups[tr], groups[vl])


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost objective
# ─────────────────────────────────────────────────────────────────────────────

def xgb_objective(trial: optuna.Trial, X_tr, y_tr, X_vl, y_vl, groups_tr):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 9),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "objective":         "reg:squarederror",
        "random_state":      RANDOM_SEED,
        "tree_method":       "hist",
        "verbosity":         0,
    }

    kf   = KFold(N_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    maes = []

    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr.iloc[ti], y_tr.iloc[ti],
            eval_set=[(X_tr.iloc[vi], y_tr.iloc[vi])],
            verbose=False,
        )
        preds = model.predict(X_tr.iloc[vi])
        fold_mae = mean_absolute_error(y_tr.iloc[vi], preds)
        maes.append(fold_mae)
        trial.report(np.mean(maes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM objective
# ─────────────────────────────────────────────────────────────────────────────

def lgbm_objective(trial: optuna.Trial, X_tr, y_tr, X_vl, y_vl, groups_tr):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 127),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "objective":        "regression",
        "random_state":     RANDOM_SEED,
        "verbose":          -1,
    }

    kf   = KFold(N_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    maes = []

    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr.iloc[ti], y_tr.iloc[ti],
            eval_set=[(X_tr.iloc[vi], y_tr.iloc[vi])],
            callbacks=[
                lgb.early_stopping(30, verbose=False),
                lgb.log_evaluation(-1),
            ],
        )
        preds = model.predict(X_tr.iloc[vi])
        fold_mae = mean_absolute_error(y_tr.iloc[vi], preds)
        maes.append(fold_mae)
        trial.report(np.mean(maes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# Neural net objective (architecture + training hypers)
# ─────────────────────────────────────────────────────────────────────────────

def nn_objective(trial: optuna.Trial, X_tr, y_tr, X_vl, y_vl, groups_tr):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.warning("PyTorch not installed — skipping NN tuning")
        raise optuna.TrialPruned()

    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden   = [trial.suggest_int(f"h_{i}", 32, 256, step=32)
                for i in range(n_layers)]
    dropout  = trial.suggest_float("dropout", 0.0, 0.5)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    bs       = trial.suggest_categorical("batch_size", [32, 64, 128])
    wd       = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    def build_net(input_dim):
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    kf   = KFold(N_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    maes = []

    for fold, (ti, vi) in enumerate(kf.split(X_tr)):
        Xti = torch.tensor(X_tr.iloc[ti].values.astype(np.float32))
        yti = torch.tensor(y_tr.iloc[ti].values.astype(np.float32)).unsqueeze(1)
        Xvi = torch.tensor(X_tr.iloc[vi].values.astype(np.float32))
        yvi = y_tr.iloc[vi].values

        net     = build_net(Xti.shape[1])
        opt     = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        loader  = DataLoader(TensorDataset(Xti, yti), batch_size=bs, shuffle=True)
        loss_fn = nn.MSELoss()

        best_mae, patience = 999, 0
        for epoch in range(100):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                ep_mae = mean_absolute_error(
                    yvi, net(Xvi).squeeze(1).numpy()
                )
            if ep_mae < best_mae - 0.001:
                best_mae = ep_mae
                patience = 0
            else:
                patience += 1
                if patience >= 15:
                    break

        maes.append(best_mae)
        trial.report(np.mean(maes), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# Meta-learner objective (Ridge alpha)
# ─────────────────────────────────────────────────────────────────────────────

def meta_objective(trial: optuna.Trial, oof_preds: np.ndarray, y: pd.Series):
    alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
    kf    = KFold(N_CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    maes  = []
    for ti, vi in kf.split(oof_preds):
        r = Ridge(alpha=alpha)
        r.fit(oof_preds[ti], y.iloc[ti])
        maes.append(mean_absolute_error(y.iloc[vi], r.predict(oof_preds[vi])))
    return float(np.mean(maes))


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance analysis with SHAP
# ─────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping SHAP analysis (pip install shap)")
        return pd.DataFrame()

    log.info("Running SHAP analysis …")
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)

    importance = pd.DataFrame({
        "feature":    X.columns,
        "shap_mean_abs": np.abs(shap_vals).mean(axis=0),
        "shap_std":      np.abs(shap_vals).std(axis=0),
    }).sort_values("shap_mean_abs", ascending=False)

    log.info("Top %d features by SHAP:", top_n)
    for _, row in importance.head(top_n).iterrows():
        bar = "█" * int(row["shap_mean_abs"] * 40 / importance["shap_mean_abs"].max())
        log.info("  %-35s %s %.4f", row["feature"], bar, row["shap_mean_abs"])

    if save_path:
        importance.to_csv(save_path, index=False)
        log.info("SHAP importance saved to %s", save_path)

    return importance


# ─────────────────────────────────────────────────────────────────────────────
# Study runner
# ─────────────────────────────────────────────────────────────────────────────

class TuningSession:
    """Orchestrates all Optuna studies and saves best params."""

    def __init__(
        self,
        X_tr: pd.DataFrame,
        y_tr: pd.Series,
        X_vl: pd.DataFrame,
        y_vl: pd.Series,
        groups_tr: np.ndarray,
        n_trials: int = 100,
    ):
        self.X_tr, self.y_tr   = X_tr, y_tr
        self.X_vl, self.y_vl   = X_vl, y_vl
        self.groups_tr         = groups_tr
        self.n_trials          = n_trials
        self.best_params: dict = {}
        self.study_results: dict = {}

    def _make_study(self, name: str, direction: str = "minimize") -> optuna.Study:
        storage = f"sqlite:///{STUDIES_DIR}/{name}.db"
        return optuna.create_study(
            study_name=name,
            direction=direction,
            sampler=TPESampler(seed=RANDOM_SEED),
            pruner=MedianPruner(n_warmup_steps=5),
            storage=storage,
            load_if_exists=True,
        )

    def tune_xgb(self) -> dict:
        log.info("─── Tuning XGBoost (%d trials) ───", self.n_trials)
        study = self._make_study("xgb_position")
        study.optimize(
            lambda t: xgb_objective(
                t, self.X_tr, self.y_tr, self.X_vl, self.y_vl, self.groups_tr
            ),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        params = study.best_params
        params.update({
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
            "verbosity": 0,
        })
        self.best_params["xgb"] = params
        self.study_results["xgb"] = {
            "best_mae": study.best_value,
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
        }
        log.info("XGBoost best CV MAE: %.4f", study.best_value)
        return params

    def tune_lgbm(self) -> dict:
        log.info("─── Tuning LightGBM (%d trials) ───", self.n_trials)
        study = self._make_study("lgbm_pace")
        study.optimize(
            lambda t: lgbm_objective(
                t, self.X_tr, self.y_tr, self.X_vl, self.y_vl, self.groups_tr
            ),
            n_trials=self.n_trials,
            show_progress_bar=True,
        )
        params = study.best_params
        params.update({
            "objective": "regression",
            "random_state": RANDOM_SEED,
            "verbose": -1,
        })
        self.best_params["lgbm"] = params
        self.study_results["lgbm"] = {
            "best_mae": study.best_value,
            "n_trials": len(study.trials),
        }
        log.info("LightGBM best CV MAE: %.4f", study.best_value)
        return params

    def tune_nn(self) -> dict:
        log.info("─── Tuning Neural Net (%d trials) ───", self.n_trials // 2)
        study = self._make_study("nn_tire_deg")
        study.optimize(
            lambda t: nn_objective(
                t, self.X_tr, self.y_tr, self.X_vl, self.y_vl, self.groups_tr
            ),
            n_trials=max(self.n_trials // 2, 20),
            show_progress_bar=True,
        )
        params = study.best_params
        self.best_params["nn"] = params
        self.study_results["nn"] = {"best_mae": study.best_value}
        log.info("NN best CV MAE: %.4f", study.best_value)
        return params

    def tune_meta(self, oof_preds: np.ndarray) -> float:
        log.info("─── Tuning Ridge meta-learner ───")
        study = self._make_study("ridge_meta")
        study.optimize(
            lambda t: meta_objective(t, oof_preds, self.y_tr),
            n_trials=50,
            show_progress_bar=True,
        )
        alpha = study.best_params["alpha"]
        self.best_params["meta_alpha"] = alpha
        log.info("Ridge best alpha: %.4f  CV MAE: %.4f", alpha, study.best_value)
        return alpha

    def run_all(self, models: Optional[list[str]] = None) -> dict:
        """Run all (or specified) tuning studies."""
        models = models or ["xgb", "lgbm", "nn"]
        start  = time.time()

        if "xgb" in models:
            self.tune_xgb()
        if "lgbm" in models:
            self.tune_lgbm()
        if "nn" in models:
            self.tune_nn()

        elapsed = time.time() - start
        log.info("Tuning complete in %.1f minutes", elapsed / 60)
        return self.best_params

    def save_params(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "best_params.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.best_params, indent=2))
        log.info("Best params saved to %s", path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Build final tuned ensemble
# ─────────────────────────────────────────────────────────────────────────────

def build_tuned_ensemble(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_vl: pd.DataFrame,
    y_vl: pd.Series,
    y_dnf_tr: pd.Series,
    best_params: dict,
) -> F1StackingEnsemble:
    """Instantiate and train the ensemble with tuned hyperparameters."""

    log.info("Building final tuned ensemble …")

    ensemble = F1StackingEnsemble(n_splits=5)

    # Inject tuned params
    if "xgb" in best_params:
        ensemble.base_models[0] = PositionXGB(params=best_params["xgb"])
    if "lgbm" in best_params:
        ensemble.base_models[1] = PaceLGBM(params=best_params["lgbm"])

    if "meta_alpha" in best_params:
        from sklearn.linear_model import Ridge
        ensemble.meta_learner = Ridge(alpha=best_params["meta_alpha"])

    # Train
    ensemble.fit(X_tr, y_tr, y_dnf=y_dnf_tr)

    # Evaluate
    preds  = ensemble.predict(X_vl)
    val_mae = mean_absolute_error(y_vl, preds["position_pred"])
    log.info("Tuned ensemble validation MAE: %.4f positions", val_mae)

    return ensemble


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: tuned vs default
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(
    X_tr, y_tr, X_vl, y_vl, y_dnf_tr,
    best_params: dict,
) -> pd.DataFrame:
    results = []

    # Default ensemble
    log.info("Training default ensemble for comparison …")
    default = F1StackingEnsemble(n_splits=5)
    default.fit(X_tr, y_tr, y_dnf=y_dnf_tr)
    default_preds = default.predict(X_vl)
    results.append({
        "model":   "default",
        "val_mae": round(mean_absolute_error(y_vl, default_preds["position_pred"]), 4),
    })

    # Tuned ensemble
    tuned = build_tuned_ensemble(X_tr, y_tr, X_vl, y_vl, y_dnf_tr, best_params)
    tuned_preds = tuned.predict(X_vl)
    results.append({
        "model":   "tuned",
        "val_mae": round(mean_absolute_error(y_vl, tuned_preds["position_pred"]), 4),
    })

    df = pd.DataFrame(results)
    improvement = (df.iloc[0]["val_mae"] - df.iloc[1]["val_mae"]) / df.iloc[0]["val_mae"] * 100
    df["improvement_pct"] = [0.0, round(improvement, 2)]

    print("\n── Benchmark ─────────────────────────────")
    print(df.to_string(index=False))
    print(f"\n  MAE improvement from tuning: {improvement:.1f}%")
    print("──────────────────────────────────────────\n")

    return df, tuned


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="F1 Model Hyperparameter Tuning")
    parser.add_argument("--csv",     required=True, help="Path to historical_results.csv")
    parser.add_argument("--trials",  type=int, default=100, help="Optuna trials per model")
    parser.add_argument("--model",   choices=["xgb", "lgbm", "nn", "all"], default="all")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--shap",    action="store_true", help="Run SHAP analysis after tuning")
    args = parser.parse_args()

    log.info("Loading data from %s …", args.csv)
    X, y_pos, y_dnf, groups = load_data(Path(args.csv))
    X_tr, X_vl, y_tr, y_vl, g_tr, g_vl = train_val_split(X, y_pos, groups)
    y_dnf_tr = y_dnf.iloc[
        np.where(np.isin(groups, g_tr))[0]
    ].reset_index(drop=True) if len(y_dnf) == len(X) else None

    session = TuningSession(X_tr, y_tr, X_vl, y_vl, g_tr, n_trials=args.trials)

    models_to_tune = ["xgb", "lgbm", "nn"] if args.model == "all" else [args.model]
    best = session.run_all(models=models_to_tune)
    params_path = session.save_params()

    if not args.no_benchmark:
        bench_df, tuned_ensemble = benchmark(
            X_tr, y_tr, X_vl, y_vl, y_dnf_tr, best
        )
        tuned_path = tuned_ensemble.save(MODEL_DIR / "ensemble_tuned.pkl")
        log.info("Tuned model saved to %s", tuned_path)

    if args.shap:
        final_xgb = xgb.XGBRegressor(**best.get("xgb", {}))
        final_xgb.fit(X_tr, y_tr, verbose=False)
        run_shap_analysis(
            final_xgb, X_vl,
            save_path=MODEL_DIR / "shap_importance.csv"
        )

    print(f"\nAll studies saved to:  {STUDIES_DIR}")
    print(f"Best params saved to:  {params_path}")
    print("\nTo use tuned model in predictions:")
    print("  python src/predict.py --model models/ensemble_tuned.pkl ...")
