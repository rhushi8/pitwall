"""
src/models/ensemble.py
──────────────────────
Stacking ensemble of:
  1. XGBoost  — grid / finish position regression
  2. LightGBM — stint pace regression
  3. Neural net (PyTorch) — tire degradation regression
  4. Logistic regression — safety car / DNF probability

A meta-learner (Ridge regression) combines base-model out-of-fold predictions
into final race position estimates.
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

from config.settings import (
    XGB_PARAMS, LGBM_PARAMS, MODEL_DIR, ALL_FEATURES
)

log = logging.getLogger(__name__)


class _TorchPlaceholder:
    """Placeholder used to unpickle legacy torch objects when torch is unavailable."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


class _TorchSafeUnpickler(pickle.Unpickler):
    """Best-effort unpickler that replaces torch classes with placeholders."""

    def find_class(self, module, name):
        if module.startswith("torch"):
            return _TorchPlaceholder
        return super().find_class(module, name)

# ─────────────────────────────────────────────────────────────────────────────
# Individual base models
# ─────────────────────────────────────────────────────────────────────────────

class PositionXGB:
    """XGBoost finish-position regressor."""
    name = "xgb_position"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or XGB_PARAMS
        self.model  = xgb.XGBRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PositionXGB":
        self.model.fit(X, y, eval_set=[(X, y)], verbose=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_,
            index=self.model.get_booster().feature_names,
        ).sort_values(ascending=False)


class PaceLGBM:
    """LightGBM race-pace regressor."""
    name = "lgbm_pace"

    def __init__(self, params: Optional[dict] = None):
        self.params = params or LGBM_PARAMS
        self.model  = lgb.LGBMRegressor(**self.params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PaceLGBM":
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(20, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)


class TireDegNN:
    """
    Simple PyTorch MLP for tire degradation prediction.
    Falls back to a LightGBM if torch is not installed.
    """
    name = "nn_tire_deg"

    def __init__(self):
        self._use_torch = False
        try:
            import torch  # noqa: F401
            self._use_torch = True
        except ImportError:
            log.warning("PyTorch not installed — using LightGBM fallback for TireDegNN")

        self.model = None
        self._fallback_model = lgb.LGBMRegressor(**LGBM_PARAMS)

    def _build_torch_model(self, input_dim: int):
        import torch.nn as nn
        from config.settings import NN_PARAMS

        layers = []
        prev = input_dim
        for h in NN_PARAMS["hidden_layers"]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(NN_PARAMS["dropout"])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TireDegNN":
        from config.settings import NN_PARAMS

        # Always train a portable fallback model.
        self._fallback_model.fit(X, y)

        if not self._use_torch:
            self.model = self._fallback_model
            return self

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X_t = torch.tensor(X.values.astype(np.float32))
        y_t = torch.tensor(y.values.astype(np.float32)).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=NN_PARAMS["batch_size"], shuffle=True)

        net  = self._build_torch_model(X_t.shape[1])
        opt  = torch.optim.Adam(net.parameters(), lr=NN_PARAMS["learning_rate"])
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(NN_PARAMS["epochs"]):
            net.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                pred = net(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= NN_PARAMS["patience"]:
                    log.info("TireDegNN early stop at epoch %d", epoch)
                    break

        self.model = net
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return self._fallback_model.predict(X)

        if (not self._use_torch) or hasattr(self.model, "predict"):
            return self.model.predict(X)

        import torch
        try:
            self.model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X.values.astype(np.float32))
                preds = self.model(X_t).squeeze(1).numpy()
            return preds
        except Exception:
            log.warning("Torch inference failed, using LightGBM fallback model")
            return self._fallback_model.predict(X)


class IncidentLogit:
    """Logistic regression for DNF / safety car probability."""
    name = "logit_incident"

    def __init__(self):
        self.dnf_model = LogisticRegression(max_iter=500, random_state=42)
        self.sc_model  = LogisticRegression(max_iter=500, random_state=42)

    def fit(
        self,
        X: pd.DataFrame,
        y_dnf: pd.Series,
        y_sc: Optional[pd.Series] = None,
    ) -> "IncidentLogit":
        self.dnf_model.fit(X, y_dnf)
        if y_sc is not None:
            self.sc_model.fit(X, y_sc)
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        dnf_proba = self.dnf_model.predict_proba(X)[:, 1]
        sc_proba  = (
            self.sc_model.predict_proba(X)[:, 1]
            if hasattr(self.sc_model, "classes_")
            else np.full(len(X), 0.15)
        )
        return pd.DataFrame(
            {"dnf_prob": dnf_proba, "safety_car_prob": sc_proba},
            index=X.index,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stacking ensemble
# ─────────────────────────────────────────────────────────────────────────────

class F1StackingEnsemble:
    """
        5-fold stacking ensemble for F1 finish-position prediction.

        Architecture:
            - Base models:
                1) XGBoost for position signal
                2) LightGBM for pace signal
                3) TireDegNN (PyTorch with LightGBM fallback) for degradation signal
            - Incident model:
                Logistic regression for DNF and safety-car effects
            - Meta learner:
                Ridge regression trained on out-of-fold base predictions

    Training:
      - Each base model produces OOF predictions via cross-validation.
      - The meta-learner (Ridge) is trained on the OOF predictions.

    Inference:
      - Each base model predicts on the full test set.
      - The meta-learner combines them into final position estimates.

        Example:
                ensemble = F1StackingEnsemble(n_splits=5)
                ensemble.fit(X_train, y_position, y_dnf=y_dnf)
                ensemble._artifacts = artifacts
                ensemble.save()

                loaded = F1StackingEnsemble.load()
                preds = loaded.predict(X_test)
                fi = loaded.feature_importance()
    """

    def __init__(self, n_splits: int = 5, feature_cols: Optional[list[str]] = None):
        self.n_splits = n_splits
        self._feature_cols = list(feature_cols) if feature_cols is not None else list(ALL_FEATURES)
        self.base_models = [
            PositionXGB(),
            PaceLGBM(),
            TireDegNN(),
        ]
        self.incident_model = IncidentLogit()
        self.meta_learner   = Ridge(alpha=1.0)
        self._fitted        = False

    def _ensure_portable_tire_model(self) -> None:
        """Guarantee TireDegNN can be loaded on machines without torch."""
        for model in self.base_models:
            if isinstance(model, TireDegNN):
                model._use_torch = False
                if getattr(model, "_fallback_model", None) is not None:
                    model.model = model._fallback_model

    @property
    def feature_cols(self) -> list[str]:
        if hasattr(self, "_trained_feature_cols") and self._trained_feature_cols:
            return [c for c in self._trained_feature_cols]
        return [c for c in self._feature_cols]

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.feature_cols
        X = df.copy()
        for col in cols:
            if col not in X.columns:
                X[col] = 0.0
        return X[cols].fillna(0.0)

    def fit(
        self,
        X: pd.DataFrame,
        y_position: pd.Series,
        y_dnf: Optional[pd.Series] = None,
        y_sc: Optional[pd.Series] = None,
    ) -> "F1StackingEnsemble":
        """
        Train all base models with cross-validation + meta-learner.
        """
        X_feat = self._get_features(X)
        self._trained_feature_cols = list(X_feat.columns)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        # OOF matrix: n_samples × n_base_models
        oof = np.zeros((len(X_feat), len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_feat)):
            log.info("Fold %d / %d", fold_idx + 1, self.n_splits)
            X_tr, X_vl = X_feat.iloc[train_idx], X_feat.iloc[val_idx]
            y_tr        = y_position.iloc[train_idx]

            for m_idx, model in enumerate(self.base_models):
                model.fit(X_tr, y_tr)
                oof[val_idx, m_idx] = model.predict(X_vl)

        # Train meta-learner on OOF
        self.meta_learner.fit(oof, y_position)
        log.info("Meta-learner trained. OOF MAE: %.3f",
                 mean_absolute_error(y_position, self.meta_learner.predict(oof)))

        # Re-fit each base model on the full dataset
        for model in self.base_models:
            model.fit(X_feat, y_position)

        # Incident model
        if y_dnf is not None:
            self.incident_model.fit(X_feat, y_dnf, y_sc)

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Returns a dict with:
          - 'position_pred'  : predicted finish position (float)
          - 'base_preds'     : per-model predictions (n_drivers × n_models)
          - 'dnf_prob'       : probability of DNF per driver
          - 'safety_car_prob': probability of safety car affecting driver
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted — call .fit() first")

        X_feat     = self._get_features(X)
        base_preds = np.column_stack(
            [m.predict(X_feat) for m in self.base_models]
        )
        position_pred = self.meta_learner.predict(base_preds)

        # Clip to valid F1 range [1, 20]
        position_pred = np.clip(position_pred, 1, 20)

        result = {
            "position_pred": position_pred,
            "base_preds":    base_preds,
        }

        if hasattr(self.incident_model.dnf_model, "classes_"):
            incidents = self.incident_model.predict_proba(X_feat)
            result["dnf_prob"]         = incidents["dnf_prob"].values
            result["safety_car_prob"]  = incidents["safety_car_prob"].values
        else:
            result["dnf_prob"]        = np.full(len(X), 0.06)
            result["safety_car_prob"] = np.full(len(X), 0.15)

        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or MODEL_DIR / "ensemble.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_portable_tire_model()
        payload = {
            "version": "2.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "feature_cols": self.feature_cols,
            "model": self,
            "artifacts": getattr(self, "_artifacts", {}),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info("Ensemble saved to %s", path)
        return path

    def save_versioned(self, base_path: Optional[Path] = None) -> Path:
        base_path = base_path or MODEL_DIR
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        version_path = base_path / f"ensemble_{stamp}.pkl"
        self.save(version_path)
        latest = base_path / "ensemble_latest.pkl"
        with open(latest, "wb") as f:
            with open(version_path, "rb") as src:
                f.write(src.read())
        return version_path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "F1StackingEnsemble":
        path = path or MODEL_DIR / "ensemble.pkl"
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            log.warning(
                "Model references torch but torch is unavailable; attempting portable load fallback"
            )
            with open(path, "rb") as f:
                obj = _TorchSafeUnpickler(f).load()
        if isinstance(obj, dict) and "model" in obj:
            model = obj["model"]
            model._artifacts = obj.get("artifacts", {})
            if "feature_cols" in obj and hasattr(model, "_feature_cols"):
                model._feature_cols = list(obj["feature_cols"])
            if hasattr(model, "_ensure_portable_tire_model"):
                model._ensure_portable_tire_model()
            meta = {k: v for k, v in obj.items() if k in {"version", "created_at"}}
            if meta:
                log.info("Loaded ensemble metadata: %s", json.dumps(meta))
            log.info("Ensemble loaded from %s", path)
            return model
        if isinstance(obj, cls):
            obj._ensure_portable_tire_model()
        log.info("Ensemble loaded from %s", path)
        return obj

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """Returns XGBoost feature importance as a DataFrame."""
        xgb_model = next(
            (m for m in self.base_models if isinstance(m, PositionXGB)), None
        )
        if xgb_model is None:
            return pd.DataFrame()
        return xgb_model.feature_importance().reset_index().rename(
            columns={"index": "feature", 0: "importance"}
        )
