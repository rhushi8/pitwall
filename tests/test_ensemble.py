import numpy as np
import pandas as pd
import pytest

from config.settings import ALL_FEATURES
from src.models.ensemble import F1StackingEnsemble


def _synthetic(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({col: rng.normal(size=n) for col in ALL_FEATURES})
    y_pos = pd.Series(rng.integers(1, 21, size=n).astype(float))
    # Ensure both DNF classes are present for LogisticRegression.
    y_dnf = pd.Series(([0, 1] * (n // 2 + 1))[:n])
    return X, y_pos, y_dnf


def test_fit_predict_shapes_and_range():
    X, y_pos, y_dnf = _synthetic()
    model = F1StackingEnsemble(n_splits=2).fit(X, y_pos, y_dnf=y_dnf)
    preds = model.predict(X)
    assert preds["position_pred"].shape == (len(X),)
    assert preds["position_pred"].min() >= 1.0
    assert preds["position_pred"].max() <= 20.0
    assert "dnf_prob" in preds


def test_predict_before_fit_raises():
    model = F1StackingEnsemble(n_splits=2)
    with pytest.raises(RuntimeError):
        model.predict(_synthetic()[0])


def test_get_features_fills_missing_columns():
    X, y_pos, y_dnf = _synthetic()
    model = F1StackingEnsemble(n_splits=2).fit(X, y_pos, y_dnf=y_dnf)
    # Drop a column; predict should still work (missing cols filled with 0).
    reduced = X.drop(columns=[ALL_FEATURES[0]])
    preds = model.predict(reduced)
    assert preds["position_pred"].shape == (len(reduced),)


def test_save_load_roundtrip(tmp_path):
    X, y_pos, y_dnf = _synthetic()
    model = F1StackingEnsemble(n_splits=2).fit(X, y_pos, y_dnf=y_dnf)

    # save() converts the tire model to its portable (torch-free) form, so take
    # the reference prediction AFTER saving to test serialization fidelity.
    path = model.save(tmp_path / "ensemble.pkl")
    before = model.predict(X)["position_pred"]

    loaded = F1StackingEnsemble.load(path)
    after = loaded.predict(X)["position_pred"]

    np.testing.assert_allclose(before, after, rtol=1e-6)
