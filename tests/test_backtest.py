"""Tests de las métricas de backtest (RPS, log-loss, accuracy) y evaluación."""
import numpy as np

from src.evaluation.backtest import _metrics, _true_result, compare_models, evaluate


def test_true_result():
    assert _true_result(2, 0) == "H"
    assert _true_result(1, 1) == "D"
    assert _true_result(0, 3) == "A"


def test_metrics_perfect_prediction():
    yi = np.array([0, 1, 2, 0])
    P = np.eye(3)[yi] * 0.98 + 0.01
    m = _metrics(P, yi)
    assert m["acc"] == 1.0
    assert m["logloss"] < 0.1
    assert 0.0 <= m["rps"] <= 1.0


def test_metrics_uniform_logloss():
    yi = np.array([0, 1, 2, 0, 1, 2])
    P = np.full((len(yi), 3), 1 / 3)
    m = _metrics(P, yi)
    assert abs(m["logloss"] - np.log(3)) < 1e-6  # azar -> log(3)


def test_compare_models_valid(ensemble, synthetic_matches):
    df = compare_models(ensemble, synthetic_matches.tail(60))
    assert "Ensemble (final)" in df.index
    assert ((df["accuracy"] >= 0) & (df["accuracy"] <= 1)).all()
    assert (df["log_loss"] > 0).all()


def test_evaluate_returns_metrics(ensemble, synthetic_matches):
    m = evaluate(ensemble, synthetic_matches.tail(60))
    assert m["n"] == 60
    assert 0.0 <= m["accuracy"] <= 1.0
    assert m["log_loss"] > 0
