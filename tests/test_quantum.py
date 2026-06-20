"""Smoke tests del clasificador cuántico variacional (4 qubits, 1-X-2).

Entrena con datos sintéticos chicos y pocas épocas: verifica que el circuito
corre, que las probabilidades son válidas y que la grilla de decisión tiene la
forma correcta. Si PennyLane no está instalado, el módulo se saltea.
"""
import math

import numpy as np
import pytest

pytest.importorskip("pennylane")

from src.models.quantum import FEATURES, QuantumMatchClassifier  # noqa: E402


def test_quantum_trains_and_predicts_valid_probs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, len(FEATURES)))
    # etiqueta correlacionada con la 1ª feature para que el modelo aprenda señal
    y = np.where(X[:, 0] > 0.4, 0, np.where(X[:, 0] < -0.4, 2, 1))

    clf = QuantumMatchClassifier().fit(X, y, epochs=8, batch=16)
    p = clf.predict_proba(X[0])

    assert set(p) == {"H", "D", "A"}
    assert math.isclose(p["H"] + p["D"] + p["A"], 1.0, abs_tol=1e-6)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert 0.0 <= clf.test_acc <= 1.0


def test_quantum_decision_grid_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, len(FEATURES)))
    y = rng.integers(0, 3, size=80)

    clf = QuantumMatchClassifier().fit(X, y, epochs=4, batch=16)
    gx, gy, Z = clf.decision_grid(n=12)

    assert Z.shape == (12, 12)
    assert len(gx) == 12 and len(gy) == 12
    assert np.all(np.abs(Z) <= 1.0 + 1e-6)  # P(local) − P(visitante) ∈ [-1, 1]
