"""Tests del benchmark contra cuotas de apuestas."""
import math

import numpy as np

from src.evaluation.odds_benchmark import implied_probs, _logloss, _rps


def test_implied_probs_sum_to_one():
    p = implied_probs(2.0, 3.5, 4.0)
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)
    assert all(0 < x < 1 for x in p)


def test_implied_probs_favorite_has_higher_prob():
    # cuota más baja = favorito = mayor probabilidad implícita
    ph, pd_, pa = implied_probs(1.3, 5.0, 9.0)
    assert ph > pa and ph > pd_


def test_devig_removes_margin():
    # cuotas con overround (~1/odds suma > 1) -> probs normalizadas suman exacto 1
    inv = 1 / 1.3 + 1 / 5.0 + 1 / 9.0
    assert inv > 1.0  # hay margen de la casa
    assert math.isclose(sum(implied_probs(1.3, 5.0, 9.0)), 1.0, abs_tol=1e-9)


def test_logloss_rps_perfect_prediction():
    P = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    yi = np.array([0, 2])
    assert _logloss(P, yi) < 0.01
    assert _rps(P, yi) < 0.01
