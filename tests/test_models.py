"""Invariantes de los modelos: las probabilidades siempre son válidas y suman 1."""
import math


def _sums_to_one(d):
    return math.isclose(d["H"] + d["D"] + d["A"], 1.0, abs_tol=1e-6)


def test_elo_probabilities_valid(elo):
    p = elo.probabilities("T0", "T1", neutral=True)
    assert _sums_to_one(p)
    assert all(0 <= p[k] <= 1 for k in "HDA")


def test_elo_unknown_team_defaults(elo):
    p = elo.probabilities("Equipo Inexistente", "Otro Inexistente")
    assert _sums_to_one(p)  # ambos en rating base -> sin crash


def test_poisson_score_matrix_normalized(poisson):
    mat = poisson.score_matrix("T0", "T1", neutral=True)
    assert math.isclose(mat.sum(), 1.0, abs_tol=1e-6)
    assert (mat >= 0).all()


def test_poisson_probabilities_valid(poisson):
    p = poisson.probabilities("T0", "T1")
    assert _sums_to_one(p)


def test_ensemble_predict_sums_to_one(ensemble):
    r = ensemble.predict("T0", "T1", neutral=True)
    assert math.isclose(r["p_home"] + r["p_draw"] + r["p_away"], 1.0, abs_tol=1e-6)


def test_ensemble_news_tilt_preserves_sum(ensemble):
    ensemble.sentiment = {"T0": 0.8, "T1": -0.5}
    ensemble.news_tilt = 0.35
    r = ensemble.predict("T0", "T1")
    assert math.isclose(r["p_home"] + r["p_draw"] + r["p_away"], 1.0, abs_tol=1e-6)
    ensemble.sentiment = {}
    ensemble.news_tilt = 0.0


def test_predict_with_lineups_sums_to_one(ensemble):
    r = ensemble.predict_with_lineups("T0", "T1", 75, 70, 80, 80, neutral=True)
    assert math.isclose(r["p_home"] + r["p_draw"] + r["p_away"], 1.0, abs_tol=1e-6)


def test_weaker_lineup_lowers_win_prob(ensemble):
    base = ensemble.predict("T0", "T1")
    weak = ensemble.predict_with_lineups("T0", "T1", 60, 80, 80, 80)  # T0 debilitado
    assert weak["p_home"] <= base["p_home"] + 1e-9
