"""Tests del modelo Elo: ranking ordenado y corrección por confederación (suma cero)."""
import pandas as pd

from src.models.elo import EloModel


def _intercon_matches():
    """Partidos inter-confederación con nombres reales (para activar la corrección)."""
    base = pd.Timestamp("2018-01-01")
    pairs = [("Brazil", "Germany", 1, 0), ("Argentina", "France", 2, 1),
             ("Spain", "Japan", 3, 0), ("England", "Mexico", 2, 2),
             ("Nigeria", "Belgium", 0, 1), ("Morocco", "Croatia", 1, 1)]
    rows = []
    for i in range(6):
        for j, (h, a, hs, as_) in enumerate(pairs):
            rows.append({"date": base + pd.Timedelta(days=i * 10 + j),
                         "home_team": h, "away_team": a, "home_score": hs,
                         "away_score": as_, "neutral": True, "match_weight": 1.0})
    return pd.DataFrame(rows)


def test_ranking_is_sorted_desc(elo):
    r = elo.ranking()
    assert list(r["elo"]) == sorted(r["elo"], reverse=True)


def test_higher_elo_wins_more(elo):
    teams = elo.ranking()["team"].tolist()
    top, bottom = teams[0], teams[-1]
    p = elo.probabilities(top, bottom, neutral=True)
    assert p["H"] > p["A"]  # el mejor rankeado es favorito


def test_confederation_correction_zero_sum():
    m = _intercon_matches()
    elo = EloModel().fit(m)
    offset = elo.apply_confederation_correction(m, beta=0.5)
    assert offset                                   # hubo cruces inter-confederación
    assert abs(sum(offset.values())) < 1e-6         # operación de suma cero
