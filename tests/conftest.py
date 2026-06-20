"""Fixtures de test: modelos entrenados con datos sintéticos (sin red ni descargas)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.elo import EloModel  # noqa: E402
from src.models.poisson import DixonColesModel  # noqa: E402
from src.models.ensemble import EnsemblePredictor  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_matches():
    """~600 partidos sintéticos entre 12 equipos, reproducible."""
    rng = np.random.default_rng(7)
    teams = [f"T{i}" for i in range(12)]
    rows = []
    base = pd.Timestamp("2020-01-01")
    for k in range(600):
        h, a = rng.choice(teams, size=2, replace=False)
        hs, as_ = int(rng.poisson(1.4)), int(rng.poisson(1.1))
        rows.append({"date": base + pd.Timedelta(days=k),
                     "home_team": h, "away_team": a,
                     "home_score": hs, "away_score": as_,
                     "neutral": False, "match_weight": 1.0})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def elo(synthetic_matches):
    return EloModel().fit(synthetic_matches)


@pytest.fixture(scope="session")
def poisson(synthetic_matches):
    return DixonColesModel().fit(synthetic_matches)


@pytest.fixture(scope="session")
def ensemble(elo, poisson):
    return EnsemblePredictor(elo=elo, poisson=poisson, ml=None,
                             weights={"elo": 0.5, "poisson": 0.5}, sentiment={}, news_tilt=0.0)
