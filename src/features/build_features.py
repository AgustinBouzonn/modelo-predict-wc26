"""
Construcción de features para el modelo ML, SIN fuga de datos.

Todas las features se calculan con información disponible ANTES del partido:
  - Elo pre-partido (pasada cronológica incremental: por construcción no filtra)
  - Forma reciente (media móvil de últimos N partidos, desplazada 1)
  - Días de descanso
  - Localía / neutralidad / peso del torneo

El sentimiento de noticias NO entra acá (sería constante en el histórico):
se aplica como ajuste en vivo en la capa de predicción del ensemble.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.elo import EloModel

FORM_WINDOW = 5

FEATURE_COLS = [
    "elo_diff", "elo_home", "elo_away",
    "form_pts_diff", "form_gf_diff", "form_ga_diff",
    "rest_diff", "neutral", "match_weight",
]


def _elo_prematch(matches: pd.DataFrame, conf_offset: dict | None = None) -> pd.DataFrame:
    """Una pasada cronológica que registra el Elo de cada equipo ANTES del partido.

    Si se pasa conf_offset (corrección por confederación), se suma a los ratings
    para que las features queden en la misma escala que usa la predicción.
    """
    conf_offset = conf_offset or {}
    from ..data.confederations import TEAM_CONF

    def _adj(team: str) -> float:
        return conf_offset.get(TEAM_CONF.get(team), 0.0)

    elo = EloModel()
    home_elo, away_elo = [], []
    df = matches.sort_values("date").reset_index(drop=True)
    for r in df.itertuples(index=False):
        home_elo.append(elo.rating(r.home_team) + _adj(r.home_team))
        away_elo.append(elo.rating(r.away_team) + _adj(r.away_team))
        elo.update_match(
            r.home_team, r.away_team, int(r.home_score), int(r.away_score),
            neutral=bool(getattr(r, "neutral", False)),
            weight=float(getattr(r, "match_weight", 1.0)),
        )
    df["elo_home"] = home_elo
    df["elo_away"] = away_elo
    df["elo_diff"] = df["elo_home"] - df["elo_away"]
    return df


def _rolling_form(df: pd.DataFrame) -> pd.DataFrame:
    """Media móvil (últimos N, sin incluir el actual) de puntos y goles por equipo."""
    # Tabla larga: una fila por equipo y partido
    rows = []
    for r in df.itertuples(index=True):
        rows.append((r.Index, r.date, r.home_team, r.home_score, r.away_score, "home"))
        rows.append((r.Index, r.date, r.away_team, r.away_score, r.home_score, "away"))
    long = pd.DataFrame(rows, columns=["mid", "date", "team", "gf", "ga", "side"])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], default=0)
    long = long.sort_values(["team", "date"])

    g = long.groupby("team", group_keys=False)
    long["form_pts"] = g["pts"].apply(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    long["form_gf"] = g["gf"].apply(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    long["form_ga"] = g["ga"].apply(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())
    long["rest"] = g["date"].apply(lambda s: (s - s.shift(1)).dt.days).clip(upper=60)

    home = long[long.side == "home"].set_index("mid")[["form_pts", "form_gf", "form_ga", "rest"]]
    away = long[long.side == "away"].set_index("mid")[["form_pts", "form_gf", "form_ga", "rest"]]
    df = df.copy()
    df["form_pts_diff"] = df.index.map(home["form_pts"]) - df.index.map(away["form_pts"])
    df["form_gf_diff"] = df.index.map(home["form_gf"]) - df.index.map(away["form_gf"])
    df["form_ga_diff"] = df.index.map(home["form_ga"]) - df.index.map(away["form_ga"])
    df["rest_diff"] = (df.index.map(home["rest"]) - df.index.map(away["rest"]))
    return df


def build_training_table(matches: pd.DataFrame, conf_offset: dict | None = None) -> pd.DataFrame:
    """Devuelve la tabla con FEATURE_COLS + columna target 'result'."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df = _elo_prematch(df, conf_offset=conf_offset)
    df = _rolling_form(df)

    df["neutral"] = df["neutral"].astype(int)
    df["match_weight"] = df.get("match_weight", 1.0)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def current_form(matches: pd.DataFrame, window: int = FORM_WINDOW) -> dict[str, dict]:
    """Forma reciente ACTUAL de cada equipo (promedio de sus últimos `window`
    partidos en el dataset): puntos, goles a favor y en contra. Se usa para
    predecir partidos futuros con las mismas features que vio el ML al entrenar."""
    df = matches.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    rows = []
    for r in df.itertuples(index=False):
        rows.append((r.home_team, r.home_score, r.away_score))
        rows.append((r.away_team, r.away_score, r.home_score))
    long = pd.DataFrame(rows, columns=["team", "gf", "ga"])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], default=0)

    form = {}
    for team, g in long.groupby("team"):
        last = g.tail(window)
        form[team] = {"form_pts": float(last["pts"].mean()),
                      "form_gf": float(last["gf"].mean()),
                      "form_ga": float(last["ga"].mean())}
    return form


def features_for_fixture(home: str, away: str, elo: EloModel, neutral: bool = True,
                         match_weight: float = 4.0, recent_form: dict | None = None) -> pd.DataFrame:
    """Vector de features para un partido futuro, usando la forma reciente real
    de cada equipo si está disponible."""
    eh, ea = elo.rating(home), elo.rating(away)
    rf = recent_form or {}
    fh = rf.get(home, {"form_pts": 0.0, "form_gf": 0.0, "form_ga": 0.0})
    fa = rf.get(away, {"form_pts": 0.0, "form_gf": 0.0, "form_ga": 0.0})
    row = {
        "elo_diff": eh - ea, "elo_home": eh, "elo_away": ea,
        "form_pts_diff": fh["form_pts"] - fa["form_pts"],
        "form_gf_diff": fh["form_gf"] - fa["form_gf"],
        "form_ga_diff": fh["form_ga"] - fa["form_ga"],
        "rest_diff": 0.0, "neutral": int(neutral), "match_weight": match_weight,
    }
    return pd.DataFrame([row])[FEATURE_COLS]
