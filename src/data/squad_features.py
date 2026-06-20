"""
Métricas derivadas del plantel (a partir de los datos de Wikipedia).

De cada selección calculamos indicadores que el Elo/Poisson NO ven directamente:
  - experiencia      : caps promedio del plantel (partidos internacionales)
  - poder ofensivo   : goles internacionales acumulados del plantel
  - edad             : edad promedio (juventud vs veteranía)
  - élite            : nº de jugadores en clubes de las grandes ligas europeas

Estas métricas alimentan la vista de Selecciones (informativo). NO son features
del modelo ML: son estáticas (sin histórico por partido), así que no se pueden
entrenar ni validar en backtest. La señal de plantel que sí usa el predictor es
el rating de jugador derivado (player_ratings.py) vía las alineaciones.
"""
from __future__ import annotations

import pandas as pd

from .squads import load_squads

# Clubes de élite (ligas top de Europa). Heurística para medir a qué nivel
# juega el plantel; lista curada, ampliable.
ELITE_CLUBS = {
    # Premier League
    "Manchester City", "Arsenal", "Liverpool", "Chelsea", "Manchester United",
    "Tottenham Hotspur", "Newcastle United", "Aston Villa", "Crystal Palace",
    "Brighton & Hove Albion", "West Ham United", "Bournemouth", "Nottingham Forest",
    "Everton", "Fulham", "Brentford", "Wolverhampton Wanderers",
    # LaLiga
    "Real Madrid", "Barcelona", "Atlético Madrid", "Athletic Bilbao",
    "Real Sociedad", "Real Betis", "Villarreal", "Valencia", "Sevilla",
    # Serie A
    "Inter Milan", "Milan", "Juventus", "Napoli", "Roma", "Lazio",
    "Atalanta", "Fiorentina", "Bologna",
    # Bundesliga
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
    "VfB Stuttgart", "Eintracht Frankfurt", "Borussia Mönchengladbach",
    # Ligue 1
    "Paris Saint-Germain", "Marseille", "Monaco", "Lyon", "Lille", "Nice",
    # Otras ligas europeas fuertes
    "PSV Eindhoven", "Ajax", "Feyenoord", "Benfica", "Porto", "Sporting CP",
    "Galatasaray", "Fenerbahçe", "Beşiktaş", "Celtic", "Rangers", "Olympiacos",
    "Shakhtar Donetsk", "Dinamo Zagreb", "Red Bull Salzburg", "Club Brugge",
    "Slavia Prague", "Sparta Prague", "Red Star Belgrade",
    # CONMEBOL (clubes top)
    "Flamengo", "Palmeiras", "Fluminense", "Botafogo", "São Paulo", "Corinthians",
    "Atlético Mineiro", "Internacional", "Grêmio", "River Plate", "Boca Juniors",
    "Racing Club", "Peñarol", "Nacional",
    # MLS / Liga MX
    "Inter Miami", "Los Angeles FC", "Seattle Sounders", "Monterrey", "Club América",
    # Saudi Pro League
    "Al-Hilal", "Al-Nassr", "Al-Ittihad", "Al-Ahli",
}


def squad_metrics() -> pd.DataFrame:
    """Devuelve un DataFrame por selección con las métricas de plantel."""
    sq = load_squads()
    if sq.empty:
        return pd.DataFrame()

    rows = []
    for team, g in sq.groupby("team"):
        caps = pd.to_numeric(g["caps"], errors="coerce")
        goals = pd.to_numeric(g["goals"], errors="coerce")
        age = pd.to_numeric(g["age"], errors="coerce")
        n_elite = int(g["club"].isin(ELITE_CLUBS).sum())
        rows.append({
            "team": team,
            "squad_size": len(g),
            "avg_age": round(age.mean(), 1),
            "avg_caps": round(caps.mean(), 1),
            "total_caps": int(caps.sum()),
            "total_goals": int(goals.sum()),
            "elite_clubs": n_elite,
            "elite_share": round(n_elite / len(g), 3),
        })
    return pd.DataFrame(rows).sort_values("avg_caps", ascending=False).reset_index(drop=True)


def squad_feature_table() -> pd.DataFrame:
    """Métricas indexadas por equipo, para mergear como features del modelo."""
    m = squad_metrics()
    if m.empty:
        return m
    return m.set_index("team")[["avg_caps", "total_goals", "avg_age", "elite_share"]]


def top_players(team: str, n: int = 5) -> pd.DataFrame:
    """Jugadores más determinantes del plantel (por goles y luego caps)."""
    sq = load_squads()
    g = sq[sq["team"] == team].copy()
    g["goals"] = pd.to_numeric(g["goals"], errors="coerce").fillna(0)
    g["caps"] = pd.to_numeric(g["caps"], errors="coerce").fillna(0)
    return g.sort_values(["goals", "caps"], ascending=False).head(n)


if __name__ == "__main__":
    print(squad_metrics().to_string(index=False))
