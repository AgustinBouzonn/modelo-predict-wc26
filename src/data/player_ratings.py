"""
Rating de jugador derivado + alineaciones (formaciones, mejor XI, fuerza).

SofaScore y FotMob están bloqueados (Cloudflare / token firmado), así que el
rating de cada jugador se DERIVA de datos robustos de Wikipedia:
  - experiencia (caps internacionales)
  - aporte de gol (goles/caps, ponderado por posición)
  - nivel de club (élite de las grandes ligas)
  - edad (pico de rendimiento ~27)

Da un rating 0-100 estable y siempre disponible. Si en el futuro se consiguen
ratings reales por partido (FotMob vía navegador headless), se enchufan acá.
"""
from __future__ import annotations

import pandas as pd

from .squads import load_squads
from .squad_features import ELITE_CLUBS

# Formaciones: líneas de defensa→ataque, cada una (tipo, cantidad). GK implícito.
FORMATIONS: dict[str, list[tuple[str, int]]] = {
    "4-3-3": [("DF", 4), ("MF", 3), ("FW", 3)],
    "4-4-2": [("DF", 4), ("MF", 4), ("FW", 2)],
    "4-2-3-1": [("DF", 4), ("MF", 2), ("MF", 3), ("FW", 1)],
    "4-3-1-2": [("DF", 4), ("MF", 3), ("MF", 1), ("FW", 2)],
    "3-5-2": [("DF", 3), ("MF", 5), ("FW", 2)],
    "3-4-3": [("DF", 3), ("MF", 4), ("FW", 3)],
    "5-3-2": [("DF", 5), ("MF", 3), ("FW", 2)],
    "4-5-1": [("DF", 4), ("MF", 5), ("FW", 1)],
}
DEFAULT_FORMATION = "4-3-3"


def player_rating(position: str, caps, goals, club, age) -> float:
    """Rating 0-100 derivado de experiencia, gol, club y edad.

    Experiencia y club pesan parejo para todas las posiciones, así que un arquero
    o defensor de élite no queda artificialmente por debajo de un delantero. El
    aporte de gol es un EXTRA acotado (con shrinkage) que solo suma a ofensivos.
    """
    caps = float(caps) if pd.notna(caps) else 0.0
    goals = float(goals) if pd.notna(goals) else 0.0
    age = float(age) if pd.notna(age) else 27.0
    pos = str(position)

    base = 48.0
    # Experiencia: satura a ~60 caps (a esa altura ya es un titular consolidado)
    exp = min(caps, 60.0) / 60.0 * 24.0              # hasta +24, aplica a todos
    elite = 16.0 if club in ELITE_CLUBS else 0.0     # club, aplica a todos
    # Aporte de gol con shrinkage (no infla a quien tiene pocos caps)
    gpm = (goals + 0.5) / (caps + 6.0)
    if pos == "FW":
        goal_score = min(gpm * 95.0, 20.0)
    elif pos == "MF":
        goal_score = min(gpm * 80.0, 13.0)
    elif pos == "DF":
        goal_score = min(gpm * 50.0, 5.0)
    else:  # GK — su valor viene de experiencia + club, no del gol
        goal_score = 0.0
    age_factor = -abs(age - 27.0) * 0.35
    return round(max(40.0, min(99.0, base + exp + elite + goal_score + age_factor)), 1)


def rated_squad(team: str) -> pd.DataFrame:
    """Plantel del equipo con columna 'rating', ordenado por rating desc."""
    sq = load_squads()
    g = sq[sq["team"] == team].copy()
    if g.empty:
        return g
    g["rating"] = [player_rating(r.position, r.caps, r.goals, r.club, r.age)
                   for r in g.itertuples(index=False)]
    return g.sort_values("rating", ascending=False).reset_index(drop=True)


def formation_lines(formation: str) -> list[tuple[str, int]]:
    return FORMATIONS.get(formation, FORMATIONS[DEFAULT_FORMATION])


def formation_slots(formation: str) -> list[dict]:
    """Coordenadas (x,y en 0..1) de los 11 puestos. y=1 es el arco propio."""
    lines = formation_lines(formation)
    slots = [{"pos": "GK", "x": 0.5, "y": 0.93, "line": -1}]
    n_lines = len(lines)
    for li, (ptype, count) in enumerate(lines):
        y = 0.75 - (li / max(n_lines - 1, 1)) * 0.63   # 0.75 (def) -> 0.12 (ataque)
        for k in range(count):
            x = (k + 1) / (count + 1)
            slots.append({"pos": ptype, "x": round(x, 3), "y": round(y, 3), "line": li})
    return slots


def best_xi(team: str, formation: str = DEFAULT_FORMATION) -> pd.DataFrame:
    """Selecciona el mejor XI del plantel para la formación dada.

    Devuelve 11 filas con jugador, posición, rating y coords (slot_x, slot_y).
    """
    g = rated_squad(team)
    if g.empty:
        return g

    slots = formation_slots(formation)
    need = {"GK": 0, "DF": 0, "MF": 0, "FW": 0}
    for s in slots:
        need[s["pos"]] += 1

    chosen: list[dict] = []
    used: set = set()
    by_pos = {p: g[g["position"] == p].to_dict("records") for p in need}
    pool = g.to_dict("records")

    def _take(records):
        for r in records:
            if r["player"] not in used:
                used.add(r["player"])
                return r
        return None

    # Asignar por puesto según el orden de los slots
    for s in slots:
        cand = _take(by_pos.get(s["pos"], [])) or _take(pool)
        if cand is None:
            continue
        chosen.append({**cand, "slot_x": s["x"], "slot_y": s["y"], "slot_pos": s["pos"]})
    return pd.DataFrame(chosen)


def assign_to_formation(players: pd.DataFrame, formation: str = DEFAULT_FORMATION) -> pd.DataFrame:
    """Asigna una lista concreta de jugadores (con 'rating' y 'position') a los
    puestos de una formación. Completa con los mejores disponibles si falta
    alguno en una posición."""
    if players.empty:
        return players
    slots = formation_slots(formation)
    pool = players.sort_values("rating", ascending=False).to_dict("records")
    used: set = set()
    chosen: list[dict] = []

    def take(pos=None):
        for r in pool:
            if r["player"] in used:
                continue
            if pos and r["position"] != pos:
                continue
            used.add(r["player"])
            return r
        return None

    for s in slots:
        cand = take(s["pos"]) or take(None)
        if cand:
            chosen.append({**cand, "slot_x": s["x"], "slot_y": s["y"], "slot_pos": s["pos"]})
    return pd.DataFrame(chosen)


def lineup_strength(ratings: list[float]) -> float:
    """Fuerza de una alineación = promedio de ratings de los 11."""
    vals = [r for r in ratings if pd.notna(r)]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def team_lineup_baseline(team: str, formation: str = DEFAULT_FORMATION) -> float:
    """Fuerza del mejor XI posible (referencia para medir alineaciones reales)."""
    xi = best_xi(team, formation)
    return lineup_strength(xi["rating"].tolist()) if not xi.empty else 60.0
