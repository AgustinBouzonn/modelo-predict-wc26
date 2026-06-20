"""
Modelo Elo para selecciones (estilo World Football Elo).

- Procesa los partidos en orden cronológico y mantiene un rating por equipo.
- El ajuste pondera por importancia del torneo y por margen de gol.
- Para predecir, convierte la diferencia de rating en probabilidades H/D/A
  con un modelo de empate paramétrico.

El estado (ratings) es incremental: cuando llegan resultados nuevos del
Mundial, basta con `update_match(...)` para mantenerlo al día sin reentrenar
todo desde cero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import joblib
import pandas as pd

from ..config import MODELS_DIR


@dataclass
class EloModel:
    base_rating: float = 1500.0
    k_factor: float = 40.0          # tasa de aprendizaje base
    home_advantage: float = 65.0    # puntos Elo de ventaja de localía
    draw_c0: float = 0.30           # prob máx de empate (equipos parejos)
    draw_c1: float = 220.0          # decaimiento del empate con |Δrating|
    ratings: dict[str, float] = field(default_factory=dict)
    n_games: dict[str, int] = field(default_factory=dict)
    conf_offset: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    def _expected(self, dr: float) -> float:
        """Expectativa Elo del local (P_H + 0.5*P_D)."""
        return 1.0 / (1.0 + 10.0 ** (-dr / 400.0))

    def probabilities(self, home: str, away: str, neutral: bool = True,
                      host_bonus: float = 0.0) -> dict[str, float]:
        """Probabilidades {'H','D','A'} para local vs visitante."""
        ha = 0.0 if neutral else self.home_advantage
        ha += host_bonus
        dr = (self.rating(home) + ha) - self.rating(away)
        e_home = self._expected(dr)

        p_draw = self.draw_c0 * math.exp(-abs(dr) / self.draw_c1)
        p_home = e_home - 0.5 * p_draw
        p_away = (1.0 - e_home) - 0.5 * p_draw

        # Saneamiento numérico + renormalización
        p_home, p_away = max(p_home, 1e-4), max(p_away, 1e-4)
        p_draw = max(p_draw, 1e-4)
        s = p_home + p_draw + p_away
        return {"H": p_home / s, "D": p_draw / s, "A": p_away / s}

    # ------------------------------------------------------------------ #
    def _margin_multiplier(self, goal_diff: int) -> float:
        """Factor por margen de gol (gana más rating ganar por goleada)."""
        gd = abs(goal_diff)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        return (11.0 + gd) / 8.0

    def update_match(self, home: str, away: str, home_score: int, away_score: int,
                     neutral: bool = True, weight: float = 1.0) -> None:
        """Actualiza los ratings con un resultado individual."""
        ha = 0.0 if neutral else self.home_advantage
        dr = (self.rating(home) + ha) - self.rating(away)
        e_home = self._expected(dr)

        if home_score > away_score:
            s_home = 1.0
        elif home_score < away_score:
            s_home = 0.0
        else:
            s_home = 0.5

        k = self.k_factor * weight * self._margin_multiplier(home_score - away_score)
        delta = k * (s_home - e_home)

        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        self.n_games[home] = self.n_games.get(home, 0) + 1
        self.n_games[away] = self.n_games.get(away, 0) + 1

    def fit(self, matches: pd.DataFrame) -> "EloModel":
        """Entrena recorriendo todos los partidos en orden cronológico."""
        df = matches.sort_values("date")
        for r in df.itertuples(index=False):
            self.update_match(
                r.home_team, r.away_team,
                int(r.home_score), int(r.away_score),
                neutral=bool(getattr(r, "neutral", False)),
                weight=float(getattr(r, "match_weight", 1.0)),
            )
        return self

    # ------------------------------------------------------------------ #
    def apply_confederation_correction(self, matches: pd.DataFrame, beta: float = 0.5) -> dict[str, float]:
        """
        Re-ancla los ratings por confederación según el desempeño REAL en
        partidos inter-confederación, para corregir el sesgo del Elo (equipos
        de confederaciones débiles inflados). Devuelve el offset aplicado por
        confederación. Operación de suma cero (no cambia la media global).
        """
        from ..data.confederations import TEAM_CONF

        # Elo a nivel confederación, usando solo cruces inter-confederación
        conf_elo: dict[str, float] = {}
        df = matches.sort_values("date")
        for r in df.itertuples(index=False):
            ch, ca = TEAM_CONF.get(r.home_team), TEAM_CONF.get(r.away_team)
            if not ch or not ca or ch == ca:
                continue
            rh, ra = conf_elo.get(ch, 1500.0), conf_elo.get(ca, 1500.0)
            e_home = 1.0 / (1.0 + 10.0 ** (-(rh - ra) / 400.0))
            s = 1.0 if r.home_score > r.away_score else (0.0 if r.home_score < r.away_score else 0.5)
            k = 12.0 * float(getattr(r, "match_weight", 1.0))
            conf_elo[ch] = rh + k * (s - e_home)
            conf_elo[ca] = ra - k * (s - e_home)

        if not conf_elo:
            return {}

        mean_conf = sum(conf_elo.values()) / len(conf_elo)
        offset = {c: beta * (v - mean_conf) for c, v in conf_elo.items()}

        # Aplicar a cada equipo según su confederación
        for team in list(self.ratings):
            conf = TEAM_CONF.get(team)
            if conf in offset:
                self.ratings[team] += offset[conf]
        self.conf_offset = offset
        return offset

    def ranking(self, teams: list[str] | None = None) -> pd.DataFrame:
        items = self.ratings.items()
        df = pd.DataFrame(items, columns=["team", "elo"]).sort_values("elo", ascending=False)
        if teams is not None:
            df = df[df["team"].isin(teams)]
        return df.reset_index(drop=True)

    def save(self, path=None):
        path = path or (MODELS_DIR / "elo.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> "EloModel":
        path = path or (MODELS_DIR / "elo.joblib")
        return joblib.load(path)
