"""
Modelo Poisson bivariado con corrección Dixon-Coles.

Estima para cada selección una fuerza de ATAQUE y de DEFENSA, más una
ventaja de localía global, ajustando una regresión de Poisson (vía
sklearn.PoissonRegressor, convexa y rápida) con:
  - ponderación temporal (los partidos recientes pesan más, half-life configurable)
  - ponderación por importancia del torneo
  - efectos por confederación en el diseño: la regularización L2 encoge a cada
    equipo hacia la media de SU confederación (prior jerárquico), no hacia la
    media global — clave para equipos con pocos cruces ante rivales top.
  - factor de eliminatorias: si el dataset trae la columna `is_knockout`, se
    estima por MLE un multiplicador de goles para partidos de eliminación
    directa (históricamente se juegan más cerrados) y se aplica al predecir
    con knockout=True.

Luego ajusta el parámetro rho de Dixon-Coles (corrige la dependencia en
marcadores bajos: 0-0, 1-0, 0-1, 1-1) por máxima verosimilitud 1-D.

A partir de eso genera la matriz de probabilidades de marcador, de la que
se derivan P(H/D/A), goles esperados y el marcador más probable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from scipy import sparse
from sklearn.linear_model import PoissonRegressor

from ..config import MODELS_DIR


def _dc_tau(h: np.ndarray, a: np.ndarray, lh: np.ndarray, la: np.ndarray,
            rho: float) -> np.ndarray:
    """Factor de corrección Dixon-Coles para marcadores bajos."""
    tau = np.ones_like(lh, dtype=float)
    m00 = (h == 0) & (a == 0)
    m01 = (h == 0) & (a == 1)
    m10 = (h == 1) & (a == 0)
    m11 = (h == 1) & (a == 1)
    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


@dataclass
class DixonColesModel:
    half_life_days: float = 730.0   # vigencia: 2 años
    alpha: float = 1e-3             # regularización L2 de la regresión Poisson
    max_goals: int = 10

    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defense: dict[str, float] = field(default_factory=dict)
    home_adv: float = 0.0
    rho: float = 0.0
    knockout_factor: float = 1.0    # multiplicador de goles en eliminatorias

    # ------------------------------------------------------------------ #
    def _weights(self, dates: pd.Series, match_weight: pd.Series) -> np.ndarray:
        ref = dates.max()
        age_days = (ref - dates).dt.days.clip(lower=0).to_numpy()
        decay = 0.5 ** (age_days / self.half_life_days)
        return decay * match_weight.to_numpy()

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        from ..data.confederations import TEAM_CONF

        df = matches.dropna(subset=["home_score", "away_score"]).copy()
        df["date"] = pd.to_datetime(df["date"])

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Confederación de cada equipo (columna extra en el diseño): el L2 encoge
        # la desviación del equipo hacia el efecto de su confederación.
        confs = sorted({TEAM_CONF.get(t, "OTRA") for t in self.teams})
        cix = {c: i for i, c in enumerate(confs)}
        conf_i = {t: cix[TEAM_CONF.get(t, "OTRA")] for t in self.teams}
        nc = len(confs)
        att0, def0 = 1, 1 + n                    # bloques equipo
        catt0, cdef0 = 1 + 2 * n, 1 + 2 * n + nc  # bloques confederación

        # Dos observaciones por partido (lado local y lado visitante).
        # Columnas: [home_flag] + att/def por equipo + att/def por confederación
        rows, cols, vals = [], [], []
        y, w_parts = [], []
        weights = self._weights(df["date"], df.get("match_weight", pd.Series(1.0, index=df.index)))

        obs = 0
        for (h, a, hs, as_, neu), wt in zip(
            df[["home_team", "away_team", "home_score", "away_score", "neutral"]].itertuples(index=False, name=None),
            weights,
        ):
            is_home = 0.0 if bool(neu) else 1.0
            # --- lado local: goles del local ---
            rows += [obs] * 5
            cols += [0, att0 + idx[h], def0 + idx[a], catt0 + conf_i[h], cdef0 + conf_i[a]]
            vals += [is_home, 1.0, 1.0, 1.0, 1.0]
            y.append(hs); w_parts.append(wt); obs += 1
            # --- lado visitante: goles del visitante ---
            rows += [obs] * 4
            cols += [att0 + idx[a], def0 + idx[h], catt0 + conf_i[a], cdef0 + conf_i[h]]
            vals += [1.0, 1.0, 1.0, 1.0]
            y.append(as_); w_parts.append(wt); obs += 1

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(obs, 1 + 2 * n + 2 * nc))
        y = np.asarray(y, dtype=float)
        sample_weight = np.asarray(w_parts, dtype=float)

        reg = PoissonRegressor(alpha=self.alpha, max_iter=500, fit_intercept=True)
        reg.fit(X, y, sample_weight=sample_weight)

        coef = reg.coef_
        self.home_adv = float(coef[0])
        base = float(reg.intercept_)
        # Fuerza final = desviación del equipo + efecto de su confederación
        # (absorbemos el intercepto en el ataque para que los lambdas escalen bien)
        self.attack = {t: float(coef[att0 + i]) + float(coef[catt0 + conf_i[t]]) + base
                       for i, t in enumerate(self.teams)}
        self.defense = {t: float(coef[def0 + i]) + float(coef[cdef0 + conf_i[t]])
                        for i, t in enumerate(self.teams)}

        self._fit_rho(df, weights)
        self._fit_knockout_factor(df, weights)
        return self

    def _fit_knockout_factor(self, df: pd.DataFrame, weights: np.ndarray) -> None:
        """MLE del multiplicador común de goles en eliminatorias: para
        y ~ Poisson(k*lambda), k = sum(w*y) / sum(w*lambda)."""
        self.knockout_factor = 1.0
        if "is_knockout" not in df.columns:
            return
        mask = df["is_knockout"].fillna(False).astype(bool).to_numpy()
        if mask.sum() < 30:                     # muy pocos: no estimar
            return
        ko = df[mask]
        w = np.asarray(weights)[mask]
        lh = np.array([self._rate(h, a, apply_home=not bool(neu)) for h, a, neu in
                       ko[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None)])
        la = np.array([self._rate(a, h, apply_home=False) for h, a, neu in
                       ko[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None)])
        goals = ko["home_score"].to_numpy(float) + ko["away_score"].to_numpy(float)
        pred = lh + la
        k = float((w * goals).sum() / max((w * pred).sum(), 1e-9))
        self.knockout_factor = float(np.clip(k, 0.7, 1.15))

    def _fit_rho(self, df: pd.DataFrame, weights: np.ndarray) -> None:
        h = df["home_score"].to_numpy(dtype=int)
        a = df["away_score"].to_numpy(dtype=int)
        triples = list(df[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None))
        lh = np.array([self._rate(ht, at, apply_home=not bool(neu)) for ht, at, neu in triples])
        la = np.array([self._rate(at, ht, apply_home=False) for ht, at, neu in triples])

        def neg_ll(rho: float) -> float:
            tau = _dc_tau(h, a, lh, la, rho)
            tau = np.clip(tau, 1e-9, None)
            return -np.sum(weights * np.log(tau))

        res = minimize_scalar(neg_ll, bounds=(-0.2, 0.2), method="bounded")
        self.rho = float(res.x)

    # ------------------------------------------------------------------ #
    def _rate(self, attacker: str, defender: str, apply_home: bool) -> float:
        att = self.attack.get(attacker, np.mean(list(self.attack.values())) if self.attack else 0.0)
        dfn = self.defense.get(defender, np.mean(list(self.defense.values())) if self.defense else 0.0)
        lo = att + dfn + (self.home_adv if apply_home else 0.0)
        return float(np.exp(np.clip(lo, -5, 4)))

    def _lambdas(self, home: str, away: str, neutral: bool,
                 knockout: bool = False) -> tuple[float, float]:
        lh = self._rate(home, away, apply_home=not neutral)
        la = self._rate(away, home, apply_home=False)
        if knockout:
            k = float(getattr(self, "knockout_factor", 1.0))
            lh, la = lh * k, la * k
        return lh, la

    def score_matrix(self, home: str, away: str, neutral: bool = True,
                     knockout: bool = False) -> np.ndarray:
        """Matriz (max_goals+1 x max_goals+1) de probabilidad de marcador."""
        lh, la = self._lambdas(home, away, neutral, knockout)

        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g, lh)
        pa = poisson.pmf(g, la)
        mat = np.outer(ph, pa)

        # Corrección Dixon-Coles en las 4 celdas bajas
        mat[0, 0] *= 1.0 - lh * la * self.rho
        mat[0, 1] *= 1.0 + lh * self.rho
        mat[1, 0] *= 1.0 + la * self.rho
        mat[1, 1] *= 1.0 - self.rho
        mat = np.clip(mat, 0, None)
        mat /= mat.sum()
        return mat

    def probabilities(self, home: str, away: str, neutral: bool = True,
                      knockout: bool = False, **kwargs) -> dict[str, float]:
        mat = self.score_matrix(home, away, neutral, knockout=knockout)
        p_home = np.tril(mat, -1).sum()   # local marca más
        p_away = np.triu(mat, 1).sum()    # visitante marca más
        p_draw = np.trace(mat)
        return {"H": float(p_home), "D": float(p_draw), "A": float(p_away)}

    def expected_goals(self, home: str, away: str, neutral: bool = True,
                       knockout: bool = False) -> tuple[float, float]:
        return self._lambdas(home, away, neutral, knockout)

    def most_likely_score(self, home: str, away: str, neutral: bool = True,
                          knockout: bool = False) -> tuple[int, int]:
        mat = self.score_matrix(home, away, neutral, knockout=knockout)
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        return int(i), int(j)

    def top_scores(self, home: str, away: str, neutral: bool = True,
                   n: int = 3, knockout: bool = False) -> list[tuple[tuple[int, int], float]]:
        """Los n marcadores exactos más probables, con su probabilidad."""
        mat = self.score_matrix(home, away, neutral, knockout=knockout)
        flat = np.argsort(mat, axis=None)[::-1][:n]
        ii, jj = np.unravel_index(flat, mat.shape)
        return [((int(i), int(j)), float(mat[i, j])) for i, j in zip(ii, jj)]

    # ------------------------------------------------------------------ #
    def save(self, path=None):
        path = path or (MODELS_DIR / "poisson.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> "DixonColesModel":
        path = path or (MODELS_DIR / "poisson.joblib")
        return joblib.load(path)
