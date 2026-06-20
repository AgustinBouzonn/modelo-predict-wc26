"""
Modelo Poisson bivariado con corrección Dixon-Coles.

Estima para cada selección una fuerza de ATAQUE y de DEFENSA, más una
ventaja de localía global, ajustando una regresión de Poisson (vía
sklearn.PoissonRegressor, convexa y rápida) con:
  - ponderación temporal (los partidos recientes pesan más, half-life configurable)
  - ponderación por importancia del torneo

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

    # ------------------------------------------------------------------ #
    def _weights(self, dates: pd.Series, match_weight: pd.Series) -> np.ndarray:
        ref = dates.max()
        age_days = (ref - dates).dt.days.clip(lower=0).to_numpy()
        decay = 0.5 ** (age_days / self.half_life_days)
        return decay * match_weight.to_numpy()

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        df = matches.dropna(subset=["home_score", "away_score"]).copy()
        df["date"] = pd.to_datetime(df["date"])

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        # Dos observaciones por partido (lado local y lado visitante).
        # Columnas del diseño: [home_flag] + att(one-hot) + def(one-hot del rival)
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
            rows += [obs, obs];  cols += [1 + idx[h], 1 + n + idx[a]];  vals += [1.0, 1.0]
            rows += [obs];       cols += [0];                            vals += [is_home]
            y.append(hs); w_parts.append(wt); obs += 1
            # --- lado visitante: goles del visitante ---
            rows += [obs, obs];  cols += [1 + idx[a], 1 + n + idx[h]];  vals += [1.0, 1.0]
            y.append(as_); w_parts.append(wt); obs += 1

        X = sparse.csr_matrix((vals, (rows, cols)), shape=(obs, 1 + 2 * n))
        y = np.asarray(y, dtype=float)
        sample_weight = np.asarray(w_parts, dtype=float)

        reg = PoissonRegressor(alpha=self.alpha, max_iter=500, fit_intercept=True)
        reg.fit(X, y, sample_weight=sample_weight)

        coef = reg.coef_
        self.home_adv = float(coef[0])
        base = float(reg.intercept_)
        # Absorbemos el intercepto en el ataque para que los lambdas queden bien escalados
        self.attack = {t: float(coef[1 + i]) + base for i, t in enumerate(self.teams)}
        self.defense = {t: float(coef[1 + n + i]) for i, t in enumerate(self.teams)}

        self._fit_rho(df, weights)
        return self

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

    def score_matrix(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        """Matriz (max_goals+1 x max_goals+1) de probabilidad de marcador."""
        lh = self._rate(home, away, apply_home=not neutral)
        la = self._rate(away, home, apply_home=False)

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
                      **kwargs) -> dict[str, float]:
        mat = self.score_matrix(home, away, neutral)
        p_home = np.tril(mat, -1).sum()   # local marca más
        p_away = np.triu(mat, 1).sum()    # visitante marca más
        p_draw = np.trace(mat)
        return {"H": float(p_home), "D": float(p_draw), "A": float(p_away)}

    def expected_goals(self, home: str, away: str, neutral: bool = True) -> tuple[float, float]:
        return (self._rate(home, away, apply_home=not neutral),
                self._rate(away, home, apply_home=False))

    def most_likely_score(self, home: str, away: str, neutral: bool = True) -> tuple[int, int]:
        mat = self.score_matrix(home, away, neutral)
        i, j = np.unravel_index(np.argmax(mat), mat.shape)
        return int(i), int(j)

    # ------------------------------------------------------------------ #
    def save(self, path=None):
        path = path or (MODELS_DIR / "poisson.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> "DixonColesModel":
        path = path or (MODELS_DIR / "poisson.joblib")
        return joblib.load(path)
