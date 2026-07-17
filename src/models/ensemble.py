"""
Ensemble: combina Elo + Poisson(Dixon-Coles) + ML con pesos configurables,
y aplica el sentimiento de noticias como un ajuste en vivo sobre las
probabilidades finales (transparente y desactivable).

P_ensemble = w_elo*P_elo + w_poisson*P_poisson + w_ml*P_ml   (renormalizado)
Luego se inclina P(H) vs P(A) según la diferencia de momentum mediático.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import joblib
import pandas as pd

from ..config import MODELS_DIR, load_config, canonical
from .elo import EloModel
from .poisson import DixonColesModel
from .ml_model import MLModel

# Intensidad del ajuste por noticias (0 = ignorar). Reducido de 0.35 a 0.20: la
# validación en backtest dio mejora leve pero está inflada por usar sentimiento
# post-hoc; un prior más conservador hasta que el tracking en vivo lo valide.
NEWS_TILT = 0.20


@dataclass
class EnsemblePredictor:
    elo: EloModel
    poisson: DixonColesModel
    ml: MLModel | None = None
    weights: dict[str, float] = field(default_factory=lambda: {"elo": 0.4, "poisson": 0.35, "ml": 0.25})
    sentiment: dict[str, float] = field(default_factory=dict)
    news_tilt: float = NEWS_TILT
    hosts: list[str] = field(default_factory=list)   # anfitriones (juegan en casa)
    host_tilt: float = 0.0                            # ventaja de localía del anfitrión
    temperature: float = 1.0                          # calibración: >1 aplana, <1 afila
    draw_tilt: float = 0.0                            # calibración del empate: pd *= e^d

    # ------------------------------------------------------------------ #
    def _norm_weights(self) -> dict[str, float]:
        w = dict(self.weights)
        if self.ml is None or self.ml.model is None:
            w.pop("ml", None)
        s = sum(w.values()) or 1.0
        return {k: v / s for k, v in w.items()}

    def _apply_news(self, probs: dict[str, float], home: str, away: str) -> dict[str, float]:
        if not self.sentiment or self.news_tilt <= 0:
            return probs
        diff = self.sentiment.get(home, 0.0) - self.sentiment.get(away, 0.0)
        shift = self.news_tilt * diff
        ph = probs["H"] * math.exp(shift / 2)
        pa = probs["A"] * math.exp(-shift / 2)
        pd_ = probs["D"]
        s = ph + pa + pd_
        return {"H": ph / s, "D": pd_ / s, "A": pa / s}

    def _apply_host(self, probs: dict[str, float], home: str, away: str) -> dict[str, float]:
        """Ventaja de localía para el anfitrión (juega en su país con su público)."""
        if not self.hosts or self.host_tilt <= 0:
            return probs
        h_host, a_host = home in self.hosts, away in self.hosts
        if h_host == a_host:           # ambos o ninguno anfitrión: sin ventaja
            return probs
        shift = self.host_tilt if h_host else -self.host_tilt
        ph = probs["H"] * math.exp(shift / 2)
        pa = probs["A"] * math.exp(-shift / 2)
        pd_ = probs["D"]
        s = ph + pa + pd_
        return {"H": ph / s, "D": pd_ / s, "A": pa / s}

    def _apply_temperature(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibración global: p^(1/T) renormalizado. T>1 aplana (modelo
        sobreconfiado), T<1 afila (modelo subconfiado)."""
        t = float(getattr(self, "temperature", 1.0))
        if t == 1.0:
            return probs
        scaled = {k: max(v, 1e-9) ** (1.0 / t) for k, v in probs.items()}
        s = sum(scaled.values())
        return {k: v / s for k, v in scaled.items()}

    def _apply_draw_tilt(self, probs: dict[str, float]) -> dict[str, float]:
        """Calibración del empate (la temperatura no puede corregirlo: afecta
        las 3 clases por igual). Validado en backtest: el modelo sub-predice
        empates. d>0 infla P(D): pd *= e^d, renormalizado."""
        d = float(getattr(self, "draw_tilt", 0.0))
        if d == 0.0:
            return probs
        pd_ = probs["D"] * math.exp(d)
        s = probs["H"] + pd_ + probs["A"]
        return {"H": probs["H"] / s, "D": pd_ / s, "A": probs["A"] / s}

    # ------------------------------------------------------------------ #
    def predict(self, home: str, away: str, neutral: bool = True,
                breakdown: bool = False, knockout: bool = False) -> dict:
        w = self._norm_weights()
        parts: dict[str, dict[str, float]] = {}
        parts["elo"] = self.elo.probabilities(home, away, neutral=neutral)
        parts["poisson"] = self.poisson.probabilities(home, away, neutral=neutral,
                                                      knockout=knockout)
        if "ml" in w:
            parts["ml"] = self.ml.probabilities(home, away, elo=self.elo, neutral=neutral)

        combined = {"H": 0.0, "D": 0.0, "A": 0.0}
        for model, weight in w.items():
            for k in combined:
                combined[k] += weight * parts[model][k]
        s = sum(combined.values()) or 1.0
        combined = {k: v / s for k, v in combined.items()}

        combined = self._apply_news(combined, home, away)
        combined = self._apply_host(combined, home, away)
        combined = self._apply_draw_tilt(combined)
        combined = self._apply_temperature(combined)

        eg_home, eg_away = self.poisson.expected_goals(home, away, neutral=neutral,
                                                       knockout=knockout)
        ml_score = self.poisson.most_likely_score(home, away, neutral=neutral,
                                                  knockout=knockout)
        top = self.poisson.top_scores(home, away, neutral=neutral, n=3, knockout=knockout)

        out = {
            "home": home, "away": away, "neutral": neutral,
            "p_home": combined["H"], "p_draw": combined["D"], "p_away": combined["A"],
            "xg_home": round(eg_home, 2), "xg_away": round(eg_away, 2),
            "likely_score": f"{ml_score[0]}-{ml_score[1]}",
            "top_scores": [(f"{i}-{j}", round(p, 4)) for (i, j), p in top],
        }
        if breakdown:
            out["breakdown"] = parts
            out["weights"] = w
        return out

    def predict_with_lineups(self, home: str, away: str,
                             home_strength: float, away_strength: float,
                             home_baseline: float, away_baseline: float,
                             neutral: bool = True, k: float = 0.08,
                             knockout: bool = False) -> dict:
        """
        Predicción ajustada por las ALINEACIONES elegidas. Compara la fuerza de
        cada XI con la de su mejor XI posible: si un equipo sale con suplentes
        o rota, baja su probabilidad; si pone a sus figuras, la sostiene.
        """
        out = dict(self.predict(home, away, neutral=neutral, knockout=knockout))
        dh = home_strength - home_baseline     # <=0 si el XI está debilitado
        da = away_strength - away_baseline
        shift = k * (dh - da)
        ph = out["p_home"] * math.exp(shift / 2)
        pa = out["p_away"] * math.exp(-shift / 2)
        pdr = out["p_draw"]
        s = ph + pa + pdr
        out.update(p_home=ph / s, p_draw=pdr / s, p_away=pa / s,
                   lineup_shift=round(shift, 3),
                   home_strength=home_strength, away_strength=away_strength)
        return out

    def predict_many(self, fixtures: list[tuple[str, str, bool]]) -> pd.DataFrame:
        return pd.DataFrame([self.predict(h, a, n) for h, a, n in fixtures])

    # ------------------------------------------------------------------ #
    def save(self, path=None):
        path = path or (MODELS_DIR / "ensemble.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> "EnsemblePredictor":
        path = path or (MODELS_DIR / "ensemble.joblib")
        return joblib.load(path)


def load_or_build_sentiment(cfg: dict | None = None) -> dict[str, float]:
    """Carga el sentimiento por equipo (canónico) si existe."""
    from ..data.news import load_sentiment
    cfg = cfg or load_config()
    s = load_sentiment()
    if s.empty:
        return {}
    return {canonical(r.team, cfg): float(r.news_sentiment) for r in s.itertuples(index=False)}
