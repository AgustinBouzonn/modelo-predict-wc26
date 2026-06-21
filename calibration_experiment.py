"""
¿Mejora la calibración de probabilidades al ensemble?

Aplica temperature scaling (p^(1/T) renormalizado) a las probabilidades 1-X-2
del ensemble. Ajusta T en la primera mitad del holdout y evalúa en la segunda
mitad (out-of-sample), comparando log-loss y RPS con vs sin calibración.

Uso:  python calibration_experiment.py [--months 12]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation.backtest import _CLASSES, _load, _metrics, train_ensemble_asof
from src.features.build_features import build_training_table


def apply_temp(P, T):
    Q = np.clip(P, 1e-9, 1.0) ** (1.0 / T)
    return Q / Q.sum(axis=1, keepdims=True)


def logloss(P, yi):
    P = np.clip(P, 1e-9, 1.0)
    return float(-np.log(P[np.arange(len(yi)), yi]).mean())


def main(months=12):
    matches = _load()
    cutoff = matches["date"].max() - pd.DateOffset(months=months)
    print(f"Datos: {len(matches):,} · corte {cutoff.date()} (holdout {months}m)\n")

    table = build_training_table(matches).copy()
    m = table["home_score"] - table["away_score"]
    table["yi"] = np.where(m > 0, 0, np.where(m == 0, 1, 2))
    test = table[(table["date"] >= cutoff) & (table["date"] < "2026-06-11")]
    test = test[test["match_weight"] >= 2.0].sort_values("date")

    ens = train_ensemble_asof(matches, cutoff.strftime("%Y-%m-%d"))
    P = np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in
                  (ens.predict(r.home_team, r.away_team,
                               neutral=bool(getattr(r, "neutral", False)))
                   for r in test.itertuples(index=False))])
    yi = test["yi"].to_numpy()

    # Split temporal: calibrar en la 1ª mitad, evaluar en la 2ª
    h = len(yi) // 2
    Pc, yc = P[:h], yi[:h]
    Pe, ye = P[h:], yi[h:]

    Ts = np.arange(0.6, 2.01, 0.05)
    best_T = min(Ts, key=lambda T: logloss(apply_temp(Pc, T), yc))

    raw = _metrics(Pe, ye)
    cal = _metrics(apply_temp(Pe, best_T), ye)
    print(f"Calibración: {h} partidos · Evaluación: {len(ye)} partidos\n")
    print(f"Temperatura óptima (1ª mitad): T = {best_T:.2f} "
          f"({'aplana' if best_T > 1 else 'agudiza'} las probabilidades)\n")
    print(f"{'':<22}{'log_loss':>10}{'RPS':>9}{'accuracy':>10}")
    print(f"{'Sin calibrar':<22}{raw['logloss']:>10.4f}{raw['rps']:>9.4f}{raw['acc']:>9.1%}")
    print(f"{'Calibrado (T)':<22}{cal['logloss']:>10.4f}{cal['rps']:>9.4f}{cal['acc']:>9.1%}")
    impr = (raw["logloss"] - cal["logloss"]) / raw["logloss"] * 100
    verdict = "✓ mejora" if impr > 0 else "✗ no mejora"
    print(f"\n{verdict}: log-loss {impr:+.2f}% con calibración por temperatura.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    main(ap.parse_args().months)
