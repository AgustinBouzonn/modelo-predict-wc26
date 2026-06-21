"""
Cierra el círculo cuántico↔clásico: ¿aporta el modelo cuántico al ensemble?

Mezcla la distribución 1-X-2 del clasificador cuántico con la del ensemble
clásico (blend convexo) y busca el peso del cuántico que minimiza el log-loss
en el holdout out-of-sample. Si el mejor blend mejora al ensemble solo, el
cuántico aporta señal complementaria.

Uso:  python quantum_ensemble.py [--months 12]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.evaluation.backtest import _CLASSES, _load, _metrics, train_ensemble_asof
from src.features.build_features import build_training_table
from src.models.quantum import FEATURES, QuantumMatchClassifier


def main(months=12):
    matches = _load()
    cutoff = matches["date"].max() - pd.DateOffset(months=months)
    print(f"Datos: {len(matches):,} partidos · corte {cutoff.date()} "
          f"(holdout {months}m)\n")

    table = build_training_table(matches).copy()
    margin = table["home_score"] - table["away_score"]
    table["yi"] = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))

    train = table[table["date"] < cutoff]
    test = table[(table["date"] >= cutoff) & (table["date"] < "2026-06-11")]
    test = test[test["match_weight"] >= 2.0]

    clf = QuantumMatchClassifier().fit(train[FEATURES].to_numpy(float),
                                       train["yi"].to_numpy(int))
    q_P = np.array([[clf.predict_proba(r)[c] for c in _CLASSES]
                    for r in test[FEATURES].to_numpy(float)])

    ens = train_ensemble_asof(matches, cutoff.strftime("%Y-%m-%d"))
    e_P = np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in
                    (ens.predict(r.home_team, r.away_team,
                                 neutral=bool(getattr(r, "neutral", False)))
                     for r in test.itertuples(index=False))])

    yi = test["yi"].to_numpy()
    base = _metrics(e_P, yi)
    print(f"Test: {len(yi)} partidos competitivos\n")
    print(f"{'blend (peso cuántico)':<24}{'log_loss':>10}{'RPS':>9}{'accuracy':>10}")
    print(f"{'Ensemble solo (w=0.00)':<24}{base['logloss']:>10.4f}"
          f"{base['rps']:>9.4f}{base['acc']:>9.1%}")

    best = (0.0, base["logloss"], base)
    for w in np.arange(0.05, 0.51, 0.05):
        blend = (1 - w) * e_P + w * q_P
        m = _metrics(blend, yi)
        if w in (0.1, 0.2, 0.3, 0.4, 0.5):
            print(f"{'+ cuántico w=' + format(w, '.2f'):<24}{m['logloss']:>10.4f}"
                  f"{m['rps']:>9.4f}{m['acc']:>9.1%}")
        if m["logloss"] < best[1]:
            best = (w, m["logloss"], m)

    print()
    if best[0] > 0:
        impr = (base["logloss"] - best[1]) / base["logloss"] * 100
        print(f"✓ Mejor blend: peso cuántico {best[0]:.2f} -> log-loss {best[1]:.4f} "
              f"(mejora {impr:.2f}% vs ensemble solo)")
    else:
        print("✗ El blend no mejora: el cuántico no aporta señal complementaria "
              "sobre el ensemble en este holdout.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    main(ap.parse_args().months)
