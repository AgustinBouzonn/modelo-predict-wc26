"""
Comparación 1-X-2 (3 clases) Cuántico vs Ensemble, out-of-sample.

Entrena ambos as-of una fecha de corte y los evalúa sobre el holdout de los
últimos N meses (partidos competitivos). Métricas: accuracy, log-loss y RPS
(la métrica correcta para 1X2 ordinal).

Uso:  python quantum_eval.py [--months 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import _CLASSES, _load, _metrics, train_ensemble_asof
from src.features.build_features import build_training_table
from src.models.quantum import FEATURES, QuantumMatchClassifier

METRICS_PATH = Path(__file__).resolve().parent / "docs" / "metricas_holdout.json"


def _result_idx(hs, as_):
    return 0 if hs > as_ else (1 if hs == as_ else 2)  # H/D/A


def main(months=12):
    matches = _load()
    cutoff = matches["date"].max() - pd.DateOffset(months=months)
    print(f"Datos: {len(matches):,} partidos hasta {matches['date'].max().date()}")
    print(f"Corte: {cutoff.date()}  (holdout = últimos {months} meses)\n")

    table = build_training_table(matches).copy()
    margin = table["home_score"] - table["away_score"]
    table["yi"] = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))

    train = table[table["date"] < cutoff]
    test = table[(table["date"] >= cutoff) & (table["date"] < "2026-06-11")]
    test = test[test["match_weight"] >= 2.0]
    print(f"Entrenamiento: {len(train):,} · Test competitivo: {len(test):,}\n")

    # ---- Cuántico (as-of, 4 features, 3 clases) ----
    clf = QuantumMatchClassifier().fit(train[FEATURES].to_numpy(float),
                                       train["yi"].to_numpy(int))
    q_P = np.array([[clf.predict_proba(r)[c] for c in _CLASSES]
                    for r in test[FEATURES].to_numpy(float)])

    # ---- Ensemble (as-of) ----
    ens = train_ensemble_asof(matches, cutoff.strftime("%Y-%m-%d"))
    e_P = []
    for r in test.itertuples(index=False):
        p = ens.predict(r.home_team, r.away_team, neutral=bool(getattr(r, "neutral", False)))
        e_P.append([p["p_home"], p["p_draw"], p["p_away"]])
    e_P = np.array(e_P)

    yi = test["yi"].to_numpy()
    n = len(yi)
    base = np.tile([0.45, 0.27, 0.28], (n, 1))

    qm = _metrics(q_P, yi)
    em = _metrics(e_P, yi)
    bm = _metrics(base, yi)
    print("=== Cuántico (4q, 1-X-2) vs Ensemble · holdout out-of-sample ===")
    print(f"{'modelo':<26}{'accuracy':>10}{'log_loss':>10}{'RPS':>8}")
    print(f"{'⚛️ Cuántico (4 qubits)':<26}{qm['acc']:>9.1%}{qm['logloss']:>10.4f}{qm['rps']:>8.4f}")
    print(f"{'🧮 Ensemble':<26}{em['acc']:>9.1%}{em['logloss']:>10.4f}{em['rps']:>8.4f}")
    print(f"{'📏 Baseline (frec.)':<26}{bm['acc']:>9.1%}{bm['logloss']:>10.4f}{bm['rps']:>8.4f}")

    # Lo usa scripts/make_readme_figure.py para que las figuras no queden desfasadas.
    def _row(m):
        return {k: round(float(m[k]), 4) for k in ("acc", "logloss", "rps")}

    payload = {
        "medido": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "datos_hasta": str(matches["date"].max().date()),
        "desde": str(cutoff.date()),
        "meses": months,
        "partidos": int(n),
        "cuantico": _row(qm),
        "ensemble": _row(em),
        "baseline_frecuencias": _row(bm),
    }
    METRICS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nMétricas guardadas en {METRICS_PATH.relative_to(METRICS_PATH.parent.parent)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=12)
    main(ap.parse_args().months)
