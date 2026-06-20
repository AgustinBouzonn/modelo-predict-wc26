"""
Tracking en vivo: el benchmark honesto y definitivo del modelo vs el mercado.

Idea: ANTES de que se juegue un partido, guardamos la predicción del modelo y
la probabilidad implícita del mercado (cuotas). Cuando el partido se juega,
comparamos ambas contra el resultado REAL. Como las predicciones se congelaron
pre-partido, no hay fuga de datos: es la medición más honesta posible.

Flujo:
  - snapshot_predictions():  para cada partido pendiente, upsert de
    {predicción modelo + prob mercado}. Los partidos ya jugados quedan congelados.
  - evaluate_tracking():     une snapshots con resultados reales y calcula
    log-loss / RPS / accuracy de modelo y mercado sobre los partidos jugados.

Conviene correr snapshot_predictions a diario (antes de reentrenar) para ir
acumulando histórico. Uso:
    python -m src.evaluation.tracking            # toma snapshot + muestra evaluación
    python -m src.evaluation.tracking --eval     # solo evaluación
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DATA_DIR, PROCESSED_DIR
from ..models.ensemble import EnsemblePredictor
from .odds_benchmark import load_odds, implied_probs

PRED_DIR = DATA_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_CSV = PRED_DIR / "predictions.csv"
COLS = ["date", "home", "away", "p_home", "p_draw", "p_away",
        "m_home", "m_draw", "m_away", "snapshot"]


def _load_preds() -> pd.DataFrame:
    if PRED_CSV.exists():
        return pd.read_csv(PRED_CSV)
    return pd.DataFrame(columns=COLS)


def snapshot_predictions(ens: EnsemblePredictor | None = None) -> pd.DataFrame:
    """Guarda/actualiza la predicción del modelo + la prob del mercado de cada
    partido PENDIENTE. Los partidos ya jugados quedan congelados (no se tocan)."""
    from ..data.sources import load_fixture
    ens = ens or EnsemblePredictor.load()
    fx = load_fixture()
    pend = fx[~fx["played"]] if not fx.empty else pd.DataFrame()
    if pend.empty:
        print("[info] no hay partidos pendientes para snapshotear.")
        return _load_preds()

    # Probabilidad de mercado por (home, away), de las cuotas disponibles
    odds = load_odds()
    mkt = {}
    if not odds.empty:
        for o in odds.itertuples(index=False):
            mkt[(o.home_team, o.away_team)] = implied_probs(o.odds_home, o.odds_draw, o.odds_away)

    stamp = datetime.now(timezone.utc).date().isoformat()
    rows = []
    for r in pend.itertuples(index=False):
        p = ens.predict(r.home, r.away, neutral=True)
        m = mkt.get((r.home, r.away), (np.nan, np.nan, np.nan))
        rows.append({"date": r.date.date().isoformat(), "home": r.home, "away": r.away,
                     "p_home": round(p["p_home"], 4), "p_draw": round(p["p_draw"], 4),
                     "p_away": round(p["p_away"], 4),
                     "m_home": m[0], "m_draw": m[1], "m_away": m[2], "snapshot": stamp})
    new = pd.DataFrame(rows)

    existing = _load_preds()
    if not existing.empty:
        keys = set(zip(new["date"], new["home"], new["away"]))
        mask = existing.apply(lambda x: (x["date"], x["home"], x["away"]) in keys, axis=1)
        existing = existing[~mask]   # quitar versiones viejas de los pendientes
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(PRED_CSV, index=False)
    n_mkt = int(new[["m_home"]].notna().sum().iloc[0])
    print(f"[info] snapshot de {len(new)} partidos pendientes ({n_mkt} con cuota de mercado) "
          f"-> {PRED_CSV}")
    return combined


def _logloss(P, yi):
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    return float(-np.log(P[np.arange(len(yi)), yi]).mean())


def _rps(P, yi):
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    oh = np.eye(3)[yi]
    return float(((np.cumsum(P, 1) - np.cumsum(oh, 1)) ** 2).sum(1).mean() / 2)


def evaluate_tracking() -> dict | None:
    """Compara las predicciones congeladas (modelo y mercado) contra los
    resultados reales de los partidos ya jugados que tengan snapshot."""
    preds = _load_preds()
    if preds.empty:
        return None
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").dropna(subset=["home_score", "away_score"])
    matches["date"] = pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d")
    res = matches.set_index(["date", "home_team", "away_team"])[["home_score", "away_score"]]

    Pm, Pk, yi, rows = [], [], [], []
    has_mkt = []
    for r in preds.itertuples(index=False):
        try:
            sc = res.loc[(r.date, r.home, r.away)]
        except KeyError:
            continue
        hs, as_ = int(sc["home_score"]), int(sc["away_score"])
        y = 0 if hs > as_ else (2 if hs < as_ else 1)
        Pm.append([r.p_home, r.p_draw, r.p_away]); yi.append(y)
        rows.append(f"{r.home} {hs}-{as_} {r.away}")
        if pd.notna(r.m_home):
            Pk.append([r.m_home, r.m_draw, r.m_away]); has_mkt.append(True)
        else:
            has_mkt.append(False)
    if not yi:
        return {"n": 0}

    Pm, yi = np.array(Pm), np.array(yi)
    out = {"n": len(yi),
           "modelo": {"logloss": _logloss(Pm, yi), "rps": _rps(Pm, yi),
                      "acc": float((Pm.argmax(1) == yi).mean())}}
    if Pk:
        mask = np.array(has_mkt)
        Pk = np.array(Pk)
        out["n_mercado"] = len(Pk)
        out["mercado"] = {"logloss": _logloss(Pk, yi[mask]), "rps": _rps(Pk, yi[mask]),
                          "acc": float((Pk.argmax(1) == yi[mask]).mean())}
        out["modelo_en_mismos"] = {"logloss": _logloss(Pm[mask], yi[mask]),
                                   "rps": _rps(Pm[mask], yi[mask])}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Solo evaluar (no snapshotear)")
    args = ap.parse_args()
    if not args.eval:
        snapshot_predictions()
    r = evaluate_tracking()
    if not r or r.get("n", 0) == 0:
        print("Todavía no hay partidos jugados con snapshot previo (recién empieza a acumular).")
    else:
        print(f"\n=== Tracking ({r['n']} partidos jugados con snapshot) ===")
        print(f"  modelo  -> log-loss {r['modelo']['logloss']:.3f} | RPS {r['modelo']['rps']:.3f} "
              f"| acc {r['modelo']['acc']:.0%}")
        if "mercado" in r:
            print(f"  mercado -> log-loss {r['mercado']['logloss']:.3f} | RPS {r['mercado']['rps']:.3f} "
                  f"| acc {r['mercado']['acc']:.0%}  ({r['n_mercado']} con cuota)")
