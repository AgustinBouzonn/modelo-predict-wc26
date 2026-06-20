"""
Backtesting / validación de precisión del modelo (sin fuga de datos).

Entrena el ensemble "as-of" una fecha de corte (solo con partidos ANTERIORES)
y mide la calidad de las predicciones sobre partidos posteriores:

  - accuracy   : % de aciertos del resultado más probable (H/D/A)
  - log-loss   : penaliza la confianza mal puesta (menor = mejor)
  - Brier      : error cuadrático multiclase (menor = mejor)
  - vs baselines: "siempre local" y "azar" (1/3 cada uno)

Dos backtests:
  • Mundial 2026: entrena hasta el 11/6 y predice los partidos ya jugados
    (out-of-sample real).
  • Holdout temporal: entrena hasta hace N meses y evalúa el último período.

Uso:
    python -m src.evaluation.backtest                 # Mundial + holdout 12m
    python -m src.evaluation.backtest --no-holdout
    python -m src.evaluation.backtest --compare-conf  # con vs sin corrección Elo
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, load_config
from ..models.elo import EloModel
from ..models.poisson import DixonColesModel
from ..models.ml_model import MLModel
from ..models.ensemble import EnsemblePredictor

_CLASSES = ["H", "D", "A"]


def _load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["home_score", "away_score"])


def train_ensemble_asof(matches: pd.DataFrame, cutoff: str,
                        conf_correction: bool = True, beta: float = 0.5) -> EnsemblePredictor:
    """Entrena el ensemble usando SOLO partidos anteriores a `cutoff`."""
    cut = pd.to_datetime(cutoff)
    train = matches[matches["date"] < cut].copy()

    elo = EloModel().fit(train)
    offset = elo.apply_confederation_correction(train, beta=beta) if conf_correction else {}
    poisson = DixonColesModel().fit(train)
    ml = MLModel().fit(train, conf_offset=offset)

    cfg = load_config()
    weights = cfg.get("ensemble_weights", {"elo": 0.4, "poisson": 0.35, "ml": 0.25})
    # Sin noticias en backtest: evaluamos solo la capacidad del modelo base
    return EnsemblePredictor(elo=elo, poisson=poisson, ml=ml,
                             weights=weights, sentiment={}, news_tilt=0.0)


def _true_result(hs: float, as_: float) -> str:
    return "H" if hs > as_ else ("A" if hs < as_ else "D")


def _metrics(P: np.ndarray, yi: np.ndarray) -> dict:
    """accuracy, log-loss, Brier y RPS (Ranked Probability Score, el métrico
    correcto para 1X2 ordinal: penaliza errar por más lugares). Menor mejor salvo accuracy."""
    P = np.clip(P, 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    onehot = np.eye(3)[yi]
    acc = float((P.argmax(1) == yi).mean())
    logloss = float(-np.log(P[np.arange(len(yi)), yi]).mean())
    brier = float(((P - onehot) ** 2).sum(1).mean())
    # RPS sobre el orden ordinal H < D < A
    cp, co = np.cumsum(P, axis=1), np.cumsum(onehot, axis=1)
    rps = float(((cp - co) ** 2).sum(1).mean() / 2)
    return {"acc": acc, "logloss": logloss, "brier": brier, "rps": rps}


def compare_models(ens: EnsemblePredictor, test: pd.DataFrame) -> pd.DataFrame:
    """Ablation: métricas de cada componente y del ensemble vs baselines.
    Muestra qué aporta cada parte y si el modelo le gana a 'siempre local'/'azar'."""
    rows_elo, rows_pois, rows_ml, rows_ens, yi = [], [], [], [], []
    for r in test.itertuples(index=False):
        neu = bool(getattr(r, "neutral", False))
        b = ens.predict(r.home_team, r.away_team, neutral=neu, breakdown=True)
        parts = b["breakdown"]
        rows_elo.append([parts["elo"][c] for c in _CLASSES])
        rows_pois.append([parts["poisson"][c] for c in _CLASSES])
        rows_ml.append([parts["ml"][c] for c in _CLASSES] if "ml" in parts else [1/3]*3)
        rows_ens.append([b["p_home"], b["p_draw"], b["p_away"]])
        yi.append(_CLASSES.index(_true_result(r.home_score, r.away_score)))
    yi = np.array(yi)
    n = len(yi)
    out = {
        "Elo solo": _metrics(np.array(rows_elo), yi),
        "Poisson solo": _metrics(np.array(rows_pois), yi),
        "ML solo": _metrics(np.array(rows_ml), yi),
        "Ensemble (final)": _metrics(np.array(rows_ens), yi),
        "Baseline: siempre local": _metrics(np.tile([0.6, 0.25, 0.15], (n, 1)), yi),
        "Baseline: azar (1/3)": _metrics(np.tile([1/3, 1/3, 1/3], (n, 1)), yi),
    }
    df = pd.DataFrame(out).T
    df.columns = ["accuracy", "log_loss", "brier", "RPS"]
    return df.round(4)


def calibration_table(ens: EnsemblePredictor, test: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Diagrama de calibración: agrupa las predicciones por la prob del resultado
    más probable y compara con la frecuencia real de acierto. Bien calibrado =
    prob ≈ frecuencia. Revela sobre/sub-confianza del modelo."""
    conf, hit = [], []
    for r in test.itertuples(index=False):
        b = ens.predict(r.home_team, r.away_team, neutral=bool(getattr(r, "neutral", False)))
        probs = {"H": b["p_home"], "D": b["p_draw"], "A": b["p_away"]}
        pick = max(probs, key=probs.get)
        conf.append(probs[pick])
        hit.append(pick == _true_result(r.home_score, r.away_score))
    df = pd.DataFrame({"conf": conf, "hit": hit})
    df["bin"] = pd.cut(df["conf"], np.linspace(0.33, 1.0, bins + 1))
    g = df.groupby("bin", observed=True).agg(
        n=("hit", "size"), confianza_media=("conf", "mean"), acierto_real=("hit", "mean"))
    return g.round(3).reset_index().astype({"bin": str})


def evaluate(ens: EnsemblePredictor, test: pd.DataFrame) -> dict:
    """Calcula métricas sobre un conjunto de test ya jugado."""
    P, Y = [], []
    for r in test.itertuples(index=False):
        p = ens.predict(r.home_team, r.away_team, neutral=bool(getattr(r, "neutral", False)))
        P.append([p["p_home"], p["p_draw"], p["p_away"]])
        Y.append(_true_result(r.home_score, r.away_score))

    P = np.clip(np.array(P), 1e-9, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    yi = np.array([_CLASSES.index(y) for y in Y])
    onehot = np.eye(3)[yi]

    pred_idx = P.argmax(axis=1)
    acc = float((pred_idx == yi).mean())
    logloss = float(-np.log(P[np.arange(len(yi)), yi]).mean())
    brier = float(((P - onehot) ** 2).sum(axis=1).mean())

    # Baselines
    home = np.tile([0.45, 0.27, 0.28], (len(yi), 1))  # frecuencias típicas locales
    ll_home = float(-np.log(np.clip(home[np.arange(len(yi)), yi], 1e-9, 1)).mean())
    acc_home = float((np.zeros(len(yi)) == yi).mean())
    ll_unif = float(-np.log(1 / 3) * 1)

    return {
        "n": len(yi), "accuracy": acc, "log_loss": logloss, "brier": brier,
        "baseline_home_acc": acc_home, "baseline_home_logloss": ll_home,
        "baseline_uniform_logloss": ll_unif,
    }


def per_match_table(ens: EnsemblePredictor, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in test.itertuples(index=False):
        p = ens.predict(r.home_team, r.away_team, neutral=bool(getattr(r, "neutral", False)))
        true = _true_result(r.home_score, r.away_score)
        probs = {"H": p["p_home"], "D": p["p_draw"], "A": p["p_away"]}
        pick = max(probs, key=probs.get)
        rows.append({
            "partido": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
            "real": true, "pred": pick, "acierto": "✓" if pick == true else "✗",
            "P_real": round(probs[true], 2),
            "P(H/D/A)": f"{probs['H']:.0%}/{probs['D']:.0%}/{probs['A']:.0%}",
        })
    return pd.DataFrame(rows)


def _print_metrics(title: str, m: dict) -> None:
    print(f"\n=== {title} ({m['n']} partidos) ===")
    print(f"  Accuracy : {m['accuracy']:.1%}   (baseline 'siempre local': {m['baseline_home_acc']:.1%})")
    print(f"  Log-loss : {m['log_loss']:.3f}   (local: {m['baseline_home_logloss']:.3f} | azar: {m['baseline_uniform_logloss']:.3f})")
    print(f"  Brier    : {m['brier']:.3f}   (menor es mejor; azar ≈ 0.667)")


def optimize_weights(matches: pd.DataFrame, months_test: int = 12,
                     step: float = 0.05, write: bool = False) -> dict:
    """
    Busca por grid los pesos del ensemble (Elo/Poisson/ML) que minimizan el
    log-loss en un holdout temporal. Entrena UNA vez y barre los pesos sobre las
    probabilidades por modelo ya calculadas (rápido). Opcionalmente los escribe
    al config.
    """
    cutoff = matches["date"].max() - pd.DateOffset(months=months_test)
    ens = train_ensemble_asof(matches, cutoff.strftime("%Y-%m-%d"), conf_correction=True)
    test = matches[(matches["date"] >= cutoff) & (matches["date"] < "2026-06-11")]
    test = test[test["match_weight"] >= 2.0]

    Pe, Pp, Pm, yi = [], [], [], []
    for r in test.itertuples(index=False):
        b = ens.predict(r.home_team, r.away_team,
                        neutral=bool(getattr(r, "neutral", False)), breakdown=True)["breakdown"]
        Pe.append([b["elo"][c] for c in _CLASSES])
        Pp.append([b["poisson"][c] for c in _CLASSES])
        Pm.append([b["ml"][c] for c in _CLASSES] if "ml" in b else [1/3, 1/3, 1/3])
        yi.append(_CLASSES.index(_true_result(r.home_score, r.away_score)))
    Pe, Pp, Pm = np.array(Pe), np.array(Pp), np.array(Pm)
    yi = np.array(yi)
    idx = np.arange(len(yi))

    def ll(P):
        P = P / P.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(P[idx, yi], 1e-9, 1)).mean())

    base = {"elo": 0.40, "poisson": 0.35, "ml": 0.25}
    base_ll = ll(base["elo"] * Pe + base["poisson"] * Pp + base["ml"] * Pm)

    best, best_ll = base, base_ll
    we = 0.0
    while we <= 1.0 + 1e-9:
        wp = 0.0
        while wp <= 1.0 - we + 1e-9:
            wm = 1.0 - we - wp
            cur = ll(we * Pe + wp * Pp + wm * Pm)
            if cur < best_ll:
                best_ll, best = cur, {"elo": round(we, 2), "poisson": round(wp, 2), "ml": round(wm, 2)}
            wp += step
        we += step

    print(f"\n=== Optimización de pesos (holdout {months_test}m, {len(yi)} partidos) ===")
    print(f"  Pesos actuales {base}  -> log-loss {base_ll:.4f}")
    print(f"  Pesos óptimos  {best}  -> log-loss {best_ll:.4f}")
    print(f"  Mejora: {(base_ll - best_ll):.4f} ({(base_ll-best_ll)/base_ll*100:.1f}%)")

    if write:
        from ..data.sources import _write_config
        cfg = load_config()
        cfg["ensemble_weights"] = best
        _write_config(cfg)
        print("  ✓ pesos escritos en config/teams_wc26.yaml (reentrená para aplicarlos)")
    return {"current": base, "best": best, "current_ll": base_ll, "best_ll": best_ll}


def evaluate_tilts(matches: pd.DataFrame) -> pd.DataFrame:
    """Mide si los ajustes heurísticos (noticias, ventaja de anfitrión) MEJORAN
    o EMPEORAN las predicciones, comparando el ensemble con cada tilt on/off sobre
    los partidos del Mundial ya jugados (out-of-sample, entrenado as-of 11/6).

    Salvedad: el sentimiento se aplica con su valor ACTUAL (no hay histórico de
    noticias), así que la validación de 'noticias' es indicativa, no pura."""
    from ..config import load_config, canonical
    from ..models.ensemble import load_or_build_sentiment

    cfg = load_config()
    ens = train_ensemble_asof(matches, "2026-06-11")
    ens.sentiment = load_or_build_sentiment(cfg)
    ens.hosts = [canonical(h, cfg) for h in cfg.get("tournament", {}).get("hosts", [])]
    news_v = float(cfg.get("news_tilt", 0.35) or 0.35)
    host_v = float(cfg.get("host_advantage_tilt", 0.25) or 0.25)

    test = matches[matches["tournament"].str.contains("FIFA World Cup", na=False)
                   & ~matches["tournament"].str.contains("qualif", case=False, na=False)]
    test = test[test["date"] >= "2026-06-11"]
    if test.empty:
        return pd.DataFrame()

    configs = {"base (sin tilts)": (0.0, 0.0), "+ noticias": (news_v, 0.0),
               "+ anfitrión": (0.0, host_v), "+ ambos": (news_v, host_v)}
    rows = {}
    for name, (nt, ht) in configs.items():
        ens.news_tilt, ens.host_tilt = nt, ht
        P, yi = [], []
        for r in test.itertuples(index=False):
            p = ens.predict(r.home_team, r.away_team, neutral=True)
            P.append([p["p_home"], p["p_draw"], p["p_away"]])
            yi.append(_CLASSES.index(_true_result(r.home_score, r.away_score)))
        rows[name] = _metrics(np.array(P), np.array(yi))
    df = pd.DataFrame(rows).T
    df.columns = ["accuracy", "log_loss", "brier", "RPS"]
    return df.round(4)


def backtest_worldcup(matches: pd.DataFrame, conf_correction: bool = True) -> dict:
    test = matches[matches["tournament"].str.contains("FIFA World Cup", na=False)
                   & ~matches["tournament"].str.contains("qualification", case=False, na=False)]
    test = test[test["date"] >= "2026-06-11"]
    if test.empty:
        print("[aviso] no hay partidos jugados del Mundial para evaluar.")
        return {}
    ens = train_ensemble_asof(matches, "2026-06-11", conf_correction=conf_correction)
    m = evaluate(ens, test)
    _print_metrics("Mundial 2026 (out-of-sample)", m)
    print()
    print(per_match_table(ens, test).to_string(index=False))
    return m


def backtest_holdout(matches: pd.DataFrame, months: int = 12,
                     conf_correction: bool = True) -> dict:
    cutoff = matches["date"].max() - pd.DateOffset(months=months)
    test = matches[(matches["date"] >= cutoff) & (matches["date"] < "2026-06-11")]
    # Filtrar partidos competitivos (no microselecciones) para una señal más limpia
    test = test[test["match_weight"] >= 2.0]
    ens = train_ensemble_asof(matches, cutoff.strftime("%Y-%m-%d"), conf_correction=conf_correction)
    m = evaluate(ens, test)
    _print_metrics(f"Holdout últimos {months} meses (competitivos)", m)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest del predictor WC26")
    ap.add_argument("--no-holdout", action="store_true", help="Saltear el holdout temporal")
    ap.add_argument("--months", type=int, default=12, help="Meses del holdout")
    ap.add_argument("--compare-conf", action="store_true",
                    help="Comparar con vs sin corrección por confederación")
    ap.add_argument("--optimize-weights", action="store_true",
                    help="Buscar los pesos óptimos del ensemble")
    ap.add_argument("--write", action="store_true",
                    help="Escribir los pesos óptimos al config")
    args = ap.parse_args()

    matches = _load()
    print(f"Datos: {len(matches):,} partidos hasta {matches['date'].max().date()}")

    if args.optimize_weights:
        optimize_weights(matches, months_test=args.months, write=args.write)
    elif args.compare_conf:
        print("\n############ CON corrección por confederación ############")
        backtest_worldcup(matches, conf_correction=True)
        if not args.no_holdout:
            backtest_holdout(matches, months=args.months, conf_correction=True)
        print("\n############ SIN corrección por confederación ############")
        backtest_worldcup(matches, conf_correction=False)
        if not args.no_holdout:
            backtest_holdout(matches, months=args.months, conf_correction=False)
    else:
        backtest_worldcup(matches, conf_correction=True)
        if not args.no_holdout:
            backtest_holdout(matches, months=args.months, conf_correction=True)
