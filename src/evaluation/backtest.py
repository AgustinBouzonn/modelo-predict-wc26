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


# =========================================================================== #
# TUNING con validación temporal rolling-origin (multi-fold, sin fuga)
# =========================================================================== #
# 7 cortes anuales + el corte pre-Mundial: entrena hasta el 01/06/2026 y
# evalúa sobre el WC26 jugado (el fold más parecido al dominio que nos importa)
FOLD_CUTOFFS = [f"{y}-01-01" for y in range(2019, 2026)] + ["2026-06-01"]
HORIZON_DAYS = 365


def _load_clean() -> pd.DataFrame:
    """Como _load(), pero re-derivando match_weight SIN el boost del torneo en
    curso (para que la config de producción no contamine el backtest) y con el
    flag is_knockout recalculado."""
    from ..data.sources import tournament_weight
    df = _load().copy()
    df["match_weight"] = df["tournament"].map(tournament_weight).fillna(1.0)
    is_wc = (df["tournament"].str.contains("FIFA World Cup", na=False)
             & ~df["tournament"].str.contains("qualification", case=False, na=False))
    df["is_knockout"] = False
    for year, g in df[is_wc].groupby(df["date"].dt.year):
        if year == 2026:
            df.loc[g.index[g["date"] >= pd.Timestamp("2026-06-29")], "is_knockout"] = True
        elif len(g) >= 48:
            df.loc[g.sort_values("date").index[-16:], "is_knockout"] = True
    df["_is_wc"] = is_wc
    return df.sort_values("date").reset_index(drop=True)


def _folds(df: pd.DataFrame):
    for cut in FOLD_CUTOFFS:
        t0 = pd.Timestamp(cut)
        t1 = t0 + pd.Timedelta(days=HORIZON_DAYS)
        train = df[df["date"] < t0]
        test = df[(df["date"] >= t0) & (df["date"] < t1)]
        test = test[test["match_weight"] >= 2.0]   # competitivos: señal más limpia
        if len(train) > 1000 and len(test) > 100:
            yield cut, train, test


def _poisson_P(model: DixonColesModel, test: pd.DataFrame, knockout: bool = False) -> np.ndarray:
    """Prob. del Poisson por fila. Si el test trae `is_knockout`, se aplica el
    factor de eliminatorias por partido (igual que en producción); el flag
    `knockout` fuerza el factor para todo el set."""
    ko_col = (test["is_knockout"].fillna(False).astype(bool).tolist()
              if "is_knockout" in test.columns else [False] * len(test))
    out = []
    for (h, a, neu), ko in zip(
            test[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None), ko_col):
        p = model.probabilities(h, a, neutral=bool(neu), knockout=knockout or ko)
        out.append([p["H"], p["D"], p["A"]])
    return np.asarray(out)


def _elo_P(model: EloModel, test: pd.DataFrame) -> np.ndarray:
    out = []
    for h, a, neu in test[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None):
        p = model.probabilities(h, a, neutral=bool(neu))
        out.append([p["H"], p["D"], p["A"]])
    return np.asarray(out)


def _yi(test: pd.DataFrame) -> np.ndarray:
    hs, as_ = test["home_score"].to_numpy(int), test["away_score"].to_numpy(int)
    return np.where(hs > as_, 0, np.where(hs < as_, 2, 1))


def _ll(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def _rps_arr(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    oh = np.eye(3)[y]
    return float(((np.cumsum(P, 1) - np.cumsum(oh, 1)) ** 2).sum(1).mean() / 2)


def grid_poisson(df: pd.DataFrame,
                 half_lives=(365.0, 550.0, 730.0, 1100.0),
                 alphas=(3e-4, 1e-3, 3e-3)) -> pd.DataFrame:
    """Grid half_life x alpha del Poisson en folds rolling (log-loss pooled)."""
    import itertools
    rows = []
    for hl, al in itertools.product(half_lives, alphas):
        Ps, ys = [], []
        for cut, train, test in _folds(df):
            m = DixonColesModel(half_life_days=hl, alpha=al).fit(train)
            Ps.append(_poisson_P(m, test)); ys.append(_yi(test))
        P, y = np.vstack(Ps), np.concatenate(ys)
        rows.append({"half_life": hl, "alpha": al, "n": len(y),
                     "logloss": _ll(P, y), "rps": _rps_arr(P, y)})
        print(f"  half_life={hl:6.0f} alpha={al:.0e} -> "
              f"logloss {rows[-1]['logloss']:.4f} rps {rows[-1]['rps']:.4f}")
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def eval_boost(df: pd.DataFrame, boosts=(1.0, 2.0, 3.0, 5.0, 8.0),
               half_life: float = 730.0, alpha: float = 1e-3,
               w_elo: float = 0.5) -> pd.DataFrame:
    """Boost del 'torneo en curso' evaluado donde importa: para cada Mundial
    pasado entrena hasta el inicio de sus knockouts (con la fase de grupos de
    ESA edición sobre-ponderada por `boost`) y evalúa el ensemble elo+poisson
    sobre sus 16 eliminatorias."""
    editions = sorted({d.year for d in df.loc[df["is_knockout"], "date"]})
    rows = []
    for boost in boosts:
        Ps, ys = [], []
        for year in editions:
            ko = df[df["is_knockout"] & (df["date"].dt.year == year)]
            cutoff = ko["date"].min()
            train = df[df["date"] < cutoff].copy()
            cur = train["_is_wc"] & (train["date"].dt.year == year)
            train.loc[cur, "match_weight"] *= boost
            po = DixonColesModel(half_life_days=half_life, alpha=alpha).fit(train)
            el = EloModel().fit(train)
            P = w_elo * _elo_P(el, ko) + (1 - w_elo) * _poisson_P(po, ko, knockout=True)
            Ps.append(P); ys.append(_yi(ko))
        P, y = np.vstack(Ps), np.concatenate(ys)
        rows.append({"boost": boost, "n": len(y), "logloss": _ll(P, y),
                     "rps": _rps_arr(P, y)})
        print(f"  boost x{boost:<4} -> logloss {rows[-1]['logloss']:.4f} "
              f"rps {rows[-1]['rps']:.4f}  (n={len(y)})")
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def tune_weights_cv(df: pd.DataFrame, half_life: float = 730.0,
                    alpha: float = 1e-3, use_ml: bool = True) -> dict:
    """Pesos elo/poisson/ml + temperatura minimizando log-loss pooled en los
    folds rolling. Más robusto que un holdout único."""
    parts_e, parts_p, parts_m, ys = [], [], [], []
    for cut, train, test in _folds(df):
        po = DixonColesModel(half_life_days=half_life, alpha=alpha).fit(train)
        el = EloModel().fit(train)
        el.apply_confederation_correction(train, beta=0.5)   # como en producción
        parts_e.append(_elo_P(el, test)); parts_p.append(_poisson_P(po, test))
        ys.append(_yi(test))
        if use_ml:
            try:
                ml = MLModel(device="cpu").fit(train)
                out = []
                for h, a, neu in test[["home_team", "away_team", "neutral"]].itertuples(index=False, name=None):
                    p = ml.probabilities(h, a, elo=el, neutral=bool(neu))
                    out.append([p["H"], p["D"], p["A"]])
                parts_m.append(np.asarray(out))
            except Exception as e:  # noqa: BLE001
                print(f"  [aviso] ML no disponible en fold {cut}: {type(e).__name__}")
                use_ml, parts_m = False, []
        print(f"  fold {cut}: {len(ys[-1])} partidos")

    Pe, Pp, y = np.vstack(parts_e), np.vstack(parts_p), np.concatenate(ys)
    Pm = np.vstack(parts_m) if (use_ml and parts_m) else None

    best = {"logloss": np.inf}
    ml_grid = (0.0, 0.05, 0.10, 0.15, 0.20) if Pm is not None else (0.0,)
    for wm in ml_grid:
        for we in np.arange(0.0, 1.0001 - wm, 0.05):
            wp = 1.0 - wm - we
            P = we * Pe + wp * Pp + (wm * Pm if Pm is not None else 0.0)
            cur = _ll(P, y)
            if cur < best["logloss"]:
                best = {"elo": round(float(we), 2), "poisson": round(float(wp), 2),
                        "ml": round(float(wm), 2), "logloss": cur}

    # Temperatura sobre la mezcla ganadora
    P = (best["elo"] * Pe + best["poisson"] * Pp
         + (best["ml"] * Pm if Pm is not None else 0.0))
    best_t, best_ll = 1.0, _ll(P, y)
    for t in np.arange(0.70, 1.41, 0.05):
        Pt = np.clip(P, 1e-9, 1.0) ** (1.0 / t)
        Pt = Pt / Pt.sum(1, keepdims=True)
        cur = _ll(Pt, y)
        if cur < best_ll:
            best_t, best_ll = float(t), cur
    best["temperature"] = round(best_t, 2)
    best["logloss_calibrado"] = round(best_ll, 4)
    best["n"] = len(y)
    return best


def dispersion(df: pd.DataFrame, half_life: float = 730.0, alpha: float = 1e-3) -> dict:
    """Índice de dispersión de Pearson out-of-sample: mean((y-lambda)^2/lambda).
    ~1.0 = varianza compatible con Poisson; >>1 = sobredispersión."""
    ratios, n = [], 0
    for cut, train, test in _folds(df):
        m = DixonColesModel(half_life_days=half_life, alpha=alpha).fit(train)
        for h, a, hs, as_, neu in test[["home_team", "away_team", "home_score",
                                        "away_score", "neutral"]].itertuples(index=False, name=None):
            lh, la = m.expected_goals(h, a, neutral=bool(neu))
            ratios.append((hs - lh) ** 2 / max(lh, 1e-6))
            ratios.append((as_ - la) ** 2 / max(la, 1e-6))
        n += len(test)
    idx = float(np.mean(ratios))
    return {"dispersion_index": round(idx, 3), "n_partidos": n,
            "veredicto": ("OK: compatible con Poisson" if idx < 1.15 else
                          "Sobredispersión moderada" if idx < 1.35 else
                          "Sobredispersión fuerte: considerar binomial negativa")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest del predictor WC26")
    ap.add_argument("--no-holdout", action="store_true", help="Saltear el holdout temporal")
    ap.add_argument("--months", type=int, default=12, help="Meses del holdout")
    ap.add_argument("--compare-conf", action="store_true",
                    help="Comparar con vs sin corrección por confederación")
    ap.add_argument("--optimize-weights", action="store_true",
                    help="Buscar los pesos óptimos del ensemble (holdout único)")
    ap.add_argument("--write", action="store_true",
                    help="Escribir los pesos óptimos al config")
    ap.add_argument("--grid", action="store_true",
                    help="Grid half_life x alpha del Poisson (folds rolling)")
    ap.add_argument("--boost", action="store_true",
                    help="Evaluar el boost del torneo en curso (knockouts WCs pasados)")
    ap.add_argument("--tune-weights", action="store_true",
                    help="Pesos del ensemble + temperatura (folds rolling)")
    ap.add_argument("--dispersion", action="store_true",
                    help="Diagnóstico de sobredispersión del Poisson")
    ap.add_argument("--half-life", type=float, default=730.0)
    ap.add_argument("--alpha", type=float, default=1e-3)
    args = ap.parse_args()

    if args.grid or args.boost or args.tune_weights or args.dispersion:
        dfc = _load_clean()
        print(f"Datos: {len(dfc):,} partidos hasta {dfc['date'].max().date()}\n")
        if args.grid:
            print("=== Grid Poisson: half_life x alpha (rolling, pooled) ===")
            res = grid_poisson(dfc)
            print("\nMejores 5:")
            print(res.head(5).to_string(index=False))
            args.half_life = float(res.iloc[0]["half_life"])
            args.alpha = float(res.iloc[0]["alpha"])
            print(f"\n>> Ganador: half_life={args.half_life:.0f} alpha={args.alpha:.0e}\n")
        if args.boost:
            print("=== Boost del torneo en curso (knockouts de WCs pasados) ===")
            res = eval_boost(dfc, half_life=args.half_life, alpha=args.alpha)
            print(f"\n>> Ganador: boost x{res.iloc[0]['boost']}\n")
        if args.tune_weights:
            print("=== Pesos del ensemble + temperatura (rolling) ===")
            best = tune_weights_cv(dfc, half_life=args.half_life, alpha=args.alpha)
            print(f"\n>> Ganador: {best}\n")
        if args.dispersion:
            print("=== Sobredispersión ===")
            print(dispersion(dfc, half_life=args.half_life, alpha=args.alpha))
        raise SystemExit(0)

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
