"""
Benchmark del modelo contra las CUOTAS DE APUESTAS (el estándar de oro).

El mercado es el rival más duro: si el modelo no se le acerca en log-loss/RPS,
no es competitivo. Esta herramienta compara las predicciones del modelo contra
las probabilidades implícitas del mercado sobre los partidos jugados.

Fuente de cuotas (pluggable, en este orden):
  1. the-odds-api.com  -> si hay API key (gratis, 500 req/mes) en
     config `odds_api_key` o variable de entorno ODDS_API_KEY.
  2. CSV manual `data/raw/wc26_odds.csv` con columnas:
     date,home_team,away_team,odds_home,odds_draw,odds_away   (cuotas decimales)

Uso:
    python -m src.evaluation.odds_benchmark            # corre el benchmark
    python -m src.evaluation.odds_benchmark --fetch    # baja cuotas de the-odds-api
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import requests

from ..config import RAW_DIR, PROCESSED_DIR, ROOT, load_config, canonical
from ..models.ensemble import EnsemblePredictor

ODDS_CSV = RAW_DIR / "wc26_odds.csv"
KEY_FILE = ROOT / "config" / "odds_api_key.txt"
API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds/"


def get_api_key() -> str | None:
    """API key de the-odds-api: archivo local (no versionado) > config > env."""
    if KEY_FILE.exists():
        k = KEY_FILE.read_text(encoding="utf-8").strip()
        if k:
            return k
    return load_config().get("odds_api_key") or os.environ.get("ODDS_API_KEY")


def implied_probs(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    """Cuotas decimales -> probabilidades implícitas, sin el margen de la casa
    (de-vig por normalización proporcional)."""
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    return float(p[0]), float(p[1]), float(p[2])


def fetch_odds_api(api_key: str, sport: str = "soccer_fifa_world_cup",
                   regions: str = "eu") -> pd.DataFrame:
    """Descarga cuotas h2h de the-odds-api y las guarda en el CSV manual."""
    cfg = load_config()
    r = requests.get(API_URL.format(sport=sport),
                     params={"apiKey": api_key, "regions": regions,
                             "markets": "h2h", "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        home, away = ev.get("home_team"), ev.get("away_team")
        books = ev.get("bookmakers", [])
        if not books:
            continue
        # promedio de cuotas entre casas
        oh, od, oa = [], [], []
        for bk in books:
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                price = {o["name"]: o["price"] for o in mk["outcomes"]}
                if home in price and away in price and "Draw" in price:
                    oh.append(price[home]); od.append(price["Draw"]); oa.append(price[away])
        if oh:
            rows.append({"date": ev.get("commence_time", "")[:10],
                         "home_team": canonical(home, cfg), "away_team": canonical(away, cfg),
                         "odds_home": np.mean(oh), "odds_draw": np.mean(od), "odds_away": np.mean(oa)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(ODDS_CSV, index=False)
        print(f"{len(df)} partidos con cuotas -> {ODDS_CSV}")
    else:
        print("[aviso] the-odds-api no devolvió cuotas (¿torneo sin mercado activo?).")
    return df


def fetch_winner_odds() -> dict[str, float]:
    """Probabilidad implícita del mercado de que cada selección sea CAMPEONA
    (cuotas outright de the-odds-api, normalizadas para quitar el margen)."""
    key = get_api_key()
    if not key:
        return {}
    cfg = load_config()
    try:
        r = requests.get(API_URL.format(sport="soccer_fifa_world_cup_winner"),
                         params={"apiKey": key, "regions": "eu", "markets": "outrights",
                                 "oddsFormat": "decimal"}, timeout=30)
        r.raise_for_status()
        books = r.json()[0].get("bookmakers", [])
        if not books:
            return {}
        outs = books[0]["markets"][0]["outcomes"]
    except Exception:  # noqa: BLE001
        return {}
    mkt = {canonical(o["name"], cfg): 1.0 / o["price"] for o in outs if o.get("price")}
    tot = sum(mkt.values()) or 1.0
    return {k: v / tot for k, v in mkt.items()}


def champion_benchmark(sim_probs: dict[str, float]) -> pd.DataFrame:
    """Compara P(campeón) del modelo (simulación) vs el mercado de apuestas."""
    mkt = fetch_winner_odds()
    if not mkt:
        return pd.DataFrame()
    rows = [{"team": t, "mercado": pm, "modelo": float(sim_probs.get(t, 0.0)),
             "dif": float(sim_probs.get(t, 0.0)) - pm}
            for t, pm in mkt.items()]
    return pd.DataFrame(rows).sort_values("mercado", ascending=False).reset_index(drop=True)


def load_odds() -> pd.DataFrame:
    cfg = load_config()
    api_key = get_api_key()
    if api_key:
        try:
            df = fetch_odds_api(api_key)
            if not df.empty:
                return df
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] the-odds-api falló ({type(e).__name__}); uso CSV manual.")
    if ODDS_CSV.exists():
        df = pd.read_csv(ODDS_CSV, comment="#").dropna(subset=["home_team", "away_team"])
        if df.empty:
            return df
        df["home_team"] = df["home_team"].map(lambda t: canonical(t, cfg))
        df["away_team"] = df["away_team"].map(lambda t: canonical(t, cfg))
        return df
    return pd.DataFrame()


def _rps(P, yi):
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    oh = np.eye(3)[yi]
    return float(((np.cumsum(P, 1) - np.cumsum(oh, 1)) ** 2).sum(1).mean() / 2)


def _logloss(P, yi):
    P = np.clip(P, 1e-9, 1.0); P = P / P.sum(1, keepdims=True)
    return float(-np.log(P[np.arange(len(yi)), yi]).mean())


def benchmark(ens: EnsemblePredictor | None = None) -> dict | None:
    """Compara modelo vs mercado sobre los partidos jugados que tengan cuotas."""
    odds = load_odds()
    if odds.empty:
        print("[aviso] no hay cuotas. Cargá data/raw/wc26_odds.csv o una ODDS_API_KEY.")
        return None
    ens = ens or EnsemblePredictor.load()
    matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").dropna(subset=["home_score", "away_score"])
    matches["date"] = pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d")

    Pm, Pmkt, yi, rows = [], [], [], []
    for o in odds.itertuples(index=False):
        m = matches[(matches["home_team"] == o.home_team) & (matches["away_team"] == o.away_team)]
        if m.empty:
            continue
        r = m.iloc[-1]
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        y = 0 if hs > as_ else (2 if hs < as_ else 1)
        pr = ens.predict(o.home_team, o.away_team, neutral=True)
        Pm.append([pr["p_home"], pr["p_draw"], pr["p_away"]])
        Pmkt.append(list(implied_probs(o.odds_home, o.odds_draw, o.odds_away)))
        yi.append(y)
        rows.append(f"{o.home_team} {hs}-{as_} {o.away_team}")
    if not yi:
        print("[aviso] ninguna cuota coincide con un partido jugado.")
        return None

    Pm, Pmkt, yi = np.array(Pm), np.array(Pmkt), np.array(yi)
    res = {
        "n": len(yi),
        "modelo": {"logloss": _logloss(Pm, yi), "rps": _rps(Pm, yi),
                   "acc": float((Pm.argmax(1) == yi).mean())},
        "mercado": {"logloss": _logloss(Pmkt, yi), "rps": _rps(Pmkt, yi),
                    "acc": float((Pmkt.argmax(1) == yi).mean())},
    }
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Bajar cuotas de the-odds-api")
    args = ap.parse_args()
    if args.fetch:
        key = load_config().get("odds_api_key") or os.environ.get("ODDS_API_KEY")
        if not key:
            print("Falta ODDS_API_KEY (env) o odds_api_key en el config.")
        else:
            fetch_odds_api(key)
    res = benchmark()
    if res:
        print(f"\n=== Modelo vs Mercado ({res['n']} partidos con cuotas) ===")
        for k in ("logloss", "rps", "acc"):
            print(f"  {k:<8} modelo {res['modelo'][k]:.4f} | mercado {res['mercado'][k]:.4f}")
