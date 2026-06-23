"""
Auto-ingesta de resultados en vivo desde the-odds-api (/scores).

El dataset histórico (martj42) llega con días de retraso. Este módulo trae los
resultados de los partidos del Mundial ya completados (últimos días) usando la
MISMA API key de cuotas, y los vuelca a data/raw/wc26_manual.csv para que el
pipeline los incorpore de inmediato. Pensado para correr a diario.

Uso:
    python -m src.data.live_results          # trae resultados y actualiza el manual
"""
from __future__ import annotations

import requests

from ..config import RAW_DIR, load_config, canonical
from .sources import append_manual_results  # noqa: F401  (mantiene la API del módulo)

SCORES_URL = "https://api.the-odds-api.com/v4/sports/{sport}/scores/"
MANUAL_CSV = RAW_DIR / "wc26_manual.csv"


def _api_key() -> str | None:
    from ..evaluation.odds_benchmark import get_api_key
    return get_api_key()


def fetch_live_results(days_from: int = 3) -> list[dict]:
    """Resultados de partidos del WC ya completados (nombres canónicos)."""
    key = _api_key()
    if not key:
        print("[aviso] sin API key (config/odds_api_key.txt) — no se traen resultados live.")
        return []
    cfg = load_config()
    try:
        r = requests.get(SCORES_URL.format(sport="soccer_fifa_world_cup"),
                         params={"apiKey": key, "daysFrom": days_from}, timeout=30)
        r.raise_for_status()
        events = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] the-odds-api /scores falló: {type(e).__name__}")
        return []

    out = []
    for ev in events:
        if not ev.get("completed"):
            continue
        sc = {s["name"]: s.get("score") for s in (ev.get("scores") or [])}
        h, a = ev.get("home_team"), ev.get("away_team")
        if h not in sc or a not in sc or sc[h] is None or sc[a] is None:
            continue
        try:
            out.append({"date": ev["commence_time"][:10],
                        "home_team": canonical(h, cfg), "away_team": canonical(a, cfg),
                        "home_score": int(sc[h]), "away_score": int(sc[a])})
        except (ValueError, TypeError):
            continue
    return out


def update_manual(days_from: int = 3) -> int:
    """Agrega los resultados live al CSV manual (sin duplicar). Devuelve cuántos
    son nuevos."""
    import pandas as pd
    live = fetch_live_results(days_from)
    if not live:
        return 0
    new = pd.DataFrame(live)

    if MANUAL_CSV.exists():
        cur = pd.read_csv(MANUAL_CSV)
    else:
        cur = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score"])

    key = {(r.date, r.home_team, r.away_team) for r in cur.itertuples(index=False)}
    fresh = new[~new.apply(lambda x: (x["date"], x["home_team"], x["away_team"]) in key, axis=1)]
    if fresh.empty:
        print(f"[info] {len(live)} resultados live, ninguno nuevo (ya estaban).")
        return 0

    combined = pd.concat([cur, fresh], ignore_index=True)
    combined.to_csv(MANUAL_CSV, index=False)
    print(f"[info] {len(fresh)} resultados nuevos agregados a {MANUAL_CSV}:")
    for r in fresh.itertuples(index=False):
        print(f"    {r.date} {r.home_team} {r.home_score}-{r.away_score} {r.away_team}")
    return len(fresh)


if __name__ == "__main__":
    update_manual()
