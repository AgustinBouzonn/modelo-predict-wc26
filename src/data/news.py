"""
Ingesta de noticias y análisis de sentimiento por selección.

Estrategia sin API paga: leemos el RSS de Google News para cada equipo
(consulta "<equipo> football national team") y calculamos un sentimiento
promedio con VADER. Esto produce un índice de "momentum mediático" en [-1, 1].

NOTA: este sentimiento NO es una feature del modelo ML (no entra en
build_features). Se aplica como un ajuste post-hoc, transparente y desactivable,
sobre las probabilidades del ensemble (ver ensemble._apply_news, news_tilt).
Es un prior heurístico, no validado en backtest.

Uso:
    python -m src.data.news
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import quote_plus

import re

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..config import NEWS_DIR, load_config, all_teams

RSS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_ANALYZER = SentimentIntensityAnalyzer()

# Titulares de logística/preview que no aportan ánimo real (suelen dar sentimiento ~0)
_NOISE = re.compile(
    r"predicted line|probable line|line[- ]?ups?\b|starting xi|confirmed (team|line)|"
    r"where to watch|how to watch|live ?stream|team news|kick[- ]?off|what time|"
    r"tv channel|preview|h2h|head to head|betting|\bodds\b|prediction|highlights|"
    r"full time|half time|live updates|as it happened|minute by minute",
    re.I,
)
# Palabras que SÍ indican un evento con carga (lesiones, crisis, hazañas, etc.)
_EVENT = re.compile(
    r"injur|doubt|ruled out|return|comeback|suspend|\bban\b|crisis|sack|controvers|"
    r"star|brilliant|stunning|sensational|victory|defeat|thrash|hero|slam|"
    r"criticis|criticiz|boost|blow|shock|upset|eliminat|knock(ed)? ?out|qualif|"
    r"advance|wonder|scandal|fitness|fit again|miss|axed|dropped",
    re.I,
)


def is_relevant(title: str) -> bool:
    """True si el titular aporta señal de ánimo (no es puro preview/logística)."""
    if _EVENT.search(title):
        return True
    return not _NOISE.search(title)


def fetch_team_news(team: str, max_items: int = 25) -> list[dict]:
    """Devuelve titulares recientes de un equipo con su score de sentimiento."""
    query = quote_plus(f"{team} football national team")
    url = RSS_TEMPLATE.format(query=query)
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "")
        # Google News pone "Titular - Medio"; separamos el medio si está
        source = ""
        src = entry.get("source")
        if src is not None:
            source = getattr(src, "title", "") or (src.get("title", "") if isinstance(src, dict) else "")
        if not source and " - " in title:
            title, source = title.rsplit(" - ", 1)
        score = _ANALYZER.polarity_scores(title)["compound"]
        items.append({
            "team": team,
            "title": title,
            "source": source,
            "published": entry.get("published", ""),
            "link": entry.get("link", ""),
            "sentiment": score,
            "relevant": is_relevant(title),
        })
    return items


def build_sentiment_table(max_items: int = 25, pause: float = 0.4) -> pd.DataFrame:
    """
    Recorre todas las selecciones del torneo y arma una tabla con el
    sentimiento agregado por equipo. Guarda titulares y agregados en data/news.
    """
    cfg = load_config()
    teams = all_teams(cfg)

    all_items: list[dict] = []
    for team in teams:
        try:
            items = fetch_team_news(team, max_items=max_items)
            all_items.extend(items)
            print(f"  {team:<22} {len(items):>2} titulares")
        except Exception as e:  # noqa: BLE001
            print(f"  [aviso] {team}: {e}")
        time.sleep(pause)  # cortesía con el servidor

    headlines = pd.DataFrame(all_items)
    if headlines.empty:
        print("[aviso] no se obtuvieron noticias.")
        return pd.DataFrame(columns=["team", "news_sentiment", "n_articles"])

    headlines.to_parquet(NEWS_DIR / "headlines.parquet", index=False)

    # El sentimiento agregado usa SOLO titulares relevantes (sin ruido de preview).
    # Si un equipo tiene <3 relevantes, cae al promedio sobre todos sus titulares.
    rows = []
    for team, g in headlines.groupby("team"):
        rel = g[g["relevant"]]
        base = rel if len(rel) >= 3 else g
        rows.append({
            "team": team,
            "news_sentiment": float(base["sentiment"].mean()),
            "n_articles": int(len(g)),
            "n_relevant": int(len(rel)),
        })
    agg = pd.DataFrame(rows)
    agg["updated"] = datetime.now(timezone.utc).isoformat()
    agg.to_parquet(NEWS_DIR / "sentiment.parquet", index=False)
    kept = int(headlines["relevant"].sum())
    print(f"\nSentimiento de {len(agg)} selecciones "
          f"({kept}/{len(headlines)} titulares relevantes) -> data/news/sentiment.parquet")
    return agg


def load_headlines() -> pd.DataFrame:
    """Carga los titulares individuales analizados (con sentimiento, medio y link)."""
    path = NEWS_DIR / "headlines.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["team", "title", "source", "published", "link", "sentiment"])


def load_sentiment() -> pd.DataFrame:
    """Carga la tabla de sentimiento si existe; si no, devuelve vacío."""
    path = NEWS_DIR / "sentiment.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["team", "news_sentiment", "n_articles"])


if __name__ == "__main__":
    build_sentiment_table()
