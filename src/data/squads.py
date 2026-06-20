"""
Planteles + técnicos del Mundial 2026 desde Wikipedia (con enlaces a las
páginas de cada persona, para luego traer sus fotos vía la API de Wikipedia).

Uso:
    python -m src.data.squads          # scrapea planteles + DT + fotos
    python -m src.data.squads --no-img # sin fotos (solo datos)
"""
from __future__ import annotations

import argparse
import re
from urllib.parse import unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..config import DATA_DIR, load_config, canonical, all_teams

SQUADS_DIR = DATA_DIR / "squads"
SQUADS_DIR.mkdir(parents=True, exist_ok=True)
WIKI_SQUADS = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
HEADERS = {"User-Agent": "WC26-Predictor/0.1 (educational use)"}


def _extract_age(dob_text: str) -> int | None:
    m = re.search(r"aged?[\s ]*(\d{2})", str(dob_text))
    return int(m.group(1)) if m else None


def _title_from_href(a) -> str:
    if not a or "/wiki/" not in (a.get("href") or ""):
        return ""
    return unquote(a["href"].split("/wiki/")[1]).replace("_", " ")


def _col_index(header: list[str]) -> dict[str, int]:
    idx = {}
    for i, h in enumerate(header):
        h = h.lower()
        for key in ("no", "pos", "player", "birth", "caps", "goals", "club"):
            if key in h and key not in idx:
                idx[key] = i
    return idx


def scrape_squads() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    teams_set = set(all_teams(cfg))
    resp = requests.get(WIKI_SQUADS, headers=HEADERS, timeout=40)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    rows: list[dict] = []
    coaches: list[dict] = []
    current: str | None = None

    for el in soup.find_all(["h2", "h3", "h4", "p", "table"]):
        if el.name in ("h2", "h3", "h4"):
            txt = el.get_text(" ", strip=True).replace("[edit]", "").strip()
            cand = canonical(txt, cfg)
            current = cand if cand in teams_set else None
        elif el.name == "p" and current:
            txt = el.get_text(" ", strip=True)
            if txt.lower().startswith("coach"):
                # El primer <a> suele ser la banderita de nacionalidad; el nombre
                # del DT es el último link con texto.
                links = [a for a in el.find_all("a")
                         if "/wiki/" in (a.get("href") or "") and a.get_text(strip=True)]
                if links:
                    a = links[-1]
                    coaches.append({"team": current, "coach": a.get_text(strip=True),
                                    "coach_title": _title_from_href(a)})
        elif el.name == "table" and "wikitable" in (el.get("class") or []):
            if current is None:
                continue
            trs = el.find_all("tr")
            if not trs:
                continue
            header = [th.get_text(" ", strip=True) for th in trs[0].find_all(["th", "td"])]
            if not any("player" in h.lower() for h in header):
                continue
            idx = _col_index(header)
            ip = idx.get("player")
            for tr in trs[1:]:
                cells = tr.find_all(["th", "td"])
                if ip is None or len(cells) <= ip:
                    continue
                pc = cells[ip]
                a = pc.find("a")
                # Quitar anotaciones tipo "( captain )" del nombre mostrado
                player = re.sub(r"\s*\([^)]*\)\s*", "", pc.get_text(" ", strip=True)).strip()
                if not player:
                    continue
                def cell(key):
                    i = idx.get(key)
                    return cells[i].get_text(" ", strip=True) if i is not None and i < len(cells) else None
                club_cell = cells[idx["club"]] if "club" in idx and idx["club"] < len(cells) else None
                rows.append({
                    "team": current,
                    "number": pd.to_numeric(cell("no"), errors="coerce"),
                    "position": cell("pos"),
                    "player": player,
                    "wiki_title": _title_from_href(a),
                    "age": _extract_age(cell("birth")),
                    "caps": pd.to_numeric(cell("caps"), errors="coerce"),
                    "goals": pd.to_numeric(cell("goals"), errors="coerce"),
                    "club": club_cell.get_text(" ", strip=True) if club_cell is not None else None,
                })

    squads = pd.DataFrame(rows)
    squads["position"] = squads["position"].astype(str).str.extract(r"([A-Z]{2})", expand=False)
    coaches_df = pd.DataFrame(coaches).drop_duplicates("team")
    print(f"Planteles: {squads['team'].nunique()} selecciones, {len(squads)} jugadores · "
          f"{len(coaches_df)} técnicos")
    return squads, coaches_df


def enrich_with_images(squads: pd.DataFrame, coaches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrega columnas player_img / coach_img con los thumbnails de Wikipedia."""
    from .wiki_images import fetch_thumbnails
    titles = [t for t in squads["wiki_title"].tolist() if t]
    titles += [t for t in coaches.get("coach_title", pd.Series(dtype=str)).tolist() if t]
    titles = list(dict.fromkeys(titles))
    print(f"Trayendo fotos de Wikipedia para {len(titles)} personas ...")

    thumbs = fetch_thumbnails(titles, size=300)
    # Segunda pasada sobre los que no trajeron foto (lotes que el rate-limit tumbó)
    missing = [t for t in titles if t not in thumbs]
    if missing:
        print(f"  reintentando {len(missing)} faltantes ...")
        thumbs.update(fetch_thumbnails(missing, size=300, pause=0.7, batch_size=20))

    squads["player_img"] = squads["wiki_title"].map(thumbs).fillna("").map(_clean_img)
    if not coaches.empty:
        coaches["coach_img"] = coaches["coach_title"].map(thumbs).fillna("").map(_clean_img)
    got = int((squads["player_img"] != "").sum())
    print(f"  fotos de jugadores: {got}/{len(squads)} ({got/len(squads)*100:.0f}%)")
    return squads, coaches


_NOT_PORTRAIT = re.compile(r"\.svg|logo|flag|coat[_ ]of|emblem|badge|escudo|\bmap\b", re.I)


def _clean_img(url: str) -> str:
    """Descarta imágenes que no son retratos (banderas, logos, escudos)."""
    return "" if (not url or _NOT_PORTRAIT.search(url)) else url


def load_squads() -> pd.DataFrame:
    path = SQUADS_DIR / "squads.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["team", "number", "position", "player", "wiki_title",
                                 "age", "caps", "goals", "club", "player_img"])


def load_coaches() -> pd.DataFrame:
    path = SQUADS_DIR / "coaches.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["team", "coach", "coach_title", "coach_img"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-img", action="store_true", help="No traer fotos")
    args = ap.parse_args()
    sq, co = scrape_squads()
    if not args.no_img:
        sq, co = enrich_with_images(sq, co)
    sq.to_parquet(SQUADS_DIR / "squads.parquet", index=False)
    co.to_parquet(SQUADS_DIR / "coaches.parquet", index=False)
    print(f"Guardado en {SQUADS_DIR}")
