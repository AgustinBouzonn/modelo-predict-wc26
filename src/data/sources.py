"""
Ingesta de datos (scraping web).

Fuentes:
  1. Histórico de partidos internacionales -> dataset martj42/international_results
     (CSV en GitHub, se actualiza con cada fecha FIFA). Es la base para Elo/Poisson/ML.
  2. Fixture y resultados del Mundial 2026 -> Wikipedia (opcional, best-effort).

Uso:
    python -m src.data.sources                 # descarga/actualiza el histórico
    python -m src.data.sources --fetch-fixture # intenta traer grupos del WC26
"""
from __future__ import annotations

import argparse
import io
from datetime import datetime, timezone

import pandas as pd
import requests
import yaml

from ..config import RAW_DIR, PROCESSED_DIR, CONFIG_PATH, load_config, canonical

HIST_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
WIKI_WC26 = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
HEADERS = {"User-Agent": "WC26-Predictor/0.1 (educational use)"}


# --------------------------------------------------------------------------- #
# 1) Histórico de resultados internacionales
# --------------------------------------------------------------------------- #
def fetch_historical(force: bool = True) -> pd.DataFrame:
    """Descarga el CSV histórico y lo guarda crudo en data/raw."""
    raw_path = RAW_DIR / "international_results.csv"
    if not force and raw_path.exists():
        return pd.read_csv(raw_path, parse_dates=["date"])

    print(f"Descargando histórico desde {HIST_URL} ...")
    resp = requests.get(HIST_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["date"])
    df.to_csv(raw_path, index=False)
    print(f"  {len(df):,} partidos guardados en {raw_path}")
    return df


def build_processed(min_year: int = 2002) -> pd.DataFrame:
    """
    Normaliza el histórico para entrenamiento:
      - filtra por año mínimo (relevancia)
      - agrega columnas derivadas (resultado, diferencia de gol, peso por torneo)
    """
    cfg = load_config()
    df = fetch_historical(force=False)

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= min_year].copy()

    # Normalizar nombres a canónicos (consistencia con el config del torneo)
    df["home_team"] = df["home_team"].map(lambda t: canonical(t, cfg))
    df["away_team"] = df["away_team"].map(lambda t: canonical(t, cfg))

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")

    # Resultado desde la perspectiva del local: H / D / A
    def _res(r):
        if r.home_score > r.away_score:
            return "H"
        if r.home_score < r.away_score:
            return "A"
        return "D"

    df["result"] = df.apply(_res, axis=1)
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]

    # Peso por importancia del torneo (para Elo y para ponderar el entrenamiento)
    df["match_weight"] = df["tournament"].map(tournament_weight).fillna(1.0)

    df = df.sort_values("date").reset_index(drop=True)
    out = PROCESSED_DIR / "matches.parquet"
    df.to_parquet(out, index=False)
    print(f"Procesados {len(df):,} partidos (desde {min_year}) -> {out}")
    return df


def tournament_weight(name: str) -> float:
    """Peso de importancia tipo FIFA según el tipo de torneo."""
    name = str(name).lower()
    if "world cup" in name and "qualification" not in name:
        return 4.0
    if "confederations" in name:
        return 3.0
    if any(k in name for k in ("uefa euro", "copa américa", "copa america",
                               "african cup", "afc asian", "gold cup", "nations league")):
        return 3.0
    if "qualification" in name:
        return 2.5
    if "friendly" in name:
        return 1.0
    return 2.0


# --------------------------------------------------------------------------- #
# 2) Fixture / resultados del Mundial 2026 desde Wikipedia (best-effort)
# --------------------------------------------------------------------------- #
def fetch_wc26_results() -> pd.DataFrame:
    """
    Intenta scrapear partidos ya jugados del WC26 desde Wikipedia.
    Devuelve un DataFrame (posiblemente vacío). Best-effort: si la
    estructura de la página cambia, no rompe el pipeline.
    """
    try:
        resp = requests.get(WIKI_WC26, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] no se pudo scrapear Wikipedia: {e}")
        return pd.DataFrame()

    # Heurística: buscar tablas con columnas tipo "Team 1 / Score / Team 2"
    frames = []
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("score" in c for c in cols) and len(t.columns) >= 3:
            frames.append(t)
    print(f"[info] {len(frames)} tablas candidatas encontradas en Wikipedia.")
    out = RAW_DIR / "wc26_wikipedia_tables.parquet"
    if frames:
        # Guardamos crudo para inspección manual; el parseo fino se hace aparte.
        try:
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            print(f"[info] tablas crudas -> {out}")
        except Exception:  # noqa: BLE001
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def append_manual_results(path: str | None = None) -> pd.DataFrame:
    """
    Carga resultados del WC26 cargados a mano en data/raw/wc26_manual.csv
    (columnas: date,home_team,away_team,home_score,away_score) y los
    fusiona al histórico procesado para reentrenar con lo último.
    """
    cfg = load_config()
    manual_path = path or (RAW_DIR / "wc26_manual.csv")
    try:
        man = pd.read_csv(manual_path, parse_dates=["date"])
    except FileNotFoundError:
        return pd.DataFrame()

    man["home_team"] = man["home_team"].map(lambda t: canonical(t, cfg))
    man["away_team"] = man["away_team"].map(lambda t: canonical(t, cfg))
    if "tournament" not in man:
        man["tournament"] = "FIFA World Cup"
    if "neutral" not in man:
        man["neutral"] = True  # sede compartida: tratamos como neutral por defecto
    print(f"[info] {len(man)} resultados manuales del WC26 incorporados.")
    return man


def load_fixture(edition_year: int = 2026) -> pd.DataFrame:
    """
    Devuelve el fixture del Mundial (nombres canónicos), marcando qué partidos
    ya se jugaron. Columnas: date, group, home, away, home_score, away_score, played.
    El grupo se infiere del config (— si los equipos no comparten grupo).
    """
    cfg = load_config()
    # Mapa equipo -> letra de grupo
    team_group = {canonical(t, cfg): g for g, teams in cfg.get("groups", {}).items()
                  for t in teams}

    df = fetch_historical(force=False)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[df["tournament"].str.contains("FIFA World Cup", na=False)
            & ~df["tournament"].str.contains("qualification", case=False, na=False)]
    wc = wc[wc["date"].dt.year == edition_year].copy()
    if wc.empty:
        return pd.DataFrame(columns=["date", "group", "home", "away",
                                     "home_score", "away_score", "played"])

    wc["home"] = wc["home_team"].map(lambda t: canonical(t, cfg))
    wc["away"] = wc["away_team"].map(lambda t: canonical(t, cfg))
    wc["played"] = wc["home_score"].notna() & wc["away_score"].notna()
    wc["group"] = [team_group.get(h) if team_group.get(h) == team_group.get(a) else "—"
                   for h, a in zip(wc["home"], wc["away"])]

    # Incorporar resultados manuales (pisan a los del dataset por fecha+equipos)
    manual = append_manual_results()
    if not manual.empty:
        manual["date"] = pd.to_datetime(manual["date"])
        for m in manual.itertuples(index=False):
            mask = ((wc["date"] == m.date) & (wc["home"] == m.home_team)
                    & (wc["away"] == m.away_team))
            wc.loc[mask, ["home_score", "away_score"]] = [m.home_score, m.away_score]
            wc.loc[mask, "played"] = True

    cols = ["date", "group", "home", "away", "home_score", "away_score", "played"]
    return wc[cols].sort_values(["date", "group"]).reset_index(drop=True)


def head_to_head(team_a: str, team_b: str, last: int = 6) -> dict:
    """Historial de enfrentamientos directos entre dos selecciones (desde el
    histórico procesado). Balance desde la perspectiva de team_a + últimos partidos."""
    path = PROCESSED_DIR / "matches.parquet"
    if not path.exists():
        return {"played": 0}
    df = pd.read_parquet(path).dropna(subset=["home_score", "away_score"])
    m = df[((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
           ((df["home_team"] == team_b) & (df["away_team"] == team_a))].copy()
    if m.empty:
        return {"played": 0}
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date")

    wins_a = draws = wins_b = gf_a = gf_b = 0
    recent = []
    for r in m.itertuples(index=False):
        hs, as_ = int(r.home_score), int(r.away_score)
        a_home = r.home_team == team_a
        a_goals, b_goals = (hs, as_) if a_home else (as_, hs)
        gf_a += a_goals; gf_b += b_goals
        if a_goals > b_goals:
            wins_a += 1
        elif a_goals < b_goals:
            wins_b += 1
        else:
            draws += 1
        recent.append({"date": r.date.date().isoformat(),
                       "result": f"{r.home_team} {hs}-{as_} {r.away_team}",
                       "tournament": r.tournament})
    return {"played": len(m), "wins_a": wins_a, "draws": draws, "wins_b": wins_b,
            "gf_a": gf_a, "gf_b": gf_b, "recent": recent[-last:][::-1]}


def load_standings() -> dict[str, pd.DataFrame]:
    """Tabla de posiciones de cada grupo, calculada con los partidos ya jugados.
    Devuelve {grupo: DataFrame} con PJ, PG, PE, PP, GF, GC, DG, Pts (ordenado)."""
    cfg = load_config()
    fx = load_fixture()
    standings: dict[str, pd.DataFrame] = {}
    for g, teams in cfg.get("groups", {}).items():
        teams = [canonical(t, cfg) for t in teams]
        st = {t: dict(PJ=0, PG=0, PE=0, PP=0, GF=0, GC=0, Pts=0) for t in teams}
        played = fx[(fx["group"] == g) & (fx["played"])]
        for r in played.itertuples(index=False):
            h, a = r.home, r.away
            if h not in st or a not in st:
                continue
            hs, as_ = int(r.home_score), int(r.away_score)
            st[h]["PJ"] += 1; st[a]["PJ"] += 1
            st[h]["GF"] += hs; st[h]["GC"] += as_
            st[a]["GF"] += as_; st[a]["GC"] += hs
            if hs > as_:
                st[h]["PG"] += 1; st[h]["Pts"] += 3; st[a]["PP"] += 1
            elif hs < as_:
                st[a]["PG"] += 1; st[a]["Pts"] += 3; st[h]["PP"] += 1
            else:
                st[h]["PE"] += 1; st[a]["PE"] += 1; st[h]["Pts"] += 1; st[a]["Pts"] += 1
        df = pd.DataFrame([{"team": t, **s} for t, s in st.items()])
        df["DG"] = df["GF"] - df["GC"]
        df = df.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
        df.index += 1
        standings[g] = df
    return standings


def infer_groups_from_fixture(edition_year: int = 2026) -> dict[str, list[str]]:
    """
    Deriva los 12 grupos reales del fixture del Mundial leyendo el histórico.

    En fase de grupos cada selección juega exactamente contra las otras 3 de su
    grupo, y esos son sus 3 primeros partidos del torneo. Tomamos, por equipo,
    sus 3 primeros rivales en orden cronológico -> ese conjunto de 4 es el grupo.
    Las etiquetas A..L se asignan por la fecha del primer partido de cada grupo.
    """
    cfg = load_config()
    df = fetch_historical(force=False)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[df["tournament"].str.contains("FIFA World Cup", na=False)
            & ~df["tournament"].str.contains("qualification", case=False, na=False)]
    wc = wc[wc["date"].dt.year == edition_year].sort_values("date")
    if wc.empty:
        print("[aviso] no hay partidos del WC en el histórico para inferir grupos.")
        return {}

    rivals: dict[str, list[str]] = {}
    first_date: dict[str, pd.Timestamp] = {}
    for r in wc.itertuples(index=False):
        h, a = canonical(r.home_team, cfg), canonical(r.away_team, cfg)
        for x, y in ((h, a), (a, h)):
            rivals.setdefault(x, [])
            if y not in rivals[x] and len(rivals[x]) < 3:
                rivals[x].append(y)
            first_date.setdefault(x, r.date)

    seen: set[frozenset] = set()
    raw_groups: list[tuple[pd.Timestamp, list[str]]] = []
    for team, rv in rivals.items():
        members = frozenset([team, *rv])
        if len(members) == 4 and members not in seen:
            seen.add(members)
            start = min(first_date[m] for m in members)
            raw_groups.append((start, sorted(members)))

    raw_groups.sort(key=lambda x: x[0])
    letters = "ABCDEFGHIJKL"
    groups = {letters[i]: g for i, (_, g) in enumerate(raw_groups[:12])}
    return groups


def regenerate_config_groups(edition_year: int = 2026) -> dict | None:
    """Reescribe la sección `groups` de config/teams_wc26.yaml con el sorteo real."""
    groups = infer_groups_from_fixture(edition_year)
    if len(groups) != 12:
        print(f"[aviso] se infirieron {len(groups)} grupos (esperados 12); no se reescribe el config.")
        return None

    cfg = load_config()
    cfg["groups"] = groups
    _write_config(cfg)
    print(f"[info] config actualizado con los 12 grupos reales del WC{edition_year}.")
    for g, teams in groups.items():
        print(f"  Grupo {g}: {', '.join(teams)}")
    return groups


def _write_config(cfg: dict) -> None:
    """Reescribe el YAML preservando el encabezado explicativo."""
    header = (
        "# ============================================================\n"
        "# Configuración del Mundial 2026 (48 equipos, 12 grupos A-L)\n"
        "# ============================================================\n"
        "# Los grupos se regeneran desde el fixture real con:\n"
        "#     python -m src.data.sources --regen-groups\n"
        "# Las letras A-L se asignan por fecha del primer partido (aproximadas);\n"
        "# la composición de cada grupo es exacta (derivada del fixture oficial).\n"
        "# ============================================================\n\n"
    )
    body = yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=False)
    CONFIG_PATH.write_text(header + body, encoding="utf-8")


def update_all() -> pd.DataFrame:
    """Pipeline de ingesta: histórico + resultados manuales del WC26 + procesado."""
    fetch_historical(force=True)
    df = build_processed()

    manual = append_manual_results()
    if not manual.empty:
        merged = pd.concat([df, manual], ignore_index=True)
        merged["match_weight"] = merged["tournament"].map(tournament_weight).fillna(1.0)
        merged["date"] = pd.to_datetime(merged["date"])
        # Dedup por (fecha, local, visitante): el resultado manual pisa al auto
        merged = (merged.sort_values("date")
                  .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
                  .reset_index(drop=True))
        merged.to_parquet(PROCESSED_DIR / "matches.parquet", index=False)
        df = merged

    stamp = datetime.now(timezone.utc).isoformat()
    (PROCESSED_DIR / "last_update.txt").write_text(stamp, encoding="utf-8")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingesta de datos WC26")
    ap.add_argument("--fetch-fixture", action="store_true",
                    help="Intentar scrapear resultados del WC26 desde Wikipedia")
    ap.add_argument("--regen-groups", action="store_true",
                    help="Regenerar los grupos del config desde el fixture real")
    args = ap.parse_args()

    update_all()
    if args.regen_groups:
        regenerate_config_groups()
    if args.fetch_fixture:
        fetch_wc26_results()
