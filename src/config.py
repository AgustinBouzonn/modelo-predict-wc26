"""Carga de configuración y rutas del proyecto."""
from __future__ import annotations

from pathlib import Path
import yaml

# Raíz del proyecto (dos niveles arriba de este archivo: src/config.py -> raíz)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NEWS_DIR = DATA_DIR / "news"
MODELS_DIR = ROOT / "models"
CONFIG_PATH = ROOT / "config" / "teams_wc26.yaml"

for _d in (RAW_DIR, PROCESSED_DIR, NEWS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Lee el YAML de configuración del torneo."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def all_teams(cfg: dict) -> list[str]:
    """Lista plana de todas las selecciones del torneo (canónicas)."""
    teams: list[str] = []
    for group in cfg.get("groups", {}).values():
        teams.extend(group)
    # Dedup preservando orden
    seen: set[str] = set()
    out: list[str] = []
    for t in teams:
        c = canonical(t, cfg)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def canonical(team: str, cfg: dict) -> str:
    """Devuelve el nombre canónico (el que usa el dataset histórico)."""
    aliases = cfg.get("aliases", {}) or {}
    return aliases.get(team, team)
