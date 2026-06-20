"""Validación de la configuración del torneo y la identidad visual."""
import math

from src.config import load_config, all_teams
from app.teams_visual import ISO, team_color


def test_config_has_12_groups_of_4():
    cfg = load_config()
    groups = cfg["groups"]
    assert len(groups) == 12, f"esperados 12 grupos, hay {len(groups)}"
    for g, teams in groups.items():
        assert len(teams) == 4, f"grupo {g} no tiene 4 equipos"


def test_48_distinct_teams():
    teams = all_teams(load_config())
    assert len(teams) == 48
    assert len(set(teams)) == 48, "hay equipos repetidos"


def test_every_team_has_flag():
    """Todo equipo del torneo debe tener código de bandera (si no, UI rota)."""
    missing = [t for t in all_teams(load_config()) if t not in ISO]
    assert not missing, f"equipos sin bandera: {missing}"


def test_team_color_always_returns_hex():
    for t in all_teams(load_config()):
        c = team_color(t)
        assert c.startswith("#") and len(c) == 7


def test_ensemble_weights_sum_to_one():
    w = load_config().get("ensemble_weights", {})
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6), f"pesos no suman 1: {w}"
