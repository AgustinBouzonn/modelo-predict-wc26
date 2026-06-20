"""Identidad visual de las selecciones: banderas (flagcdn) y colores de camiseta."""
from __future__ import annotations

# Nombre canónico (dataset) -> código de bandera para flagcdn.com
ISO = {
    "Mexico": "mx", "South Africa": "za", "Korea Republic": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "Qatar": "qa", "Switzerland": "ch",
    "Australia": "au", "Paraguay": "py", "Turkey": "tr", "United States": "us",
    "Brazil": "br", "Haiti": "ht", "Morocco": "ma", "Scotland": "gb-sct",
    "Japan": "jp", "Netherlands": "nl", "Sweden": "se", "Tunisia": "tn",
    "Curaçao": "cw", "Côte d'Ivoire": "ci", "Ecuador": "ec", "Germany": "de",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Cabo Verde": "cv", "Saudi Arabia": "sa", "Spain": "es", "Uruguay": "uy",
    "Algeria": "dz", "Argentina": "ar", "Austria": "at", "Jordan": "jo",
    "France": "fr", "Iraq": "iq", "Norway": "no", "Senegal": "sn",
    "Colombia": "co", "DR Congo": "cd", "Portugal": "pt", "Uzbekistan": "uz",
    "Croatia": "hr", "England": "gb-eng", "Ghana": "gh", "Panama": "pa",
}

# Color primario de camiseta (para la cancha). Fallback gris azulado.
COLORS = {
    "Argentina": "#6CACE4", "Brazil": "#FBC02D", "Spain": "#C8102E", "France": "#1E3A8A",
    "England": "#E2E8F0", "Germany": "#0F172A", "Netherlands": "#EA580C", "Portugal": "#A01C1C",
    "Mexico": "#0E7A3B", "United States": "#1D3A8A", "Canada": "#D52B1E", "Croatia": "#C8102E",
    "Belgium": "#7A1010", "Uruguay": "#5AC8FA", "Colombia": "#FDD835", "Japan": "#1E40AF",
    "Morocco": "#B91C1C", "Senegal": "#16A34A", "Switzerland": "#D52B1E", "Norway": "#B91C1C",
    "Australia": "#FBC02D", "Korea Republic": "#C8102E", "Iran": "#15803D", "Egypt": "#B91C1C",
    "Saudi Arabia": "#16A34A", "Qatar": "#7A1230", "Ghana": "#111827", "Nigeria": "#16A34A",
    "Scotland": "#1D3A8A", "Sweden": "#FBC02D", "Austria": "#DC2626", "Paraguay": "#C8102E",
    "Ecuador": "#FDD835", "Turkey": "#E11D2E", "Tunisia": "#B91C1C", "Algeria": "#16A34A",
    "Côte d'Ivoire": "#EA7317", "Cabo Verde": "#1D4ED8", "Panama": "#B91C1C", "Haiti": "#1D4ED8",
    "Curaçao": "#1D4ED8", "Bosnia and Herzegovina": "#1D3A8A", "Jordan": "#0F172A",
    "Iraq": "#16A34A", "Uzbekistan": "#1D4ED8", "New Zealand": "#0F172A",
    "Czech Republic": "#C8102E", "DR Congo": "#1D74D8", "South Africa": "#16A34A",
}


# flagcdn solo sirve este set de anchos; mapeamos al más cercano para no dar 404.
_FLAGCDN_W = (20, 40, 80, 160, 320, 640)


def flag_url(team: str, w: int = 40) -> str:
    code = ISO.get(team)
    if not code:
        return ""
    valid = min(_FLAGCDN_W, key=lambda x: abs(x - w))
    return f"https://flagcdn.com/w{valid}/{code}.png"


def flag_img(team: str, w: int = 24, css: str = "") -> str:
    """<img> de la bandera (cae a vacío si no hay código)."""
    url = flag_url(team, w * 2)
    if not url:
        return ""
    return (f'<img src="{url}" width="{w}" '
            f'style="border-radius:3px;vertical-align:middle;box-shadow:0 1px 3px rgba(0,0,0,.3);{css}" '
            f'alt="{team}">')


def team_color(team: str) -> str:
    return COLORS.get(team, "#64748B")
