"""
Mapa selección -> confederación y utilidades para corregir el sesgo del Elo.

El Elo "World Football" infla a equipos de confederaciones débiles porque
acumulan rating contra rivales flojos y juegan pocos partidos inter-confederación.
Acá estimamos, a partir de los enfrentamientos REALES entre confederaciones,
cuánto más fuerte/débil es cada confederación, y derivamos un offset por
confederación para re-anclar los ratings (ver elo.apply_confederation_correction).

Los nombres son los canónicos del dataset (post-alias).
"""
from __future__ import annotations

CONFEDERATIONS: dict[str, list[str]] = {
    "UEFA": [
        "Spain", "France", "England", "Portugal", "Netherlands", "Belgium",
        "Croatia", "Italy", "Germany", "Switzerland", "Austria", "Norway",
        "Denmark", "Ukraine", "Poland", "Serbia", "Sweden", "Scotland",
        "Czech Republic", "Bosnia and Herzegovina", "Turkey", "Hungary",
        "Wales", "Romania", "Greece", "Russia", "Iceland", "Finland",
        "Slovakia", "Slovenia", "Republic of Ireland", "Northern Ireland",
        "Albania", "North Macedonia", "Georgia", "Montenegro", "Kosovo",
        "Bulgaria", "Israel", "Luxembourg",
    ],
    "CONMEBOL": [
        "Argentina", "Brazil", "Uruguay", "Colombia", "Ecuador", "Paraguay",
        "Chile", "Peru", "Bolivia", "Venezuela",
    ],
    "CONCACAF": [
        "United States", "Canada", "Mexico", "Panama", "Costa Rica", "Jamaica",
        "Haiti", "Honduras", "El Salvador", "Curaçao", "Trinidad and Tobago",
        "Guatemala", "Dominican Republic",
    ],
    "CAF": [
        "Morocco", "Senegal", "Egypt", "Nigeria", "Algeria", "Tunisia",
        "Côte d'Ivoire", "Ghana", "Cameroon", "South Africa", "Cabo Verde",
        "DR Congo", "Mali", "Burkina Faso", "Angola", "Madagascar", "Guinea",
        "Mauritania", "Niger", "Benin", "Togo", "Sierra Leone", "Liberia",
    ],
    "AFC": [
        "Japan", "Korea Republic", "Iran", "Australia", "Saudi Arabia", "Qatar",
        "Uzbekistan", "Jordan", "Iraq", "China", "Thailand", "Oman", "Bahrain",
        "India", "Indonesia", "Tajikistan", "Kuwait", "Palestine",
    ],
    "OFC": [
        "New Zealand", "Fiji", "Vanuatu", "Papua New Guinea", "Solomon Islands",
    ],
}

# Índice inverso equipo -> confederación
TEAM_CONF: dict[str, str] = {t: conf for conf, teams in CONFEDERATIONS.items() for t in teams}


def confederation_of(team: str) -> str | None:
    return TEAM_CONF.get(team)
