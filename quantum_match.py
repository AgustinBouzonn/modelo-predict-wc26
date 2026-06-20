"""
Entrena el Clasificador Cuántico Variacional multiclase (1-X-2) y guarda artefactos.

Usa 4 features reales (derivables de Elo + forma reciente por selección):
    elo_diff, form_pts_diff, form_gf_diff, form_ga_diff
Target: H (gana local) / D (empate) / A (gana visitante).

Produce:
    models/quantum.joblib              -> modelo entrenado (lo usa el dashboard)
    data/processed/quantum_demo.json   -> grilla 2D + features por selección

Corre en el simulador local de PennyLane. Uso:  python quantum_match.py
"""
from __future__ import annotations

import json

import numpy as np

from src.config import all_teams, load_config
from src.features.build_features import build_training_table, current_form
from src.models.elo import EloModel
from src.models.quantum import FEATURES, QuantumMatchClassifier
from src.pipeline import load_matches

GRID = 60


def main():
    print("Cargando partidos historicos...")
    matches = load_matches(fetch=False)
    table = build_training_table(matches).copy()

    margin = table["home_score"] - table["away_score"]
    y = np.where(margin > 0, 0, np.where(margin == 0, 1, 2))  # H/D/A
    X = table[FEATURES].to_numpy(dtype=float)
    print(f"Partidos: {len(X)} | H={np.mean(y==0):.0%} D={np.mean(y==1):.0%} A={np.mean(y==2):.0%}")

    print("Entrenando clasificador cuantico multiclase (4 qubits)...")
    clf = QuantumMatchClassifier().fit(
        X, y, log=lambda ep, acc: print(f"  epoca {ep:3d} | test acc {acc:.2%}"))
    print(f"Exactitud final en test: {clf.test_acc:.2%}")
    print(f"Modelo guardado: {clf.save()}")

    # ---------------- exportar JSON para el demo / dashboard -------------- #
    gx, gy, Z = clf.decision_grid(n=GRID)
    cfg = load_config()
    elo = EloModel.load()
    form = current_form(matches)

    team_data = []
    for t in all_teams(cfg):
        f = form.get(t, {"form_pts": 1.0, "form_gf": 1.0, "form_ga": 1.0})
        team_data.append({
            "name": t,
            "elo": round(float(elo.rating(t)), 1),
            "form_pts": round(float(f["form_pts"]), 3),
            "form_gf": round(float(f["form_gf"]), 3),
            "form_ga": round(float(f["form_ga"]), 3),
        })
    team_data.sort(key=lambda d: d["name"])

    out = {
        "features": FEATURES, "classes": ["H", "D", "A"],
        "x_min": clf.x_min.tolist(), "x_max": clf.x_max.tolist(),
        "grid": GRID, "zmax": round(clf.zmax, 4),
        "gx": [round(v, 2) for v in gx.tolist()],
        "gy": [round(v, 3) for v in gy.tolist()],
        "z": [[round(v, 4) for v in row] for row in Z.tolist()],
        "teams": team_data,
        "test_acc": round(clf.test_acc, 4), "n_matches": int(len(X)),
    }
    path_json = "data/processed/quantum_demo.json"
    with open(path_json, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False)
    print(f"Exportado: {path_json} | equipos: {len(team_data)} | grilla {GRID}x{GRID}")


if __name__ == "__main__":
    main()
