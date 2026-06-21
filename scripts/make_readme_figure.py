"""Genera las figuras del README (carpeta docs/), en tema oscuro como el dashboard:
  - docs/quantum.png      frontera de decisión del VQC + comparación vs ensemble
  - docs/ranking.png      ranking Elo de las selecciones del Mundial
  - docs/performance.png  rendimiento del ensemble vs baselines (out-of-sample)

Lee data/processed/quantum_demo.json (no necesita PennyLane). Uso:
    python scripts/make_readme_figure.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

with open(os.path.join(ROOT, "data/processed/quantum_demo.json"), encoding="utf-8") as f:
    D = json.load(f)

BG = "#0e1117"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155", "font.size": 11,
})


def fig_quantum():
    models = ["Baseline", "Cuántico (4q)", "Ensemble"]
    acc = [46.4, 61.2, 64.1]
    rps = [0.234, 0.175, 0.160]
    colors = ["#64748b", "#a855f7", "#16a34a"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("⚛️  Clasificador Cuántico Variacional  ·  Mundial 2026",
                 fontsize=15, fontweight="bold", color="#f1f5f9", y=0.98)

    z = np.array(D["z"])
    gx, gy = D["gx"], D["gy"]
    im = ax1.imshow(z, origin="lower", aspect="auto", cmap="RdBu",
                    vmin=-D["zmax"], vmax=D["zmax"], extent=[gx[0], gx[-1], gy[0], gy[-1]])
    ax1.contour(np.linspace(gx[0], gx[-1], z.shape[1]),
                np.linspace(gy[0], gy[-1], z.shape[0]), z, levels=[0],
                colors="white", linewidths=1.5)
    ax1.set_title("Frontera de decisión aprendida", color="#e2e8f0", fontsize=12, pad=8)
    ax1.set_xlabel("ventaja de Elo (local − visitante)")
    ax1.set_ylabel("ventaja de forma reciente")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.set_label("P(gana local) − P(gana visitante)", color="#cbd5e1", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="#94a3b8")
    ax1.text(0.04, 0.93, "gana local", transform=ax1.transAxes, color="#bfdbfe", fontsize=10)
    ax1.text(0.62, 0.07, "gana visitante", transform=ax1.transAxes, color="#fecaca", fontsize=10)

    y = np.arange(len(models))
    ax2.barh(y, acc, color=colors, height=0.6)
    ax2.set_yticks(y); ax2.set_yticklabels(models)
    ax2.set_xlim(0, 75)
    ax2.set_xlabel("Accuracy (%) · holdout out-of-sample")
    ax2.set_title("Cuántico vs Ensemble (1-X-2)", color="#e2e8f0", fontsize=12, pad=8)
    for i, (a, r) in enumerate(zip(acc, rps)):
        ax2.text(a + 1, i, f"{a:.1f}%   ·   RPS {r:.3f}", va="center",
                 color="#e2e8f0", fontsize=10)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    fig.text(0.5, 0.01, "4 qubits · data re-uploading · entrenado sobre 23.298 partidos "
             "(PennyLane) · RPS menor = mejor", ha="center", color="#64748b", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig.savefig(os.path.join(DOCS, "quantum.png"), dpi=130)
    plt.close(fig)


def fig_ranking():
    teams = sorted(D["teams"], key=lambda t: t["elo"], reverse=True)[:16]
    names = [t["name"] for t in teams][::-1]
    elos = [t["elo"] for t in teams][::-1]
    cmap = plt.cm.viridis(np.linspace(0.25, 0.9, len(elos)))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(elos)), elos, color=cmap, height=0.7)
    ax.set_yticks(range(len(elos))); ax.set_yticklabels(names)
    ax.set_xlim(min(elos) - 60, max(elos) + 70)
    ax.set_xlabel("Rating Elo")
    ax.set_title("Ranking Elo · Top 16 selecciones del Mundial 2026",
                 color="#f1f5f9", fontsize=14, fontweight="bold", pad=10)
    for i, e in enumerate(elos):
        ax.text(e + 6, i, f"{e:.0f}", va="center", color="#e2e8f0", fontsize=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "ranking.png"), dpi=130)
    plt.close(fig)


def fig_performance():
    # Métricas del backtest out-of-sample (holdout 12m, 601 partidos competitivos)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.suptitle("Rendimiento del ensemble · holdout out-of-sample",
                 fontsize=14, fontweight="bold", color="#f1f5f9", y=0.99)

    labels = ["Azar", "Siempre\nlocal", "Ensemble"]
    acc = [33.3, 46.4, 64.1]
    ll = [1.099, 1.061, 0.834]
    cols_a = ["#64748b", "#64748b", "#16a34a"]

    ax1.bar(labels, acc, color=cols_a, width=0.6)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_ylim(0, 75)
    ax1.set_title("Accuracy (mayor = mejor)", color="#e2e8f0", fontsize=11)
    for i, a in enumerate(acc):
        ax1.text(i, a + 1.5, f"{a:.1f}%", ha="center", color="#e2e8f0", fontsize=10)

    ax2.bar(labels, ll, color=["#64748b", "#64748b", "#16a34a"], width=0.6)
    ax2.set_ylabel("Log-loss"); ax2.set_ylim(0, 1.25)
    ax2.set_title("Log-loss (menor = mejor)", color="#e2e8f0", fontsize=11)
    for i, v in enumerate(ll):
        ax2.text(i, v + 0.03, f"{v:.3f}", ha="center", color="#e2e8f0", fontsize=10)

    for ax in (ax1, ax2):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    fig.text(0.5, 0.005, "601 partidos competitivos · sin fuga de datos (entrenado as-of)",
             ha="center", color="#64748b", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(DOCS, "performance.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_quantum()
    fig_ranking()
    fig_performance()
    print("figuras generadas en", DOCS)
