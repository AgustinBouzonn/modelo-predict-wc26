"""Genera la figura del módulo cuántico para el README (docs/quantum.png):
frontera de decisión del VQC + comparación de resultados vs el ensemble.

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
with open(os.path.join(ROOT, "data/processed/quantum_demo.json"), encoding="utf-8") as f:
    D = json.load(f)

# --- Métricas medidas out-of-sample (holdout 12m, 601 partidos) ---
models = ["Baseline", "Cuántico (4q)", "Ensemble"]
acc = [46.4, 61.2, 63.7]
rps = [0.234, 0.175, 0.160]
colors = ["#64748b", "#a855f7", "#16a34a"]

BG = "#0e1117"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": "#e2e8f0", "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "axes.edgecolor": "#334155", "font.size": 11,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})
fig.suptitle("⚛️  Clasificador Cuántico Variacional  ·  Mundial 2026",
             fontsize=15, fontweight="bold", color="#f1f5f9", y=0.98)

# --- Panel 1: frontera de decisión ---
z = np.array(D["z"])
gx, gy = D["gx"], D["gy"]
im = ax1.imshow(z, origin="lower", aspect="auto", cmap="RdBu",
                vmin=-D["zmax"], vmax=D["zmax"],
                extent=[gx[0], gx[-1], gy[0], gy[-1]])
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

# --- Panel 2: accuracy + RPS ---
y = np.arange(len(models))
ax2.barh(y, acc, color=colors, height=0.6)
ax2.set_yticks(y)
ax2.set_yticklabels(models)
ax2.set_xlim(0, 75)
ax2.set_xlabel("Accuracy (%) · holdout out-of-sample")
ax2.set_title("Cuántico vs Ensemble (1-X-2)", color="#e2e8f0", fontsize=12, pad=8)
for i, (a, r) in enumerate(zip(acc, rps)):
    ax2.text(a + 1, i, f"{a:.1f}%   ·   RPS {r:.3f}", va="center",
             color="#e2e8f0", fontsize=10)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

fig.text(0.5, 0.01,
         "4 qubits · data re-uploading · entrenado sobre 23.298 partidos (PennyLane) · "
         "RPS menor = mejor",
         ha="center", color="#64748b", fontsize=9)
fig.tight_layout(rect=[0, 0.03, 1, 0.94])

out = os.path.join(ROOT, "docs", "quantum.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=130)
print("guardado:", out)
