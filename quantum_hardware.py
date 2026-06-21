"""
Corre el clasificador cuántico en un backend de QISKIT — simulador local o una
QPU REAL de IBM Quantum — y compara contra el simulador exacto de PennyLane.
El objetivo es ver el efecto del ruido del hardware NISQ sobre el modelo.

SEGURIDAD: el token de IBM nunca se pasa por código ni se versiona. Se lee de:
    1) la variable de entorno  IBM_QUANTUM_TOKEN
    2) el archivo              config/ibm_token.txt   (gitignored)

Cómo obtener el token (gratis):
    1. Crear cuenta en https://quantum.ibm.com  (plan Open, ~10 min de QPU/mes).
    2. Copiar el "API token" del panel.
    3. Guardarlo:  setx IBM_QUANTUM_TOKEN "tu_token"   (Windows, reabrí la terminal)
       o crear el archivo config/ibm_token.txt con el token adentro.

Uso:
    python quantum_hardware.py                # valida en simulador qiskit (sin token)
    python quantum_hardware.py --hardware     # corre en una QPU real de IBM
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from src.config import CONFIG_PATH
from src.models.quantum import N_QUBITS, QuantumMatchClassifier

MATCHUPS = [("Argentina", "Brazil"), ("Spain", "Cabo Verde"), ("Germany", "Japan")]


def load_token():
    tok = os.environ.get("IBM_QUANTUM_TOKEN")
    if tok:
        return tok.strip()
    path = os.path.join(os.path.dirname(CONFIG_PATH), "ibm_token.txt")
    if os.path.exists(path):
        return open(path, encoding="utf-8").read().strip()
    return None


def feats_for(qd, home, away):
    f = {t["name"]: t for t in qd["teams"]}
    h, a = f[home], f[away]
    return np.array([h["elo"] - a["elo"], h["form_pts"] - a["form_pts"],
                     h["form_gf"] - a["form_gf"], h["form_ga"] - a["form_ga"]])


def make_circuit(dev, weights):
    @qml.qnode(dev)
    def circuit(x):
        for layer in range(weights.shape[0]):
            qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
            qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]
    return circuit


def probs_from_logits(clf, logits):
    z = clf.scale * np.asarray(logits) + clf.bias
    e = np.exp(z - z.max())
    return e / e.sum()


def run(clf, qd, dev, label):
    circ = make_circuit(dev, pnp.array(clf.weights))
    print(f"\n--- {label} ---")
    out = {}
    for home, away in MATCHUPS:
        x = pnp.array(clf._angles(feats_for(qd, home, away).reshape(1, -1))[0])
        p = probs_from_logits(clf, np.array(circ(x)))
        out[(home, away)] = p
        print(f"  {home} vs {away}:  H {p[0]:.0%}  D {p[1]:.0%}  A {p[2]:.0%}")
    return out


def main(hardware=False):
    clf = QuantumMatchClassifier.load()
    with open("data/processed/quantum_demo.json", encoding="utf-8") as f:
        qd = json.load(f)

    # Referencia exacta (PennyLane)
    ref = run(clf, qd, qml.device("default.qubit", wires=N_QUBITS),
              "Simulador exacto (PennyLane default.qubit)")

    if not hardware:
        # Validación: mismo circuito vía qiskit (simulador local con shots)
        try:
            qdev = qml.device("qiskit.basicsim", wires=N_QUBITS, shots=4096)
            run(clf, qd, qdev, "Simulador qiskit local (4096 shots)")
            print("\n✓ Integración con qiskit OK. Para correr en hardware real:")
            print("  conseguí el token (ver cabecera) y ejecutá:  python quantum_hardware.py --hardware")
        except Exception as e:  # noqa: BLE001
            print(f"\n[aviso] no se pudo usar el simulador qiskit local: {e}")
        return

    # Hardware real de IBM Quantum
    token = load_token()
    if not token:
        print("\n✗ Falta el token de IBM. Ponelo en IBM_QUANTUM_TOKEN o en "
              "config/ibm_token.txt (ver cabecera del archivo).")
        return
    from qiskit_ibm_runtime import QiskitRuntimeService
    channel = os.environ.get("IBM_QUANTUM_CHANNEL", "ibm_quantum_platform")
    print(f"\nConectando a IBM Quantum (channel={channel}) ...")
    service = QiskitRuntimeService(channel=channel, token=token)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=N_QUBITS)
    print(f"Backend menos ocupado: {backend.name} ({backend.num_qubits} qubits). "
          "La cola puede tardar — paciencia.")
    hw = run(clf, qd, qml.device("qiskit.remote", wires=N_QUBITS, backend=backend, shots=4096),
             f"QPU REAL · {backend.name}")

    print("\n=== Ruido del hardware (|Δ| de P(gana local) vs simulador) ===")
    for k in MATCHUPS:
        d = abs(hw[k][0] - ref[k][0])
        print(f"  {k[0]} vs {k[1]}:  Δ {d:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", action="store_true", help="Correr en QPU real de IBM")
    main(ap.parse_args().hardware)
