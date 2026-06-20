"""
Clasificador Cuántico Variacional (VQC) multiclase para resultados 1-X-2.

Versión mejorada: 4 qubits, 4 features y salida de 3 clases (gana local /
empate / gana visitante), entrenada con entropía cruzada. Esto lo hace
directamente comparable al ensemble clásico (accuracy, log-loss, RPS).

Arquitectura:
  - AngleEmbedding (RY): 4 features -> 4 qubits.
  - StronglyEntanglingLayers: ansatz variacional entrenable.
  - Se miden ⟨PauliZ⟩ en 3 qubits -> 3 logits -> softmax (con escala y sesgo
    entrenables) -> probabilidades H/D/A.

Corre en el simulador local 'default.qubit' de PennyLane.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from ..config import MODELS_DIR

N_LAYERS = 6
# 4 features (mejor configuración medida; 6 qubits empeoró — ver README Paso 7)
FEATURES = ["elo_diff", "form_pts_diff", "form_gf_diff", "form_ga_diff"]
N_QUBITS = len(FEATURES)
CLASSES = ["H", "D", "A"]


def _make_circuit(n_qubits=N_QUBITS):
    """Circuito con DATA RE-UPLOADING: los datos se re-codifican antes de cada
    capa variacional, lo que aumenta la expresividad del modelo (más capacidad
    de aprender fronteras no lineales que codificando una sola vez)."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights, x):
        for layer in range(weights.shape[0]):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights[layer:layer + 1], wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(3)]  # 3 logits

    return circuit


def _softmax(z):
    z = z - pnp.max(z, axis=1, keepdims=True)
    e = pnp.exp(z)
    return e / pnp.sum(e, axis=1, keepdims=True)


class QuantumMatchClassifier:
    """VQC multiclase 1-X-2 (H=gana local, D=empate, A=gana visitante)."""

    def __init__(self):
        self.circuit = _make_circuit()
        self.weights = None
        self.bias = np.zeros(3)
        self.scale = 2.0
        self.x_min = None
        self.x_max = None
        self.zmax = 1.0
        self.test_acc = None

    # ----------------------------- interno -------------------------------- #
    def _angles(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min) * np.pi - np.pi / 2

    def _logits(self, weights, Xa):
        return pnp.stack([pnp.stack(self.circuit(weights, x)) for x in Xa])

    def _proba(self, weights, bias, scale, Xa):
        return _softmax(scale * self._logits(weights, Xa) + bias)

    # ----------------------------- API ------------------------------------ #
    def fit(self, X, y, epochs=90, batch=24, lr=0.06, seed=42, log=None):
        """X: (n,4) reales; y: (n,) en {0=H, 1=D, 2=A}.

        Selecciona los MEJORES pesos por log-loss de validación (early-stopping
        implícito), no los del último epoch: el entrenamiento variacional es
        ruidoso y el último paso no suele ser el mejor.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.x_min, self.x_max = X.min(axis=0), X.max(axis=0)
        Xa = self._angles(X)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xa))
        cut = int(0.8 * len(Xa))
        tr, te = idx[:cut], idx[cut:]
        Xtr = pnp.array(Xa[tr], requires_grad=False)
        ytr = y[tr]
        # Validación sobre una submuestra (rápida) para early-stopping
        te = te[:1500]
        Xte = pnp.array(Xa[te], requires_grad=False)
        yte = y[te]
        onehot_tr = pnp.array(np.eye(3)[ytr], requires_grad=False)
        onehot_te = np.eye(3)[yte]
        VAL_EVERY = 3

        wshape = qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS)
        w = pnp.array(rng.normal(0, 0.1, wshape), requires_grad=True)
        b = pnp.array(np.zeros(3), requires_grad=True)
        s = pnp.array(2.0, requires_grad=True)
        opt = qml.AdamOptimizer(lr)

        def cost(w, b, s, Xb, yb_oh):
            p = self._proba(w, b, s, Xb)
            return -pnp.mean(pnp.sum(yb_oh * pnp.log(p + 1e-9), axis=1))

        best = {"loss": np.inf, "w": None, "b": None, "s": None, "acc": 0.0}
        for ep in range(epochs):
            bi = rng.permutation(len(Xtr))[:batch]
            w, b, s, _, _ = opt.step(cost, w, b, s, Xtr[bi], onehot_tr[bi])
            if ep % VAL_EVERY and ep != epochs - 1:
                continue
            Pte = np.array(self._proba(w, b, s, Xte))
            val_loss = float(-np.mean(np.sum(onehot_te * np.log(Pte + 1e-9), axis=1)))
            if val_loss < best["loss"]:
                acc = float((Pte.argmax(1) == yte).mean())
                best = {"loss": val_loss, "w": np.array(w), "b": np.array(b),
                        "s": float(s), "acc": acc}
            if log and (ep % 15 == 0 or ep == epochs - 1):
                log(ep, float((Pte.argmax(1) == yte).mean()))

        self.weights, self.bias, self.scale = best["w"], best["b"], best["s"]
        self.test_acc = best["acc"]

        gx, gy, Z = self.decision_grid(n=40)
        self.zmax = float(np.abs(Z).max()) or 1.0
        return self

    def predict_proba(self, feats):
        """feats: array (4,) de [elo_diff, form_pts_diff, form_gf_diff, form_ga_diff].
        Devuelve dict {H, D, A}."""
        x = self._angles(np.asarray(feats, dtype=float).reshape(1, -1))
        p = np.array(self._proba(pnp.array(self.weights), pnp.array(self.bias),
                                 pnp.array(self.scale), pnp.array(x)))[0]
        return {"H": float(p[0]), "D": float(p[1]), "A": float(p[2])}

    def decision_grid(self, n=60):
        """Grilla 2D (elo_diff × form_pts_diff) con el resto de features fijas
        (0, salvo 'neutral'=1 → cancha neutral, como en el Mundial).
        Z = P(gana local) − P(gana visitante), en [-1, 1] (azul/rojo)."""
        gx = np.linspace(self.x_min[0], self.x_max[0], n)
        gy = np.linspace(self.x_min[1], self.x_max[1], n)
        XX, YY = np.meshgrid(gx, gy)
        feats = np.zeros((XX.size, len(self.x_min)))
        feats[:, 0] = XX.ravel()
        feats[:, 1] = YY.ravel()
        if "neutral" in FEATURES:
            feats[:, FEATURES.index("neutral")] = 1.0
        Xa = pnp.array(self._angles(feats))
        P = np.array(self._proba(pnp.array(self.weights), pnp.array(self.bias),
                                 pnp.array(self.scale), Xa))
        Z = (P[:, 0] - P[:, 2]).reshape(n, n)
        return gx, gy, Z

    # ------------------------- serialización ------------------------------ #
    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("circuit", None)
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.circuit = _make_circuit()

    def save(self, path=None):
        import joblib
        path = path or (MODELS_DIR / "quantum.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None):
        import joblib
        path = path or (MODELS_DIR / "quantum.joblib")
        return joblib.load(path)
