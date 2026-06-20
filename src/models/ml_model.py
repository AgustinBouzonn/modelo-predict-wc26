"""
Modelo ML (XGBoost) de 3 clases para el resultado: Local (H) / Empate (D) / Visitante (A).

Usa las features sin fuga de datos de features.build_features. Se entrena con
ponderación temporal + por torneo y devuelve probabilidades calibradas por
softmax del clasificador.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

from ..config import MODELS_DIR
from ..features.build_features import FEATURE_COLS, build_training_table

_CLASSES = ["H", "D", "A"]
_CLASS_TO_IDX = {c: i for i, c in enumerate(_CLASSES)}

# A partir de este nº de filas, la GPU compensa el overhead de transferencia.
# Por debajo, la CPU es más rápida (benchmark: 23k filas → CPU 1.3x más veloz).
_GPU_ROW_THRESHOLD = 150_000
_gpu_cache: bool | None = None


def gpu_available() -> bool:
    """True si XGBoost fue compilado con CUDA y hay una GPU usable (cacheado)."""
    global _gpu_cache
    if _gpu_cache is not None:
        return _gpu_cache
    try:
        if not xgb.build_info().get("USE_CUDA", False):
            _gpu_cache = False
            return False
        # Confirmar con un fit mínimo en GPU
        XGBClassifier(n_estimators=2, device="cuda", tree_method="hist").fit(
            np.zeros((8, 2)), np.array([0, 1] * 4))
        _gpu_cache = True
    except Exception:  # noqa: BLE001
        _gpu_cache = False
    return _gpu_cache


def resolve_device(device: str, n_rows: int) -> str:
    """Resuelve 'auto'/'cpu'/'cuda' al device óptimo (con fallback a CPU)."""
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if gpu_available() else "cpu"
    # auto: GPU solo si el dataset es grande y hay GPU
    if n_rows >= _GPU_ROW_THRESHOLD and gpu_available():
        return "cuda"
    return "cpu"


@dataclass
class MLModel:
    half_life_days: float = 1095.0  # 3 años
    device: str = "auto"            # "auto" | "cpu" | "cuda"
    model: XGBClassifier | None = None
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    recent_form: dict = field(default_factory=dict)

    def _sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        ref = df["date"].max()
        age = (ref - df["date"]).dt.days.clip(lower=0).to_numpy()
        decay = 0.5 ** (age / self.half_life_days)
        return decay * df.get("match_weight", pd.Series(1.0, index=df.index)).to_numpy()

    def fit(self, matches: pd.DataFrame, conf_offset: dict | None = None) -> "MLModel":
        from ..features.build_features import current_form
        self.recent_form = current_form(matches)
        table = build_training_table(matches, conf_offset=conf_offset)
        table = table.dropna(subset=["result"])
        X = table[self.feature_cols].to_numpy(dtype=float)
        y = table["result"].map(_CLASS_TO_IDX).to_numpy()
        w = self._sample_weights(table)

        dev = resolve_device(self.device, len(X))

        def _make(device):
            return XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", n_jobs=-1, reg_lambda=1.5,
                tree_method="hist", device=device,
            )

        try:
            self.model = _make(dev)
            self.model.fit(X, y, sample_weight=w)
        except Exception as e:  # noqa: BLE001 — si la GPU falla, caemos a CPU
            if dev == "cuda":
                print(f"[aviso] entrenamiento en GPU falló ({type(e).__name__}); usando CPU.")
                self.model = _make("cpu")
                self.model.fit(X, y, sample_weight=w)
            else:
                raise
        self.device_used = dev
        return self

    def predict_proba_row(self, feat: pd.DataFrame) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("El modelo ML no está entrenado.")
        p = self.model.predict_proba(feat[self.feature_cols].to_numpy(dtype=float))[0]
        return {c: float(p[i]) for c, i in _CLASS_TO_IDX.items()}

    def probabilities(self, home: str, away: str, elo, neutral: bool = True,
                      **kwargs) -> dict[str, float]:
        from ..features.build_features import features_for_fixture
        feat = features_for_fixture(home, away, elo, neutral=neutral, recent_form=self.recent_form)
        return self.predict_proba_row(feat)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame()
        imp = self.model.feature_importances_
        return (pd.DataFrame({"feature": self.feature_cols, "importance": imp})
                .sort_values("importance", ascending=False).reset_index(drop=True))

    def save(self, path=None):
        path = path or (MODELS_DIR / "ml.joblib")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> "MLModel":
        path = path or (MODELS_DIR / "ml.joblib")
        return joblib.load(path)
