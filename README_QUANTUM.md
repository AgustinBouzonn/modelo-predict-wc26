# ⚛️ Predictor Cuántico de Partidos (QML)

Módulo de **Quantum Machine Learning** integrado al Predictor Mundial 2026. Un
**clasificador cuántico variacional** (VQC) **multiclase** predice el resultado
1-X-2 de un partido a partir de features reales del proyecto, y se compara lado a
lado contra el ensemble clásico (Elo · Poisson · XGBoost).

Todo corre en el **simulador local** `default.qubit` de [PennyLane](https://pennylane.ai/):
no hace falta hardware cuántico real.

## Qué hace

- Predice **1-X-2** (gana local / empate / gana visitante).
- En el holdout out-of-sample (601 partidos competitivos): **61.2% accuracy**,
  **RPS 0.175** — a ~2.5 pts del ensemble completo (63.7%, RPS 0.160) y muy por
  encima del baseline (46.4%, RPS 0.234).
- Se integra al dashboard de Streamlit como pestaña **⚛️ Cuántico**, donde se
  compara la distribución 1-X-2 contra el ensemble y se muestran las mayores
  discrepancias.

## Cómo funciona el modelo (versión actual)

```
  4 features          Data re-uploading × 6 capas              Medición
 (Elo, forma     ─►   [ AngleEmbedding(x) → StronglyEntangling ] ─►  ⟨Z⟩ en 3 qubits
  pts/GF/GA)            (los datos se re-codifican cada capa)        → softmax
                                                                          │
                                                                          ▼
                                                              P(H) · P(D) · P(A)
```

1. **Features** (sin fuga de datos, vía `src/features/build_features.py`),
   derivables de Elo + forma reciente por selección:
   `elo_diff`, `form_pts_diff`, `form_gf_diff`, `form_ga_diff`.
2. **Codificación con re-uploading**: cada feature se escala a `[-π/2, π/2]` y se
   re-inyecta como rotación `RY` **antes de cada una de las 6 capas variacionales**
   (data re-uploading, Pérez-Salinas et al. 2020). Esto aumenta la expresividad
   del circuito frente a codificar una sola vez.
3. **Ansatz** (`StronglyEntanglingLayers`): rotaciones parametrizadas +
   entrelazamiento. Sus pesos `θ` son lo que se entrena.
4. **Medición y salida**: ⟨PauliZ⟩ en 3 qubits → 3 logits → **softmax** (con
   escala y sesgo entrenables) → probabilidades H/D/A.
5. **Entrenamiento**: Adam minimizando **entropía cruzada**; gradientes
   analíticos por *parameter-shift*. Se quedan los **mejores pesos por log-loss
   de validación** (early-stopping), porque el entrenamiento variacional es
   ruidoso y el último epoch no suele ser el mejor.

## 🧭 Evolución del modelo (paso a paso)

El módulo se construyó iterando y **midiendo cada cambio** out-of-sample. Este es
el registro del progreso:

### Paso 0 — Prototipo de QML (clasificador de Iris)
Antes de tocar fútbol, se validó el flujo de un VQC con un clasificador
variacional sobre Iris (2 features, frontera de decisión visual). Sirvió de base
conceptual: codificación → ansatz → medición → entrenamiento.

### Paso 1 — VQC binario sobre fútbol (2 qubits)
Primer modelo real: 2 qubits, 2 features (`elo_diff`, `form_pts_diff`), target
**binario** (gana local vs visitante, sin empates). Entrenado sobre el histórico
del proyecto.
- Resultado: ~73-76% de accuracy sobre partidos sin empate. Sólido como prueba
  de concepto.
- Entregables: demo interactivo (`quantum_predictor.html`) con banderas y
  "medición del qubit" animada.

### Paso 2 — Integración al dashboard + comparación con el ensemble
Se sumó la pestaña **⚛️ Cuántico** al dashboard de Streamlit: predicción del
partido, mapa de la frontera de decisión y **tabla de discrepancias** contra el
ensemble clásico. Se encapsuló el modelo en la clase reutilizable
`QuantumMatchClassifier` (fit / save / load / predict).

### Paso 3 — Datos frescos + reentrenamiento completo
Se scrapearon datos nuevos (**49.477** partidos crudos → **23.298** procesados,
hasta 2026-06-18) y se reentrenó **todo**: Elo, Poisson, XGBoost, noticias y el
cuántico. Backtest del ensemble: **63.7%** accuracy en el holdout de 12 meses
(vs 46.4% del baseline).

### Paso 4 — Salto a multiclase 1-X-2 (4 qubits)
Para que fuera **directamente comparable** al ensemble, se reescribió el modelo:
- **4 qubits**, **4 features** (`elo_diff`, `form_pts_diff`, `form_gf_diff`,
  `form_ga_diff`).
- Salida **3 clases** (H/D/A): ⟨Z⟩ en 3 qubits → softmax, entrenado con
  **entropía cruzada**.
- Problema inicial: el entrenamiento era inestable y, al quedarse con los pesos
  del último epoch, terminaba en **~49%** (apenas el baseline de "siempre local").

### Paso 5 — Estabilizar el entrenamiento (early-stopping)
Se cambió la selección de pesos: quedarse con los **mejores por log-loss de
validación** (sobre una submuestra, evaluada cada pocos epochs) en vez del último
epoch, más tuning de learning rate y épocas.
- Resultado: **60.7%** accuracy · log-loss 0.880 · **RPS 0.173** en el holdout.
  Salto enorme respecto al 49% inestable.

### Paso 6 — Data re-uploading (versión actual)
Se reemplazó la codificación única por **data re-uploading**: re-inyectar los
datos antes de cada una de las 6 capas variacionales, aumentando la expresividad.
- Resultado: **61.2%** accuracy · log-loss 0.892 · **RPS 0.175**. Esta es la
  **versión final** del modelo.
- Lectura honesta: mejora **marginal** (+0.5 pt accuracy, RPS casi igual). Señal
  de que con estas 4 features tabulares el modelo ya está cerca de su techo.

### Paso 7 — Experimento: 6 qubits (descartado)
Hipótesis: sumar features → más qubits → más capacidad. Se probó con **6 qubits**
agregando `rest_diff` (descanso) y `neutral` (localía / cancha neutral).
- Resultado: **59.1%** accuracy · log-loss 0.916 · RPS 0.184 — **peor en todo**
  que la versión de 4 qubits.
- Por qué: las features extra son débiles frente al Elo, y el circuito más grande
  (más parámetros) es **más difícil de entrenar** con el mismo presupuesto de
  épocas — optimización más ruidosa, típico de los *barren plateaus* en VQCs.
- Decisión: **revertido**. El modelo final se queda en 4 qubits. Lección: en VQCs,
  más qubits no es gratis; hay que acompañarlo con mejor optimización (más épocas,
  otro ansatz, o reducir parámetros), no solo agregar features débiles.

## 📊 Resultados (out-of-sample, holdout 12m, 601 partidos competitivos)

| Modelo | Accuracy | Log-loss | RPS |
|---|---|---|---|
| 🧮 Ensemble (Elo · Poisson · XGBoost) | 63.7% | 0.837 | 0.160 |
| ⚛️ Cuántico (4 qubits, re-uploading) | 61.2% | 0.892 | 0.175 |
| 📏 Baseline (frecuencias 1-X-2) | 46.4% | 1.061 | 0.234 |

> RPS (Ranked Probability Score) es la métrica correcta para 1X2 ordinal
> (menor = mejor). El cuántico, con 4 features, queda muy cerca del ensemble.

## Archivos

| Archivo | Rol |
|---------|-----|
| `src/models/quantum.py` | Clase `QuantumMatchClassifier` (multiclase + re-uploading) |
| `quantum_match.py` | Entrena con los datos del proyecto y guarda los artefactos |
| `quantum_eval.py` | Comparación 1-X-2 cuántico vs ensemble (accuracy / log-loss / RPS) |
| `app/dashboard.py` | Pestaña ⚛️ Cuántico (comparación con el ensemble) |
| `models/quantum.joblib` | Modelo entrenado (lo carga el dashboard) |
| `data/processed/quantum_demo.json` | Grilla de decisión + features por selección |
| `quantum_predictor.html` | Demo interactivo autónomo (versión binaria del Paso 1) |

## Uso

```powershell
python quantum_match.py            # entrena (quantum.joblib + quantum_demo.json)
python quantum_eval.py             # cuántico vs ensemble (out-of-sample)
streamlit run app/dashboard.py     # dashboard -> pestaña ⚛️ Cuántico
```

Requiere PennyLane: `pip install pennylane`.

## 🚀 Próximas mejoras (lo que falta probar)

- **Más qubits PERO con mejor optimización**: el Paso 7 mostró que sumar qubits
  a lo bruto empeora. Para que rinda hay que acompañarlo con más épocas,
  inicialización que evite *barren plateaus*, o un ansatz más eficiente.
- **Features más fuertes**: en vez de descanso/localía (débiles), probar Elo por
  confederación o fuerza de plantel (requiere histórico de planteles sin fuga).
- **Otra codificación**: `AmplitudeEmbedding` (codifica más info por qubit, no
  necesita 1 qubit por feature).
- **Ensamblar**: incorporar la distribución cuántica como un 4º modelo del
  ensemble y validar si baja el log-loss en la pestaña Rendimiento.
- **Hardware real**: correr el circuito final en un backend de IBM Quantum (vía
  qiskit) y comparar con el simulador (incluye mitigación de ruido).

## Stack

Python · PennyLane (autograd) · NumPy · pandas · scikit-learn · Streamlit · Plotly

> Proyecto educativo de QML: muestra el flujo completo de un clasificador
> cuántico variacional multiclase —codificación con re-uploading, ansatz,
> entrenamiento con early-stopping, evaluación y comparación contra modelos
> clásicos— sobre datos reales, iterado y medido paso a paso.
