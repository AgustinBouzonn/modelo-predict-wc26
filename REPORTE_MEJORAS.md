# 📊 Reporte de evaluación y mejoras — WC26 (2026-06-19)

Datos: **23.298 partidos** (2002 → 2026-06-18), 49.477 crudos scrapeados.
Incluye 8 resultados del Mundial ya cargados. Todas las métricas son
**out-of-sample** (entrenado as-of, evaluado sobre lo posterior).

## 1. Resultados actuales

### Ensemble clásico (1X2: local/empate/visitante)
| Conjunto | Accuracy | Log-loss | Baseline "siempre local" |
|---|---|---|---|
| Holdout 12m (601 competitivos) | **63.7%** | **0.837** | 46.4% / 1.061 |
| Mundial 2026 (28 jugados) | 50.0% | 1.039 | 53.6% / 1.032 |

> El holdout (muestra grande) muestra un modelo **sólido**: +17 pts de accuracy
> y mucho mejor log-loss que el baseline. En el Mundial (muestra chica, 28
> partidos, mucho empate y sorpresa) todavía no le gana a "siempre local" —
> esperable y honesto.

### Cuántico vs Ensemble — 1-X-2 multiclase (holdout 601 competitivos)
Tras el upgrade a **4 qubits, multiclase (1-X-2)** con early-stopping y
**data re-uploading**:

| Modelo | Accuracy | Log-loss | RPS |
|---|---|---|---|
| 🧮 Ensemble | **63.7%** | **0.837** | **0.160** |
| ⚛️ Cuántico (4q, re-uploading) | 61.2% | 0.892 | 0.175 |
| 📏 Baseline (frecuencias) | 46.4% | 1.061 | 0.234 |

> Ahora es **comparable directo** por RPS (la métrica de 1X2). El cuántico queda
> a ~2.5 pts de accuracy y RPS 0.175 vs 0.160: muy competitivo. El re-uploading
> dio una mejora marginal sobre la versión sin él (60.7% / 0.173) → el techo con
> 4 features está cerca; el próximo salto pide más features/qubits.
> La evolución completa paso a paso está en `README_QUANTUM.md`.

## 2. Hallazgos clave

1. **El XGBoost casi no aporta hoy.** La optimización de pesos da óptimo
   `{elo: 0.5, poisson: 0.5, ml: 0.0}` (vs actual `0.4/0.35/0.25`), mejora de
   solo 0.8% en log-loss. El ML está aportando poco sobre Elo+Poisson.
2. **El Mundial es difícil de batir.** 28 partidos con muchos empates (varios
   1-1) y sorpresas (USA 4-1, Iraq 1-4). El log-loss del modelo ≈ el del
   baseline: hay poco margen con tan pocos datos.
3. **El cuántico es competitivo** pese a usar 2 de las 9 features del ensemble.

## 3. Cómo mejorar — priorizado

### Ensemble (alto impacto)
- **Revisar el XGBoost** (peso óptimo 0). Probar: regularización, más features
  (planteles, descanso real, head-to-head), o reemplazarlo por LightGBM. Si no
  mejora, bajarle el peso oficialmente a ~0.1.
- **Aplicar los pesos óptimos**: `python -m src.evaluation.backtest --optimize-weights --write` y reentrenar.
- **Calibración**: revisar la pestaña Rendimiento → calibración; si hay
  sobre-confianza, aplicar Platt/Isotónica sobre las probabilidades finales.
- **Modelar mejor el empate**: en el Mundial fallaron varios empates (Brasil-Marruecos,
  Países Bajos-Japón). Ajustar el `draw_c0/draw_c1` del Elo o el ρ de Dixon-Coles.

### Cuántico (medio impacto, alto valor de portfolio)
- **Más features / qubits**: pasar de 2 a 4-6 qubits sumando descanso, localía y
  fuerza del plantel. Es lo que más subiría la accuracy.
- **3 clases (1-X-2)**: predecir el empate con varias salidas → comparable
  directo por log-loss/RPS contra el ensemble.
- **Ensamblarlo**: meter la probabilidad cuántica como 4º modelo del ensemble y
  validar si baja el log-loss.

### Datos / pipeline (bajo esfuerzo)
- **Cargar resultados del Mundial a mano** en `data/raw/wc26_manual.csv` a medida
  que se juegan, y reentrenar — el tracking en vivo dará el veredicto real.
- **Automatizar** el update diario (ya hay `scripts/daily_update.ps1` + Task Scheduler).

## 4b. Mejoras aplicadas y medidas (actualización)

- **Pesos óptimos del ensemble aplicados**: `{elo: 0.5, poisson: 0.5, ml: 0.0}`
  (escritos al config y reentrenado). El XGBoost queda con peso 0 — confirma que
  hoy no aporta sobre Elo+Poisson.
- **Círculo cuántico↔clásico cerrado**: se ensambló la distribución 1-X-2 del
  modelo cuántico con la del ensemble (blend convexo) sobre el holdout:

  | Blend | Log-loss | RPS | Accuracy |
  |-------|:--------:|:---:|:--------:|
  | Ensemble solo | 0.8342 | 0.1595 | 64.1% |
  | + cuántico (peso 0.10) | **0.8324** | 0.1591 | 64.2% |

  El mejor blend (10% cuántico) **mejora el log-loss 0.21%**. Marginal pero
  positivo: el cuántico aporta algo de señal complementaria, no es redundante.
  Reproducible con `python quantum_ensemble.py`.

## 4c. Despliegue

- **Demo en vivo** del predictor cuántico interactivo (GitHub Pages):
  https://agustinbouzonn.github.io/modelo-predict-wc26/
- **Dashboard** listo para Streamlit Cloud con bootstrap automático (ver
  `DEPLOY.md`).

## 4. Cómo reproducir
```powershell
python -m src.pipeline                              # scrape + reentrena ensemble
python quantum_match.py                             # reentrena cuántico
python -m src.evaluation.backtest                   # backtest ensemble
python -m src.evaluation.backtest --optimize-weights
python quantum_eval.py                              # cuántico vs ensemble
streamlit run app/dashboard.py                      # dashboard
```
