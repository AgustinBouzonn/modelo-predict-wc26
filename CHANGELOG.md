# Changelog

Hitos del proyecto (de lo más nuevo a lo más viejo).

## Módulo cuántico (QML)
- Clasificador cuántico variacional **multiclase 1-X-2** (4 qubits, *data
  re-uploading*, PennyLane) → 61.2% accuracy, RPS 0.175 (holdout out-of-sample).
- **Ensamblado** con el ensemble clásico: mejora marginal del log-loss (0.21%).
- Integrado al dashboard (pestaña ⚛️ Cuántico) con comparación vs el ensemble.
- Demo interactivo en vivo (GitHub Pages).
- Evolución completa paso a paso en [README_QUANTUM.md](README_QUANTUM.md).

## Modelo clásico
- Ensemble **Elo + Poisson (Dixon-Coles) + XGBoost** con ajuste por sentimiento
  de noticias.
- Backtest out-of-sample: **64.1% accuracy**, log-loss 0.834 (vs 46.4% / 1.061
  del baseline "siempre local").
- **Pesos óptimos** del ensemble buscados por grid y aplicados
  (`{elo:0.5, poisson:0.5, ml:0}`).
- **Calibración**: temperature scaling encuentra T≈1.0 → el ensemble ya está bien
  calibrado, no necesita corrección.
- Simulación Monte Carlo del torneo (grupos + llaves) y benchmark contra el
  mercado de apuestas.

## Infraestructura
- Dashboard Streamlit con 10 pestañas + bootstrap para deploy sin modelos.
- CI (ruff + pytest + cobertura/Codecov) en cada push/PR.
- Dockerfile, Dependabot, licencia MIT.
- 30+ tests sobre datos sintéticos (sin red).
