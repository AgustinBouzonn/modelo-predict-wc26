# ⚽ Modelo Predictivo Mundial 2026

[![CI](https://github.com/AgustinBouzonn/modelo-predict-wc26/actions/workflows/ci.yml/badge.svg)](https://github.com/AgustinBouzonn/modelo-predict-wc26/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-QML-7B3FE4)
![License](https://img.shields.io/badge/license-MIT-green)

Modelo **ensemble** que predice los partidos del Mundial 2026 y se **actualiza**
con nuevos datasets, resultados y noticias a medida que avanza el torneo. Incluye
un dashboard interactivo en Streamlit y un módulo experimental de **Machine
Learning cuántico** comparado contra los modelos clásicos.

> 🔮 **[Probá el predictor cuántico en vivo →](https://agustinbouzonn.github.io/modelo-predict-wc26/)**
> (elegí dos selecciones y "medí el qubit" — corre en tu navegador, sin instalar nada)

Combina tres enfoques complementarios:

| Modelo | Qué aporta |
|--------|-----------|
| **Elo** | Fuerza relativa de cada selección, actualizable partido a partido |
| **Poisson (Dixon-Coles)** | Goles esperados, marcador más probable y probabilidad de cada resultado |
| **XGBoost** | Patrones no lineales sobre Elo, forma reciente y descanso |

Además incorpora **sentimiento de noticias** (RSS de Google News + VADER) como
ajuste en vivo del "momentum mediático" de cada selección.

> ⚛️ **Extra — módulo de Quantum Machine Learning.** Un clasificador cuántico
> variacional (PennyLane, 4 qubits, *data re-uploading*) predice el resultado
> 1-X-2 y se compara contra el ensemble en la pestaña **⚛️ Cuántico** del
> dashboard. Toda la evolución paso a paso está en
> **[README_QUANTUM.md](README_QUANTUM.md)**.

## 📊 Resultados (out-of-sample, holdout 12 meses · 601 partidos competitivos)

![Módulo cuántico: frontera de decisión y comparación con el ensemble](docs/quantum.png)

| Modelo | Accuracy | Log-loss | RPS |
|--------|:--------:|:--------:|:---:|
| 🧮 **Ensemble** (Elo · Poisson · XGBoost) | **63.7%** | **0.837** | **0.160** |
| ⚛️ Cuántico (4 qubits, 1-X-2) | 61.2% | 0.892 | 0.175 |
| 📏 Baseline (frecuencias 1-X-2) | 46.4% | 1.061 | 0.234 |

Métricas calculadas **sin fuga de datos** (modelo entrenado as-of una fecha de
corte, evaluado sobre lo posterior). RPS = *Ranked Probability Score*, la métrica
estándar para 1X2 (menor = mejor). Ambos modelos superan claramente al baseline.

![Rendimiento del ensemble vs baselines](docs/performance.png)

## 🖥️ Dashboard

`streamlit run app/dashboard.py` abre un panel con:

`🗓️ Fixture` · `🗺️ Torneo` · `🏆 Llaves` · `👥 Selecciones` · `⚽ Partido`
`⚛️ Cuántico` · `📊 Ranking` · `🎲 Simular` · `📈 Rendimiento` · `🗞️ Datos`

Incluye predicción por partido con alineaciones en cancha (SVG), proyección del
cuadro eliminatorio, simulación Monte Carlo del torneo y un benchmark medible del
modelo contra el mercado de apuestas.

![Ranking Elo de las selecciones](docs/ranking.png)

## 🏗️ Arquitectura

```
config/teams_wc26.yaml        Grupos del torneo, alias y pesos del ensemble
src/
  config.py                   Rutas y carga de configuración
  data/sources.py             Ingesta del histórico (scraping) + resultados WC26
  data/news.py                Noticias + sentimiento por selección
  features/build_features.py  Features sin fuga de datos (Elo pre-partido, forma)
  models/elo.py               Modelo Elo
  models/poisson.py           Dixon-Coles (sklearn PoissonRegressor + ρ)
  models/ml_model.py          Clasificador XGBoost H/D/A
  models/quantum.py           Clasificador cuántico variacional (PennyLane)
  models/ensemble.py          Combina los 3 + ajuste por noticias
  simulation/tournament.py    Monte Carlo del torneo (grupos + llaves)
  evaluation/backtest.py      Backtest out-of-sample + optimización de pesos
  pipeline.py                 Orquesta ingesta → entrenamiento → ensemble
app/dashboard.py              Dashboard Streamlit
quantum_match.py              Entrena el modelo cuántico · quantum_eval.py lo compara
```

## ⚙️ Instalación

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## ▶️ Uso

**1. Entrenar (descarga datos + entrena los 4 modelos):**
```bash
python -m src.pipeline              # ensemble clásico (Elo, Poisson, XGBoost)
python quantum_match.py             # clasificador cuántico
```
Opciones del pipeline: `--no-fetch` (no redescargar), `--no-news` (saltear
noticias), `--sims 5000` (simular el torneo al final).

**2. Evaluar (out-of-sample):**
```bash
python -m src.evaluation.backtest               # backtest del ensemble
python -m src.evaluation.backtest --optimize-weights
python quantum_eval.py                          # cuántico vs ensemble (RPS)
```

**3. Abrir el dashboard:**
```bash
streamlit run app/dashboard.py
```

> ☁️ También se puede **publicar gratis en Streamlit Community Cloud** (la app
> entrena sola en el primer arranque si faltan los modelos). Pasos en
> **[DEPLOY.md](DEPLOY.md)**.

## 🔄 Cómo actualizarlo con el torneo en curso

- **Resultados automáticos:** el histórico se baja del dataset público de
  partidos internacionales (se actualiza con cada fecha FIFA). Volvé a correr
  `python -m src.pipeline` para reentrenar con lo último.
- **Resultados del Mundial a mano:** agregá filas a `data/raw/wc26_manual.csv`
  con `date,home_team,away_team,home_score,away_score` y reentrená.
- **Noticias:** `python -m src.data.news` refresca el sentimiento por selección.
- **Grupos / fixture:** editá `config/teams_wc26.yaml` para que coincida con el
  sorteo oficial (los grupos cargados son un placeholder editable).

> Los modelos y los datos pesados no se versionan (ver `.gitignore`): se generan
> corriendo el pipeline. La API key de cuotas (`config/odds_api_key.txt`) tampoco
> se sube; el benchmark de mercado es opcional.

## 🪟 Actualización automática (Windows)

Para que se actualice solo todos los días, programá el script de update con el
Task Scheduler:
```powershell
schtasks /create /tn "WC26 Update" /sc daily /st 09:00 ^
  /tr "powershell -ExecutionPolicy Bypass -File `"%CD%\scripts\daily_update.ps1`""
```
El log queda en `data/update.log`. En el dashboard, el botón **🔄 Recargar datos**
(pestaña Datos) toma lo reentrenado sin reiniciar el proceso.

## 🧪 Desarrollo

```bash
pip install -r requirements-dev.txt
ruff check src app tests     # lint
pytest                       # tests (modelos, simulación, config y cuántico)
```
El CI (`.github/workflows/ci.yml`) corre `ruff` + `pytest` en cada push/PR sobre
Python 3.13. Los tests usan datos sintéticos, así que corren sin red ni descargas.

## 🛠️ Stack

Python · pandas · NumPy · scikit-learn · XGBoost · PennyLane · Streamlit · Plotly · pytest · ruff

## 📄 Licencia

[MIT](LICENSE) — uso libre con atribución.
