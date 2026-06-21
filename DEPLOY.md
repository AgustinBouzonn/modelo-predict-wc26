# 🚀 Deploy del dashboard en Streamlit Community Cloud

El dashboard se puede publicar gratis con una URL pública. Como los modelos y los
datos no se versionan, la app trae un **bootstrap**: si al arrancar no encuentra
los modelos, descarga los datos y los entrena sola (la primera carga tarda unos
minutos; después queda cacheado).

## Pasos

1. Entrá a **[share.streamlit.io](https://share.streamlit.io)** y logueate con tu
   cuenta de GitHub.
2. **New app** → elegí el repo `AgustinBouzonn/modelo-predict-wc26`, branch `main`.
3. **Main file path:** `app/dashboard.py`
4. **Deploy.** En el primer arranque la app entrena el ensemble (y el cuántico al
   abrir su pestaña) con una barra de progreso.

> Streamlit Cloud detecta `requirements.txt` automáticamente. No hace falta
> configurar nada más. El benchmark de cuotas queda desactivado en la nube (no se
> sube la API key); el resto del dashboard funciona completo.

## Notas

- **Primera carga lenta** (~2-4 min): descarga ~50k partidos y entrena. Las
  siguientes son instantáneas (cache de Streamlit).
- **Recursos**: el free tier (1 GB RAM) alcanza para entrenar con ~23k partidos.
- **Actualizar**: cada push a `main` redeploya solo. Para forzar reentrenamiento,
  usá el botón «🔄 Recargar datos» de la pestaña Datos.

## Alternativa: correrlo local

```bash
pip install -r requirements.txt
python -m src.pipeline        # entrena el ensemble
python quantum_match.py       # entrena el cuántico
streamlit run app/dashboard.py
```
