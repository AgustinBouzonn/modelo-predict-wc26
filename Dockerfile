# Imagen para correr el dashboard del Predictor Mundial 2026.
# Build:  docker build -t wc26 .
# Run:    docker run -p 8501:8501 wc26   ->  http://localhost:8501
#
# La app entrena los modelos en el primer arranque si faltan (bootstrap),
# así que la imagen no necesita modelos pre-entrenados.
FROM python:3.13-slim

WORKDIR /app

# Dependencias del sistema mínimas para algunas ruedas (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/dashboard.py", \
            "--server.port=8501", "--server.address=0.0.0.0", \
            "--browser.gatherUsageStats=false"]
