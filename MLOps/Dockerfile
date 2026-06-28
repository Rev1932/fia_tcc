# Imagem base para os serviços de predição (API + dashboard)
FROM python:3.12-slim

WORKDIR /app

# Dependências de sistema mínimas para LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código (artefatos do modelo são montados via volume)
COPY Model ./Model
COPY MLOps ./MLOps

EXPOSE 8000 8501
CMD ["uvicorn", "MLOps.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
