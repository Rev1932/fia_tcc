# Imagem dos serviços de predição (API + dashboard).
# Python 3.14 alinhado ao ambiente que gera o artifacts/model.joblib: pickles
# de sklearn/LightGBM entre versões diferentes falham no pior momento possível.
FROM python:3.14-slim

WORKDIR /app

# libgomp1: dependência de runtime do LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código. Os dados (Dados/) e os artefatos (artifacts/) entram por volume.
COPY Model ./Model
COPY MLOps ./MLOps
COPY DataPipeline ./DataPipeline

EXPOSE 8000 8501
CMD ["uvicorn", "MLOps.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
