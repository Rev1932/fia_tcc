# Imagem do Airflow com as dependências do projeto.
#
# Por que python3.14: é o MESMO interpretador que gera o artifacts/model.joblib
# (ver MLOps/Dockerfile). Pickle de sklearn/LightGBM entre versões diferentes
# falha no pior momento possível — aqui quem treina e quem serve usam a mesma.
FROM apache/airflow:3.3.1-python3.14

# libgomp1: runtime do LightGBM
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# Só o subconjunto que as tasks usam. Instalar o requirements.txt inteiro
# sobrescreveria o fastapi do Airflow (ver requirements-pipeline.txt).
COPY MLOps/airflow/requirements-pipeline.txt /tmp/requirements-pipeline.txt
RUN pip install --no-cache-dir -r /tmp/requirements-pipeline.txt
