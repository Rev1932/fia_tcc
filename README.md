# Credit Scoring — Home Credit Default Risk

Projeto Final (TCC) do MBA Big Data e Analytics — FIA/LABDATA.
Solução de **Machine Learning para risco de crédito**, do dado bruto ao serviço de
predição, seguindo o ciclo **CRISP-DM**.

## 🎯 Objetivo de negócio

Uma financeira de crédito ao consumidor perde dinheiro em dois extremos: **aprovando
quem não paga** (perda de capital — ~8% da carteira entra em default) e **negando quem
pagaria** (receita perdida e exclusão financeira). A decisão hoje depende de scores de
bureau que **faltam em 20–56% dos casos**.

**Solução:** um modelo de *credit scoring* que estima `P(default)` e, via uma régua de
decisão calibrada por **matriz de custo**, recomenda APROVAR/NEGAR maximizando o
resultado financeiro.

## 📊 Resumo da metodologia (CRISP-DM)

1. **Data Understanding** — EDA em `DataPipeline/exp_analysis.ipynb`.
2. **Data Preparation** — limpeza (`data_sanitization.py`) + ABT agregando as 9 tabelas
   relacionais (`abt_transform.py`) → 1 linha por cliente, ~470 features.
3. **Modeling** — baseline interpretável (Regressão Logística) + campeão **LightGBM**
   (categóricas nativas, `is_unbalance`, early stopping) — `Model/train.py`.
4. **Evaluation** — AUC, KS, threshold por custo e **SHAP** em `Model/evaluation.ipynb`.
5. **Deployment (MLOps)** — serviço FastAPI/Streamlit, orquestração e docker-compose
   (ver `MLOps/Readme.md`).

**Resultados (conjunto de teste):** LightGBM **AUC ≈ 0,785 · KS ≈ 0,44** (baseline LogReg AUC ≈ 0,77).

## 📁 Estrutura

```
Dados/            raw_data.csv · clean_data.csv · abt.csv   (gerados; fora do git)
DataPipeline/     data_sanitization.py · abt_transform.py · exp_analysis.ipynb · config.yaml
Model/            train.py · predict.py · evaluation.ipynb · config.yaml
MLOps/            Readme.md · Dockerfile · docker-compose.yml · pipeline_orchestration.py · app/
artifacts/        model.joblib · metrics.json · feature_metadata.json   (gerados)
requirements.txt
```

> A base bruta (>3 GB, 9 CSVs) fica em `data/risco_fraude/home-credit-default-risk/`
> e **não** é versionada. Fonte:
> <https://www.kaggle.com/competitions/home-credit-default-risk>.

## 🚀 Como reproduzir

```bash
# 1) Ambiente
python -m venv .venv && source .venv/bin/activate   # (fish: source .venv/bin/activate.fish)
pip install -r requirements.txt

# 2) Pipeline + treino (tudo de uma vez)
python MLOps/pipeline_orchestration.py
#   ou passo a passo:
python DataPipeline/data_sanitization.py    # -> Dados/clean_data.csv
python DataPipeline/abt_transform.py        # -> Dados/abt.csv
python Model/train.py                        # -> artifacts/model.joblib

# 3) Avaliação
jupyter notebook Model/evaluation.ipynb
```

## 🔮 Serviço de predição

```bash
# Predição via CLI
python Model/predict.py --json '{"NAME_CONTRACT_TYPE":"Cash loans","AMT_CREDIT":500000,"AMT_INCOME_TOTAL":150000}'

# API + dashboard via Docker
docker compose -f MLOps/docker-compose.yml up --build
#   API:        http://localhost:8000/docs
#   Dashboard:  http://localhost:8501
```

Detalhes de arquitetura, monitoramento e ações automatizadas: [`MLOps/Readme.md`](MLOps/Readme.md).
Planejamento completo do projeto: [`PLAN.md`](PLAN.md) · Análise de escopo: [`OKR.md`](OKR.md).
