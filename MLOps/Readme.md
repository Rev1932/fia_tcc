# MLOps — Arquitetura da Solução (Etapa Individual)

Desenho da arquitetura de deploy do modelo de **credit scoring** como serviço de
predição, com orquestração, monitoramento e ações automatizadas.

## 1. Arquitetura

```
                      ┌─────────────────── Orquestração (Airflow) ───────────────────┐
                      │                                                               │
  Fonte de dados      │   raw_data ──► data_sanitization ──► abt_transform ──► train  │
  (CSV / DB / API)    │   (bruta)        (clean_data)          (abt.csv)     (model)  │
                      └───────────────────────────────────────────────┬──────────────┘
                                                                       │ artifacts/model.joblib
                                                                       ▼
                                          ┌──────────────── Serviço de predição ───────────────┐
                                          │   FastAPI  (POST /predict)   +   Streamlit (demo)   │
                                          └──────────────────────────┬──────────────────────────┘
                                                                     ▼
                                            Ações automatizadas (auto-decisão / fila / alerta)
```

- **Ingestão & pipeline**: `DataPipeline/data_sanitization.py` → `DataPipeline/abt_transform.py`.
- **Treino**: `Model/train.py` gera `artifacts/model.joblib` (modelo + threshold + metadados).
- **Serving**: `MLOps/app/api.py` (FastAPI) e `MLOps/app/streamlit_app.py` (demo).
- **Orquestração**: `MLOps/pipeline_orchestration.py` (standalone ou DAG Airflow).
- **Infra**: `MLOps/docker-compose.yml` sobe `api` (porta 8000) e `dashboard` (8501).

## 2. Como executar

```bash
# 1) Treinar (gera artifacts/model.joblib)
python Model/train.py

# 2) Subir os serviços
docker compose -f MLOps/docker-compose.yml up --build

# 3) Testar a API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"NAME_CONTRACT_TYPE":"Cash loans","AMT_CREDIT":500000,"AMT_INCOME_TOTAL":150000}]}'

# Dashboard: http://localhost:8501
```

## 3. Monitoramento em produção (item iii)

| Dimensão | Métrica | Ação ao desviar |
|---|---|---|
| **Data drift** | PSI / KS das features de entrada vs. treino | Alerta + investigar fonte |
| **Concept drift** | AUC/KS em janela móvel (back-testing) | Re-treino agendado |
| **Qualidade dos dados** | % nulos, % fora do domínio, falhas de schema | Bloquear batch + notificar |
| **Performance do serviço** | latência, taxa de erro, throughput | Auto-scaling / rollback |
| **Negócio** | taxa de aprovação e inadimplência observada | Recalibrar threshold |

Stack sugerida: logs estruturados → Prometheus/Grafana; PSI calculado em job diário no
Airflow; *model registry* (MLflow) para versionar modelos e habilitar rollback.

## 4. Ações automatizadas + agentes de IA (item iv)

A partir do `P(default)` e do threshold de negócio:

- **Auto-aprovação** para risco baixo (decisão instantânea, sem fila humana).
- **Auto-negação / oferta alternativa** para risco alto (ex.: limite menor, garantia).
- **Faixa cinza → revisão humana** (human-in-the-loop) com o relatório SHAP do caso.
- **Early-warning de cobrança**: scores altos disparam régua de comunicação proativa.
- **Agente de IA** que monta o resumo explicável (SHAP → linguagem natural) para o
  analista de crédito e registra a justificativa para fins de governança/auditoria.

## 5. Próximos passos

- Versionamento de modelo (MLflow) + CI/CD do pipeline.
- Feature store para reuso das agregações da ABT em tempo real.
- Testes automatizados de dados (Great Expectations) e de contrato da API.
- Calibração de probabilidade (Platt/Isotonic) para leitura direta do risco.
