# MLOps — Arquitetura da Solução (Etapa Individual)

Desenho da arquitetura de deploy do modelo de **credit scoring** como serviço de
predição, com orquestração, monitoramento e ações automatizadas.

## 1. Arquitetura

```
   ┌───────────── Orquestração — Airflow, a cada 7 dias ──────────────────────┐
   │                                                                          │
   │  checar_fontes ─► sanitizacao ─► abt_transform ─► validar_abt            │
   │                                                        │                 │
   │    resumo ◄─ calcular_psi ◄─ validar_metricas ◄─ treino ◄─ perfil_colunas│
   │                    (drift)      (GATE de AUC)                            │
   └──────────────────────────────────────────┬───────────────────────────────┘
                                              │ artifacts/  (rodada canônica)
                                              ▼
              ┌──────────────── Serviço de predição ─────────────────┐
              │   FastAPI (27 endpoints)   +   Streamlit (dashboard) │
              └──────────────────────────┬───────────────────────────┘
                                         ▼
                        Ações automatizadas (3 faixas: aprovar / revisar / negar)
```

O DAG é **executável**, não é diagrama: `MLOps/airflow/` sobe a instância e
`dags/treino_credit_scoring.py` define as 9 tasks. Ver
[`MLOps/airflow/README.md`](airflow/README.md).

- **Ingestão & pipeline**: `DataPipeline/data_sanitization.py` → `DataPipeline/abt_transform.py`.
- **Treino**: `Model/train.py` gera `artifacts/model.joblib` (modelo + threshold + metadados).
- **Serving**: `MLOps/app/api.py` (FastAPI) e `MLOps/app/streamlit_app.py` (demo).
- **Orquestração**: `dags/treino_credit_scoring.py` no Airflow, a cada 7 dias,
  com gate de qualidade e cálculo de drift. `MLOps/pipeline_orchestration.py`
  continua como o caminho manual, para rodar o pipeline sem subir a stack.
- **Infra**: `MLOps/docker-compose.yml` sobe `api` (8000) e `dashboard` (8501);
  `MLOps/airflow/docker-compose.yml` sobe o Airflow (8080). Stacks separadas:
  a API fica no ar, o Airflow acorda a cada 7 dias.

## 2. Como executar

### Agendado (Airflow)

```bash
cd MLOps/airflow
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d --build      # http://localhost:8080  (admin/admin)
```

O DAG `treino_credit_scoring` roda a cada 7 dias. Para demonstrar em ~1 min,
defina a Variable `hc_sample=30000` e dispare manualmente — detalhes em
[`MLOps/airflow/README.md`](airflow/README.md).

### Manual

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

| Dimensão | Métrica | Ação ao desviar | Estado |
|---|---|---|---|
| **Data drift** | PSI do score e das features | Alerta + investigar fonte | ✅ `calcular_psi` no DAG e `GET /model/psi` |
| **Concept drift** | AUC vs. a rodada anterior aceita | Barrar a promoção do modelo | ✅ `validar_metricas` — **falha a DAG** |
| **Qualidade dos dados** | fontes presentes, granularidade da ABT, vazamento | Bloquear antes do treino | ✅ `checar_fontes` e `validar_abt` |
| **Re-treino** | — | Agendado | ✅ a cada 7 dias |
| **Performance do serviço** | latência, taxa de erro, throughput | Auto-scaling / rollback | ⬜ proposta |
| **Negócio** | aprovação e inadimplência observadas | Recalibrar threshold | ⬜ proposta (o cálculo existe em `/model/threshold-analysis`) |

As quatro primeiras linhas **executam**: são tasks do DAG, com log por etapa e
`artifacts/psi_report.json` gravado a cada rodada. As duas últimas seguem como
proposta — exigiriam telemetria de produção, que este projeto não tem.

Evolução: logs estruturados → Prometheus/Grafana; *model registry* (MLflow) para
versionar modelos e habilitar rollback.

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
- Testes automatizados de dados (Great Expectations) — hoje as checagens de
  qualidade são as tasks `checar_fontes` e `validar_abt`.
- Notificação ativa (e-mail/Slack) quando o gate de AUC barra uma rodada.

> Concluídos em ciclos anteriores: **calibração isotônica** (o score virou
> P(inadimplência) real, Brier 0,1668 → 0,0658) e **orquestração no Airflow**
> com agendamento de 7 dias.
