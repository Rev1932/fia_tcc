# Credit Scoring — Home Credit Default Risk

Projeto Final (TCC) do MBA Big Data e Analytics — FIA/LABDATA.
Solução de **Machine Learning para risco de crédito**, do dado bruto ao serviço de
predição, seguindo o ciclo **CRISP-DM**.

> **Todos os números deste repositório saem de `artifacts/`.** Para reproduzir a
> tabela oficial da rodada: `python Model/run_summary.py --markdown`.
> Nada de número copiado à mão.

## 🎯 Objetivo de negócio

Uma financeira de crédito ao consumidor perde dinheiro em dois extremos: **aprovando
quem não paga** (perda de capital — 8,07% da carteira entra em inadimplência) e **negando
quem pagaria** (receita perdida e exclusão financeira). A decisão hoje depende de scores
de bureau que **faltam em 20–56% dos casos**.

**Solução:** um modelo de *credit scoring* que estima `P(inadimplência)` e, via uma régua
de decisão calibrada por **matriz de custo**, recomenda APROVAR / REVISAR / NEGAR
maximizando o resultado financeiro.

## 📊 Metodologia (CRISP-DM)

1. **Data Understanding** — EDA em `DataPipeline/exp_analysis.ipynb`.
2. **Data Preparation** — limpeza (`data_sanitization.py`) + ABT agregando as 9 tabelas
   relacionais (`abt_transform.py`) → 1 linha por cliente, **1.018 features**.
3. **Modeling** — baseline interpretável (Regressão Logística) + campeão **LightGBM**
   + **calibração isotônica** — `Model/train.py`.
4. **Evaluation** — AUC, KS, Brier, threshold por custo, fairness com intervalo de
   confiança e **SHAP** — `Model/evaluation.ipynb` e `artifacts/`.
5. **Deployment (MLOps)** — API FastAPI com 26 endpoints, dashboard Streamlit,
   orquestração e docker-compose (ver `MLOps/Readme.md` e `MLOps/app/README.md`).
   Stack verificada em Docker; monitoramento de drift via `GET /model/psi`.

## 📈 Resultados (conjunto de teste)

| Métrica | Baseline (Reg. Logística) | Campeão (LightGBM calibrado) |
|---|---|---|
| AUC | 0,7776 | **0,7868** |
| KS | 0,4228 | **0,4342** |
| Brier | 0,1874 | **0,0658** |


> **Servido vs. cru.** Estes são os números do modelo que a API entrega — o
> campeão com calibração isotônica. O campeão cru marca AUC 0,7871 / KS 0,4354;
> a isotônica é monotônica, mas cria empates no score, o que move a terceira
> casa. O que ela de fato muda é o Brier: 0,1668 → 0,0658.

- Overfitting sob controle: AUC treino 0,8753 → validação 0,7835 → **teste 0,7871** (modelo cru)
  (validação e teste empatam).
- Régua de custo (falso negativo custa 10× o falso positivo) → threshold **0,09**,
  **68,7%** de aprovação.
- As 9 tabelas relacionais respondem por **39,7%** da importância do modelo.

### O que foi corrigido neste ciclo

| Rodada | Features | AUC | KS | Brier | Corte |
|---|---|---|---|---|---|
| v1 (original) | 471 | 0,7846 | 0,4349 | 0,1639 | 0,47 |
| v2 (ABT corrigida) | 1.018 | 0,7880 | 0,4402 | 0,1649 | 0,50 |
| **v3 (calibrado)** | 1.018 | 0,7868 | 0,4342 | **0,0658** | **0,09** |

1. **A ABT descartava as variáveis categóricas** das tabelas relacionais —
   `bureau_balance.STATUS` (histórico mês a mês de atraso) e
   `previous_application.NAME_CONTRACT_STATUS` (histórico de recusa) ficavam de fora.
   Corrigido: 473 → 1.020 colunas.
2. **Scores externos combinados** (`EXT_SOURCE_MEAN` virou a variável nº 1, com 24% da
   importância) em vez de tratados isoladamente.
3. **Calibração isotônica** — o score passou a ser P(inadimplência) real: Brier caiu 2,5×
   e o corte ótimo saiu de 0,50 (arbitrário) para 0,09 (interpretável).
4. **Régua de três faixas** — os segmentos de baixa confiança medida vão para revisão
   humana, em vez de decisão automática.

### Onde o modelo ainda falha (e por que dizemos isso com números)

Cada AUC por segmento vem com **intervalo de confiança bootstrap**. Um grupo só conta
como fraqueza real quando seu IC **não sobrepõe** o geral (0,7806–0,7935):

| Segmento | n | AUC [IC 95%] | Fraqueza real? |
|---|---|---|---|
| `<25 anos` | 2.355 | 0,7319 [0,7012–0,7597] | **sim** |
| `55-65 anos` | 12.166 | 0,7465 [0,7281–0,7672] | **sim** |
| Thin-file | 8.776 | 0,7745 [0,7577–0,7899] | não — sobrepõe |
| Gênero M vs F | 20.940 / 40.561 | 0,7872 / 0,7795 | não — discrimina risco, não pessoas |

As correções melhoraram o modelo geral mas **não resolveram a fraqueza em `<25`**.
A resposta para esse caso é de processo, não de modelagem: faixa cinza ampliada →
revisão humana (`GET /model/decision-policy`).

## 📁 Estrutura

```
Dados/            raw_data.csv · clean_data.csv · abt.csv · abt.parquet   (gerados; fora do git)
DataPipeline/     data_sanitization.py · abt_transform.py · to_parquet.py · exp_analysis.ipynb · config.yaml
Model/            train.py · predict.py · metrics_lib.py · derived.py · run_summary.py · evaluation.ipynb · config.yaml
MLOps/            Readme.md · Dockerfile · docker-compose.yml · pipeline_orchestration.py
MLOps/app/        api.py · db.py · schemas.py · explain.py · policy.py · routers/ · README.md · requests.http
tests/            test_metrics_lib.py · test_derived.py · test_api.py · test_security.py
artifacts/        model.joblib · metrics.json · curves.json · fairness.json ·
                  feature_importance.json · improvement_log.json · scores.parquet   (gerados)
```

> A base bruta (>3 GB, 9 CSVs) fica em `data/risco_fraude/home-credit-default-risk/`
> e **não** é versionada. Fonte:
> <https://www.kaggle.com/competitions/home-credit-default-risk>.

## 🚀 Como reproduzir

```bash
# 1) Ambiente (Python 3.14)
python -m venv .venv && source .venv/bin/activate    # fish: source .venv/bin/activate.fish
pip install -r requirements.txt

# 2) Pipeline completo: raw -> clean -> ABT -> parquet -> treino
python MLOps/pipeline_orchestration.py

#    ou passo a passo:
python DataPipeline/data_sanitization.py    # -> Dados/clean_data.csv
python DataPipeline/abt_transform.py        # -> Dados/abt.csv + abt.parquet
python DataPipeline/to_parquet.py           # -> artifacts/abt_profile.json
python Model/train.py                        # -> artifacts/

# 3) Números oficiais da rodada
python Model/run_summary.py --markdown

# 4) Testes
pytest -q
```

**Modo demonstração** — treino em menos de 1 minuto, para alterar um hiperparâmetro e
ver o efeito na hora (a rodada oficial roda sem a flag):

```bash
python Model/train.py --sample 30000 --tag demo
```

## 🔮 Serviço de predição

```bash
# API + dashboard via Docker
docker compose -f MLOps/docker-compose.yml up --build
#   API:        http://localhost:8000/docs
#   Dashboard:  http://localhost:8501

# ou local
uvicorn MLOps.app.api:app --port 8000
streamlit run MLOps/app/streamlit_app.py
```

A API entrega os dados da análise por **26 endpoints filtráveis** em quatro famílias
(`/clients`, `/stats`, `/model`, predição/simulação/explicabilidade).
Referência completa e exemplos: [`MLOps/app/README.md`](MLOps/app/README.md) ·
chamadas prontas em [`MLOps/app/requests.http`](MLOps/app/requests.http).

```bash
# "e se aprovar um mau pagador custasse 20x em vez de 10x?" — sem re-treinar
curl "localhost:8000/model/threshold-analysis?cost_fn=20&cost_fp=1"

# inadimplência por escolaridade, só entre thin-file com menos de 25 anos
curl "localhost:8000/stats/default-rate?by=education&thin_file=true&age_max=25&min_count=5"

# por que este cliente foi negado
curl "localhost:8000/clients/100052/explain?top=5"

# como detectar que o modelo envelheceu (PSI — monitoramento de drift)
curl "localhost:8000/model/psi?referencia=train&comparado=test"
```

---

**[Dossiê do projeto](https://claude.ai/code/artifact/40f858c1-2775-40a9-accc-e1ac97221284)**
— o registro de construção: da exploração dos dados às fraquezas que foram (e não foram)
corrigidas. Fonte em [`docs/dossie/`](docs/dossie/).
Tarefas em aberto: [`TODO.md`](TODO.md).

Arquitetura, monitoramento e ações automatizadas: [`MLOps/Readme.md`](MLOps/Readme.md).
Planejamento do projeto: [`PLAN.md`](PLAN.md) · Análise de escopo: [`OKR.md`](OKR.md) ·
Documento de TCC: [`docs/TCC.md`](docs/TCC.md).
