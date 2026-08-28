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
5. **Deployment (MLOps)** — API FastAPI com 27 endpoints, dashboard Streamlit,
   e **re-treino orquestrado no Airflow a cada 7 dias** (ver `MLOps/Readme.md`,
   `MLOps/airflow/README.md` e `MLOps/app/README.md`). Stack verificada em
   Docker; drift via `calcular_psi` no DAG e `GET /model/psi`.

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

Um grupo conta como fraqueza quando o **IC bootstrap da diferença** entre o AUC dele e o
dos demais grupos do mesmo eixo exclui o zero. O critério anterior comparava o segmento
com o AUC **geral** — que contém o próprio segmento e é composto 77,6% por pares entre
faixas etárias. Ver [`docs/diagnostico-faixa-etaria.md §5`](docs/diagnostico-faixa-etaria.md).

| Segmento | n | AUC [IC 95%] | Δ vs. demais [IC 95%] | p | Fraqueza? |
|---|---|---|---|---|---|
| `<25 anos` | 2.355 | 0,7319 [0,7012–0,7597] | −0,0514 [−0,0827 – −0,0205] | 0,008 | **sim** |
| `55-65 anos` | 12.166 | 0,7465 [0,7281–0,7672] | −0,0415 [−0,0619 – −0,0209] | 0,004 | **sim** |
| `65+ anos` | 1.725 | 0,7631 [0,7084–0,8100] | −0,0198 [−0,0696 – +0,0331] | 0,395 | não |
| Thin-file | 8.776 | 0,7745 [0,7577–0,7899] | −0,0133 [−0,0299 – +0,0043] | 0,132 | não |
| Gênero M vs F | 20.940 / 40.561 | 0,7872 / 0,7795 | ±0,0078 [−0,0195 – +0,0195] | 0,240 | não |

**A fraqueza em `<25` foi diagnosticada e tem veredito: é teto de dado, não defeito de
modelo — mas não pela causa que este projeto vinha declarando.**

- Não é ausência de histórico. Uma coorte de 25–45 anos reamostrada até reproduzir o
  perfil de informação do jovem (71 estratos, 99,9% de cobertura) atinge **AUC 0,7803**
  contra 0,7319 do jovem. **A causa declarada foi refutada.**
- ~36% do efeito nem é sobre idade: é sobre estar na região baixa do score externo, onde
  o modelo separa pior em qualquer faixa.
- Não há conserto disponível: seis variantes de modelo dedicado (`<25` e `<30`) ficam
  **todas abaixo** do modelo geral; a reponderação já havia sido rejeitada em `v4`.
- Achado novo: `<25` é a **única** faixa com viés de calibração — prevê 13,4% onde
  ocorrem 11,8% (+1,6 pp, IC [+0,4 – +2,9]). Causa: a isotônica é ajustada globalmente.

**A faixa `55-65` também foi diagnosticada — e falha por outro motivo.** Não é
aposentadoria (dentro da faixa, aposentado 0,7488 e ativo 0,7427 são indistinguíveis,
apesar de 68,4% dela ser aposentada) nem perfil de informação (coorte pareada chega a
0,7906, e a faixa é pior em 100% das réplicas). A causa é que os três scores externos
rendem ali o **pior de todas as faixas**. Ali o modelo agrega o normal (0,0744) sobre um
sinal ruim — é deficiência da fonte; no `<25` o modelo agrega o **mínimo** (0,0518), então
falta também no que ele extrai do resto.

A resposta continua sendo de processo: faixa cinza ampliada → revisão humana
(`GET /model/decision-policy`), agora fundamentada num teste válido.

## 📁 Estrutura

```
Dados/            raw_data.csv · clean_data.csv · abt.csv · abt.parquet   (gerados; fora do git)
DataPipeline/     data_sanitization.py · abt_transform.py · to_parquet.py · exp_analysis.ipynb · config.yaml
Model/            train.py · predict.py · metrics_lib.py · derived.py · run_summary.py · evaluation.ipynb · config.yaml
                  diagnostico_idade.py · experimento_teto_idade.py   (diagnóstico da faixa etária)
scripts/          restaurar_improvement_log.py · regenerar_fairness.py
dags/             treino_credit_scoring.py · callables.py   (o DAG de re-treino)
MLOps/            Readme.md · Dockerfile · docker-compose.yml · pipeline_orchestration.py
MLOps/airflow/    Dockerfile · docker-compose.yml · README.md   (a instância do Airflow)
MLOps/app/        api.py · db.py · schemas.py · explain.py · policy.py · routers/ · README.md · requests.http
tests/            test_metrics_lib.py · test_derived.py · test_api.py · test_policy.py ·
                  test_split.py · test_security.py · test_dashboard.py · test_dags.py
artifacts/        model.joblib · metrics.json · curves.json · fairness.json ·
                  feature_importance.json · improvement_log.json · scores.parquet ·
                  diagnostico_idade.json · experimentos/teto_idade.json   (gerados)
```

> A base bruta (>3 GB, 9 CSVs) fica em `data/risco_fraude/home-credit-default-risk/`
> e **não** é versionada. Fonte:
> <https://www.kaggle.com/competitions/home-credit-default-risk>.

## 🚀 Como reproduzir

### Agendado — Airflow, a cada 7 dias

```bash
cd MLOps/airflow
echo "AIRFLOW_UID=$(id -u)" > .env
docker compose up -d --build      # http://localhost:8080  (admin/admin)
```

O DAG `treino_credit_scoring` quebra o pipeline em 9 tasks — cada etapa com log
próprio na interface — e inclui um **gate de qualidade**: se o AUC cair além do
limiar frente à última rodada aceita, a execução falha e os artefatos anteriores
permanecem. Detalhes em [`MLOps/airflow/README.md`](MLOps/airflow/README.md).

### Manual — sem subir infraestrutura

```bash
# 1) Ambiente (Python 3.12+; a rodada oficial reproduz bit a bit em 3.12 e 3.14)
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
Veredito sobre a faixa `<25`: [`docs/diagnostico-faixa-etaria.md`](docs/diagnostico-faixa-etaria.md).
