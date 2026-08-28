# API de Credit Scoring — referência

Serviço que entrega os dados da análise de risco de crédito por meio de
endpoints especializados e filtráveis.

- **Swagger interativo:** <http://localhost:8000/docs>
- **ReDoc:** <http://localhost:8000/redoc>
- **Coleção pronta para executar:** [`requests.http`](requests.http)

---

## 1. Por que a API tem esta forma

A versão anterior tinha dois endpoints (`GET /health`, `POST /predict`), nenhum
filtro, nenhum parâmetro de consulta e **nenhum acesso aos dados da análise**.
Para responder "qual a inadimplência por escolaridade?" era preciso abrir um
notebook.

Esta versão troca o método único por **quatro famílias de endpoints**, todas
combináveis por filtro:

| Família | Para quê |
|---|---|
| `/clients` | consultar a carteira: filtros, ficha e score |
| `/stats` | a análise exploratória servida por SQL |
| `/model` | métricas congeladas e recálculos ao vivo |
| `/predict`, `/simulate`, `/explain` | pontuar, simular cenários e explicar decisões |

**Regra de ouro do projeto:** nenhum número em slide, documento ou notebook
pode divergir de `GET /model/metrics`. Tudo sai dos artefatos da rodada
congelada em `artifacts/`.

---

## 2. Arquitetura

```
Dados/abt.parquet ──┐
                    ├── DuckDB (views abt, scores, clients) ── /clients  /stats
artifacts/          │
  scores.parquet ───┘
  model.joblib ──────── LightGBM + calibrador isotônico ────── /predict  /simulate
                        └─ TreeExplainer (SHAP) ────────────── /explain
  metrics.json    ─┐
  curves.json      ├── leitura direta ───────────────────────── /model/*
  fairness.json    │
  improvement_log ─┘
```

**DuckDB sobre Parquet**, e não a base em memória: a ABT tem 307.511 clientes ×
1.020 colunas. O DuckDB lê direto do disco e faz *pushdown* de projeção e
filtro — pedir 12 colunas custa 12 colunas, e o serviço sobe em segundos com
uso de memória baixo.

---

## 3. Como subir

```bash
# local
uvicorn MLOps.app.api:app --host 0.0.0.0 --port 8000

# docker
docker compose -f MLOps/docker-compose.yml up --build
```

Pré-requisitos (nesta ordem):

```bash
python MLOps/pipeline_orchestration.py   # gera Dados/abt.csv e artifacts/
python DataPipeline/to_parquet.py        # gera Dados/abt.parquet + abt_profile.json
```

Todos os caminhos são sobrescrevíveis por variável de ambiente — é como o
`docker-compose.yml` aponta o serviço para os volumes montados:

| Variável | Padrão |
|---|---|
| `HC_ABT_PARQUET` | `Dados/abt.parquet` |
| `HC_SCORES_PARQUET` | `artifacts/scores.parquet` |
| `HC_MODEL_PATH` | `artifacts/model.joblib` |
| `HC_ARTIFACTS_DIR` | `artifacts/` |
| `HC_EXPLAINER_EAGER` | `1` (constrói o SHAP no startup) |
| `HC_DUCKDB_THREADS` / `HC_DUCKDB_MEMORY_LIMIT` | `4` / `2GB` |

---

## 4. Perguntas da banca → chamada da API

A tabela para deixar aberta durante a defesa.

| Pergunta | Chamada |
|---|---|
| "E se aprovar um mau pagador custasse 20× em vez de 10×?" | `GET /model/threshold-analysis?cost_fn=20&cost_fp=1` |
| "Vocês acharam onde o modelo falha — e o que fizeram?" | `GET /model/improvements` |
| "O modelo discrimina por gênero?" | `GET /model/fairness?by=gender` |
| "E por idade?" | `GET /model/fairness?by=age_band` |
| "Mostra um cliente negado e o porquê" | `GET /clients?decision=NEGAR&page_size=1` → `GET /clients/{id}/explain` |
| "Se a renda dele dobrasse, aprovaria?" | `POST /simulate` com `changes` |
| "A ABT das 9 tabelas valeu a pena?" | `GET /model/feature-importance` → campo `by_source` |
| "Qual a inadimplência por escolaridade?" | `GET /stats/default-rate?by=education` |
| "Quantos thin-file existem e qual o risco deles?" | `GET /stats/default-rate?by=thin_file` |
| "O score é uma probabilidade de verdade?" | `GET /model/calibration` |
| "Quantos casos vão para revisão humana?" | `GET /model/decision-policy` |
| "Qual a matriz de confusão no corte escolhido?" | `GET /model/confusion-matrix` |
| "Esse número do slide bate com o modelo em disco?" | `GET /model/metrics` |

---

## 5. Convenções

### Paginação
Toda listagem devolve o mesmo envelope:
```json
{"meta": {"page": 1, "page_size": 50, "total": 3419,
          "total_pages": 69, "has_next": true, "has_prev": false},
 "items": [ ... ]}
```

### Erros
```json
{"error": {"code": "BAD_REQUEST", "message": "Coluna desconhecida: 'XPTO'. Veja GET /meta/columns.", "detail": null}}
```

| Código | Quando |
|---|---|
| `400` | coluna, dimensão ou feature fora da whitelist |
| `404` | cliente ou artefato inexistente |
| `422` | parâmetro com tipo/faixa inválidos (inclui erro de digitação no nome do filtro) |
| `503` | modelo ou dados não carregados — a mensagem diz qual comando rodar |

### Decisão
O score vira decisão em **três faixas** (ver §7):
`APROVAR` · `REVISAR` (análise humana) · `NEGAR`.

---

## 6. Filtros

Aceitos por `GET /clients` **e por todos os `/stats/*`**. É o que permite
perguntar "inadimplência por escolaridade, só entre thin-file com menos de
25 anos" numa chamada só.

| Filtro | Tipo | Observação |
|---|---|---|
| `age_min` / `age_max` | número | `AGE_YEARS` |
| `income_min` / `income_max` | número | renda anual |
| `credit_min` / `credit_max` | número | valor do crédito |
| `annuity_min` / `annuity_max` | número | valor da parcela |
| `children_min` / `children_max` | inteiro | |
| `employed_years_min` / `_max` | número | |
| `gender` | `M` \| `F` | |
| `contract_type`, `education`, `income_type`, `family_status`, `housing_type`, `occupation` | texto | vários valores separados por **vírgula** |
| `score_min` / `score_max` | 0–1 | P(default) |
| `decision` | `APROVAR` \| `NEGAR` | aplica o threshold vigente |
| `target` | `0` \| `1` | rótulo real de inadimplência |
| `thin_file` | booleano | `true` = sem nenhum registro no bureau |
| `split` | `train` \| `valid` \| `test` | |

Um nome de filtro escrito errado devolve **422** em vez de ser ignorado em
silêncio.

### Dimensões de agrupamento (`?by=`)
`gender` · `contract_type` · `education` · `family_status` · `income_type` ·
`housing_type` · `occupation` · `organization` · `children` · `split` ·
`target` · `age_band` · `income_band` · `credit_band` · `score_band` ·
`thin_file`

Lista viva em `GET /meta/dimensions`.

---

## 7. Endpoints

### Saúde e metadados

#### `GET /health`
Devolve **503** quando o modelo ou os dados não carregaram — e é isso que faz o
healthcheck do `docker-compose` ter significado. A versão anterior respondia
200 mesmo sem modelo, então o container ficava `healthy` sem conseguir pontuar
nada.

```bash
curl -s localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true, "data_loaded": true,
 "run_id": "20260822-154818-d73d85c", "threshold": 0.09,
 "n_features": 1018, "n_clients": 307511, "errors": {}}
```

#### `GET /meta/columns` · `GET /meta/dimensions` · `POST /admin/reload`
Catálogo de colunas (com % de nulos), dimensões aceitas em `?by=`, e recarga
do modelo sem reiniciar o serviço — útil depois de um re-treino ao vivo.

---

### Clientes

#### `GET /clients`
Filtros + `page`, `page_size` (≤500), `sort`, `order`, `fields`.

```bash
curl -s "localhost:8000/clients?age_min=20&age_max=25&thin_file=true&page_size=3"
curl -s "localhost:8000/clients?fields=SK_ID_CURR,AMT_CREDIT,proba_champion&sort=proba_champion&order=desc"
```

`?fields=` aceita qualquer uma das 1.020 colunas da ABT (até 60 por chamada) e
é validado contra a whitelist derivada do próprio esquema.

#### `GET /clients/{id}`
Ficha organizada em blocos (identificação, financeiro, histórico, scores
externos) + o score. `?include=all` acrescenta as 1.018 features do modelo.

#### `GET /clients/{id}/score`
`?recompute=true` recalcula pelo modelo em memória e devolve `agreement_error`
comparando com o valor do lote — **prova ao vivo que o artefato em disco e o
modelo servido são o mesmo modelo**.

---

### Estatísticas (a EDA por SQL)

| Endpoint | Devolve |
|---|---|
| `GET /stats/overview` | KPIs: nº de clientes, inadimplência, % thin-file, cobertura de EXT_SOURCE, taxa de aprovação |
| `GET /stats/default-rate?by=` | n, inadimplência, aprovação, score médio e **lift** por segmento |
| `GET /stats/distribution?feature=` | histograma (numérica) ou contagem (categórica), com `by_target=true` |
| `GET /stats/missing` | nulos por coluna, servido do perfil pré-computado |
| `GET /stats/crosstab?rows=&cols=` | tabela cruzada entre duas dimensões |

```bash
curl -s "localhost:8000/stats/default-rate?by=education"
```
Reproduz o corte da EDA: de **1,83%** (doutorado) a **10,93%** (fundamental
incompleto).

---

### Modelo

| Endpoint | Devolve |
|---|---|
| `GET /model/metrics` | **a fonte única de verdade**. O bloco `served` traz as métricas do modelo que a API de fato entrega — é dele que sai todo número de capa |
| `GET /model/roc` · `GET /model/ks` | curvas e tabela de decis |
| `GET /model/calibration` | curva de confiabilidade e Brier antes/depois |
| `GET /model/feature-importance` | ranking + `by_source` (importância por tabela de origem) |
| `GET /model/threshold-analysis` | **recalcula a régua de custo ao vivo** |
| `GET /model/confusion-matrix` | matriz em qualquer threshold |
| `GET /model/fairness?by=` | desempenho por segmento, **com intervalo de confiança** |
| `GET /model/improvements` | antes e depois das correções, por segmento |
| `GET /model/decision-policy` | as três faixas e quantos clientes caem em cada uma |
| `GET /model/psi` | estabilidade populacional entre duas fatias — o alerta de drift |

#### `GET /model/threshold-analysis` — o mais útil numa arguição

Varre os thresholds sobre `artifacts/scores.parquet` e aponta o ótimo para
**qualquer** par de custos, **sem re-treinar nada** (responde em milissegundos):

```bash
curl -s "localhost:8000/model/threshold-analysis?cost_fn=20&cost_fp=1"
```

| custo FN : FP | threshold ótimo | aprovação |
|---|---|---|
| 1 : 0,1 (10×) | 0,09 | 68,7% |
| 5 : 1 (5×) | 0,16 | 87,6% |
| 20 : 1 (20×) | 0,04 | 49,8% |

#### `GET /model/psi` — monitoramento de drift

Mede o quanto a distribuição de cada variável mudou em relação a uma fatia de
referência, com a leitura fixa de mercado:

| PSI | Leitura |
|---|---|
| < 0,10 | estável |
| 0,10 – 0,25 | atenção — investigar a fonte |
| > 0,25 | mudança relevante — o modelo pode ter deixado de valer |

```bash
# o score é o sinal de drift mais importante: resume todas as variáveis
curl -s "localhost:8000/model/psi?referencia=train&comparado=test"

# variáveis específicas
curl -s "localhost:8000/model/psi?features=AGE_YEARS,EXT_SOURCE_2,AMT_CREDIT"
```

Entre `train` e `test` o PSI dá ~0,0004 — o esperado, já que são a mesma safra.
Isso confirma que o split não introduziu viés. **O valor é operacional:** em
produção basta apontar `comparado` para a safra nova, e o mesmo cálculo vira o
alerta de drift descrito em `MLOps/Readme.md`.

#### `GET /model/fairness` — como uma fraqueza é declarada

Cada grupo vem com IC bootstrap do AUC e com `vs_referencia`: o IC da **diferença**
entre o AUC do grupo e o dos demais grupos do mesmo eixo. É esse o critério, exposto
em `fraqueza_confirmada`.

O campo `overlaps_overall` compara com o AUC **geral** e está **deprecado**: o grupo é
subconjunto do geral, e o geral é composto em sua maior parte por pares entre grupos —
comparações que nenhum AUC intra-grupo realiza. A resposta traz também `decomposicao`,
que mostra qual fração dos pares do AUC agregado é intra-grupo (22,4% no eixo etário).

No modelo atual, `<25` (−0,0514, p = 0,008) e `55-65` (−0,0415, p = 0,004) são as
fraquezas confirmadas. `65+`, thin-file e gênero não são. Veredito sobre o `<25` em
`docs/diagnostico-faixa-etaria.md`.

---

### Predição, simulação e explicabilidade

#### `POST /predict`
Aceita payload parcial (o que faltar vira nulo, que o LightGBM trata
nativamente) e informa `coverage` e `unknown_features` — campos desconhecidos
não são mais descartados em silêncio. As variáveis derivadas
(`CREDIT_INCOME_RATIO`, `EXT_SOURCE_MEAN`, ...) são recalculadas a partir do
que veio.

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "records": [{"NAME_CONTRACT_TYPE":"Cash loans","AMT_INCOME_TOTAL":150000,
               "AMT_CREDIT":500000,"AMT_ANNUITY":25000,"EXT_SOURCE_2":0.55}]}'
```

#### `POST /simulate`
Parte de um cliente real ou de um payload livre, aplica `changes` e devolve
antes/depois. **As derivadas são propagadas**: mudar `AMT_CREDIT` move também
`CREDIT_INCOME_RATIO` e `CREDIT_TERM`, como aconteceria numa proposta real.

```bash
curl -s -X POST localhost:8000/simulate -H 'Content-Type: application/json' \
  -d '{"sk_id_curr": 100002, "changes": {"AMT_CREDIT": 900000}}'

# varredura de uma variável inteira
curl -s -X POST localhost:8000/simulate -H 'Content-Type: application/json' \
  -d '{"sk_id_curr": 100002, "sweep": {"feature":"EXT_SOURCE_2","start":0,"stop":1,"steps":11}}'
```

#### `POST /predict/csv`
Pontuação em lote no formato que uma mesa de crédito usa de fato: exporta a fila
do dia, pontua, reimporta. Devolve o CSV original acrescido de
`probability_default`, `decision` e `score_band`, preservando as colunas que
vieram. Limite de 50 mil linhas e 20 MB.

```bash
curl -X POST localhost:8000/predict/csv -F 'arquivo=@fila.csv' -o pontuado.csv
```

Cabeçalhos da resposta trazem `X-Linhas-Pontuadas`, `X-Features-Reconhecidas`
e `X-Threshold`.

#### `GET /clients/{id}/explain`
Contribuições SHAP em log-odds, separadas em fatores de risco e favoráveis,
com **`consistency_check`**: `base_value + Σ shap` reconstrói a probabilidade
do modelo (erro < 1e-9). É a prova de que a explicação é fiel, e não uma
aproximação. Traz também uma narrativa pronta em português — o texto que iria
ao analista de crédito junto com a decisão.

> O SHAP explica o modelo **cru**; o score servido passa pela calibração
> isotônica. Como a isotônica é monotônica, a ordem e o sinal das contribuições
> continuam válidos. `raw_probability` traz o valor antes da calibração.

---

## 8. A régua de três faixas

Um corte único trata como iguais uma probabilidade de 8,9% e uma de 9,1%,
quando o modelo não tem essa precisão. Em vez disso:

```
APROVAR    p <  threshold − largura/2
REVISAR    faixa cinza  → análise humana
NEGAR      p >= threshold + largura/2
```

A **largura não é fixa**: dobra nos segmentos cujo IC de AUC ficou
comprovadamente abaixo do geral. Quanto menos o modelo enxerga aquele perfil,
mais casos vão para um humano — em vez de fingir a mesma confiança.

Os segmentos entram nessa lista **por medição**, não por opinião:
`GET /model/decision-policy` mostra quais são e por quê.

> Diferenciar o **critério de risco** por gênero ou idade seria discriminação
> direta e não é feito. O que varia é **quanto vai para revisão humana**.

---

## 9. Segurança

- Todo **valor** de filtro vai parametrizado (`?`); nenhum texto de usuário
  entra na string SQL.
- Todo **identificador** (coluna em `fields`/`sort`, dimensão em `by`, feature
  em `feature`) é validado contra whitelist derivada de `DESCRIBE clients`.
- Fora da whitelist → `400` com a indicação de onde ver os valores válidos.
- Coberto por `tests/test_security.py`.

Não há autenticação: é um projeto acadêmico, servindo dados públicos do Kaggle.
Em produção entrariam autenticação, rate limit e CORS restrito.

---

## 10. Testes

```bash
pytest -q
```

A suíte **não depende dos 1,3 GB da ABT**: uma fixture gera uma base sintética
de 300 clientes com os nomes de coluna reais, treina um LightGBM minúsculo e
escreve artefatos de brinquedo no mesmo formato. Isso é possível porque todos
os caminhos vêm de variável de ambiente — o mesmo mecanismo do compose.

| Arquivo | Cobre |
|---|---|
| `test_metrics_lib.py` | trava a semântica de `business_threshold` e `ks_statistic` |
| `test_derived.py` | as fórmulas do what-if batem com o que o pipeline gravou |
| `test_api.py` | saúde, filtros, paginação, métricas, predição, simulação, SHAP |
| `test_security.py` | injeção de SQL por `sort`, `fields` e `by` |
