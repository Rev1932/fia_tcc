# Documento de TCC — Problema, Objetivo e Método

### Credit Scoring — Home Credit Default Risk · MBA Big Data e Analytics (FIA/LABDATA)

> Base textual consolidada para o PPT (5 slides) e a defesa na banca. Consolida as
> seções 1–3 de `PLAN.md`, com os números reais já produzidos em
> `DataPipeline/exp_analysis.ipynb` e `Model/evaluation.ipynb`.

---

## 1. Problema (Business Understanding)

**Empresa fictícia:** financeira de crédito ao consumidor (perfil Home Credit), atuando
em público sub-bancarizado.

**A dor de negócio:** a decisão de conceder crédito gera perda nos dois extremos:

- **(a) Aprovar quem não paga** — perda direta de capital. Na base analisada, **8,07%**
  dos clientes têm dificuldade de pagamento (`TARGET=1`).
- **(b) Negar quem pagaria** — receita perdida e exclusão financeira.

Hoje a decisão depende fortemente de scores externos de bureau (`EXT_SOURCE_1/2/3`),
que **faltam em 20–56% dos casos** (`EXT_SOURCE_1` sozinho tem 56,4% de nulos) — mesmo
sendo, junto com `EXT_SOURCE_2` e `EXT_SOURCE_3`, uma das variáveis com maior correlação
com o target (~ -0,16 a -0,18). Ou seja: o sinal mais forte é o que menos está disponível.

**Pergunta de negócio:** *Dado um pedido de crédito, qual a probabilidade de o cliente
ter dificuldade de pagamento, e qual ponto de corte maximiza o resultado financeiro
(perda evitada vs. volume aprovado)?*

**Formulação em Machine Learning:** classificação binária supervisionada,
`TARGET ∈ {0,1}`, saída `P(default)` + régua de decisão (threshold) orientada a custo.

**Evidências da EDA que sustentam o problema** (`exp_analysis.ipynb`):

- Base de 307.511 clientes, 124 variáveis na tabela principal.
- Risco desigual por segmento: `NAME_INCOME_TYPE` varia de 0% (estudante/empresário) a
  36–40% (desempregado/licença-maternidade); `NAME_EDUCATION_TYPE` varia de 1,8%
  (doutorado) a 10,9% (ensino fundamental incompleto).
- Nulos concentrados em dados de imóvel (`COMMONAREA_*` ~70%) e em `EXT_SOURCE_1`.

---

## 2. Objetivo

**Objetivo geral:** desenvolver e disponibilizar um modelo de credit scoring que estime
o risco de inadimplência e suporte a decisão de aprovação, com performance e
explicabilidade defensáveis perante uma banca.

**Objetivos específicos:**

1. Construir pipeline reprodutível: `raw → clean → ABT`, agregando as 9 tabelas
   relacionais em ~470 features por cliente.
2. Treinar um baseline interpretável (Regressão Logística) **e** um modelo campeão
   (LightGBM) com controle de overfitting.
3. Avaliar com métricas técnicas e **traduzir em métrica de negócio** (matriz de custo).
4. Explicar o modelo (SHAP) e discutir limitações/governança.
5. Empacotar como serviço de predição (API + dashboard), com infraestrutura
   docker-compose e estratégia de monitoramento.

**Métricas de sucesso:**

- **Técnicas:** AUC-ROC (primária), KS, recall na classe default.
- **Negócio:** taxa de aprovação resultante do threshold de custo, perda esperada
  evitada.

**Resultados obtidos (conjunto de teste):**


| Modelo                            | AUC       | KS        |
| --------------------------------- | --------- | --------- |
| Baseline — Regressão Logística | 0,771     | 0,406     |
| **Campeão — LightGBM**          | **0,785** | **0,438** |

- AUC treino = 0,872 · AUC validação = 0,781 · AUC teste = 0,785 → validação e teste
  praticamente empatados, indicando generalização estável (early stopping na iteração
  654 evitou overfitting descontrolado, apesar do gap natural treino→validação).
- Threshold de negócio calibrado em **0,50**, com **taxa de aprovação de 72,1%**, usando
  matriz de custo (custo falso negativo = 1,0 · custo falso positivo = 0,10 — aprovar um
  mau pagador custa 10x mais que negar um bom pagador).

---

## 3. Método (CRISP-DM)

### 3.1 Data Understanding (EDA) — `DataPipeline/exp_analysis.ipynb`

Perfil da base, distribuição do target, nulos, análise por segmento e poder preditivo
dos scores externos. Conclusões: desbalanceamento ~8% → priorizar AUC/KS/recall;
nulos relevantes → tratamento nativo no LightGBM; EXT_SOURCE forte mas com baixa
cobertura → reforçar com a ABT.

### 3.2 Data Preparation — `data_sanitization.py` + `abt_transform.py`

- **Limpeza:** sentinela `DAYS_EMPLOYED=365243` → NaN; `CODE_GENDER='XNA'` → NaN;
  clip de outliers de renda no percentil 99,9%; variáveis derivadas legíveis
  (`AGE_YEARS`, `YEARS_EMPLOYED`).
- **ABT:** 1 linha por `SK_ID_CURR`, agregando bureau (+bureau_balance),
  previous_application, POS_CASH, credit_card e installments com
  mean/sum/max/min/count, mais ratios de negócio (credit/income, annuity/income,
  credit_term, employed/age). Validação: `assert` de unicidade de `SK_ID_CURR`.
- Tudo parametrizado via `DataPipeline/config.yaml`.

### 3.3 Modeling — `Model/train.py`

Split estratificado 60/20/20 (treino/validação/teste). Baseline com imputação +
scaling + one-hot + `class_weight=balanced`. Campeão LightGBM com categóricas
nativas, `is_unbalance=true`, `learning_rate=0,02`, `num_leaves=34`, `max_depth=8`,
regularização L1/L2 e early stopping (100 rounds de paciência).

### 3.4 Evaluation — `Model/evaluation.ipynb`

ROC, Precision-Recall, KS, matriz de confusão no threshold de negócio, importância de
variáveis (gain) e **SHAP** (summary plot) para explicabilidade caso a caso —
essencial para governança de crédito.

### 3.5 Deployment / MLOps

`Model/predict.py` expõe a função de predição; `MLOps/app/api.py` (FastAPI) e
`MLOps/app/streamlit_app.py` (dashboard); `MLOps/pipeline_orchestration.py` orquestra
`raw → clean → abt → train` (compatível com Airflow); `MLOps/docker-compose.yml` sobe
API (8000) e dashboard (8501). Estratégia de monitoramento e ações automatizadas
detalhadas em `MLOps/Readme.md`.

---

## 4. Diagnóstico crítico — pontos de atenção para a banca

> Calculado em `Model/evaluation.ipynb` (seções 6 e 7), com o modelo campeão sobre o
> conjunto de teste.

- **Overfitting:** controlado — AUC treino 0,883 vs. validação 0,781 vs. teste 0,785;
  validação e teste praticamente empatados (early stopping na iteração 783).
- **Viés por gênero:** AUC muito próximo entre M (0,7834) e F (0,7783) — o modelo
  discrimina risco igualmente bem nos dois grupos. A taxa de aprovação é menor para
  homens (60,6% vs. 74,3%), mas acompanha a taxa de default real observada (10,2% vs.
  7,0%) — a diferença reflete risco real, não viés injustificado.
- **Viés por idade:** AUC cai nos extremos — clientes com menos de 25 anos (AUC 0,739)
  e a faixa 55-65 (AUC 0,744) têm poder de ordenação mais fraco que a faixa 35-55
  (AUC ~0,79). O modelo é menos confiável exatamente para o segmento mais jovem, que
  também tem a maior taxa de default real (11,7%).
- **Desempenho no grupo thin-file:** 14,3% da base de teste não tem nenhum registro em
  `bureau` (`BUREAU_COUNT` nulo). Esse grupo tem AUC menor (0,773 vs. 0,785 do restante)
  e taxa de aprovação mais baixa (57,3% vs. 71,7%), apesar de default real mais alto
  (10,1% vs. 7,7%) — confirma que menos sinal disponível deixa o score menos confiável.
- **Cenários de falha:** concentrar decisões 100% automáticas nos segmentos com AUC mais
  baixo (jovens <25, thin-file) é o principal risco operacional — recomenda-se rota de
  revisão humana para esses casos em vez de aprovação/negação totalmente automática.

---

## 5. Referências

- Dataset: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)
- Repositório: estrutura conforme `PLAN.md` e `OKR.md`
