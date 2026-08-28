# Documento de TCC — Problema, Objetivo e Método

### Credit Scoring — Home Credit Default Risk · MBA Big Data e Analytics (FIA/LABDATA)

> **Números desta versão:** rodada canônica `20260828-003844`, gerada de
> `artifacts/`. Reimprima com `python Model/run_summary.py --markdown`.
> Nenhum número aqui é digitado à mão.

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
   relacionais em 1.018 features por cliente.
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
| Baseline — Regressão Logística | 0,7776     | 0,4228     |
| **Campeão — LightGBM (servido)** | **0,7868** | **0,4342** |

- AUC treino = 0,8753 · AUC validação = 0,7835 · AUC teste = 0,7871 → validação e teste
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

## 4. Diagnóstico crítico — o que foi medido, corrigido e o que restou

> Calculado em `Model/train.py` (que grava `artifacts/fairness.json`) sobre o conjunto
> de teste, com **intervalo de confiança bootstrap** para cada segmento.

### 4.1 Overfitting — controlado

AUC treino **0,8753** → validação **0,7835** → teste **0,7871**.
O gap treino→validação é esperado com mais de mil variáveis; o que importa é que
**validação e teste empatam**. Early stopping parou na iteração **507** de 2.000.

### 4.2 Fraqueza real vs. ruído amostral

Um AUC menor num grupo pequeno pode ser apenas tamanho de amostra. Por isso cada segmento
vem com IC bootstrap. E a comparação é contra os **demais grupos do mesmo eixo**, nunca
contra o AUC geral: o segmento é subconjunto do geral, e o geral é composto 77,6% por
pares formados entre faixas etárias diferentes — comparações que nenhum AUC intra-faixa
realiza. Um grupo é fraqueza quando o IC da **diferença** exclui o zero.

| Segmento | n | AUC | IC 95% | Δ vs. demais | IC da diferença | p | Fraqueza? | Aprovação | Inadimpl. |
|---|---|---|---|---|---|---|---|---|---|
| Homens | 20.940 | 0,7872 | [0,7778–0,7963] | +0,0078 | [−0,0066 – +0,0195] | 0,240 | não | 60,4% | 10,17% |
| Mulheres | 40.561 | 0,7795 | [0,7709–0,7886] | −0,0078 | [−0,0195 – +0,0066] | 0,240 | não | 73,4% | 6,99% |
| < 25 anos | 2.355 | **0,7319** | [0,7012–0,7597] | **−0,0514** | [−0,0827 – −0,0205] | 0,008 | **sim** | 41,5% | 11,80% |
| 55–65 anos | 12.166 | **0,7465** | [0,7281–0,7672] | **−0,0415** | [−0,0619 – −0,0209] | 0,004 | **sim** | 79,9% | 5,61% |
| 65+ anos | 1.725 | 0,7631 | [0,7084–0,8100] | −0,0198 | [−0,0696 – +0,0331] | 0,395 | não | 89,9% | 3,71% |
| Thin-file | 8.776 | 0,7745 | [0,7577–0,7899] | −0,0133 | [−0,0299 – +0,0043] | 0,132 | não | 56,5% | 10,14% |

**Gênero:** o poder de ordenação é praticamente idêntico entre os grupos. A aprovação
menor para homens acompanha a inadimplência real observada — o modelo **discrimina
risco, não pessoas**.

**Idade:** aqui há fraqueza genuína, nos dois extremos, e ela sobrevive ao critério
correto. As duas foram diagnosticadas nesta rodada (ver 4.4), e **falham por motivos
diferentes**: em `55-65` o modelo agrega o normal sobre um score externo ruim — é
deficiência da fonte; em `<25` o sinal é ruim **e** o modelo agrega o mínimo de todas as
faixas.

**Thin-file:** ao contrário do que versões anteriores deste trabalho afirmavam, a
diferença **não é estatisticamente distinguível** do modelo geral. O IC sobrepõe.

### 4.3 Calibração

Com `is_unbalance=true` o score ordenava bem mas **não era P(inadimplência) real** — o
corte ótimo caía em 0,47, um número indefensável como probabilidade. Uma regressão
isotônica ajustada em fatia exclusiva corrigiu isso:

| | Antes | Depois |
|---|---|---|
| Brier | 0,1668 | **0,0658** |
| Ponto de corte | 0,47 | **0,09** |
| AUC | 0,7871 | 0,7868 |

O AUC não muda porque a isotônica é monotônica: ela corrige a probabilidade, não a
ordenação. Verificação independente: a média do score em toda a base é 0,081, contra
inadimplência real de 8,07%.

### 4.4 A faixa `<25`: diagnóstico e veredito

As correções da ABT melhoraram o modelo geral, mas **a faixa abaixo de 25 anos
praticamente não se moveu** (0,7364 → 0,7358 → 0,7319 em três rodadas, enquanto as faixas
do meio subiram). Um AUC congelado enquanto o número de variáveis dobra é assinatura de
teto de informação. Esta rodada testou as causas candidatas em vez de supô-las.

**A explicação que este trabalho vinha dando estava errada.** Dizíamos que "é o grupo com
menos histórico por definição, e histórico é o insumo que falta". Testável: construiu-se
uma coorte de clientes de **25 a 45 anos** reamostrada estrato a estrato até reproduzir o
perfil de informação do jovem — mesma taxa de thin-file, mesmo número de scores externos
presentes, mesma faixa de tempo de emprego, mesmo comprimento de histórico. 71 estratos,
99,9% de cobertura.

| Coorte pareada por | AUC | IC 95% | AUC `<25` | Δ | Réplicas em que `<25` é pior |
|---|---|---|---|---|---|
| perfil de informação | 0,7803 | [0,7527–0,8096] | 0,7319 | −0,0484 | 99,8% |
| perfil de informação + nível do score externo | 0,7627 | [0,7378–0,7876] | 0,7319 | −0,0308 | 98,8% |

Dê a um cliente de 33 anos exatamente a pobreza informacional de um de 22 e o modelo
continua ordenando-o normalmente. **Escassez de histórico não explica o buraco.** Cerca de
36% do efeito é estar na região baixa do score externo — onde o modelo separa pior em
qualquer idade — e não ser jovem. Restam −0,0308 específicos da idade.

**Não há conserto disponível.** Além da reponderação já rejeitada, seis variantes de
modelo dedicado (`<25` e `<30`, três capacidades cada, mesma partição, mesmos 2.355
clientes de teste) ficam **todas abaixo** do modelo geral: a melhor marca 0,7296 contra
0,7319. O modelo global já está no teto do segmento para este conjunto de variáveis.

**Achado novo:** `<25` é a única faixa com viés de calibração significativo — prevê 13,4%
onde ocorrem 11,8% (+1,6 pp, IC [+0,4; +2,9]), por a isotônica ser ajustada globalmente.
Corrigível, mas o conserto não melhora Brier nem AUC e **reduz** a aprovação da faixa de
47,6% para 44,8%: corrige contabilidade de carteira, não decisão individual.

A resposta continua sendo de processo: **régua de três faixas**, com a faixa cinza dobrada
nos segmentos de baixa confiança medida, encaminhando esses casos a análise humana com o
relatório SHAP (`GET /model/decision-policy`). É a conduta certa diante de um teto de dado
medido. Detalhamento completo em `docs/diagnostico-faixa-etaria.md`.

**A faixa `55-65`.** Diagnosticada com o mesmo instrumental e com causa distinta. A
hipótese natural era a aposentadoria: 68,4% da faixa é aposentada (contra 0,4–7,8% nas do
meio) e a sanitização anula o bloco de emprego desse público. Refutada — dentro da faixa,
aposentado (0,7488) e ativo (0,7427) são indistinguíveis, e os aposentados de `45-55` vão
acima da média (0,8065). Coorte pareada por perfil de informação atinge 0,7906 contra
0,7465, com a faixa pior em 100% das réplicas. A causa medida é que os três scores
externos rendem ali o pior de todas as faixas (0,6288 / 0,6301 / 0,6506). O ganho que o
modelo acrescenta sobre o sinal bruto é **normal** (0,0744): ele parte de uma base pior,
não trabalha pior. É limitação da fonte de dado, sem conserto de modelagem disponível.

### 4.5 Cenários de falha

Decidir 100% automaticamente nos segmentos de AUC comprovadamente menor é o principal
risco operacional. É exatamente o que a régua de três faixas evita.

---

## 5. Referências

- Dataset: [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/competitions/home-credit-default-risk)
- Repositório: estrutura conforme `PLAN.md` e `OKR.md`
