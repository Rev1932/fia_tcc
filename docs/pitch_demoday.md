# Roteiro do Pitch — Demoday (15 min)

> Guia para ensaiar a apresentação em grupo, defendendo a solução como para uma banca.
> Alinhado aos 5 slides do PPT (`docs/credit_scoring_deck.pptx`) e aos números reais
> produzidos em `DataPipeline/exp_analysis.ipynb` e `Model/evaluation.ipynb`.

## Distribuição do tempo (15 min)

| Tempo | Bloco | Conteúdo |
|---|---|---|
| 0:00–1:00 | Abertura | Gancho + apresentação do grupo |
| 1:00–4:00 | Slide 1 — Problema de negócio | A dor, os dois tipos de erro, por que importa |
| 4:00–6:00 | Slide 2 — EDA | Achados que sustentam o problema |
| 6:00–8:00 | Slide 3 — ABT | Como os dados viram features |
| 8:00–11:00 | Slide 4 — Modelo | Técnica, overfitting, hiperparâmetros |
| 11:00–14:00 | Slide 5 — Avaliação | Performance, explicabilidade, diagnóstico crítico |
| 14:00–15:00 | Fechamento | Resultado de negócio + próximo passo (deploy) |

Regra prática: 1 minuto de sobra é buffer — se atrasar em algum bloco, corte tempo do
fechamento, não da avaliação (é o bloco mais cobrado pela banca).

---

## 1. Abertura (1 min)

Gancho sugerido:
> "Uma financeira de crédito perde dinheiro nos dois lados: aprova quem não paga —
> 8% da carteira entra em default — e nega quem pagaria, perdendo receita e excluindo
> gente do sistema financeiro. Hoje essa decisão depende de scores de bureau que
> **faltam em até 56% dos casos**. A gente construiu um modelo pra resolver isso."

Apresentar o grupo rapidamente (nomes + quem fez o quê, se a banca pedir depois).

## 2. Problema de negócio (3 min)

- Empresa fictícia: financeira de crédito ao consumidor, público sub-bancarizado.
- A dor nos dois extremos: falso negativo (aprovar mau pagador) vs. falso positivo
  (negar bom pagador).
- Por que o problema é difícil: o sinal mais forte (`EXT_SOURCE_1/2/3`, correlação
  ~-0,16 a -0,18 com o target) é o que **menos está disponível** (20–56% de nulos).
- Pergunta de negócio: qual a probabilidade de default e qual threshold maximiza o
  resultado financeiro?
- Métricas de sucesso definidas: técnicas (AUC, KS) **e** de negócio (taxa de
  aprovação, perda evitada via matriz de custo).

## 3. EDA (2 min)

Números para citar de cabeça:
- 307.511 clientes, 124 variáveis na tabela principal, 8,07% de default.
- Risco desigual por segmento: `NAME_INCOME_TYPE` varia de 0% a 36–40%;
  `NAME_EDUCATION_TYPE` de 1,8% a 10,9%.
- Anomalia tratada: `DAYS_EMPLOYED = 365243` (~1000 anos) → NaN; `CODE_GENDER = 'XNA'`
  → NaN.
- Conclusão que direciona a modelagem: desbalanceamento pede AUC/KS/recall como
  métrica primária, não acurácia.

## 4. ABT (2 min)

- 1 linha por `SK_ID_CURR`, agregando as 9 tabelas relacionais → **473 colunas**.
- Agregações de bureau, previous_application, POS_CASH, credit_card, installments
  (mean/sum/max/min/count) + ratios de negócio (credit/income, annuity/income).
- Cuidado explícito com vazamento: só features conhecidas no momento do pedido de
  crédito entram na ABT.
- Validação: `assert` de unicidade de `SK_ID_CURR` (garantia de que a granularidade
  está certa).

## 5. Modelo (3 min)

- Baseline interpretável: Regressão Logística (imputação + scaling + one-hot,
  `class_weight=balanced`). AUC teste = **0,7712**, KS = 0,4062.
- Campeão: LightGBM — categóricas nativas, `is_unbalance=true`, early stopping.
  AUC teste = **0,7846**, KS = **0,4349**.
- Controle de overfitting: AUC treino 0,883 → validação 0,781 → teste 0,785.
  **Validação e teste ficam praticamente empatados** — o gap treino→validação é
  esperado (473 features), mas não vaza para o teste. Early stopping parou na
  iteração 783.
- Por que LightGBM venceu: lida nativamente com nulos (que são estruturais no
  dataset, não erro) e com alta cardinalidade categórica, sem precisar de imputação
  artificial.

## 6. Avaliação (3 min) — bloco mais importante para a banca

Performance:
- AUC 0,785 / KS 0,435 no teste — poder de ordenação suficiente para sustentar uma
  régua de decisão.
- Threshold de negócio calibrado por matriz de custo (custo de aprovar mau pagador =
  10x o custo de negar bom pagador) → taxa de aprovação de ~69–72%.

Explicabilidade (SHAP):
- Traduz o modelo caixa-preta em contribuição por variável, por cliente — essencial
  pra justificar uma negação de crédito (governança).

Diagnóstico crítico — **é aqui que a banca vai testar se vocês entendem o modelo**:
- **Gênero:** AUC quase idêntico entre M (0,783) e F (0,778) — o modelo não perde
  poder discriminativo por gênero. A menor taxa de aprovação para homens (60,6% vs.
  74,3%) reflete a taxa de default real observada (10,2% vs. 7,0%), não é viés
  arbitrário do modelo.
- **Idade:** AUC cai nos extremos — menores de 25 anos (AUC 0,739) e 55–65 anos
  (AUC 0,744) — o modelo é menos confiável justamente nos jovens, que têm a maior
  taxa de default real (11,7%).
- **Thin-file:** 14,3% da base não tem histórico de bureau. AUC menor (0,773 vs.
  0,785) e menos aprovação (57,3% vs. 71,7%) — confirma que menos dado disponível =
  score menos confiável.
- **Mitigação proposta:** revisão humana (human-in-the-loop) para os segmentos de
  AUC mais baixo, em vez de decisão 100% automática — conectado à arquitetura de
  deploy (`MLOps/Readme.md`).

## 7. Fechamento (1 min)

> "Com AUC de 0,785 e uma régua de decisão calibrada por custo, o modelo sustenta uma
> aprovação de ~70% da carteira mantendo o risco sob controle — e sabemos exatamente
> onde ele é mais fraco (jovens e thin-file), então já propomos revisão humana pra
> esses casos. O próximo passo é o deploy como serviço de predição, que é a etapa
> individual de cada um de nós."

---

## Perguntas prováveis da banca (e como responder)

| Pergunta | Resposta-chave |
|---|---|
| "Por que LightGBM e não outro modelo?" | Lida nativamente com nulos e categóricas de alta cardinalidade; testado contra baseline interpretável (regressão logística) para ter referência. |
| "Como vocês sabem que não é overfitting?" | AUC validação (0,781) ≈ AUC teste (0,785) — se fosse overfitting, o teste cairia bem abaixo da validação. Early stopping (783 iterações) foi o mecanismo de controle. |
| "O modelo discrimina por gênero/idade?" | Não por gênero (AUC quase igual, diferença de aprovação reflete risco real medido). Por idade, sim há uma fraqueza real: AUC mais baixo em <25 e 55-65 — assumimos isso como limitação e propomos revisão humana nesses casos. |
| "E os clientes sem histórico de crédito (thin-file)?" | 14,3% da base, AUC menor (0,773 vs 0,785) — mitigado com revisão humana, não decisão 100% automática. |
| "Qual o impacto financeiro real?" | Threshold calibrado por matriz de custo (FN custa 10x mais que FP) resulta em ~70% de aprovação — cada ponto de threshold pode ser traduzido em R$ evitado vs. volume aprovado (ver seção 3 do `evaluation.ipynb`). |
| "Como isso vira produto?" | Etapa individual: API FastAPI + dashboard Streamlit, orquestração via `pipeline_orchestration.py`, infra em `docker-compose`, com monitoramento de drift e ações automatizadas descritas no `MLOps/Readme.md`. |

**Lembrete:** o edital avisa que a banca pode pedir para **modificar o projeto ao
vivo** — revisem `Model/config.yaml` e `Model/train.py` para saber alterar
hiperparâmetros ou o threshold de custo na hora, sem precisar procurar no código.
