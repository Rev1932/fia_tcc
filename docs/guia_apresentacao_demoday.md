# Guia de Apresentação — Demoday (Cola + Discurso)

> Versão aprofundada do `pitch_demoday.md`: explica cada termo técnico como você deve
> falar para a banca, traz o discurso pronto para memorizar bloco a bloco, e uma cola
> de números para revisar 10 minutos antes de subir no palco.

---

## PARTE 1 — Glossário técnico: o que é e como explicar em voz alta

A regra aqui: para cada termo, memorize a **definição de 1 frase** (para usar no meio
do discurso) e entenda a **explicação completa** (para quando a banca perguntar).

### Termos de negócio

**Default (inadimplência)**
- *1 frase:* "Default é quando o cliente pega o crédito e não paga."
- *Completo:* No dataset, `TARGET = 1` significa que o cliente teve dificuldade de
  pagamento. 8,07% da base é default — ou seja, de cada 100 clientes aprovados no
  histórico, 8 não pagaram.

**Falso negativo vs. falso positivo (os dois erros)**
- *1 frase:* "Falso negativo é aprovar quem não paga; falso positivo é negar quem pagaria."
- *Completo:* O modelo prevê risco. Errar para um lado custa o crédito inteiro
  (aprovou, cliente deu default). Errar para o outro custa a margem/receita (negou um
  bom pagador) e ainda exclui gente do sistema financeiro. Os dois erros têm custos
  diferentes — por isso não basta acertar "na média", é preciso pesar cada erro.

**Bureau de crédito**
- *1 frase:* "Bureau é a agência externa de crédito, tipo Serasa — dá um score pronto do cliente."
- *Completo:* As variáveis `EXT_SOURCE_1/2/3` são scores de bureaus externos. São o
  sinal mais forte do dataset, mas faltam em 20% a 56% dos casos — exatamente porque
  o público é sub-bancarizado. Esse é o coração do problema: **o melhor dado é o que
  menos existe**.

**Thin-file**
- *1 frase:* "Thin-file é o cliente sem histórico de crédito — a 'pasta fina', sem registro em bureau."
- *Completo:* 14,3% da nossa base não tem nenhum registro de bureau. Para esses, o
  modelo tem menos informação. O AUC é menor (0,7745 vs 0,7871), mas o
  intervalo de confiança sobrepõe o geral — a diferença NÃO está estabelecida (AUC 0,773 vs 0,7871
  no geral). É o público que mais precisa de crédito e o mais difícil de avaliar.

**Threshold (régua de decisão / ponto de corte)**
- *1 frase:* "Threshold é a nota de corte: acima dessa probabilidade de default, negamos; abaixo, aprovamos."
- *Completo:* O modelo devolve uma probabilidade (ex.: 0,12 = 12% de chance de
  default). Alguém precisa decidir onde cortar. Nós não escolhemos o corte "no olho":
  calibramos pela **matriz de custo**.

**Matriz de custo**
- *1 frase:* "É colocar preço em cada erro: aprovar um mau pagador custa 10x mais do que negar um bom pagador."
- *Completo:* No `config.yaml`: `cost_false_negative = 1.0` e `cost_false_positive =
  0.10` — razão de 10 para 1. Testamos todos os thresholds possíveis e escolhemos o
  que minimiza o custo total esperado. Resultado: taxa de aprovação de ~69–72% da
  carteira. Isso transforma métrica técnica em decisão financeira.

**Taxa de aprovação**
- *1 frase:* "É o percentual da carteira que o modelo aprova com o threshold escolhido — no nosso caso, ~70%."

### Termos de dados

**EDA (Exploratory Data Analysis / Análise Exploratória)**
- *1 frase:* "EDA é a etapa de conhecer os dados antes de modelar: distribuições, nulos, anomalias e relação com o target."
- *Completo:* Foi na EDA que descobrimos o desbalanceamento (8% de default), os
  segmentos de risco muito desigual e as anomalias que precisavam de tratamento.

**Anomalia `DAYS_EMPLOYED = 365243`**
- *1 frase:* "Era um código de 'não se aplica' disfarçado de número — 365 mil dias seria ~1000 anos de emprego."
- *Completo:* Valor sentinela usado para pensionistas/desempregados. Se deixado como
  número, o modelo aprenderia um padrão falso. Convertido para NaN (nulo explícito).
  Mesma lógica para `CODE_GENDER = 'XNA'` (gênero não informado).

**ABT (Analytical Base Table)**
- *1 frase:* "ABT é a tabela final de modelagem: uma linha por cliente, com todas as features consolidadas — 1.020 colunas."
- *Completo:* Os dados brutos vêm em 9 tabelas relacionais (cadastro, bureau,
  pedidos anteriores, parcelas, cartão...). Um cliente pode ter 20 empréstimos
  anteriores — mas o modelo precisa de **uma linha por cliente**. A ABT resolve isso
  agregando o histórico (média, soma, máximo, mínimo, contagem) e somando ratios de
  negócio como crédito/renda.

**Granularidade**
- *1 frase:* "Granularidade é 'o que cada linha representa' — na nossa ABT, cada linha é exatamente um cliente."
- *Completo:* Garantida por `assert` de unicidade de `SK_ID_CURR` (o ID do cliente).
  Se houvesse duplicata, o mesmo cliente contaria duas vezes no treino — erro clássico.

**Vazamento de dados (data leakage)**
- *1 frase:* "Vazamento é deixar o modelo ver informação do futuro — dados que não existiriam no momento da decisão."
- *Completo:* Exemplo: usar o comportamento de pagamento *do próprio empréstimo* para
  prever se ele será pago. O modelo pareceria ótimo no papel e falharia em produção.
  Nossa regra: **só entra na ABT o que é conhecido no momento do pedido de crédito**.

**Feature / feature engineering**
- *1 frase:* "Feature é cada variável de entrada do modelo; feature engineering é criar variáveis novas com conhecimento de negócio."
- *Completo:* Ex.: `credit/income` (quanto o crédito pesa na renda) e
  `annuity/income` (quanto a parcela pesa na renda) — variáveis que não existem
  cruas, mas carregam a lógica de comprometimento de renda que um analista usaria.

### Termos de modelagem

**Regressão Logística (baseline)**
- *1 frase:* "É o modelo estatístico clássico de crédito: simples, linear e interpretável — nossa régua de comparação."
- *Completo:* Serve como *baseline*: se o modelo complexo não bater o simples, o
  complexo não se justifica. Exigiu preparação: imputação de nulos, scaling
  (padronizar escalas) e one-hot encoding (transformar categorias em colunas 0/1).
  Resultado: AUC 0,7776.

**`class_weight=balanced` / `is_unbalance=true`**
- *1 frase:* "É dizer ao modelo que errar na classe rara (default, 8%) pesa mais — senão ele ignoraria a minoria."
- *Completo:* Com 92% de bons pagadores, um modelo ingênuo que aprova todo mundo
  teria 92% de acurácia e seria inútil. Esses parâmetros reponderam as classes para
  que o modelo aprenda a distinguir o default.

**LightGBM (campeão)**
- *1 frase:* "É um modelo de gradient boosting: centenas de árvores de decisão pequenas, cada uma corrigindo os erros da anterior."
- *Completo:* Três motivos para vencer aqui: (1) trata **nulos nativamente** — e os
  nossos nulos são estruturais, carregam informação ("não tem bureau" é um fato, não
  um erro de coleta); (2) trata **categóricas nativamente**, sem explodir em centenas
  de colunas one-hot; (3) captura relações não-lineares e interações que a regressão
  logística não vê. AUC 0,7871.

**Hiperparâmetros (os que estão no `config.yaml`)**
- `n_estimators: 2000` — teto de árvores (o early stopping decide onde parar).
- `learning_rate: 0.02` — passo pequeno: cada árvore corrige pouco, aprendizado suave.
- `num_leaves: 34` / `max_depth: 8` — limitam a complexidade de cada árvore.
- `min_child_samples: 70` — cada folha precisa de ≥70 clientes (evita decorar exceções).
- `subsample: 0.8` / `colsample_bytree: 0.8` — cada árvore vê só 80% das linhas e
  80% das colunas (aleatoriedade que reduz overfitting).
- `reg_alpha` / `reg_lambda: 0.1` — regularização L1/L2, penaliza complexidade.
- *Resumo falável:* "Os hiperparâmetros foram escolhidos para **conter a
  complexidade**: árvores rasas, aprendizado lento, amostragem parcial e
  regularização — tudo mirando generalização, não decoreba."

**Overfitting**
- *1 frase:* "Overfitting é o modelo decorar o treino em vez de aprender o padrão — vai bem no papel, mal na vida real."
- *Completo:* Nosso diagnóstico usa 3 conjuntos: treino (60%), validação (20%) e
  teste (20%). AUC: treino 0,8753 → validação 0,7835 → teste 0,7871. O gap
  treino→validação é esperado com 473 features; o que importa é que **validação e
  teste empatam** — o modelo performa em dado nunca visto igual ao que estimamos.
  Se houvesse overfitting real, o teste despencaria.

**Early stopping**
- *1 frase:* "É o freio automático: o treino para quando a validação para de melhorar — no nosso caso, na iteração 507 de 2000 possíveis."
- *Completo:* A cada árvore adicionada, medimos o AUC na validação. Se ficar 100
  rodadas sem melhorar (`early_stopping_rounds: 100`), o treino para e volta ao
  melhor ponto. É o principal mecanismo anti-overfitting do boosting.

**Split estratificado**
- *1 frase:* "Dividimos treino/validação/teste mantendo os mesmos 8% de default em cada fatia, com semente fixa para reprodutibilidade."

### Termos de avaliação

**AUC (Area Under the ROC Curve)**
- *1 frase:* "AUC mede o poder de ordenação: pegando um mau e um bom pagador ao acaso, é a probabilidade de o modelo dar score de risco maior ao mau pagador."
- *Completo:* Vai de 0,5 (aleatório, moeda) a 1,0 (perfeito). Nosso 0,7868 significa:
  em ~78,5% dos pares (bom, mau), o modelo ordena certo. Para crédito ao consumidor
  com público sub-bancarizado, 0,78+ é um patamar sólido de mercado. **Não depende do
  threshold** — mede a qualidade do ranking, não da decisão final.

**KS (Kolmogorov–Smirnov)**
- *1 frase:* "KS mede a separação máxima entre a curva dos bons e a dos maus pagadores ao longo do score — é a métrica clássica de crédito no Brasil."
- *Completo:* Vai de 0 a 1 (na prática fala-se em pontos: KS 0,4342 = "KS de 43,4").
  Regra de bolso do mercado: KS acima de 0,40 já é considerado um modelo bom para
  concessão de crédito. É a métrica que um banco vai reconhecer na hora.

**Acurácia (e por que NÃO usamos como métrica principal)**
- *1 frase:* "Com 92% de bons pagadores, aprovar todo mundo dá 92% de acurácia e zero utilidade — por isso usamos AUC e KS."

**Recall**
- *1 frase:* "Recall é: de todos os que realmente deram default, quantos o modelo pegou — a métrica do erro mais caro."

**SHAP (SHapley Additive exPlanations)**
- *1 frase:* "SHAP abre a caixa-preta: mostra, cliente a cliente, quanto cada variável empurrou o score para cima ou para baixo."
- *Completo:* Baseado em teoria dos jogos (valores de Shapley): distribui a
  "responsabilidade" da predição entre as features de forma justa. Uso prático em
  crédito: **justificar uma negação** ("seu pedido foi negado principalmente por
  comprometimento de renda alto e histórico curto") — exigência de governança e
  potencial exigência regulatória. Sem explicabilidade, modelo de crédito não vai
  para produção.

**Análise de fairness (viés) por segmento**
- *1 frase:* "Medimos AUC e taxa de aprovação separadamente por gênero, idade e thin-file, para saber ONDE o modelo é confiável e onde não é."
- *Completo:* Distinção crucial que a banca vai testar:
  - **Diferença de aprovação ≠ viés.** Homens são menos aprovados (60,6% vs 74,3%)
    porque dão mais default de fato (10,2% vs 7,0%). O AUC por gênero é quase igual
    (0,7872 M vs 0,7795 F) → o modelo **discrimina risco, não pessoas**.
  - **Queda de AUC = fraqueza real.** Em <25 anos (AUC 0,7319) e 55–65 (0,7465) o
    modelo enxerga pior. E os jovens são justamente o grupo de maior default real
    (11,7%). Isso nós **assumimos como limitação** e mitigamos.

**Human-in-the-loop**
- *1 frase:* "Nos segmentos onde o modelo é fraco, a decisão não é 100% automática: vai para um analista humano com o relatório SHAP do caso."

### Termos de deploy (fechamento / etapa individual)

**API (FastAPI)** — serviço que recebe os dados do cliente via `POST /predict` e
devolve a probabilidade de default em tempo real.
**Streamlit** — dashboard de demonstração para simular pedidos.
**Orquestração (Airflow)** — agendador que roda o pipeline (dados brutos → limpeza →
ABT → treino) de ponta a ponta, automaticamente.
- *1 frase:* "É o que transforma o re-treino de um comando que alguém precisa
  lembrar de rodar num processo agendado, com histórico e log de cada etapa."
- *Completo:* No projeto, o DAG `treino_credit_scoring` roda **a cada 7 dias** e
  quebra o pipeline em **9 tasks** — cada etapa com log próprio na interface, em
  `localhost:8080`. Três delas não existiam no pipeline manual: `checar_fontes`
  (falha em segundos se um CSV sumiu, em vez de descobrir 11 min depois),
  `validar_abt` (granularidade e vazamento, antes de gastar 15 min treinando) e
  **`validar_metricas`** — o gate: se o AUC cair além do limiar frente à última
  rodada aceita, a execução **falha** e o modelo anterior continua servindo.
  Mais `calcular_psi`, que mede o drift do score a cada rodada.
- *Para demonstrar em 1 minuto:* Variable `hc_sample=30000` e disparo manual —
  o mesmo DAG, sobre uma amostra, gravando em `artifacts/demo/`.
**Docker / docker-compose** — empacota tudo em containers reprodutíveis (API na porta
8000, dashboard na 8501).
**Data drift / PSI** — mudança na distribuição dos dados de entrada vs. o treino;
medida pelo PSI (Population Stability Index). Drift alto = alerta.
**Concept drift** — a relação entre features e default muda (ex.: crise econômica);
monitorado por AUC/KS em janela móvel → dispara re-treino.

---

## PARTE 2 — O DISCURSO (para memorizar)

> Como usar: memorize as **frases em negrito** palavra por palavra (são as âncoras).
> O resto é o "recheio" — entenda a ideia e fale com suas palavras. Cada bloco tem
> uma **[PONTE]** que conecta ao próximo slide: decore as pontes, elas evitam o
> branco entre slides.

### BLOCO 0 — Abertura (0:00–1:00)

*(Olhar para a banca, sem ler slide. Frase de impacto direto:)*

> **"Uma financeira de crédito perde dinheiro dos dois lados: quando aprova quem não
> paga — e 8% da carteira entra em default — e quando nega quem pagaria, perdendo
> receita e excluindo gente do sistema financeiro."**
>
> "E o pior: a ferramenta que o mercado usa pra essa decisão — o score de bureau,
> tipo Serasa — **falta em até 56% dos casos** no nosso público, que é
> sub-bancarizado."
>
> **"A gente construiu um modelo de credit scoring que decide com o dado que existe —
> e que sabe dizer onde ele mesmo não é confiável."**
>
> "Somos [nomes]. Nos próximos 15 minutos: o problema, os dados, o modelo e — a parte
> mais importante — a avaliação crítica dele."

*Memorização: a abertura tem 3 números — **8% default, 56% sem bureau, 15 minutos**.*

### BLOCO 1 — Problema de negócio (1:00–4:00) [SLIDE 1]

> "Nossa empresa é uma financeira de crédito ao consumidor para público
> sub-bancarizado — gente com pouco ou nenhum histórico bancário."
>
> **"Todo erro de crédito tem um preço, mas os preços são diferentes."** "Aprovar um
> mau pagador — o falso negativo — custa o crédito inteiro. Negar um bom pagador — o
> falso positivo — custa a margem e um cliente. Na nossa matriz de custo, o primeiro
> erro custa **10 vezes mais** que o segundo."
>
> "Por que esse problema é difícil? Porque o sinal mais forte que existe — os scores
> externos de bureau, as variáveis EXT_SOURCE — tem correlação de -0,16 a -0,18 com o
> default... **e é justamente o dado que mais falta: de 20 a 56% de nulos.**"
>
> *(Pausa. Essa é a tensão central do pitch:)*
> **"O melhor dado é o que menos existe. É esse o problema que fomos resolver."**
>
> "A pergunta de negócio ficou assim: qual a probabilidade de cada cliente dar
> default, e qual o ponto de corte que maximiza o resultado financeiro? E definimos
> sucesso em dois níveis: métricas técnicas — AUC e KS — e métricas de negócio —
> taxa de aprovação e perda evitada."

**[PONTE →]** *"Pra responder isso, primeiro fomos entender os dados. E a análise
exploratória confirmou que o problema era real — e nos deu três direções."*

### BLOCO 2 — EDA (4:00–6:00) [SLIDE 2]

> "Trabalhamos com **307.511 clientes e 124 variáveis** na tabela principal —
> **8,07% de default**."
>
> "Três achados que direcionaram tudo:"
>
> "**Um: o risco é muito desigual por segmento.** Por tipo de renda, o default vai de
> 0% a quase 40%. Por escolaridade, de 1,8% a 10,9%. Ou seja: tem sinal nos dados —
> dá pra separar bons e maus pagadores."
>
> "**Dois: os dados mentem se você não olhar.** Encontramos clientes com 365 mil dias
> de emprego — mil anos trabalhando. Era um código de 'não se aplica' pra
> pensionistas, disfarçado de número. Convertemos pra nulo explícito, senão o modelo
> aprenderia um padrão falso. Mesmo tratamento pro gênero 'XNA'."
>
> "**Três: a base é desbalanceada** — 92 bons pra cada 8 maus. Isso descarta a
> acurácia como métrica: um modelo que aprova todo mundo teria 92% de acurácia e
> seria inútil. **Por isso nossas métricas primárias são AUC, KS e recall.**"

**[PONTE →]** *"Entendido o dado, o próximo desafio: ele está espalhado em 9 tabelas
relacionais — e o modelo precisa de uma linha por cliente. É aí que entra a ABT."*

### BLOCO 3 — ABT (6:00–8:00) [SLIDE 3]

> "ABT — Analytical Base Table — é a tabela final de modelagem: **uma linha por
> cliente, 1.020 colunas**."
>
> "O desafio: um cliente pode ter 20 empréstimos anteriores, dezenas de parcelas,
> histórico de cartão... Agregamos as 9 tabelas — bureau, pedidos anteriores, POS,
> cartão, parcelas — com média, soma, máximo, mínimo e contagem. E criamos ratios de
> negócio: **crédito sobre renda e parcela sobre renda** — o comprometimento de renda
> que qualquer analista de crédito olharia."
>
> *(Os dois cuidados de engenharia — a banca valoriza isso:)*
>
> "**Primeiro cuidado: vazamento de dados.** Só entra na ABT o que é conhecido **no
> momento do pedido de crédito**. Nada do futuro. Vazamento é o erro que faz o modelo
> parecer ótimo no notebook e falhar em produção."
>
> "**Segundo cuidado: granularidade.** Temos um assert de unicidade do ID do cliente —
> garantia automática de que ninguém conta duas vezes no treino."

**[PONTE →]** *"Com a ABT pronta, fomos modelar. E a nossa regra foi: nenhum modelo
complexo se justifica sem antes bater um modelo simples."*

### BLOCO 4 — Modelo (8:00–11:00) [SLIDE 4]

> "**Começamos pelo baseline: regressão logística** — o modelo clássico de crédito,
> simples e interpretável. Com imputação, scaling, one-hot e balanceamento de
> classes: **AUC de 0,7776 no teste, KS de 0,4228.** Essa é a régua."
>
> "**O campeão: LightGBM** — gradient boosting, centenas de árvores pequenas em que
> cada uma corrige os erros da anterior. **AUC 0,7868, KS 0,4342.** Bateu o baseline."
>
> "Por que o LightGBM venceu **neste** dataset? Dois motivos estruturais:
> **ele trata nulos nativamente** — e os nossos nulos não são erro de coleta, são
> informação: 'não ter bureau' é um fato sobre o cliente. E **trata categóricas
> nativamente**, sem explodir one-hot. Além de capturar não-linearidades que a
> regressão não vê."
>
> *(Overfitting — decorar esta sequência de 3 números:)*
> **"AUC de treino 0,8753, validação 0,7835, teste 0,7871."**
> "O gap treino-validação é esperado com 473 features. O que importa é que
> **validação e teste empatam** — se fosse overfitting, o teste despencaria abaixo da
> validação. Não despencou: o modelo generaliza."
>
> "O mecanismo de controle foi o **early stopping**: demos teto de 2000 árvores, e o
> treino parou sozinho na **iteração 507**, quando a validação parou de melhorar. Os
> hiperparâmetros seguem a mesma filosofia: árvores rasas, aprendizado lento,
> regularização — tudo contendo complexidade."

**[PONTE →]** *"Modelo treinado e sob controle. Mas AUC não paga boleto — a pergunta
final é: o que esse modelo entrega pro negócio, e onde ele falha? Esse é o slide mais
importante."*

### BLOCO 5 — Avaliação (11:00–14:00) [SLIDE 5]

> "**Performance: AUC 0,7868, KS 0,4342 no teste.** KS acima de 0,40 é o patamar que o
> mercado de crédito considera bom. Isso dá poder de ordenação suficiente pra
> sustentar uma régua de decisão."
>
> "**A régua:** calibramos o threshold pela matriz de custo — errar aprovando custa
> 10x errar negando. O corte ótimo aprova **cerca de 70% da carteira** com o risco
> sob controle. Não é um corte arbitrário: é o corte que minimiza perda esperada."
>
> "**Explicabilidade: SHAP.** Pra cada cliente, sabemos quanto cada variável empurrou
> o score. Isso é o que permite justificar uma negação de crédito — governança. Sem
> isso, modelo de crédito não vai pra produção."
>
> *(Diagnóstico crítico — falar com confiança, é o diferencial da apresentação:)*
>
> "E agora a parte que a gente considera o diferencial do trabalho: **fomos procurar
> onde o nosso modelo falha.**"
>
> "**Gênero:** o AUC é praticamente igual — 0,7872 nos homens, 0,7795 nas mulheres.
> Homens são menos aprovados, sim — 60,6% contra 74,3% — mas porque dão mais default
> de fato: 10,2% contra 7,0%. **O modelo discrimina risco, não pessoas.**"
>
> "**Idade: aqui há uma fraqueza real.** O AUC cai nos extremos — 0,739 nos menores
> de 25 anos, 0,744 na faixa 55 a 65. E os jovens são justamente o grupo de maior
> default: 11,7%. **O modelo enxerga pior exatamente onde o risco é maior.**"
>
> "**Thin-file:** 14,3% da base não tem histórico de bureau. AUC menor — 0,773 — e
> aprovação bem menor: 57,3% contra 71,7%. Confirma a tese: menos dado, score menos
> confiável."
>
> "**Nossa mitigação: human-in-the-loop.** Nos segmentos de AUC baixo — jovens e
> thin-file — a decisão não é 100% automática: vai pra um analista, acompanhada do
> relatório SHAP do caso. Assumir a limitação e desenhar o processo em volta dela é
> mais honesto do que fingir que o modelo é perfeito."

**[PONTE →]** *"Fechando a conta:"*

### BLOCO 6 — Fechamento (14:00–15:00)

> **"Com AUC de 0,7868 e uma régua calibrada por custo, o modelo sustenta aprovação de
> ~70% da carteira com risco sob controle. E sabemos exatamente onde ele é mais fraco
> — jovens e thin-file — e já desenhamos revisão humana pra esses casos."**
>
> "O próximo passo é o deploy como serviço de predição — API, dashboard, orquestração
> e monitoramento de drift — que é a etapa individual de cada um de nós. Obrigado(a);
> estamos à disposição pras perguntas."

---

## PARTE 3 — COLA DE NÚMEROS (revisar 10 min antes)

### Os 12 números que você NÃO pode errar

> Gerados de `artifacts/` pela rodada `20260828-003844`.
> Para reimprimir depois de um re-treino: `python Model/run_summary.py --markdown`.

| # | Número | O que é | Truque de memória |
|---|---|---|---|
| 1 | **307.511** | clientes na base | "307 mil" basta |
| 2 | **124 → 1.020** | colunas: tabela principal → ABT | "a ABT quase octuplica" |
| 3 | **8,07%** | taxa de inadimplência | "8 em 100 não pagam" |
| 4 | **20–56%** | nulos nos EXT_SOURCE | "o melhor dado falta em até metade" |
| 5 | **9** | tabelas relacionais agregadas | — |
| 6 | **0,7776 → 0,7868** | AUC baseline → servido | "78 e pouco, dos dois lados" |
| 7 | **0,4228 → 0,4342** | KS baseline → servido | "42 vira 43" |
| 8 | **0,8753 / 0,7835 / 0,7871** | AUC treino / validação / teste | "87 cai pra 78 e FICA em 78" |
| 9 | **507** | iteração do early stopping (teto 2000) | "parou num quarto do teto" |
| 10 | **10×** | custo FN vs FP na matriz | "aprovar errado custa 10x" |
| 11 | **0,09** | ponto de corte (agora é P real) | "9% de risco é o limite" |
| 12 | **68,7%** | taxa de aprovação nesse corte | "aprova quase 7 em 10" |

**Se perguntarem por que o AUC do notebook difere do slide na quarta casa:**
o slide usa o modelo **servido** (calibrado), o notebook mostrava o cru. A isotônica
é monotônica, mas cria empates — move AUC/KS na terceira casa e não muda nenhuma
conclusão. `GET /model/metrics` traz os dois, rotulados.

**Os dois números da calibração** (a correção mais defensável do trabalho):
Brier **0,1668 → 0,0658** e corte
**0,47 → 0,09**. O AUC não muda — a isotônica é monotônica.

### Números do diagnóstico crítico (o quadro que a banca vai atacar)

AUC geral: **0,7868**, intervalo de confiança **[0,7806–0,7935]**.

Um segmento só é fraqueza real quando o **topo do IC dele fica abaixo do piso do IC geral**.
Sem o intervalo, você não consegue distinguir fraqueza de ruído amostral — e é essa
distinção que muda a conclusão do trabalho.

| Segmento | AUC | IC 95% | Aprovação | Inadimplência real | Leitura |
|---|---|---|---|---|---|
| Homens | 0,7872 | [0,7778–0,7963] | 60,4% | 10,17% | AUC igual ao das mulheres → **não é viés** |
| Mulheres | 0,7795 | [0,7709–0,7886] | 73,4% | 6,99% | aprovação maior porque o risco real é menor |
| < 25 anos | **0,7319** | [0,7012–0,7597] | 41,5% | 11,80% | **fraqueza real** — IC não sobrepõe |
| 55–65 anos | **0,7465** | [0,7281–0,7672] | 79,9% | 5,61% | **fraqueza real** — IC não sobrepõe |
| Thin-file | 0,7745 | [0,7577–0,7899] | 56,5% | 10,14% | diferença **dentro do ruído** |

*Padrão para decorar: "**gênero OK; idade é fraqueza real e medida; thin-file entra no
ruído — mitigação: faixa cinza ampliada, com humano no loop**".*

> ⚠️ Mudança em relação à versão anterior do trabalho: os decks antigos afirmavam que
> thin-file era uma fraqueza comprovada. Com intervalo de confiança, **não é** — a
> diferença é explicável por tamanho de amostra. Assumir isso é mais forte que insistir
> num número que não se sustenta.

### Hiperparâmetros (se pedirem para mexer ao vivo)

Arquivo: `Model/config.yaml` → seção `champion.params`. Treino: `python Model/train.py`.

- Subir/descer complexidade: `num_leaves` (34), `max_depth` (8), `min_child_samples` (70)
- Velocidade de aprendizado: `learning_rate` (0.02), `n_estimators` (2000)
- Anti-overfitting: `subsample`/`colsample_bytree` (0.8), `reg_alpha`/`reg_lambda` (0.1)
- Threshold de custo: seção `business` → `cost_false_negative: 1.0`, `cost_false_positive: 0.10`

---

## PARTE 4 — Perguntas da banca: respostas prontas em fala

Decore a **primeira frase** de cada resposta — ela ganha tempo e mostra segurança.

**"Por que LightGBM e não [XGBoost / rede neural / outro]?"**
> "Porque as duas maiores dificuldades deste dataset — nulos estruturais e
> categóricas de alta cardinalidade — o LightGBM resolve nativamente, sem imputação
> artificial. E ele foi testado contra um baseline interpretável, a regressão
> logística, então sabemos que a complexidade extra se paga: +0,013 de AUC e +0,029
> de KS."

**"Como vocês sabem que não é overfitting?"**
> "Pelo empate entre validação e teste: 0,7835 contra 0,7871. Overfitting apareceria
> como teste bem abaixo da validação — e não apareceu. O mecanismo de controle foi o
> early stopping, que parou o treino na iteração 507, mais regularização e
> amostragem parcial em cada árvore."

**"O modelo discrimina por gênero?"**
> "Não. O poder discriminativo é praticamente igual — AUC 0,7872 nos homens e 0,778
> nas mulheres. A diferença de aprovação existe, mas reflete a taxa de default real
> observada: 10,2% contra 7,0%. O modelo mede risco; a diferença está nos dados, não
> num viés do algoritmo."

**"E por idade?"**
> "Por idade, sim, há uma fraqueza real e nós a assumimos: o AUC cai para 0,739 nos
> menores de 25 e 0,744 na faixa 55–65. Nossa resposta não é esconder isso, é
> desenhar o processo em volta: revisão humana com relatório SHAP para esses
> segmentos."

**"E os clientes sem histórico (thin-file)?"**
> "São 14,3% da base. O modelo perde poder — AUC 0,773 contra 0,7871 — e aprova menos:
> 57,3% contra 71,7%. Mesma mitigação: human-in-the-loop. E, como evolução, dados
> alternativos — telecom, utilities — para dar sinal a esse público."

**"Qual o impacto financeiro real?"**
> "O threshold foi calibrado por matriz de custo — falso negativo custa 10x o falso
> positivo — e resulta em ~70% de aprovação minimizando a perda esperada. Cada ponto
> de threshold se traduz em reais evitados versus volume aprovado; a curva completa
> está na seção 3 do evaluation.ipynb."

**"Como isso vira produto?"**
> "É a etapa individual de cada um: o modelo vira um serviço — API FastAPI com POST
> /predict, dashboard Streamlit, orquestração do pipeline no Airflow, tudo em Docker.
> Em produção, monitoramos data drift por PSI, concept drift por AUC em janela móvel,
> e a taxa de aprovação observada — com re-treino e recalibração do threshold como
> ações automáticas."

**"Por que não usaram acurácia?"**
> "Porque com 92% de bons pagadores, aprovar todo mundo dá 92% de acurácia e zero
> valor. Em base desbalanceada, as métricas certas são de ordenação e separação — AUC
> e KS — e recall na classe rara."

**"O que é AUC / KS?" (se pedirem a definição)**
> AUC: "É a probabilidade de o modelo dar score de risco maior a um mau pagador do
> que a um bom pagador, escolhidos ao acaso — 0,5 é aleatório, o nosso é 0,7871."
> KS: "É a separação máxima entre a distribuição de score dos bons e dos maus — 43,5
> pontos, acima do patamar de 40 que o mercado considera bom."

**"Como o modelo é re-treinado? Alguém roda na mão?"**
> "Não. Tem um DAG no Airflow que roda a cada 7 dias, com uma task por etapa do
> pipeline — dá pra ver o log de cada uma na interface. E tem um gate: se o
> re-treino piorar o AUC além do limite, a execução falha e o modelo antigo
> continua servindo. A regra de aceite que a gente usou no desenvolvimento virou
> verificação automática." *(Se pedirem para ver: `localhost:8080`, e com a
> Variable `hc_sample` o DAG inteiro roda em um minuto.)*

**"Se eu pedir pra mudar um hiperparâmetro agora, vocês conseguem?"**
> "Sim — tudo está centralizado no Model/config.yaml: hiperparâmetros do LightGBM na
> seção champion, custos do threshold na seção business. Mudou o YAML, roda
> python Model/train.py e o pipeline regenera modelo, métricas e threshold."

---

## PARTE 5 — Técnica de memorização (como ensaiar)

1. **Decore a espinha, não o texto.** A apresentação inteira é UMA frase por bloco:
   - Abertura: *"perde dos dois lados e o melhor dado falta"*
   - Problema: *"dois erros com preços diferentes; o melhor dado é o que menos existe"*
   - EDA: *"tem sinal, tem sujeira, é desbalanceado"*
   - ABT: *"9 tabelas viram 1 linha por cliente; sem vazamento, sem duplicata"*
   - Modelo: *"simples primeiro, campeão depois; 88 cai pra 78 e fica"*
   - Avaliação: *"78/43 sustenta a régua; gênero OK, idade fraca, thin-file fraco, humano no loop"*
   - Fechamento: *"70% de aprovação com risco sob controle; próximo passo, deploy"*

2. **Decore as 6 PONTES.** O branco dá na transição de slide, não no meio do slide.
   As pontes estão marcadas com [PONTE →] no discurso.

3. **Regra do 78.** Quase tudo converge pra 78: AUC 0,7871, validação 0,7835, early
   stopping na 507. Se travar, "78" te salva.

4. **Ensaio em 3 passadas:** (1ª) lendo o discurso em voz alta com timer; (2ª) só com
   a cola de números; (3ª) só com os slides. Meta: 14 min na 3ª passada — o minuto
   de folga é o buffer.

5. **Se estourar o tempo:** corte do fechamento, nunca da avaliação (Bloco 5) — é o
   bloco que a banca mais cobra.

6. **Na pergunta que você não sabe:** "Não medimos isso, mas o caminho seria [X]" vale
   mais que enrolar. A banca testa honestidade técnica — o diagnóstico crítico do
   Bloco 5 já mostrou que vocês sabem onde o modelo falha; mantenham essa postura.
