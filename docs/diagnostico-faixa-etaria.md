# Diagnóstico da fraqueza por faixa etária — veredito sobre o segmento `<25 anos`

| Campo | Valor |
|---|---|
| Versão | 1.1 |
| Data | 2026-08-28 |
| Status | Concluído — veredito emitido para as duas faixas fracas; consertos testados e descartados |
| Escopo | Segmentos `<25 anos` **e `55-65 anos`** do modelo de *credit scoring* Home Credit; por extensão, todo o eixo `age_band` |
| Rodada de referência | `v3-rebuild-py312` (`run_id` 20260828-090029-7d78823) |
| Stack | Python 3.12.3 · LightGBM 4.6.0 · scikit-learn 1.9.0 · pandas 3.0.3 |
| Substitui | `TODO.md §2`, `docs/TCC.md §4.4`, seção 09 do dossiê |

> **Todos os números deste documento são gerados**, por `python Model/diagnostico_idade.py --markdown`,
> `python Model/experimento_teto_idade.py` e `python Model/run_summary.py --markdown`. Nenhum foi digitado à mão.

---

## Sumário

1. [Como usar este documento](#1-como-usar-este-documento)
2. [Glossário](#2-glossário)
3. [Problema, premissas e escopo](#3-problema-premissas-e-escopo)
4. [Fundamentos: o que o AUC mede e o que não mede](#4-fundamentos-o-que-o-auc-mede-e-o-que-não-mede)
5. [O critério de aceite anterior era inválido](#5-o-critério-de-aceite-anterior-era-inválido)
6. [A fraqueza é real — confirmada sob o teste correto](#6-a-fraqueza-é-real--confirmada-sob-o-teste-correto)
7. [As quatro causas candidatas, e o que sobrou](#7-as-quatro-causas-candidatas-e-o-que-sobrou)
8. [O teto do segmento: nenhum modelo dedicado supera o geral](#8-o-teto-do-segmento-nenhum-modelo-dedicado-supera-o-geral)
8b. [A faixa `55-65`: mesma fraqueza, causa diferente](#8b-a-faixa-55-65-mesma-fraqueza-causa-diferente)
9. [Nível de risco: os jovens são mais inadimplentes — e o modelo exagera](#9-nível-de-risco-os-jovens-são-mais-inadimplentes--e-o-modelo-exagera)
10. [Veredito](#10-veredito)
11. [Runbook — como reproduzir](#11-runbook--como-reproduzir)
12. [Riscos e limitações](#12-riscos-e-limitações)
13. [Checklists](#13-checklists)
14. [Alternativas consideradas](#14-alternativas-consideradas)
15. [Resumo executivo](#15-resumo-executivo)
16. [Referências](#16-referências)

---

## 1. Como usar este documento

| Papel | Seções |
|---|---|
| Banca / avaliador | 15, depois 5, 7 e 10 |
| Quem vai defender o trabalho | 4, 5, 6, 7, 8, 9 — nessa ordem |
| Quem for mexer no modelo | 7, 8, 12, 13 |
| Quem for reproduzir | 11 e 13 |
| Risco / crédito | 9 e 10 |

> **A seção mais importante é a 7.** Ela mostra que a causa que o projeto vinha
> declarando — falta de histórico de crédito — **não sobrevive à medição**, e que
> a recomendação derivada dela (dados alternativos) não se sustenta como consequência.

---

## 2. Glossário

| Termo | Significado |
|---|---|
| ABT (*Analytical Base Table*) | Tabela analítica com uma linha por cliente, 1.020 colunas |
| AUC (*Area Under the ROC Curve*) | Probabilidade de o modelo pontuar um inadimplente acima de um adimplente sorteado ao acaso. Mede **ordenação**, não nível |
| KS (Kolmogorov–Smirnov) | Distância máxima entre as distribuições acumuladas de bons e maus pagadores |
| *Brier score* | Erro quadrático médio da probabilidade prevista. Mede **calibração**, não ordenação |
| IC (Intervalo de Confiança) | Faixa de valores plausíveis do estimador; aqui sempre por *bootstrap* |
| *Bootstrap* | Reamostragem com reposição para estimar a incerteza de uma estatística |
| Isotônica (regressão) | Transformação monotônica que converte o *score* bruto em probabilidade calibrada |
| *Thin-file* | Cliente sem nenhum registro no *bureau* de crédito |
| `EXT_SOURCE` | Scores de crédito externos fornecidos na base; a família mais importante do modelo (24% do ganho) |
| Faixa cinza | Intervalo em torno do corte em que a decisão vai para revisão humana em vez de automática |
| Coorte pareada | Grupo construído por reamostragem para reproduzir o perfil de outro grupo |

---

## 3. Problema, premissas e escopo

### O problema

O modelo de *credit scoring* apresenta AUC 0,7319 na faixa `<25 anos` contra 0,7868 no
conjunto de teste inteiro. O repositório registrava isso como **débito técnico não
resolvido** (`TODO.md:32`, única linha ❌ do quadro de situação), com uma tentativa de
conserto executada e rejeitada (`v4-pesos-idade`).

O pedido que originou este documento: **corrigir o modelo, ou emitir veredito** de que a
base está correta e os jovens são de fato mais inadimplentes.

### O que conta como solução aceitável

Regra de aceite herdada de `TODO.md §4`, mantida: mesmo *split*, mesma semente, mesmo
conjunto de teste; uma mudança por rodada; um conserto só fica se melhorar o segmento-alvo
**sem** piorar o global; o que não funcionar é registrado.

### Premissas assumidas

| Premissa | Justificativa |
|---|---|
| O conjunto de teste da rodada congelada é intocável | `tests/test_split.py` re-deriva a partição e confere contra `artifacts/scores.parquet` |
| Python 3.12 é equivalente a 3.14 para esta análise | A rodada reproduziu **bit a bit**: todas as métricas do LightGBM idênticas a 16 casas, inclusive `best_iteration=507` |
| A faixa `<25` da base cobre 20,5 a 25,0 anos | Mínimo etário do *dataset* Home Credit |
| O *score* servido é o calibrado (`proba_champion`) | É o que a API entrega e o que `fairness.json` mede |

### Fora de escopo

- Dados externos ao *dataset* do Kaggle (telecom, contas de consumo, transacional).
- Re-treino com arquitetura diferente de LightGBM.
- Regeneração dos `.pptx`.
- Faixas que **não** são fraqueza confirmada (`65+`, gênero, *thin-file*): medidas, não
  investigadas.
- A qualidade do *score* externo em si — este documento mostra que ele discrimina pior nos
  extremos etários, mas não tem acesso a como ele é construído.

---

## 4. Fundamentos: o que o AUC mede e o que não mede

O pedido original funde duas perguntas que exigem métricas diferentes. Separá-las é o
primeiro resultado deste trabalho.

| Pergunta | Métrica correta | Resposta |
|---|---|---|
| "Jovens são mais inadimplentes?" | Taxa de inadimplência observada, calibração | **Sim** — 11,80% contra 8,07% geral, monotônico na idade. E o modelo captura |
| "O modelo ordena mal os jovens?" | AUC dentro do segmento | **Sim** — 0,7319 contra 0,7833 dos demais |
| "O modelo erra o nível de risco dos jovens?" | *Gap* previsto − observado | **Sim, para mais** — prevê 13,4% onde ocorre 11,8% |

O AUC é a probabilidade de ordenar corretamente um par (inadimplente, adimplente) sorteado
**dentro** do grupo. Um AUC menor num segmento **não** significa que o modelo subestima ou
superestima o risco daquele grupo — significa que, entre dois jovens, distinguir qual vai
falhar é mais difícil. São afirmações independentes, e o repositório vinha tratando a
primeira como se fosse consequência da segunda.

---

## 5. O critério de aceite anterior era inválido

O projeto classificava um grupo como "fraqueza real" quando o topo do IC do grupo ficava
abaixo do piso do IC **geral**. Isso estava implementado em quatro lugares — e em **duas
versões diferentes**, o que já indicava o problema:

| Arquivo | Regra |
|---|---|
| `MLOps/app/policy.py` | unilateral: `ci_high(grupo) < ci_low(geral)` |
| `Model/run_summary.py` | unilateral |
| `docs/dossie/build_data.py` | unilateral |
| `MLOps/app/routers/model.py` | **simétrica**: sobreposição em qualquer direção |

Três defeitos, em ordem de gravidade:

1. **O grupo é subconjunto do geral.** As duas estimativas são aninhadas e correlacionadas;
   comparar seus ICs não é um teste da diferença entre elas.
2. **O AUC geral não é comparável a um AUC intra-grupo.** Ele conta pares formados por
   clientes de faixas etárias **diferentes** — comparações que nenhum AUC dentro de uma
   faixa realiza. Medido: apenas **22,4%** dos pares do AUC agregado são intra-faixa.
3. **Sobreposição de IC não é teste de hipótese**, nem para amostras independentes. É um
   teste conservador de nível ≈0,006, não 0,05.

Os dois primeiros erros puxam em direções opostas — o critério não era sistematicamente
severo nem brando, era **descalibrado**, que é pior de defender.

### O que substituiu

`Model.metrics_lib.auc_diff_bootstrap` e `auc_diff_all_groups`: *bootstrap* estratificado da
**diferença** entre o AUC do grupo e o AUC de referência — a média, ponderada por pares, do
AUC medido **dentro** de cada um dos demais grupos do mesmo eixo. As réplicas são
compartilhadas entre grupos, de modo que dois grupos do mesmo eixo são julgados na mesma
reamostragem. Um grupo é fraqueza quando o IC 95% da diferença exclui o zero pelo lado
negativo.

`Model.metrics_lib.auc_within_between` decompõe o AUC agregado em pares intra e entre
grupos. A decomposição é **identidade exata**, travada em teste
(`test_decomposicao_reconstroi_o_auc_agregado`).

> **O critério novo não muda quem vai para revisão humana.** `<25` e `55-65` continuam
> sendo as duas fraquezas do eixo etário; gênero e *thin-file* continuam fora. A faixa
> cinza dobrada segue valendo para exatamente os mesmos clientes — agora sobre base válida.
> Isso está travado em `tests/test_policy.py`.

---

## 6. A fraqueza é real — confirmada sob o teste correto

Trocar o critério **não** dissolve o achado. Ele encolhe pouco:

| Referência | Δ do `<25` |
|---|---|
| AUC geral agregado (critério antigo) | −0,0549 |
| AUC intra-faixa dos demais grupos (critério correto) | **−0,0514** |

A inflação por agregação explica ~6% do buraco. O resto é real: diferença −0,0514, IC 95%
[−0,0827; −0,0205], p = 0,008.

### Robustez ao corte de faixas

O efeito não é artefato da largura das faixas do projeto (`<25` cobre ~4,5 anos; as do meio,
10 anos). Com **janelas de largura fixa de 5 anos** o padrão é um U limpo:

| Janela | n | AUC | Δ vs. demais | p | Fraqueza? |
|---|---|---|---|---|---|
| 20-25 | 2.355 | 0,7319 | −0,0509 | 0,004 | **sim** |
| 25-30 | 6.460 | 0,7682 | −0,0157 | 0,104 | não |
| 30-35 | 7.844 | 0,7953 | +0,0162 | 0,120 | não |
| 35-40 | 8.656 | 0,7844 | +0,0031 | 0,683 | não |
| 40-45 | 8.256 | 0,7939 | +0,0140 | 0,164 | não |
| 45-50 | 7.038 | 0,7952 | +0,0150 | 0,148 | não |
| 50-55 | 7.003 | 0,7922 | +0,0114 | 0,291 | não |
| 55-60 | 6.621 | 0,7532 | −0,0310 | 0,044 | **sim** |
| 60-65 | 5.545 | 0,7374 | −0,0468 | 0,008 | **sim** |
| 65-70 | 1.725 | 0,7631 | −0,0188 | 0,487 | não |

A janela mais jovem é a pior de todas. Com **sextis** (mesma frequência, n≈10.250) o efeito
se dilui para −0,0234, porque o sexto mais jovem se estende até 31 anos e mistura a faixa
boa com a ruim — o que confirma que o problema está concentrado na ponta, não espalhado.

---

## 7. As quatro causas candidatas, e o que sobrou

**Esta é a seção mais importante do documento.**

| Hipótese | Origem | Veredito | Evidência |
|---|---|---|---|
| Empates da calibração isotônica | Levantada nesta análise | **Refutada** | Δ cru→calibrado de −0,0011 no `<25`, mesma ordem dos demais (−0,0002 a −0,0008) |
| Restrição de amplitude (*range restriction*) | Levantada nesta análise | **Refutada** | Desvio-padrão de `EXT_SOURCE_MEAN`: 0,1455 no `<25` contra 0,1389–0,1521 nas demais. Mesma dispersão, média deslocada |
| **Ausência de histórico de crédito** | **`TODO.md:164`, dossiê §09** | **Refutada** | Coorte madura pareada ao perfil de informação do jovem: AUC **0,7803** contra **0,7319** do jovem |
| Concentração na região baixa do *score* externo | Levantada nesta análise | **Confirmada, parcial** | Pareando também pelo nível do *score*, o buraco cai de −0,0484 para −0,0308 |

### 7.1 O teste que derruba a causa declarada

O projeto afirmava, em `TODO.md:164` e repetido no dossiê e no `docs/TCC.md`:

> "a dificuldade nos jovens não parece ser de *alocação de capacidade*, e sim de
> **ausência de sinal**" · "a fraqueza vem de ausência de histórico, e histórico é
> exatamente o que um cliente de 22 anos não tem"

Isso é testável e foi testado. Construiu-se uma coorte de clientes **maduros (25–45 anos)**
reamostrada estrato a estrato até reproduzir o perfil de informação do `<25`: mesma taxa de
*thin-file*, mesmo número de *scores* externos presentes, mesma faixa de tempo de emprego e
mesmo comprimento de histórico de parcelas. 71 estratos, cobrindo 99,9% dos jovens.

| Coorte pareada por | Estratos | Cobertura | AUC da coorte | IC 95% | AUC `<25` | Δ | Réplicas em que `<25` é pior |
|---|---|---|---|---|---|---|---|
| perfil de informação | 71 | 99,9% | 0,7803 | [0,7527 – 0,8096] | 0,7319 | −0,0484 | 99,8% |
| perfil de informação **+ nível do *score* externo** | 340 | 99,7% | 0,7627 | [0,7378 – 0,7876] | 0,7319 | −0,0308 | 98,8% |

**Leitura:** dê a um cliente de 33 anos exatamente a pobreza informacional de um de 22 anos
e o modelo continua ordenando-o normalmente (0,7803 ≈ média da base). A escassez de
histórico **não** explica o buraco.

O que explica parte dele é o **nível** do *score* externo: acrescentando o decil de
`EXT_SOURCE_MEAN` ao pareamento, o buraco cai de −0,0484 para −0,0308. Ou seja, ~36% do
efeito é "estar na região baixa do *score* externo, onde o modelo separa pior **para
qualquer idade**" — e não "ser jovem".

Sobra um resíduo de **−0,0308**, genuinamente específico da idade, presente em 98,8% das
réplicas.

### 7.2 O sinal disponível é mais fraco, não mais escasso

A cobertura das variáveis dominantes é praticamente igual entre faixas (`EXT_SOURCE_MEAN`
tem 100% de cobertura em todas). O que muda é o **poder discriminante**:

| Feature | `<25` | 25-35 | 35-45 | 45-55 | 55-65 | 65+ |
|---|---|---|---|---|---|---|
| `EXT_SOURCE_MEAN` (AUC univariado) | 0,680 | 0,707 | 0,726 | 0,720 | 0,672 | 0,691 |
| `EXT_SOURCE_MIN` | 0,655 | 0,673 | 0,695 | 0,697 | 0,657 | 0,677 |
| AUC do modelo completo | 0,7319 | 0,7834 | 0,7899 | 0,7942 | 0,7465 | 0,7631 |

O *score* externo — 24% do ganho do modelo — é intrinsecamente menos preditivo nos extremos
etários. O padrão do univariado acompanha o do modelo. A conclusão correta não é "falta
dado", é **"o dado que existe discrimina menos nesse grupo"**.

---

## 8. O teto do segmento: nenhum modelo dedicado supera o geral

Se o resíduo viesse de o modelo global aplicar aos jovens uma função ajustada à maioria
madura, um modelo treinado só neles deveria superá-lo. Foi testado, reusando a **mesma
partição** da rodada congelada (nunca refeita — lida de `artifacts/scores.parquet`) e
avaliando nos **mesmos 2.355 clientes** de teste.

| Variante | n de treino | Hiperparâmetros | AUC no teste `<25` | IC 95% |
|---|---|---|---|---|
| **geral (rodada congelada)** | 153.755 | `num_leaves=34, min_child=70` | **0,7319** | [0,7015 – 0,7583] |
| segmentado `<25` | 6.063 | `num_leaves=34, min_child=70` | 0,7131 | [0,6830 – 0,7410] |
| segmentado `<25` | 6.063 | `num_leaves=16, min_child=40` | 0,7007 | [0,6680 – 0,7313] |
| segmentado `<25` | 6.063 | `num_leaves=8, min_child=20` | 0,7164 | [0,6883 – 0,7429] |
| segmentado `<30` | 22.556 | `num_leaves=34, min_child=70` | 0,7296 | [0,7005 – 0,7566] |
| segmentado `<30` | 22.556 | `num_leaves=16, min_child=40` | 0,7281 | [0,6986 – 0,7562] |
| segmentado `<30` | 22.556 | `num_leaves=8, min_child=20` | 0,7280 | [0,6994 – 0,7556] |

**Nenhuma das seis variantes supera o modelo geral.** A melhor fica 0,0023 abaixo. O modelo
global já extrai, para esse segmento, tudo o que este conjunto de variáveis oferece —
inclusive aproveitando o que aprende com os clientes maduros.

Isso fecha o cerco junto com a rodada `v4-pesos-idade` já registrada: reponderar não
funcionou, segmentar não funciona, e a causa que justificaria dados alternativos foi
refutada na seção 7.

---

## 8b. A faixa `55-65`: mesma fraqueza, causa diferente

`55-65` é a outra fraqueza confirmada do eixo (−0,0415, IC [−0,0619; −0,0209], p = 0,004,
n = 12.166). Foi diagnosticada com o mesmo instrumental. **A causa não é a mesma do `<25`.**

### 8b.1 Não é aposentadoria

A hipótese natural: `55-65` é **68,4% aposentado**, contra 0,4%–7,8% nas faixas do meio, e
a sanitização transforma `DAYS_EMPLOYED = 365243` em nulo — o bloco de emprego inteiro
morre para esse público. Se a fraqueza viesse daí, o aposentado seria pior que o ativo
**dentro** da mesma faixa. Não é o que acontece:

| Faixa | % aposentado | AUC aposentado | AUC ativo | Δ (apos − ativo) |
|---|---|---|---|---|
| 25-35 | 0,4% | 0,8110 (n=55) | 0,7833 (n=14.249) | +0,0278 |
| 35-45 | 1,1% | 0,7281 (n=193) | 0,7905 (n=16.719) | −0,0624 |
| 45-55 | 7,8% | 0,8065 (n=1.090) | 0,7934 (n=12.951) | +0,0132 |
| **55-65** | **68,4%** | **0,7488 (n=8.327)** | **0,7427 (n=3.839)** | **+0,0061** |
| 65+ | 90,4% | 0,7602 (n=1.559) | — | — |

Dentro de `55-65`, aposentado e ativo são indistinguíveis (0,7488 × 0,7427). E os
aposentados de `45-55` vão **melhor** que a média (0,8065). Medido sem controlar idade, o
eixo aposentadoria parece explicativo (Δ −0,0313, p = 0,012) — mas é confundimento puro:
aposentadoria correlaciona com idade, e é a idade que carrega o efeito.

> Esta é a mesma armadilha que o critério antigo cometia em outra forma: uma diferença
> agregada que desaparece quando se estratifica. Vale para qualquer causa candidata —
> **medir antes de afirmar**.

### 8b.2 Também não é o perfil de informação

Coorte tirada de `35-45` e `45-55`, reamostrada até reproduzir o perfil do `55-65` —
inclusive a faixa de emprego nulo, para que a coorte reproduza um grupo majoritariamente
inativo:

| Coorte pareada por | Estratos | Cobertura | AUC da coorte | IC 95% | AUC `55-65` | Δ | Réplicas em que `55-65` é pior |
|---|---|---|---|---|---|---|---|
| perfil de informação | 136 | 99,9% | 0,7906 | [0,7755 – 0,8073] | 0,7465 | −0,0441 | **100,0%** |
| perfil de informação + nível do *score* | 775 | 97,5% | 0,7830 | [0,7676 – 0,7987] | 0,7465 | −0,0365 | **100,0%** |

Mais nítido que no `<25`: o `55-65` é pior em **100%** das réplicas nas duas variantes. E,
diferente do `<25`, parear pelo nível do *score* externo quase não ajuda — explica ~17% do
buraco contra ~36% no caso dos jovens.

### 8b.3 A fraqueza é uniforme dentro da faixa

Não se concentra em nenhum recorte: gênero (0,7408 F × 0,7421 M), tipo de contrato
(0,7456 × 0,7584), *thin-file* (0,7465 × 0,7281), número de *scores* externos disponíveis
(0,7545 / 0,7389 / 0,7527). É um efeito difuso, compatível com queda de qualidade de sinal
— não com um subgrupo problemático escondido.

### 8b.4 A causa: os scores externos são intrinsecamente piores nessa idade

Os três *scores* externos, **cada um isoladamente**, atingem seu pior desempenho em
`55-65`:

| Faixa | `EXT_SOURCE_1` | `EXT_SOURCE_2` | `EXT_SOURCE_3` | `EXT_SOURCE_MEAN` |
|---|---|---|---|---|
| `<25` | 0,6736 | 0,6382 | 0,6504 | 0,6801 |
| 25-35 | 0,6505 | 0,6577 | 0,6596 | 0,7073 |
| 35-45 | 0,6610 | 0,6733 | 0,6701 | 0,7262 |
| 45-55 | 0,6602 | 0,6474 | 0,6893 | 0,7202 |
| **55-65** | **0,6288** | **0,6301** | **0,6506** | **0,6722** |
| 65+ | 0,6426 | 0,6533 | 0,6729 | 0,6913 |

### 8b.5 O contraste que separa os dois casos

Quanto o modelo completo **acrescenta** sobre o melhor sinal isolado:

| Faixa | AUC do modelo | `EXT_SOURCE_MEAN` sozinho | Ganho do modelo |
|---|---|---|---|
| **`<25`** | 0,7319 | 0,6801 | **0,0518** ← o menor de todos |
| 25-35 | 0,7834 | 0,7073 | 0,0761 |
| 35-45 | 0,7899 | 0,7262 | 0,0636 |
| 45-55 | 0,7942 | 0,7202 | 0,0740 |
| **`55-65`** | 0,7465 | 0,6722 | **0,0744** ← normal |
| 65+ | 0,7631 | 0,6913 | 0,0718 |

**As duas faixas fracas falham por motivos diferentes:**

- **`55-65`** — o modelo trabalha **normalmente** (ganho 0,0744, em linha com as melhores
  faixas). Ele apenas parte de um sinal pior: o *score* externo vale 0,6722 ali contra
  0,7262 aos 35-45. É deficiência **da fonte de dado**, não do modelo.
- **`<25`** — o sinal também é fraco (0,6801), **e** o modelo acrescenta o mínimo de todas
  as faixas (0,0518). Além do *score* externo ruim, o restante das variáveis rende menos
  para jovens.

Isso reforça o veredito da seção 10 e refina a recomendação: para `55-65`, a via seria uma
**fonte de score externo melhor para o público maduro** — problema do fornecedor do
*bureau*, não de modelagem. Para `<25`, nem isso basta, porque lá o déficit é também no
que o modelo consegue extrair do resto.

---

## 9. Nível de risco: os jovens são mais inadimplentes — e o modelo exagera

Esta seção responde diretamente à segunda metade do pedido original.

**Os jovens são de fato mais inadimplentes.** A taxa observada cai monotonicamente com a
idade: 11,80% → 10,55% → 8,24% → 7,39% → 5,61% → 3,71%. Não há erro de base aqui, e o
modelo captura o fato — aprova só 41,5% da faixa `<25` contra 68,7% da carteira.

**Mas exagera.** A faixa `<25` é a **única** com viés de calibração estatisticamente
significativo:

| Faixa | n | Previsto | Observado | *Gap* | IC 95% do *gap* | Enviesado? | Brier |
|---|---|---|---|---|---|---|---|
| 25-35 | 14.304 | 10,6% | 10,5% | +0,0005 | [−0,0042 – 0,0050] | não | 0,0826 |
| 35-45 | 16.912 | 8,1% | 8,2% | −0,0018 | [−0,0058 – 0,0019] | não | 0,0669 |
| 45-55 | 14.041 | 7,1% | 7,4% | −0,0027 | [−0,0065 – 0,0013] | não | 0,0604 |
| 55-65 | 12.166 | 5,7% | 5,6% | +0,0010 | [−0,0029 – 0,0047] | não | 0,0492 |
| 65+ | 1.725 | 3,8% | 3,7% | +0,0008 | [−0,0070 – 0,0085] | não | 0,0345 |
| **`<25`** | **2.355** | **13,4%** | **11,8%** | **+0,0156** | **[0,0039 – 0,0294]** | **sim** | 0,0960 |

**Causa identificada:** a isotônica é ajustada **globalmente**. Reajustando-a apenas sobre
os jovens, o viés praticamente desaparece — *gap* de +0,0143 para −0,0005.

**Mas o conserto não entrega o que parece.** Medido na fatia de teste `<25`:

| | Previsto | *Gap* | Brier | AUC | Aprovação no corte 0,09 |
|---|---|---|---|---|---|
| Isotônica global | 0,1324 | +0,0143 | 0,09584 | 0,7337 | 47,6% |
| Isotônica por faixa | 0,1176 | −0,0005 | 0,09583 | 0,7308 | **44,8%** |

O Brier não melhora e a taxa de aprovação **cai**. A calibração por segmento corrige um
viés **agregado** — que importa para estimativa de perda esperada de carteira — mas não
melhora decisões individuais e não amplia o acesso ao crédito para jovens. Registrada como
achado; **não** recomendada como conserto sem uma decisão de negócio explícita (seção 14).

---

## 10. Veredito

**O modelo está correto sobre o nível de risco, e a fraqueza de ordenação é um teto de
dado — mas não pela razão que o projeto vinha declarando.**

Ponto a ponto:

1. **A base está certa.** Jovens são mais inadimplentes (11,80% × 8,07%), o efeito é
   monotônico na idade, e o modelo o reflete na decisão (41,5% de aprovação).
2. **A fraqueza de ordenação é real**, sobrevive à correção do critério (−0,0514,
   p = 0,008) e é robusta ao corte de faixas.
3. **A causa declarada estava errada.** Não é ausência de histórico: coorte madura com o
   mesmo perfil de informação atinge 0,7803. Não é restrição de amplitude nem empate de
   calibração. É que as variáveis disponíveis — dominadas por *scores* externos —
   discriminam menos nesse grupo, e ~36% do efeito nem sequer é sobre idade, e sim sobre a
   região baixa do *score* externo.
4. **Não há conserto disponível neste conjunto de dados.** Reponderar falhou (`v4`);
   segmentar falha nas seis variantes testadas; o modelo global já está no teto do segmento.
5. **Há um defeito novo, menor e mensurado:** viés de calibração de +1,6 pp no `<25`,
   causado pela isotônica global. Corrigível, com contrapartida (seção 9).
6. **A recomendação de "dados alternativos" perde o fundamento que tinha.** Ela era
   justificada por ausência de histórico. Continua sendo uma via plausível, mas agora
   precisa ser justificada por outro argumento: seria preciso mostrar que a fonte nova
   discrimina **dentro** do grupo jovem, não apenas que preenche lacunas.

### Mitigação vigente: mantida

A régua de três faixas com faixa cinza dobrada para `<25` e `55-65` continua correta e
continua valendo para os mesmos clientes. Ela é a resposta certa a um teto de dado medido:
onde o modelo comprovadamente ordena pior, a decisão não deve ser 100% automática.

---

## 11. Runbook — como reproduzir

```bash
# ambiente (Python 3.12 basta; a rodada reproduz bit a bit)
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 0. salvar o histórico ANTES do primeiro treino numa árvore limpa
.venv/bin/python scripts/restaurar_improvement_log.py

# 1. pipeline (~15 min)
.venv/bin/python DataPipeline/data_sanitization.py
.venv/bin/python DataPipeline/abt_transform.py
.venv/bin/python DataPipeline/to_parquet.py
.venv/bin/python Model/train.py --tag <tag>

# 2. o teste que prova que a partição não foi tocada
.venv/bin/python -m pytest tests/test_split.py -q

# 3. o diagnóstico completo (~6 min)
.venv/bin/python Model/diagnostico_idade.py --n-boot 500 --markdown

# 4. o teto do segmento (~20 min)
.venv/bin/python Model/experimento_teto_idade.py --n-boot 500

# 5. números oficiais e dossiê
.venv/bin/python Model/run_summary.py --markdown
.venv/bin/python docs/dossie/build_data.py
```

Para trazer uma rodada já treinada ao critério novo sem re-treinar:
`.venv/bin/python scripts/regenerar_fairness.py`

### Diagnóstico de problemas

| Sintoma | Causa provável | Investigação |
|---|---|---|
| `improvement_log.json` com 1 rodada só | Treinou antes de restaurar o histórico | `git checkout` não recupera: `artifacts/` é *gitignored*. Rodar `scripts/restaurar_improvement_log.py` a partir do dossiê |
| `test_split.py` falha | O *split* mudou — semente, `calib_size` ou n de linhas | Nenhuma comparação entre rodadas vale até fechar. Conferir `Model/config.yaml → split` |
| `build_data.py` aborta no fim | Falta `artifacts/abt_profile.json` | Rodar `DataPipeline/to_parquet.py`; `abt_transform.py` sozinho não gera o perfil |
| AUC do rebuild ≠ documentado | Versão de biblioteca ou nº de *threads* | Conferir `metrics.json → run.versions`. A regressão logística varia ~1e-5; o LightGBM não deveria |
| `/model/fairness` sem `vs_referencia` | Artefato de rodada anterior à mudança | O código cai no critério antigo por compatibilidade. Rodar `scripts/regenerar_fairness.py` |

---

## 12. Riscos e limitações

| Risco | Natureza | Mitigação |
|---|---|---|
| O modelo segmentado tem 6.063 linhas de treino para 1.000 variáveis | Estatística | Testadas 3 configurações de capacidade e uma variante `<30` com 22.556 linhas. Nenhuma superou o geral |
| A coorte pareada usa reamostragem com reposição em estratos pequenos | Estatística | IC de 500 réplicas incorpora a variância; o resultado se mantém em 99,8% delas |
| `65+` tem n=1.725 e IC de largura 0,10 | Amostral | Não sustenta afirmação estrutural. Não é classificado como fraqueza (p = 0,395) |
| A causa do `55-65` é atribuída ao *score* externo, que é uma caixa-preta | Inferência | Mede-se que os três *scores* rendem menos ali (0,6288 / 0,6301 / 0,6506, o pior de todas as faixas). Por que rendem menos não é observável nesta base |
| `AGE_YEARS` e `DAYS_BIRTH` são ambos variáveis vivas e colineares | Higiene | Divide o ganho entre duas colunas e subestima a importância da idade em `feature_importance.json`. Não afeta o AUC |
| `scores.parquet` não inclui a fatia de calibração | Documental | 30.751 clientes ficam com *score* nulo em `GET /clients`. Travado em `tests/test_split.py` |
| O gate do Airflow compara AUC calibrado com AUC cru gravado | Latente | Viés de +0,0003 contra limiar de 0,01 — nunca disparou, mas está errado |

### O que este documento NÃO prova

- **Não prova que 0,7319 é o teto teórico** do segmento. Prova que o modelo global não é
  superado por um modelo dedicado com este conjunto de variáveis, esta família de algoritmo
  e este volume de dados. Outra família (rede neural, modelo aditivo com interações
  explícitas de idade) não foi testada.
- **Não prova que dados alternativos não ajudariam.** Prova que o argumento que os
  justificava — ausência de histórico — não se sustenta. A hipótese continua aberta sob
  outra justificativa.
- **Não explica o resíduo de −0,0308 do `<25`.** Ele é medido e é específico da idade; sua
  causa mecânica permanece desconhecida.
- **Não explica *por que* o score externo discrimina pior nos extremos etários.** Constata
  o fato; a construção do *score* é externa ao *dataset*.
- **Não testou o `65+` a fundo** — n = 1.725 e IC de largura 0,10 não sustentam conclusão.
- **Não avalia justiça sob nenhuma definição formal** (paridade demográfica, *equalized
  odds*). Mede desempenho por segmento, que é outra coisa.

---

## 13. Checklists

### Antes de alterar o modelo

- [ ] `pytest tests/test_split.py` passa — a partição não mudou
- [ ] `scripts/restaurar_improvement_log.py` já rodou, ou `improvement_log.json` tem as rodadas anteriores
- [ ] Uma única mudança por rodada, com `--tag` descritivo
- [ ] A rodada de comparação foi treinada **no mesmo ambiente** que a nova

### Go / No-Go de um conserto

- [ ] O AUC do segmento-alvo melhorou
- [ ] O AUC geral **não** piorou
- [ ] Nenhum outro segmento passou a ser fraqueza confirmada
- [ ] O `gap` de calibração do segmento não piorou
- [ ] Se reprovado: `status: "rejeitada"` e `motivo` gravados em `improvement_log.json`

### Depois de qualquer re-treino

- [ ] `python Model/run_summary.py --markdown`
- [ ] `python Model/diagnostico_idade.py --n-boot 500 --markdown`
- [ ] `python docs/dossie/build_data.py`
- [ ] `pytest -q`
- [ ] Documentos reconciliados a partir da saída gerada, nunca por edição manual de número

---

## 14. Alternativas consideradas

| Abordagem | Quando considerar | Limitação |
|---|---|---|
| Reponderar o treino (`sample_weight`) | Quando o segmento é subrepresentado **e** o sinal existe | **Testada e rejeitada** (`v4-pesos-idade`): `<25` piorou 0,0032 |
| Modelo segmentado | Quando a relação variável→risco difere no segmento | **Testada e rejeitada**: 6 variantes, nenhuma supera o geral |
| Percentil dentro da faixa etária | Quando o valor absoluto não distingue dentro do grupo | **Descartada sem teste.** É estatística populacional: `Model/derived.py` não consegue calculá-la para um cliente isolado, então `POST /predict` divergiria da ABT. Além disso vaza distribuição do teste |
| Razões normalizadas por tempo de histórico | Quando o problema é histórico curto | **Descartada pelo diagnóstico**: a causa que ela ataca (escassez de informação) foi refutada na seção 7 |
| Calibração isotônica por segmento | Quando o viés de nível do segmento importa para perda esperada | Corrige o *gap* (+0,0143 → −0,0005) mas não melhora Brier nem AUC, e **reduz** a aprovação de 47,6% para 44,8% |
| Faixa cinza ampliada (vigente) | Quando o teto é de dado e a decisão pode ter revisão humana | Custa revisão manual. **É a resposta adotada** |
| Dados alternativos | Quando se pode demonstrar poder discriminante **dentro** do grupo | Fora do *dataset*. O argumento que a justificava foi refutado; precisa de outro |

---

## 15. Resumo executivo

1. O segmento `<25 anos` tem AUC 0,7319 contra 0,7833 dos demais grupos etários. A
   diferença é **−0,0514, IC 95% [−0,0827; −0,0205], p = 0,008**: real, não ruído amostral.
2. O critério que o projeto usava para declarar isso era **estatisticamente inválido** —
   comparava o segmento com o AUC geral, que o contém e que é composto 77,6% por pares
   entre faixas. Foi substituído por um *bootstrap* da diferença contra os demais grupos do
   eixo, em `Model/metrics_lib.py`, e unificado nos quatro pontos do código que o
   duplicavam em duas versões divergentes.
3. **A conclusão sobrevive ao critério correto** — encolhe de −0,0549 para −0,0514.
4. **A causa declarada pelo projeto — ausência de histórico de crédito — foi refutada.**
   Uma coorte de clientes de 25–45 anos, reamostrada até reproduzir exatamente o perfil de
   informação dos jovens (99,9% de cobertura, 71 estratos), atinge AUC 0,7803. O jovem é
   pior em 99,8% das réplicas.
5. Cerca de **36% do efeito não é sobre idade**: é sobre estar na região baixa do *score*
   externo, onde o modelo separa pior em qualquer faixa. O resíduo específico da idade é
   **−0,0308**.
6. **Não há conserto disponível.** Seis variantes de modelo dedicado (`<25` e `<30`, três
   capacidades cada) ficam todas **abaixo** do modelo geral no mesmo conjunto de teste. A
   reponderação já havia sido rejeitada.
7. **Os jovens são de fato mais inadimplentes** (11,80% contra 8,07%, monotônico na idade)
   e o modelo captura isso. **Mas exagera:** é a única faixa com viés de calibração
   significativo, prevendo 13,4% onde ocorrem 11,8%. Causa: a isotônica é global.
8. **A outra fraqueza, `55-65` (−0,0415, p = 0,004), tem causa diferente.** Não é
   aposentadoria — dentro da faixa, aposentado (0,7488) e ativo (0,7427) são
   indistinguíveis, e aposentados de 45-55 vão acima da média. Não é perfil de informação:
   coorte pareada atinge 0,7906, e o `55-65` é pior em **100%** das réplicas. A causa é
   que os três *scores* externos rendem ali o pior de todas as faixas (0,6288 / 0,6301 /
   0,6506).
9. **As duas faixas falham de formas distintas.** Em `55-65` o modelo agrega o normal
   (0,0744) sobre um sinal ruim — é deficiência da fonte. Em `<25` o sinal é ruim **e** o
   modelo agrega o mínimo de todas as faixas (0,0518) — o resto das variáveis também rende
   menos para jovens.
10. **Veredito:** teto de dado, não defeito de modelo — porém com a causa corrigida, um
    defeito de calibração novo identificado, e a recomendação de dados alternativos
    perdendo o fundamento que a sustentava. A mitigação vigente (faixa cinza dobrada →
    revisão humana) permanece correta e inalterada.

---

## 16. Referências

### Artefatos desta análise
- `artifacts/diagnostico_idade.json` — todos os cortes, testes e perfis
- `artifacts/experimentos/teto_idade.json` — as sete variantes do teto
- `artifacts/fairness.json` — segmentos sob o critério novo
- `artifacts/fairness_criterio_anterior.json` — o mesmo sob o critério antigo
- `artifacts/improvement_log.json` — v1 a v4 reconstruídas + rodadas novas

### Código
- `Model/metrics_lib.py` — `auc_diff_bootstrap`, `auc_diff_all_groups`, `auc_within_between`
- `Model/diagnostico_idade.py` · `Model/experimento_teto_idade.py`
- `scripts/restaurar_improvement_log.py` · `scripts/regenerar_fairness.py`
- `tests/test_policy.py` · `tests/test_split.py` · `tests/test_metrics_lib.py`

### Documentos do projeto substituídos ou alterados
- `TODO.md §2` · `README.md` · `docs/TCC.md §4.2 e §4.4` · `docs/pitch_demoday.md` ·
  `MLOps/app/README.md` · `docs/dossie/index.html`

### Fonte dos dados
- Home Credit Default Risk — <https://www.kaggle.com/competitions/home-credit-default-risk>

---

*Arquivo: `docs/diagnostico-faixa-etaria.md`. Incrementar a versão quando: houver novo
re-treino que mude os números por segmento, quando a faixa `55-65` for investigada, ou
quando a calibração por segmento for adotada ou definitivamente descartada.*
