# Home Credit — Crédito com risco medido, e limite declarado

> **Resumo do projeto, escrito para ser contado.**

| | |
|---|---|
| **Versão** | 1.0 — 31/08/2026 |
| **Rodada de referência** | `20260828-144848`, tag `airflow-scheduled-2026-08-28T1423598413650000` |
| **Origem dos números** | `artifacts/*.json`. Reimprimir com `python Model/run_summary.py --markdown` |
| **Substitui** | `docs/pitch_demoday.md` e `docs/guia_apresentacao_demoday.md`, que cobrem só a fase em grupo e trazem números da rodada anterior |

## Como usar este documento

Ele tem **duas partes com funções opostas**. Confundi-las é o erro que fez o material
anterior não funcionar.

| | **Parte I — A narrativa** | **Parte II — O arsenal** |
|---|---|---|
| Serve para | ser **contada** | ser **consultada** |
| Como usar | ler em voz alta, três vezes | procurar durante a arguição |
| Se você só decorar isto | você apresenta bem | você não apresenta, mas responde tudo |

**Estude a Parte I. Consulte a Parte II.** Dentro da Parte I, o **§0.1 é a linha do tempo
da apresentação em três atos** — é por ele que se ensaia. A Parte I é uma corrente: cada movimento
termina numa pergunta que só o próximo responde. Se você lembrar da pergunta, o próximo
movimento vem sozinho — é isso que substitui a decoreba. Os números aparecem no papel que
exercem na história, nunca em lista, porque um número que você sabe *o que derrubou* se
reconstrói pelo raciocínio quando a memória falha.

---
---

# PARTE I — A NARRATIVA

## §0. A espinha

Seis frases. Cada uma causa a seguinte. Se você souber só isto, você conta o projeto.

1. Numa financeira, **os dois erros de crédito têm preços diferentes** — aprovar quem não
   paga custa cerca de dez vezes negar quem pagaria —, e isso derruba a acurácia como
   métrica e obriga o modelo a devolver uma probabilidade de verdade, não uma nota.
2. Só que **o dado que melhor prevê inadimplência falta em mais da metade dos casos**,
   exatamente porque o público é sub-bancarizado; o resto do sinal tem de vir das nove
   tabelas de histórico.
3. Ao construir essa tabela, **achamos um bug que valia mais que qualquer hiperparâmetro**:
   a agregação descartava em silêncio toda variável categórica das relacionais. 473 colunas
   viraram 1.020.
4. O modelo passou a ordenar bem, **mas o score ainda não era probabilidade** — o corte
   ótimo caía em 0,47 numa carteira com 8% de inadimplência. Calibração isotônica numa
   fatia reservada de 10% levou o Brier de 0,1668 para 0,0658 e o corte para 0,09.
5. Com probabilidade de verdade, **o corte deixou de ser escolha e virou conta**: o mínimo
   de custo aprova 68,65% da carteira com 3,48% de inadimplência entre os aprovados,
   contra 8,07% sem modelo.
6. Aí fomos medir onde ele falha — e **descobrimos que o nosso próprio critério de fraqueza
   era inválido**. Trocamos; a conclusão sobreviveu; a causa que declarávamos era falsa;
   não há conserto disponível. Então a resposta foi desenhar o processo em volta do limite,
   e servir tudo isso de verdade.

---

## §0.1. A linha do tempo da apresentação — três atos

O §0 é a lógica da história. Esta seção é a **encenação** dela: onde cada movimento entra,
quanto tempo tem, e qual é a função dramática de cada ato. É o que você olha ao ensaiar.

A diferença entre a corrente causal e a encenação está num ponto só, e é deliberado: **os
números de resultado são construídos no Ato II e apresentados no Ato III.** No meio você
mostra *como* chegou neles; no fim você os entrega como resultado. Anunciar 68,65% de
aprovação no meio da apresentação queima o fecho.

| Ato | Movimentos | Slides | Tempo | Função dramática |
|---|---|---|---|---|
| **I — A dor** | §1, §2 | 1–3 | 0:00 – 2:30 | Estabelecer que o problema é **econômico e assimétrico**, não estatístico |
| **II — O desenvolvimento** | §3 a §8 | 4–10 | 2:30 – 8:30 | Mostrar **como se chega a um número em que se pode confiar** |
| **III — A entrega** | §6 (recapitulado) e §9 | 11–14 | 8:30 – 12:00 | Converter tudo em **coisa que existe e que outra pessoa consegue operar** |

---

### ATO I — A dor de origem (2:30)

**Abre com** uma financeira de crédito ao consumidor que perde dinheiro nos dois extremos
da mesma decisão: aprova quem não paga e perde o capital; nega quem pagaria e perde a
margem — e, num público sub-bancarizado, ainda empurra a pessoa para fora do sistema.

**O que precisa ficar estabelecido antes de sair daqui**, nesta ordem:

1. Os dois erros existem, e **não custam a mesma coisa**: dez para um. É premissa de
   negócio declarada, não descoberta no dado.
2. Por isso **acurácia não serve**. Numa base 92 contra 8, aprovar todo mundo acerta 92%
   das vezes e é o modelo mais caro possível.
3. Logo, o que se precisa não é um classificador: é uma **probabilidade de verdade** mais
   um **corte derivado de custo**.
4. E o dado que melhor prevê inadimplência — o score de bureau — **falta em 56,4% dos
   casos**, exatamente porque o público é sub-bancarizado.

**Fecha com o paradoxo**, que é a frase que sustenta o trabalho inteiro: *o sinal mais
forte é o que menos está disponível.*

> **Dobradiça para o Ato II** — não pule esta frase, é ela que impede a apresentação de
> virar uma lista de etapas:
> *"Então este trabalho não começou num modelo. Começou em construir a informação que não
> vinha pronta."*

---

### ATO II — O desenvolvimento (6:00)

É o ato mais longo, e é onde a banca decide se você entende o que fez. Seis blocos, na
ordem em que uma decisão obriga a seguinte.

| # | Bloco | O que se mostra | O que fica provado |
|---|---|---|---|
| 1 | **O que os dados disseram** | 307.511 clientes, 8,07% de inadimplência; risco desigual por escolaridade (1,8% → 10,9%); sentinela `DAYS_EMPLOYED = 365243` | Há sinal, ele é estrutural, e os dados mentem em pontos que precisam ser tratados |
| 2 | **A ABT e o bug** | Nove tabelas com granularidades diferentes viram uma linha por cliente. A agregação descartava em silêncio toda categórica: **473 → 1.020 colunas** | O ganho maior do projeto veio de **corrigir dado**, não de ajustar modelo |
| 3 | **Os modelos testados** | Regressão logística **0,7776** contra LightGBM **0,7871** | E, mais importante, **por que** um ganhou — abaixo |
| 4 | **O que o AUC mede** | Probabilidade de ordenar certo um par (inadimplente, adimplente) | E o que ele **não** mede: se o nível do risco está certo. Essa distinção volta no bloco 6 |
| 5 | **O score não era probabilidade** | Corte ótimo caía em **0,47** numa carteira com 8% de inadimplência → calibração isotônica → **Brier 0,1668 → 0,0658**, corte **0,09** | Calibrar não melhora o modelo: melhora o **significado do número** |
| 6 | **Onde o modelo falha** | Nosso próprio critério de fraqueza era inválido — **77,6%** dos pares do AUC agregado são entre faixas. Trocado por bootstrap da diferença; a conclusão sobreviveu (**−0,0514**, p 0,008); a causa que declarávamos era falsa (coorte pareada **0,7803** contra 0,7319); seis modelos dedicados ficaram abaixo | É **teto de dado**, e isso foi medido, não suposto |

**Por que o LightGBM ganhou — a resposta que a banca cobra, e que não é "porque deu AUC
maior":** os nulos deste dataset são **estruturais, não erro**. "Não tem score de bureau" é
informação, e imputar pela mediana apagaria justamente o sinal do público que o trabalho
quer atender. O LightGBM trata nulo e categórica de alta cardinalidade nativamente. A
regressão logística precisa imputar e codificar tudo — e por isso ela é o **piso de
comparação**, não uma alternativa descartada por capricho. Sem ela, "0,78" não significaria
nada.

**As rodadas que foram rejeitadas, e por quê** — este quadro é o que prova método em vez de
sorte, e cabe em quinze segundos de fala:

| Rodada | O que mudou | Resultado |
|---|---|---|
| v1 | 471 features, corte 0,47 | AUC 0,7846 — aceita como linha de base |
| v2 | ABT corrigida, 1.018 features | AUC 0,7880 — aceita |
| v3 | calibração isotônica | Brier 0,1668 → 0,0658, corte → 0,09 — **é a que está servida** |
| v4 | peso maior para jovens no treino | **rejeitada**: piorou o próprio alvo em 0,0032 |

**Sobre overfitting**, o argumento não é o gap treino → validação (0,8753 → 0,7835), que é
esperado com mil variáveis. É o **empate validação × teste**: 0,7835 contra 0,7871. Com
vazamento, o teste desabaria abaixo da validação. Ele empata.

**Fecha com:** sabemos o que o modelo faz, o que ele não faz, e **onde ele não é
confiável** — medido, não estimado.

> **Dobradiça para o Ato III:**
> *"Com isso definido, o que sobrou foi entregar. E entregar aqui significa que outra
> pessoa consiga operar isso sem mim."*

---

### ATO III — O que está sendo entregue (3:30)

Quatro blocos. **É aqui que os números finais aparecem como resultado**, e não como etapa
de um processo.

| # | Bloco | O que se entrega |
|---|---|---|
| 1 | **O modelo final** | LightGBM calibrado. AUC **0,7868**, KS **0,4342**, Brier **0,0658**. Corte **0,09**, que é probabilidade de verdade — a média do score na base é 0,081 contra inadimplência real de 8,07%. Aprova **68,65%** da carteira com **3,48%** de inadimplência entre os aprovados, contra 8,07% sem modelo, e barra **70,4%** dos maus pagadores |
| 2 | **A régua de decisão** | Três faixas, não duas: APROVAR abaixo de 0,065, NEGAR acima de 0,115, **REVISAR** no meio. A faixa cinza **dobra** nos dois segmentos onde a fraqueza foi medida. O critério de risco é o mesmo para todos; o que muda é quanto vai para revisão humana — **15,5%** da carteira |
| 3 | **A disponibilização** | API com **27 endpoints**, servindo o mesmo artefato que o treino produziu. SHAP por cliente com verificação de consistência. E `GET /model/threshold-analysis`, que **recalcula a régua para qualquer razão de custo ao vivo, sem re-treinar** — se a banca pedir 20:1 agora, a resposta sai em segundos: corte 0,04, aprovação 49,8% |
| 4 | **O re-treino e o monitoramento** | DAG do Airflow com **9 tasks**, a cada **7 dias**, com um **gate que falha a DAG** se o AUC cair mais de 0,01 contra a última rodada aceita. Drift medido por PSI. Prometheus e Grafana com **5 alertas**, incluindo o que distingue "a API respondeu" de "a API consegue pontuar um cliente" |

**A prova de que nada disso é encenação:** a rodada que está sendo servida agora **foi
produzida pelo próprio DAG do Airflow**, não por alguém rodando um script na mão. A tag
dela diz isso.

**Fecha com os limites declarados** — e este é o fecho certo para uma banca, não uma
promessa: não está pronto para produção, e o trabalho diz onde não está. Falta
autenticação, registry com rollback e notificação ativa. A fraqueza em jovens é **mitigada,
não resolvida**. E o resíduo de −0,0308 que sobra depois de controlar tudo que sabíamos
controlar continua sem explicação mecânica — está escrito no laudo.

---

### Onde cada número aparece

O mesmo número dito duas vezes perde força. Esta tabela evita isso: cada valor tem **um**
lugar. Os quatro marcados com ▸ são construídos no Ato II e **pagos** no Ato III.

| Ato | Números que só aparecem aqui |
|---|---|
| **I** | 10:1 · 8,07% · 56,4% · 307.511 |
| **II** | 473 → 1.020 · 24,14% · 39,71% · 0,7776 × 0,7871 · 0,8753 / 0,7835 / 0,7871 · 0,47 · ▸ 0,1668 → 0,0658 · 77,6% · −0,0514 (p 0,008) · 0,7803 × 0,7319 · v4 rejeitada |
| **III** | ▸ AUC 0,7868 · ▸ corte 0,09 · ▸ 68,65% e 3,48% · 15,5% · 27 endpoints · 9 tasks / 7 dias · 5 alertas |

### As duas dobradiças que não podem falhar

Se você decorar duas frases desta seção, que sejam estas — são as transições entre atos, e
é onde uma apresentação desmonta:

1. **I → II:** *"Este trabalho não começou num modelo. Começou em construir a informação
   que não vinha pronta."*
2. **II → III:** *"Com isso definido, o que sobrou foi entregar — e entregar significa que
   outra pessoa consiga operar isso sem mim."*

---

## §1. Dois erros, dois preços

Uma financeira de crédito ao consumidor perde dinheiro nos dois extremos da mesma decisão.
Se aprova alguém que não paga, perde o capital emprestado. Se nega alguém que pagaria,
perde a margem daquele contrato — e, num público sub-bancarizado, ainda empurra a pessoa
para fora do sistema financeiro. Os dois são erros, mas **não custam a mesma coisa**.

No projeto, essa assimetria virou um número explícito, em `Model/config.yaml`:
`cost_false_negative = 1,0` e `cost_false_positive = 0,10`. Uma razão de **dez para um**.
Não é um número descoberto no dado; é uma premissa de negócio, e é assim que ela deve ser
apresentada — declarada, parametrizada e sujeita a revisão pelo negócio.

Assumir isso tem uma consequência imediata e pouco intuitiva: **acurácia deixa de servir
como métrica.** A base tem 8,07% de inadimplência, ou seja, 92 contra 8. Um modelo que
aprova todo mundo acerta 92% das vezes e é inútil — pior, é exatamente o modelo mais caro
possível, porque concentra todo o erro no lado que custa dez vezes mais. Qualquer métrica
que trate os dois erros como iguais está estruturalmente errada para este problema.

O que sobra é medir duas coisas separadamente: **se o modelo ordena bem** os clientes por
risco — e para isso serve o AUC — e **onde cortar** essa ordenação, que é uma pergunta de
custo, não de estatística.

> **A pergunta que fica:** se acurácia não serve e o corte é uma conta de custo, o modelo
> precisa devolver uma probabilidade de verdade. De onde vem essa informação?

---

## §2. O melhor dado é o que menos existe

Do dataset Home Credit: 307.511 clientes, sete arquivos, 2,5 GB. A tabela principal traz
122 colunas de cadastro e situação; as outras seis trazem histórico — 1,7 milhão de linhas
de bureau, 27,3 milhões de saldos mensais, 13,6 milhões de parcelas pagas.

A exploração deu três achados que orientaram tudo o que veio depois.

O primeiro: **há sinal, e ele é desigual.** A inadimplência por escolaridade vai de 1,8%
(doutorado) a 10,9% (fundamental incompleto). Por tipo de renda, vai de 0% a 36–40%. Não
é ruído — é estrutura, e um modelo consegue capturá-la.

O segundo: **os dados mentem em alguns pontos.** `DAYS_EMPLOYED` traz o valor 365243 para
uma parcela grande da base, o que daria mil anos de emprego. É uma sentinela de "não se
aplica" — aposentados, na maioria. Tratar como número teria envenenado toda variável
derivada de tempo de emprego. Vira nulo na sanitização, junto com `CODE_GENDER = 'XNA'`.

O terceiro é o que define a tese do trabalho. As variáveis mais correlacionadas com
inadimplência são os três scores de bureau externo, `EXT_SOURCE_1/2/3` — correlação de
−0,16 a −0,18, muito acima de qualquer outra. **E são justamente as que mais faltam.**
`EXT_SOURCE_1` está ausente em **56,4%** dos casos; `EXT_SOURCE_3` em 19,8%. E não é acaso:
falta porque o público é sub-bancarizado. Quem menos tem histórico externo é exatamente
quem mais precisa da decisão de crédito.

Essa é a frase que resume o problema: **o sinal mais forte é o que menos está disponível.**

> **A pergunta que fica:** se o melhor dado falta em mais da metade dos casos, de onde vem
> o resto?

---

## §3. O bug que valia mais que qualquer hiperparâmetro

Das outras seis tabelas. Mas elas têm granularidade incompatível: uma linha por contrato,
por mês, por parcela. Foi preciso construir uma **ABT** (Analytical Base Table — tabela
analítica com uma linha por cliente), agregando tudo por `SK_ID_CURR` com média, soma,
máximo, mínimo e contagem.

Foi aí que a primeira versão do trabalho tropeçou. A função de agregação selecionava as
colunas com `select_dtypes("number")` — e **descartava em silêncio todas as variáveis
categóricas** das tabelas relacionais. Nenhum erro, nenhum aviso: o pipeline rodava e
entregava uma ABT que parecia completa.

O que foi jogado fora era exatamente o histórico de comportamento. `bureau_balance.STATUS`,
que é o atraso mês a mês de cada contrato. `previous_application.NAME_CONTRACT_STATUS`,
que diz se o pedido anterior foi aprovado ou **recusado**. `CODE_REJECT_REASON`, o motivo
da recusa. `CREDIT_ACTIVE`, se a dívida ainda está viva.

Corrigido, a ABT foi de **473 para 1.020 colunas**. E junto entraram cinco famílias de
variáveis novas: sinalizadores de presença de cada fonte, combinações dos scores externos,
razões financeiras (crédito sobre renda, prestação sobre renda, prazo), comportamento de
pagamento (atraso em dias, razão pago/devido, utilização do cartão) e uma janela recente
de 12 e 24 meses em paralelo à vitalícia — para distinguir quem piorou de quem melhorou.

O resultado mais eloquente: a variável mais importante do modelo passou a ser
**`EXT_SOURCE_MEAN`**, a média dos scores externos que aquele cliente *tem*, com **24,14%**
de toda a importância — contra 1,45% do melhor score isolado. Ou seja, o problema nunca
foi não ter os scores; era **desperdiçar os que existem**.

E as seis tabelas relacionais respondem hoje por **39,71%** da importância total. A ABT se
pagou, e isso é mensurável, não opinião.

> **A pergunta que fica:** com o dado certo na mesa, o modelo já resolve?

---

## §4. Ordena bem — mas não é probabilidade

A partição é estratificada por inadimplência, com semente 42, em quatro partes:
**50% treino, 20% validação, 10% calibração, 20% teste**. A fatia de teste, 61.503
clientes, não é tocada por nenhuma decisão de modelagem — nem escolha de hiperparâmetro,
nem escolha de corte.

Primeiro um baseline interpretável: regressão logística com imputação, padronização e
one-hot, `class_weight=balanced`. AUC de teste **0,7776**. Ele existe para dar referência
— sem ele, "0,78" não significa nada.

Depois o campeão: LightGBM, que trata nulos e categóricas de alta cardinalidade
nativamente. Isso importa muito aqui, porque **os nulos deste dataset são estruturais, não
erro**: "não tem score de bureau" é informação, e imputar pela mediana apagaria o sinal.
AUC de teste **0,7871**, KS 0,4354, parando por early stopping na iteração 507 de 2.000.

Sobre overfitting, o argumento correto **não é o gap**. O AUC de treino é 0,8753 e o de
validação 0,7835 — uma distância de 0,088, que assusta à primeira vista e é esperada com
mil variáveis. O que importa é o **empate entre validação (0,7835) e teste (0,7871)**. Se
houvesse vazamento ou ajuste ao ruído, o teste desabaria abaixo da validação. Ele não
desaba; ele empata. Essa é a prova.

Só que havia um problema que o AUC não enxerga. Para lidar com o desbalanceamento, o
LightGBM roda com `is_unbalance=true`, o que **distorce a escala da saída**. O modelo
ordenava bem, mas o número que ele devolvia não era a probabilidade de inadimplência. A
evidência prática: o corte ótimo de custo caía em **0,47**. Num problema em que 8 clientes
em 100 não pagam, dizer "nego quem tem 47% de risco" é indefensável diante de qualquer
banca — e, pior, impossível de explicar a um analista de crédito.

> **A pergunta que fica:** como transformar uma boa ordenação numa probabilidade que se
> possa defender?

---

## §5. Calibrar antes de cortar

Com **regressão isotônica** — uma transformação monotônica que reescreve os valores do
score sem alterar a ordem. Ajustada sobre a fatia de 10% reservada só para isso, que
**nunca entrou no treino nem no early stopping**. É por essa razão que a partição tem
quatro partes e não três: calibrar na mesma fatia que treinou seria calibrar sobre o
otimismo do próprio modelo.

O efeito é o esperado e é dramático onde deveria ser. O **Brier** — que mede o erro
quadrático da probabilidade, ou seja, se o número está no nível certo — cai de **0,1668
para 0,0658**, uma melhora de 2,5 vezes. O AUC praticamente não se move: de 0,7871 para
**0,7868**. E não deveria mesmo mover: a isotônica é monotônica, preserva a ordem; o que
ela muda é a régua, não o ranking. A terceira casa que oscila vem dos empates que a
transformação cria.

Esse ponto merece cuidado, porque é onde uma banca testa se você entende o que fez:
**calibrar não melhora o modelo, melhora o significado do número.** Quem espera ganho de
AUC não entendeu para que serve.

E o corte ótimo saiu de 0,47 para **0,09** — que é a ordem de grandeza que se espera de uma
carteira com 8% de inadimplência e custo dez para um. Há uma verificação independente que
fecha o argumento: a média do score calibrado em toda a base é **0,081**, contra
inadimplência real de **8,07%**. O modelo passou a falar na mesma unidade do mundo.

> **A pergunta que fica:** agora dá para cortar. Onde exatamente?

---

## §6. O corte deixa de ser escolha e vira conta

Varrendo 99 pontos de corte sobre a fatia de **validação** — nunca sobre o teste — e
calculando, em cada um, o custo total: dez unidades para cada mau pagador aprovado, uma
para cada bom pagador negado. O mínimo está em **0,09**, com custo 3.049,6. Aprovar todo
mundo custaria 4.965,0 — o corte por custo **economiza 38,6%** dessa perda.

O que esse corte entrega, em linguagem de negócio: aprova **68,65%** da carteira, e entre
os aprovados a inadimplência cai para **3,48%**, contra 8,07% da carteira sem modelo. É
uma redução de cerca de 57% no risco da carteira aprovada, mantendo quase sete em cada dez
pedidos aprovados. Do lado que ficou de fora, o modelo barra **70,4%** de todos os maus
pagadores.

Mas um corte único joga fora informação. Um cliente com 8,9% e outro com 9,1% recebem
decisões opostas por uma diferença que está dentro da incerteza do modelo. Por isso a
decisão final não tem duas saídas, tem três: uma **faixa cinza** de 0,05 de largura em
torno do corte. Abaixo de 0,065, **APROVAR**. Acima de 0,115, **NEGAR**. No meio,
**REVISAR** — análise humana.

E essa faixa **dobra** para os segmentos em que o modelo comprovadamente ordena pior.
Aqui há um ponto de governança que precisa ser dito com precisão, porque é onde se
confunde tudo: **o critério de risco é idêntico para todos.** Ninguém é negado por ser
jovem. O que muda é *quanto daquele grupo vai para revisão humana*, e isso é decidido pela
confiabilidade medida do modelo naquele perfil — não pelo perfil. Diferenciar o corte por
idade ou gênero seria discriminação direta, e não é feito.

Isso custa: **15,5%** da carteira passa a exigir análise manual. A régua torna esse custo
visível, em vez de escondê-lo dentro de uma decisão automática mal fundamentada.

> **A pergunta que fica:** e onde, exatamente, esse modelo erra?

---

## §7. Fomos medir — e o erro estava no nosso critério

O projeto já declarava duas fraquezas: clientes jovens e clientes sem histórico de bureau
(*thin-file*). Ao instrumentar isso com intervalo de confiança, apareceram duas coisas, e
a segunda é maior que a primeira.

A primeira: **thin-file não é fraqueza.** O AUC é menor — 0,7745 contra 0,7878 de quem tem
bureau —, mas a diferença é −0,0133 com p = 0,132. Está dentro do ruído amostral. Assumir
isso é mais forte do que sustentar um número que não se sustenta.

A segunda: **o critério que usávamos para declarar fraqueza era estatisticamente
inválido.** A regra era "o segmento é fraco quando o topo do intervalo de confiança dele
fica abaixo do piso do intervalo do AUC geral". Ela estava implementada em quatro lugares
do código — e **em duas versões diferentes**, o que já era o sintoma.

Tem três defeitos, e vale saber os três:

O primeiro é que **o grupo está dentro do geral**. As duas estimativas são aninhadas e
correlacionadas; comparar seus intervalos não é um teste da diferença entre elas.

O segundo é o mais interessante, e é o que dá o número que se cita. **O AUC geral não é
comparável a um AUC dentro de um grupo.** AUC é a probabilidade de ordenar corretamente um
par (inadimplente, adimplente). O AUC geral conta pares formados por clientes de faixas
etárias *diferentes* — comparações que nenhum AUC intra-faixa realiza. Medimos: apenas
**22,4%** dos pares do AUC agregado são dentro da mesma faixa. **77,6% são entre faixas.**
Estávamos comparando duas coisas que medem eventos diferentes.

O terceiro é que **sobreposição de intervalo não é teste de hipótese**, nem para amostras
independentes — é um teste conservador com nível efetivo perto de 0,006, não 0,05.

O detalhe que fecha o argumento: **os dois primeiros defeitos puxam em direções opostas.**
O critério não era sistematicamente severo nem sistematicamente brando — era
*descalibrado*, que é muito pior de defender.

Trocamos por um **bootstrap estratificado da diferença**: reamostra-se, e em cada réplica
compara-se o AUC do grupo contra o AUC medido *dentro* dos demais grupos do mesmo eixo,
ponderado por pares. Um grupo é fraqueza quando o intervalo de 95% dessa diferença exclui
o zero pelo lado negativo. As réplicas são compartilhadas entre grupos, então todos são
julgados na mesma reamostragem.

E o resultado importa: **a conclusão sobreviveu.** A faixa `<25` tem diferença de
**−0,0514**, intervalo [−0,0827; −0,0205], **p = 0,008**. A inflação causada pelo critério
errado explicava só cerca de 6% do buraco. Os mesmos clientes continuam indo para revisão
humana — agora sobre base válida.

> **A pergunta que fica:** está confirmado que o modelo ordena pior os jovens. Então
> conserta.

---

## §8. Tentamos consertar. E o que caiu foi a nossa explicação

O projeto afirmava a causa: *"a fraqueza vem de ausência de histórico, e histórico é
exatamente o que um cliente de 22 anos não tem."* Soa óbvio. Testamos.

Construímos uma **coorte pareada**: clientes maduros, de 25 a 45 anos, reamostrados estrato
a estrato até reproduzir o perfil de informação de um jovem — mesma taxa de thin-file,
mesmo número de scores externos presentes, mesma faixa de tempo de emprego, mesmo
comprimento de histórico de parcelas. Foram 71 estratos, cobrindo 99,9% da faixa.

Se a causa fosse falta de histórico, essa coorte deveria pontuar como os jovens. Ela marca
**0,7803**, contra **0,7319** dos jovens — e é melhor em **99,8%** das réplicas. Dito de
outro jeito: **dê a um cliente de 33 anos exatamente a pobreza informacional de um de 22, e
o modelo continua ordenando-o normalmente.** A explicação que dávamos estava errada.

Testamos mais duas hipóteses nossas, e as duas caíram: não é efeito dos empates criados
pela calibração (o impacto nos jovens é da mesma ordem das outras faixas), e não é
restrição de amplitude (a dispersão do score entre os jovens é igual à das demais faixas —
o que muda é a média, não o espalhamento).

O que sobrou, e é parcial: cerca de **36%** do efeito é estar concentrado na **região baixa
do score externo**, onde o modelo separa pior *para qualquer idade*. Isso não é sobre ser
jovem. Sobra um resíduo de **−0,0308** genuinamente etário, presente em 98,8% das réplicas,
e a causa mecânica dele **nós não sabemos** — está declarado no laudo.

E o conserto? Duas tentativas, ambas registradas e rejeitadas. Reponderar o treino para dar
mais peso aos jovens (`v4-pesos-idade`) **piorou o próprio alvo** em 0,0032. Treinar um
modelo dedicado: seis variantes, duas faixas, três capacidades — **todas abaixo do modelo
geral** no mesmo conjunto de teste, a melhor 0,7296 contra 0,7319.

O veredito, então, tem duas metades. Sobre o **nível de risco**, o modelo está certo:
jovens são mesmo mais inadimplentes, 11,8% contra 8,07%, e a taxa cai monotonicamente com
a idade. Sobre a **ordenação**, a fraqueza é real e é **teto de dado** — o sinal disponível
não distingue bem dentro desse grupo.

E há um achado novo, que encontramos e **decidimos não corrigir**. A faixa `<25` é a única
com viés de calibração: o modelo prevê 13,4% onde ocorrem 11,8%. A causa é a isotônica ser
global. Ajustá-la por faixa zera o viés — e **derruba a aprovação dos jovens de 47,6% para
44,8%**, sem melhorar nenhuma decisão individual. Está medido, documentado e pendente de
decisão de negócio. Achar o conserto e recusá-lo por honestidade sobre a contrapartida é
uma decisão, não uma omissão.

A resposta certa a um teto de dado não é insistir em engenharia de variáveis. É **desenhar
o processo em volta dele** — que é exatamente a régua de três faixas do movimento anterior.

> **A pergunta que fica:** tudo isso é slide, ou existe rodando?

---

## §9. Existe, e se defende sozinho

O modelo é servido por uma **API** FastAPI com **27 endpoints**, que consultam a base por
DuckDB direto sobre Parquet. Não são 27 rotas decorativas: cada uma responde a uma pergunta
que uma banca faz. `GET /model/fairness` devolve o quadro por segmento com o critério novo.
`GET /clients/{id}/explain` devolve as contribuições SHAP daquele cliente, com um
`consistency_check` que prova que a soma reconstrói a probabilidade do modelo. E
`GET /model/threshold-analysis` **recalcula a régua de custo ao vivo para qualquer razão
FN:FP, sem re-treinar nada** — se a banca pedir para ver o cenário 20:1 agora, a resposta
sai em segundos.

Sobre esse serviço há um **dashboard Streamlit** de quatro abas, que consome a API por
HTTP como qualquer outro cliente — não importa o modelo no próprio processo, para que o
diagrama de arquitetura seja literal.

O re-treino é orquestrado no **Airflow 3.3.1**, numa DAG de **nove tasks** que roda **a
cada 7 dias**: confere as fontes, sanitiza, monta a ABT, valida a ABT contra vazamento,
perfila as colunas, treina, valida as métricas, calcula drift e imprime o resumo. A sétima
task é um **gate**: se o AUC cair mais de 0,01 contra a última rodada aceita, ela levanta
exceção e **a DAG falha**. Modelo pior não é promovido em silêncio.

E há **telemetria**: Prometheus raspando a API a cada 10 segundos, Grafana com 18 painéis e
cinco regras de alerta. A decisão de projeto que vale citar: **não usamos uma sonda externa
de disponibilidade**, porque ela responderia "está no ar" mesmo com o modelo não carregado
— o servidor continua vivo servindo métricas. Em vez disso a própria API expõe
`hc_api_pronta`, que replica exatamente a regra do healthcheck, e `hc_erro_componente`, que
diz *qual* peça falhou. A diferença entre "respondeu" e "consegue pontuar um cliente" é a
diferença entre monitorar e fingir que monitora.

Tudo isso é sustentado por **159 testes** que rodam sem os 1,3 GB da base, sobre uma base
sintética de 300 clientes — inclusive os que travam a partição contra a rodada congelada e
os que tentam injeção de SQL nos filtros.

E o fecho, que é o melhor argumento de que nada disso é encenação: **a rodada que está
sendo servida hoje foi produzida pelo próprio DAG do Airflow**, não por alguém rodando um
script na mão. A tag dela diz isso: `airflow-scheduled-2026-08-28T1423...`.

---

## §10. As três versões da mesma história

Não são resumos diferentes. É a mesma espinha com mais carne — estudar uma reforça as
outras.

### 60 segundos

> "Numa financeira, aprovar quem não paga custa dez vezes mais do que negar quem pagaria.
> Isso descarta acurácia e exige uma probabilidade de verdade. O problema é que o dado que
> melhor prevê inadimplência falta em 56% dos casos, porque o público é sub-bancarizado.
> Construímos a informação a partir das nove tabelas de histórico — e no caminho achamos um
> bug que descartava em silêncio metade das variáveis. Calibramos o score para que ele
> fosse mesmo probabilidade, e o corte por custo passou a aprovar 68,65% da carteira com
> 3,48% de inadimplência entre os aprovados, contra 8,07% sem modelo. E medimos onde ele
> falha: descobrimos que o nosso próprio critério de fraqueza era inválido, trocamos, e a
> conclusão sobreviveu — o modelo ordena pior os jovens, não há conserto disponível nestes
> dados, e a resposta é revisão humana onde ele é comprovadamente fraco. Está tudo servido
> em API, re-treinado a cada 7 dias com gate de qualidade e monitorado em tempo real."

### 3 minutos

Os nove movimentos, um parágrafo cada, sem os números de segunda ordem. A regra: cite
**um** número por movimento — o que carrega a virada. São eles: 10:1 · 56% · 473→1.020 ·
0,47 · 0,0658 · 68,65% · 77,6% · 0,7803 · 9 tasks.

### 12 minutos

É a apresentação em si, e ela **não** é a Parte I lida na ordem: é a Parte I **encenada em
três atos**, conforme o §0.1. A diferença que importa: os números de resultado são
construídos no Ato II e apresentados no Ato III. Anunciar 68,65% de aprovação no meio
queima o fecho.

Ato I — a dor (2:30) · Ato II — o desenvolvimento (6:00) · Ato III — a entrega (3:30).
O Ato II é o mais longo porque é onde a banca decide se você entende o que fez.

---

## §11. Se você travar

Três perguntas que devolvem você ao trilho a partir de qualquer ponto da apresentação.
Responder qualquer uma delas em voz alta recoloca a narrativa nos eixos.

1. **"Qual dos dois erros custa mais aqui?"** → volta ao §1, e de lá a história inteira
   destrava para a frente.
2. **"O número que o modelo devolve é probabilidade de verdade?"** → volta ao §4–§5, o
   miolo técnico.
3. **"Onde este modelo falha, e como vocês sabem?"** → volta ao §7–§8, que é o bloco mais
   forte do trabalho.

E se a pergunta pegar você em cheio, a resposta honesta tem lugar aqui: o trabalho tem uma
seção inteira sobre o que ele **não** prova (Parte II, §F). Dizer "isso nós não medimos, e
está declarado no laudo" é mais forte do que improvisar.

---
---

# PARTE II — O ARSENAL

Daqui para baixo nada é para memorizar. É para consultar.

## §A. Perguntas da banca

A coluna **↩** diz de qual movimento da Parte I a resposta sai. Responda **narrando o
movimento**, não recitando o número — é assim que a resposta soa como entendimento e não
como cola.

### Sobre a modelagem

| ↩ | Pergunta | Resposta |
|---|---|---|
| §4 | **Por que LightGBM?** | Porque os nulos deste dataset são estruturais, não erro: "não tem score de bureau" é informação. O LightGBM trata nulo e categórica de alta cardinalidade nativamente, sem imputar. E testamos contra um baseline interpretável — regressão logística, 0,7776 — para ter referência. Ganho de 0,0092 no AUC e de 2,5× no Brier. |
| §4 | **E por que não uma rede neural?** | Em tabular com mil colunas, muita categórica e nulo estrutural, gradient boosting é o estado da prática, e o baseline linear já dá o piso. Uma rede exigiria imputar e codificar tudo — justamente o que o LightGBM dispensa. Não testamos, e isso está declarado como limitação. |
| §4 | **Como sabem que não é overfitting?** | O argumento não é o gap treino→validação (0,8753 → 0,7835), que é esperado com mil variáveis. É o **empate validação × teste**: 0,7835 contra 0,7871. Com vazamento ou ajuste ao ruído, o teste desabaria abaixo da validação. Early stopping parou na iteração 507 de 2.000. |
| §4 | **Por que não validação cruzada?** | Holdout único estratificado com semente fixa, porque a regra de aceite do projeto exige que **toda rodada seja comparável à anterior no mesmo conjunto de teste**. Com k-fold, cada rodada mediria numa partição diferente e o gate de regressão do Airflow perderia sentido. Com 61.503 clientes no teste, o erro amostral já é pequeno. |
| §5 | **Por que calibrar, se o AUC não melhora?** | Porque não é para melhorar o AUC — a isotônica é monotônica, preserva a ordem por construção. É para o número **significar** alguma coisa. Brier 0,1668 → 0,0658 e corte 0,47 → 0,09. Verificação: média do score 0,081 contra inadimplência real 8,07%. |
| §5 | **Por que uma fatia só para calibrar?** | Calibrar na mesma fatia que treinou seria calibrar sobre o otimismo do próprio modelo. Os 10% de calibração não entram no fit nem no early stopping. Custou 0,0009 de AUC — está registrado. |
| §6 | **Por que o corte é 0,09 e não 0,5?** | 0,5 é o padrão de classificação binária balanceada, e esta carteira tem 8% de inadimplência com custo assimétrico. O 0,09 sai de varrer 99 cortes na validação e escolher o de custo mínimo, com FN valendo 10× FP. Não foi escolhido, foi calculado. |
| §3 | **A ABT valeu a pena, ou é só volume?** | Mensurável: as seis tabelas relacionais respondem por **39,71%** da importância total. E a correção que as recuperou fez a variável nº 1 do modelo virar `EXT_SOURCE_MEAN`, com 24,14% — contra 1,45% do melhor score isolado. |
| §3 | **Como garantem que não há vazamento?** | Só entram variáveis conhecidas no momento do pedido. A ABT é validada por uma task própria do DAG, que rejeita cinco padrões de nome (`PROBA`, `PREDIC`, `SCORE_MODEL`, `Y_TRUE`, `Y_PRED`) e confere unicidade de `SK_ID_CURR`. O teste nunca é tocado por decisão de modelagem. |

### Sobre o diagnóstico — o bloco mais cobrado

| ↩ | Pergunta | Resposta |
|---|---|---|
| §8 | **O modelo discrimina jovens?** | São três perguntas diferentes, e a resposta é diferente em cada uma. *Jovens são mais inadimplentes?* Sim — 11,8% contra 8,07%, e cai monotonicamente com a idade. *O modelo ordena pior os jovens?* Sim — 0,7319 contra 0,7833 dos demais, p = 0,008. *O modelo erra o nível de risco deles?* Sim, **para mais**: prevê 13,4% onde ocorrem 11,8%. Confundir as três é o erro mais comum aqui. |
| §6 | **Então vocês tratam jovem diferente?** | Não no critério de risco — o corte é idêntico para todos, e ninguém é negado por idade. O que muda é **quanto do grupo vai para revisão humana**: a faixa cinza dobra onde a confiabilidade do modelo foi medida como menor. É decidido pela medição, não pelo perfil. |
| §7 | **Vocês trocaram o próprio critério estatístico. O anterior estava errado?** | Estava. Comparávamos o IC do segmento com o IC do AUC geral — que contém o próprio segmento e é composto em **77,6%** por pares entre faixas etárias, comparações que nenhum AUC intra-faixa faz. E sobreposição de IC não é teste de hipótese. Trocamos por bootstrap pareado da diferença. **A conclusão sobreviveu**: −0,0514, p = 0,008. O critério errado explicava só 6% do buraco. |
| §7 | **Por que só descobriram isso agora?** | Porque o critério estava implementado em quatro lugares do código e em **duas versões divergentes** — foi essa divergência que nos fez olhar. Está corrigido nos quatro, com o campo antigo depreciado na API em vez de removido em silêncio. |
| §8 | **Vocês testaram consertar a faixa `<25`?** | Duas vezes, ambas registradas e rejeitadas. Reponderar o treino piorou o próprio alvo em 0,0032 — está no `improvement_log.json` com o motivo escrito. Modelo dedicado: seis variantes, todas abaixo do geral, a melhor 0,7296 contra 0,7319. É o que sustenta o veredito de teto em vez de opinião. |
| §8 | **Como sabem que é falta de dado e não falha do modelo?** | Pela coorte pareada: clientes de 25 a 45 anos reamostrados até ter o mesmo perfil de informação de um jovem — 71 estratos, 99,9% de cobertura — marcam **0,7803** contra 0,7319, e são melhores em 99,8% das réplicas. Dê a um cliente de 33 anos a pobreza informacional de um de 22 e o modelo continua ordenando bem. |
| §8 | **Isso derruba a recomendação de dados alternativos?** | Derruba o **argumento** que a sustentava, não a ideia. Ela era justificada por "falta histórico" — e isso foi refutado. Continua plausível, mas agora precisa mostrar que a fonte nova discrimina **dentro** do grupo jovem, não apenas que preenche lacunas. |
| §E | **E a faixa 55-65?** | Fraqueza confirmada, −0,0415, p = 0,004 — e por outro motivo. Não é aposentadoria: 68,4% da faixa é aposentada, mas dentro dela aposentado (0,7488) e ativo (0,7427) empatam. A causa é que os três scores externos rendem ali o pior de todas as faixas. E o contraste que separa os dois casos: em 55-65 o modelo **acrescenta o normal** sobre um sinal ruim (ganho 0,0744); em `<25` acrescenta o mínimo de todas as faixas (0,0518). Lá é deficiência da fonte; aqui é fonte fraca **e** modelo agregando pouco. |
| §7 | **E os thin-file, não eram fraqueza?** | Não. AUC 0,7745 contra 0,7878, mas a diferença é −0,0133 com p = 0,132 — dentro do ruído. Retiramos a alegação. Mantivemos a mitigação de revisão humana mesmo assim, por prudência. |
| §8 | **Vocês acharam um viés de calibração e não corrigiram?** | Achamos, medimos e **decidimos não adotar** o conserto. Ele zera o viés agregado, mas não melhora nenhuma decisão individual e **derruba a aprovação dos jovens de 47,6% para 44,8%**. Está documentado como pendente de decisão de negócio — não é omissão, é uma escolha com contrapartida declarada. |
| §8 | **O que vocês não conseguem explicar?** | O resíduo de −0,0308 que sobra depois de controlar perfil de informação e nível do score. Ele é real e específico da idade, e a causa mecânica é desconhecida. Está escrito no laudo, na seção do que o documento não prova. |

### Sobre negócio

| ↩ | Pergunta | Resposta |
|---|---|---|
| §6 | **Qual o impacto financeiro em R$?** | **Não temos valor absoluto, e não vou inventar um.** O que temos é a razão de custo 10:1 e a economia relativa: o corte por custo evita **38,6%** da perda de aprovar todo mundo, e a inadimplência da carteira aprovada cai de 8,07% para 3,48%. Converter em reais exige o ticket médio e a margem, que são dados da financeira, não do dataset. O `GET /model/threshold-analysis` recalcula tudo assim que o negócio informar os custos reais. |
| §1 | **De onde veio o 10:1?** | É premissa de negócio, declarada e parametrizada, não descoberta no dado. Está em `Model/config.yaml` e é revisável em uma linha. É por isso que o endpoint de análise de threshold existe: o modelo não precisa ser re-treinado quando a premissa mudar. |
| §9 | **Se eu pedir para mudar o custo de FN para 20:1 agora, quanto tempo leva?** | Segundos, sem re-treinar. `GET /model/threshold-analysis?cost_fn=20&cost_fp=1` devolve o corte novo, a taxa de aprovação e a inadimplência resultante. O dashboard tem esse controle deslizante na aba Modelo. Em 20:1 o corte vai para 0,04 e a aprovação cai para 49,8%. |
| §6 | **Quanto custa a revisão humana?** | 15,5% da carteira. A régua torna esse custo explícito. A alternativa — decidir tudo automaticamente — não elimina o custo, só o transfere para a inadimplência e para decisões erradas em segmentos onde o modelo é comprovadamente pior. |

### Sobre a entrega em produção

| ↩ | Pergunta | Resposta |
|---|---|---|
| §9 | **Isso está pronto para produção?** | Não, e o trabalho diz onde não está. Falta autenticação, registry de modelo com rollback, logs estruturados e notificação ativa. O que existe roda: API, dashboard, DAG semanal com gate, drift e telemetria com alertas. |
| §9 | **A API não tem autenticação?** | Não, deliberadamente: é projeto acadêmico servindo dados públicos do Kaggle. Em produção entrariam autenticação, rate limit e CORS restrito. O que **existe** de segurança é defesa contra injeção de SQL: todo valor é parametrizado e todo identificador é validado contra whitelist derivada do schema. Coberto por seis testes. |
| §9 | **O que o `/metrics` expõe?** | Volume, latência, distribuição das decisões e saúde dos componentes. **Nenhum dado de cliente.** É aberto como o resto da API, e isso está registrado como limitação. |
| §9 | **E se o modelo piorar no re-treino?** | A sétima task da DAG compara o AUC servido contra a última rodada aceita. Queda maior que 0,01 levanta exceção e **a DAG falha** — o histórico de rodadas fica registrado com status e motivo. Já rejeitamos uma rodada assim (`v4-pesos-idade`). |
| §9 | **Como detectam drift?** | PSI (Population Stability Index — índice de estabilidade populacional) do score contra a rodada anterior, calculado numa task da DAG e servido em `GET /model/psi`. Faixas de leitura: abaixo de 0,10 estável, 0,10–0,25 atenção, acima de 0,25 mudança relevante. Hoje está em 0,00038. |
| §9 | **Por que Airflow e não um cron?** | Porque o pipeline tem nove etapas com dependência, uma delas leva 11 minutos e outra 15, e uma delas é um **gate que precisa falhar visivelmente**. Cron dá agendamento; não dá retentativa por task, timeout por task, histórico de execuções nem a distinção entre "falhou o treino" e "o treino rodou e o modelo piorou". |
| §9 | **Por que três stacks Docker separadas?** | Ciclos de vida diferentes. A API sobe em segundos e precisa ficar de pé; o Airflow tem cinco serviços e ~4 GB de imagem; o monitoramento acompanha a API mas não deve derrubá-la se cair. São nove containers ao todo, e o Makefile sobe as três de uma vez ou cada uma isolada. |
| §9 | **Por que um único worker no uvicorn?** | Porque cada processo teria seu próprio registro de métricas e o `/metrics` devolveria valores diferentes a cada raspagem. Além disso, um explainer e um pool DuckDB só, com memória previsível. É carga estrutural e está documentada — escalar exigiria métricas em agregador externo. |
| §9 | **O SHAP explica o modelo calibrado?** | Não — explica o LightGBM cru, antes da isotônica. Ordem e sinal das contribuições continuam válidos, porque a isotônica é monotônica, e a resposta traz a probabilidade crua junto. Mas a soma reconstrói a probabilidade crua, não a calibrada. Está registrado como dívida técnica: é possível e não foi feito. |
| §9 | **Por que não uma sonda externa de disponibilidade?** | Porque ela responderia "está no ar" mesmo com o modelo não carregado: o servidor continua vivo servindo métricas. A API expõe `hc_api_pronta`, que replica a regra do healthcheck, e `hc_erro_componente`, que diz qual peça falhou. A diferença entre "respondeu" e "consegue pontuar" é a diferença entre monitorar e fingir. |
| §9 | **Como sabem que os testes cobrem algo real?** | 159 testes sobre base sintética de 300 clientes com os nomes de coluna reais — a suíte não depende dos 1,3 GB. Entre eles: os que re-derivam a partição e conferem contra a rodada congelada, os que travam a régua de três faixas, os que tentam injeção de SQL, e o que garante que criar a aplicação duas vezes não quebra o registro de métricas. |

## §A.1. Divergências entre documentos — resposta pronta

A banca pode abrir dois arquivos do repositório e encontrar números diferentes. Todas
abaixo são conhecidas. A resposta geral: **a fonte de verdade é `artifacts/`, e
`python Model/run_summary.py --markdown` reimprime os números oficiais.**

| Onde | O que diverge | Resposta |
|---|---|---|
| `docs/TCC.md` §2 vs §4 | §2 diz *early stopping 654*, *threshold 0,50*, *aprovação 72,1%* | São os números da rodada **v2**, pré-calibração. O §4 do mesmo arquivo já traz os certos: 507, 0,09 e 68,65%. O §2 não foi reconciliado. |
| Contagem de endpoints | 27 · 26 · 25 em arquivos diferentes | **27** no OpenAPI. O `/metrics` foi deliberadamente deixado fora do schema justamente para não mexer nessa contagem. |
| Contagem de testes | 148 · 99 · 78 em documentos diferentes | **159** coletados hoje. Os números menores são de PRs anteriores; a diferença entre "funções `def test_`" e "testes coletados" vem de parametrização. |
| Dossiê §12 | Diz que telemetria de serviço "ficou de fora" | Ficou, até a PR #6. O texto do dossiê não é regerado automaticamente — só o bloco de dados é. |
| `README.md` linha 51 | 0,7871 ali, 0,7868 na tabela acima | Correto nos dois: **0,7871 é o modelo cru, 0,7868 é o servido** (calibrado). A isotônica move a terceira casa por criar empates. `GET /model/metrics` devolve os dois, rotulados. |
| `docs/guia_apresentacao_demoday.md` linha 436 | Ensina o critério antigo como se fosse válido | É material da fase em grupo, anterior à correção. **Este documento o substitui.** Se a banca citar aquela regra, a resposta é o §7 da Parte I. |

---

## §B. Cola de números — por movimento da narrativa

Ordenada pela história, não por tema. Um número por virada.

| § | Virada | Número |
|---|---|---|
| 1 | Os dois erros custam diferente | **10 : 1** (`cost_fn` 1,0 · `cost_fp` 0,10) |
| 1 | Base desbalanceada | **8,07%** de inadimplência · 307.511 clientes |
| 2 | O melhor dado falta | **56,4%** de nulos em `EXT_SOURCE_1` · 19,8% no `EXT_SOURCE_3` |
| 3 | O bug da ABT | **473 → 1.020** colunas · 1.018 features |
| 3 | O que a ABT rendeu | `EXT_SOURCE_MEAN` = **24,14%** da importância · relacionais = **39,71%** |
| 4 | Baseline → campeão | AUC **0,7776 → 0,7871** (cru) · KS 0,4219 → 0,4354 |
| 4 | Overfitting é o empate | treino 0,8753 · **validação 0,7835 · teste 0,7871** · iteração 507 |
| 4 | O score não era probabilidade | corte ótimo caía em **0,47** |
| 5 | A calibração | Brier **0,1668 → 0,0658** · AUC 0,7871 → **0,7868** |
| 5 | A verificação | média do score **0,081** × inadimplência real **8,07%** |
| 6 | O corte por custo | **0,09** · custo 3.049,6 vs 4.965,0 → economia de **38,6%** |
| 6 | O resultado de negócio | aprova **68,65%** · inadimplência dos aprovados **3,48%** · barra **70,4%** dos maus |
| 6 | A régua de três faixas | APROVAR < 0,065 · REVISAR até 0,115 · NEGAR acima · **15,5%** em revisão |
| 7 | Thin-file não é fraqueza | −0,0133 · **p = 0,132** |
| 7 | O critério antigo era inválido | só **22,4%** dos pares são intra-faixa · **77,6%** são entre faixas |
| 7 | A fraqueza sobreviveu | `<25`: **−0,0514**, IC [−0,0827; −0,0205], **p = 0,008** |
| 8 | A causa declarada caiu | coorte pareada **0,7803** × jovens **0,7319** · pior em 99,8% das réplicas |
| 8 | Não há conserto | 6 modelos dedicados, melhor **0,7296** < 0,7319 · reponderar piorou 0,0032 |
| 8 | O viés de calibração | prevê **13,4%**, ocorre **11,8%** · conserto derruba aprovação 47,6% → 44,8% |
| 9 | A entrega | **27** endpoints · **9** tasks · **7** dias · **5** alertas · **159** testes |

---

## §C. Ficha técnica — modelagem

**Rodada:** `20260828-144848` · 307.511 linhas · **1.018 features** (1.002 numéricas, 16
categóricas) · Python 3.14.3, LightGBM 4.6.0, scikit-learn 1.9.0.

**Partição** (estratificada por `TARGET`, `random_state=42`):
treino 153.755 (50%) · validação 61.502 (20%) · **calibração 30.751 (10%)** · teste 61.503 (20%).

**Baseline** — Regressão Logística: imputação (mediana / moda) → padronização → one-hot
(`max_categories=20`), `class_weight=balanced`, `max_iter=1000`.

**Campeão** — LightGBM: `n_estimators=2000` (parou em **507**), `learning_rate=0.02`,
`num_leaves=34`, `max_depth=8`, `min_child_samples=70`, `subsample=0.8`,
`colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`, `is_unbalance=true`,
`early_stopping_rounds=100`. Categóricas nativas, sem one-hot.

**Calibração** — isotônica via `CalibratedClassifierCV(FrozenEstimator(...))`, ajustada na
fatia exclusiva de 30.751.

| | AUC | KS | Brier |
|---|---|---|---|
| Baseline (LogReg) | 0,7776875 | 0,4218820 | 0,1874067 |
| Campeão cru (LGBM) | 0,7870868 | 0,4353562 | 0,1668004 |
| **Servido** (campeão + isotônica) | **0,7868061** | **0,4341797** | **0,0657601** |

**Importância por origem:** application 60,29% · PREV 11,88% · INST 10,78% · BUREAU 7,63%
· POS 4,16% · CC 3,73% · BUREAU_BALANCE 1,53%.

**Top 8 por gain:** `EXT_SOURCE_MEAN` 24,14% · `ORGANIZATION_TYPE` 9,23% ·
`EXT_SOURCE_MIN` 2,88% · `CREDIT_TERM` 2,00% · `EXT_SOURCE_MAX` 1,81% ·
`CREDIT_GOODS_RATIO` 1,54% · `EXT_SOURCE_3` 1,45% · `EXT_SOURCE_PROD` 1,29%.

**Histórico de rodadas** (`improvement_log.json`): v1-baseline 0,7846 (471 features) ·
v2-abt-corrigida 0,7880 (1.018) · v3-calibrado 0,7871 (Brier 0,0658, corte 0,09) ·
**v4-pesos-idade REJEITADA** · e três reproduções idênticas, incluindo a atual, gerada
pelo Airflow.

---

## §D. Ficha técnica — a entrega

**API** — FastAPI 0.138.1, **27 endpoints** em 6 grupos: Saúde (`/health`,
`/meta/columns`, `/meta/dimensions`, `/admin/reload`) · Clientes (3) · Estatísticas (5,
sob `/stats`) · Modelo (11, sob `/model`) · Score e explicabilidade (4). Fora do schema:
`/` (redireciona ao Swagger) e `/metrics`. Documentação interativa em `/docs`.

**Persistência** — DuckDB 1.5.5 **em memória**, como motor de consulta sobre Parquet: não
há arquivo de banco. Três views; `clients` é a ABT com os scores anexados. 307.511 linhas
× 1.020 colunas, 288 MB em Parquet contra 1,3 GB em CSV.

**Dashboard** — Streamlit 1.58.0, quatro abas (Carteira · Cliente · Simulação · Modelo),
consumindo a API por HTTP.

**Explicabilidade** — SHAP 0.52.0, `TreeExplainer`. A resposta traz `consistency_check`
com o erro de reconstrução, provando que base + contribuições reproduzem a saída do modelo.

**Orquestração** — Airflow 3.3.1 sobre PostgreSQL 16, LocalExecutor, cinco serviços. DAG
`treino_credit_scoring`, **schedule de 7 dias**, `max_active_runs=1`, nove tasks lineares:
`checar_fontes` → `sanitizacao` → `abt_transform` (11 min) → `validar_abt` →
`perfil_colunas` → `treino` (15 min) → **`validar_metricas`** → `calcular_psi` →
`resumo_da_rodada`. Rodada completa ~30 min. Gate: queda de AUC acima de **0,01** falha a
DAG. Modo demonstração com amostra roda em ~1 min e grava em pasta separada.

**Observabilidade** — Prometheus v3.14.0 (raspagem a cada 10 s) + Grafana 13.1.4 (18
painéis em três blocos). Cinco alertas: `APIForaDoAr`, `ModeloNaoCarregado`,
`ComponenteDegradado`, `LatenciaAlta`, `TaxaErro5xx`. Catorze métricas `hc_*`, incluindo
`hc_api_pronta`, `hc_predicoes_total{decision,endpoint}` e `hc_score_previsto`.

**Infra** — três stacks Docker, nove containers, cinco portas: 8000 API · 8501 dashboard ·
8080 Airflow · 9090 Prometheus · 3000 Grafana. Treino e serving no mesmo Python 3.14, de
propósito: pickle de modelo entre versões diferentes falha no pior momento possível.

**Testes** — **159** coletados em nove arquivos, sobre base sintética de 300 clientes.

**Makefile** — 76 alvos em dez seções. `make up` sobe as três stacks; `make urls` imprime
os endereços; `make obs-alertas` lista os alertas disparando.

---

## §E. O diagnóstico em profundidade

### As três perguntas que não podem ser confundidas

| Pergunta | Métrica correta | Resposta para `<25` |
|---|---|---|
| Jovens são mais inadimplentes? | taxa observada | **Sim** — 11,80% contra 8,07%, monotônico na idade |
| O modelo ordena mal os jovens? | AUC dentro do segmento | **Sim** — 0,7319 contra 0,7833 dos demais |
| O modelo erra o nível de risco deles? | previsto − observado | **Sim, para mais** — 13,4% contra 11,8% |

### Quadro por segmento (teste, n = 61.503)

| Eixo | Grupo | n | AUC | Δ vs. demais | p | Fraqueza | Aprovação | Inadimplência |
|---|---|---|---|---|---|---|---|---|
| — | geral | 61.503 | 0,7868 | — | — | — | 69,00% | 8,07% |
| gênero | F | 40.561 | 0,7795 | −0,0078 | 0,240 | não | 73,4% | 6,99% |
| gênero | M | 20.940 | 0,7872 | +0,0078 | 0,240 | não | 60,4% | 10,17% |
| idade | **<25** | 2.355 | **0,7319** | **−0,0514** | **0,008** | **sim** | 41,5% | 11,80% |
| idade | 25-35 | 14.304 | 0,7834 | +0,0008 | 0,934 | não | 57,2% | 10,55% |
| idade | 35-45 | 16.912 | 0,7899 | +0,0107 | 0,120 | não | 69,1% | 8,24% |
| idade | 45-55 | 14.041 | 0,7942 | +0,0145 | 0,080 | não | 73,5% | 7,39% |
| idade | **55-65** | 12.166 | **0,7465** | **−0,0415** | **0,004** | **sim** | 79,9% | 5,61% |
| idade | 65+ | 1.725 | 0,7631 | −0,0198 | 0,395 | não | 89,9% | 3,71% |
| bureau | com histórico | 52.727 | 0,7878 | +0,0133 | 0,132 | não | 71,1% | 7,73% |
| bureau | thin-file | 8.776 | 0,7745 | −0,0133 | 0,132 | não | 56,5% | 10,14% |

> A diferença de aprovação entre homens e mulheres **não é viés do modelo**: o AUC é
> praticamente igual, e a aprovação menor acompanha a inadimplência real observada.

### Decomposição do AUC agregado — a prova de que o critério antigo não servia

| Eixo | pares **dentro** do grupo | pares **entre** grupos |
|---|---|---|
| idade | 22,4% (AUC 0,7828) | **77,6%** (AUC 0,7880) |
| gênero | 52,4% | 47,6% |
| thin-file | 73,1% | 26,9% |

Identidade exata, travada por teste. É o número que mostra que o AUC geral e um AUC
intra-faixa medem eventos diferentes.

### Robustez ao corte de faixas (janelas de 5 anos)

O padrão é um U limpo, com fraqueza nas duas pontas: 20-25 (−0,0509, p 0,004) · 25-30
(n.s.) · 30-35 (n.s.) · 35-40 (n.s.) · 40-45 (n.s.) · 45-50 (n.s.) · 50-55 (n.s.) · **55-60
(−0,0310, p 0,044)** · **60-65 (−0,0468, p 0,008)** · 65-70 (n.s.). Com sextis o efeito se
dilui, porque o sexto mais jovem chega a 31 anos — o que confirma que o problema é **de
ponta**, não espalhado.

### Por que `55-65` falha por outro motivo

Não é aposentadoria — 68,4% da faixa é aposentada, mas dentro dela aposentado (0,7488) e
ativo (0,7427) empatam, e aposentados de 45-55 marcam 0,8065. Sem estratificar por idade, o
eixo aposentadoria *parece* explicativo (−0,0313, p 0,012): **é confundimento puro** — a
mesma armadilha do critério antigo, em outra forma. Não é perfil de informação: a coorte
pareada marca 0,7906 contra 0,7465, melhor em **100%** das réplicas. É uniforme dentro da
faixa (gênero, contrato, thin-file, número de scores). A causa medida: os três scores
externos rendem ali o pior de todas as faixas — **0,6288 · 0,6301 · 0,6506**.

### O contraste que separa os dois casos

Quanto o modelo **acrescenta** sobre o melhor sinal isolado:

| Faixa | AUC do modelo | só `EXT_SOURCE_MEAN` | ganho |
|---|---|---|---|
| **<25** | 0,7319 | 0,6801 | **0,0518** ← o menor de todos |
| 25-35 | 0,7834 | 0,7073 | 0,0761 |
| 35-45 | 0,7899 | 0,7262 | 0,0636 |
| 45-55 | 0,7942 | 0,7202 | 0,0740 |
| **55-65** | 0,7465 | 0,6722 | **0,0744** ← normal |
| 65+ | 0,7631 | 0,6913 | 0,0718 |

Em 55-65 o modelo trabalha normalmente sobre sinal ruim → deficiência **da fonte**. Em
`<25` o sinal é ruim **e** o modelo agrega o mínimo → o resto das variáveis também rende
menos para jovens.

### Calibração por faixa

`<25` é a **única** faixa com viés estatisticamente significativo: +1,56 pp, IC
[+0,39; +2,94]. Todas as demais têm gap entre −0,27 e +0,10 pp, com IC cruzando zero.
Causa: a isotônica é global. Reajustada por faixa, o gap vai de +1,43 pp para −0,05 pp — e
a aprovação da faixa cai de 47,6% para 44,8%, sem ganho de Brier.

---

## §F. O que este trabalho NÃO resolve

Dito antes que perguntem.

**Do diagnóstico:**
- Não prova que 0,7319 é o teto teórico do segmento — prova que o modelo global não é
  superado por um dedicado, **com estas variáveis, esta família de algoritmo e este
  volume**. Rede neural e modelo aditivo com interações explícitas de idade não foram
  testados.
- Não prova que dados alternativos não ajudariam; prova que o argumento que os justificava
  caiu.
- **Não explica o resíduo de −0,0308.** É medido e específico da idade; a causa mecânica é
  desconhecida.
- Não explica *por que* o score externo discrimina pior nos extremos etários — a construção
  desse score é externa ao dataset.
- Não testou `65+` a fundo: n = 1.725 e IC de largura 0,10.
- **Não avalia justiça sob nenhuma definição formal** (paridade demográfica, *equalized
  odds*). Mede desempenho por segmento, que é outra coisa.

**Da entrega:**
- Sem autenticação, rate limit ou CORS restrito — deliberado, e nomeado.
- Sem registry de modelo: o gate detecta regressão e falha a DAG, mas não faz rollback
  automático.
- Sem logs estruturados e sem notificação ativa quando o gate barra uma rodada.
- O SHAP explica o modelo cru, não o calibrado.
- O gate compara o AUC servido contra um valor cru gravado no histórico — viés de 0,0003
  contra limiar de 0,01. Nunca disparou, **mas está errado**, e está registrado.
- 30.751 clientes (a fatia de calibração) aparecem sem score na listagem, por decisão de
  não misturar escalas.
- Segredos do Airflow fixos no compose, com valores de desenvolvimento.

---

## §G. Glossário falado

Como dizer em voz alta, uma frase cada.

| Termo | Como falar |
|---|---|
| **Inadimplência / default** | "É quando o cliente pega o crédito e não paga." |
| **Falso negativo × falso positivo** | "Falso negativo é aprovar quem não paga; falso positivo é negar quem pagaria." |
| **Bureau** | "É a agência externa de crédito, tipo Serasa — dá um score pronto do cliente." |
| **Thin-file** | "É o cliente de pasta fina: não tem histórico em bureau nenhum." |
| **AUC** | "É a chance de o modelo colocar um inadimplente à frente de um bom pagador quando sorteio um de cada." |
| **KS** | "É a maior distância entre as duas curvas acumuladas — o quanto o modelo separa os dois grupos no melhor ponto." |
| **Brier** | "É o erro da probabilidade em si. AUC diz se a ordem está certa; Brier diz se o número está certo." |
| **Calibração** | "É consertar a régua sem mexer na ordem: o modelo já sabia quem é mais arriscado, passou a saber quanto." |
| **Isotônica** | "É a transformação que só pode subir — por isso não bagunça o ranking." |
| **Threshold** | "É a nota de corte: acima dessa probabilidade, nega." |
| **Matriz de custo** | "É pôr preço em cada erro — aqui, aprovar mau pagador custa dez vezes negar um bom." |
| **ABT** | "É a tabela final, uma linha por cliente, com tudo que se sabe dele agregado." |
| **Bootstrap** | "É reamostrar a base muitas vezes para ver se o resultado é sólido ou foi sorte da amostra." |
| **Intervalo de confiança da diferença** | "Em vez de comparar dois números, eu meço a diferença entre eles muitas vezes e vejo se ela é consistentemente negativa." |
| **SHAP** | "É a conta de quanto cada variável empurrou aquele cliente para cima ou para baixo." |
| **PSI** | "É o alarme de que a população mudou desde o treino." |
| **Drift** | "É o mundo mudar e o modelo continuar o mesmo." |
| **Gate** | "É a trava: se o modelo novo for pior que o antigo, o pipeline falha em vez de promover." |

---

## §H. Manutenção

Este arquivo vive em `docs/apresentacao/resumo-projeto.md`. O prompt para gerar os slides
está ao lado, em `prompt-notebooklm.md`, e foi derivado da Parte I.

**Regere quando:** houver novo treino aceito (todos os números da rodada mudam) · o
critério de fraqueza por segmento mudar · a régua de decisão mudar · a matriz de custo
mudar · endpoints ou tasks forem adicionados.

**Como conferir os números antes de apresentar:**

```bash
python Model/run_summary.py --markdown        # tabela oficial da rodada
make urls                                      # endereços das três stacks
make health                                    # a API responde e sabe qual rodada serve
```

⚠️ **Nunca edite número à mão neste documento.** Foi exatamente a cópia manual entre
arquivos que produziu os dois conjuntos de métricas conflitantes que o projeto teve de
reconciliar.
