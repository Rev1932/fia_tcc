# Prompt para o NotebookLM — slides da apresentação (12 min)

| | |
|---|---|
| **Versão** | 1.0 — 31/08/2026 |
| **Fonte obrigatória** | `docs/apresentacao/resumo-projeto.md` |
| **Saída esperada** | 14 slides + roteiro de fala por slide |

## Como usar

1. Abra o NotebookLM e crie um notebook novo.
2. **Suba `resumo-projeto.md` como fonte** (Adicionar fonte → Enviar arquivo, ou cole o
   conteúdo como texto). Sem essa fonte o prompt não funciona: o modelo não conhece o
   projeto e vai inventar números.
3. Cole o bloco abaixo inteiro no chat.
4. Se a saída vier com parágrafos nos slides, responda apenas:
   *"Os slides estão com texto demais. Reescreva respeitando o limite de 20 palavras
   visíveis por slide — o conteúdo vai no [FALA], não no slide."*

> **Sobre o encadeamento:** o pedido mais importante do prompt é o `[GANCHO]`. É ele que
> transforma uma lista de slides numa história — e é a ausência dele que fez o material
> anterior não se sustentar na hora de contar.

> **Sobre o tema visual — leia antes de esperar demais.** O NotebookLM controla pouco da
> aparência do que gera: dependendo do formato de saída, ele pode aplicar o tema, aplicar
> só em parte, ou ignorá-lo e devolver texto puro. O bloco `IDENTIDADE VISUAL` do prompt
> serve para os dois casos: se ele aplicar, ótimo; se não, **aquele bloco é a especificação
> pronta para você montar o tema à mão** no PowerPoint ou no Google Slides — são códigos
> hexadecimais e nomes de fonte, prontos para colar. Leva cerca de dez minutos criar o
> slide-mestre a partir dele, e vale, porque um deck coerente é lido como trabalho cuidado.

### O tema, em uma frase

Fundo quase branco frio (`#F7F8FA`), texto grafite (`#1F2933`), um único destaque em
azul-petróleo dessaturado (`#2F5D7C`), tipografia IBM Plex — serifada no título, sans no
corpo, mono nos números. As três cores da régua de decisão (verde-sálvia, âmbar,
terracota) aparecem **só** nos slides que falam da régua. Fora deles, o slide é
monocromático.

---

## O prompt

```
Você vai montar os slides de uma apresentação acadêmica de TCC, de 12 minutos, a
partir do documento que carreguei como fonte.

CONTEXTO
O documento tem duas partes. Use a PARTE I (A narrativa) como espinha dos slides.
Use a PARTE II apenas para conferir números. Não transforme a Parte II em slides.

REGRA CENTRAL — A FALA É O CONTEÚDO, O SLIDE É APOIO
Quem apresenta sou eu. O slide existe para ancorar o olho da plateia, não para ser
lido. Um slide que a banca consegue ler inteiro está competindo com a minha voz.

Restrições absolutas de cada slide:
- No máximo 20 palavras visíveis, sem contar o título e os números.
- NENHUMA frase completa no corpo. Fragmentos e números.
- Nenhum bullet com mais de 8 palavras. No máximo 4 bullets.
- Um slide, uma ideia. Slide de número tem UM número grande, não uma tabela.
- Proibido: parágrafo, texto explicativo, citação longa, "conclusão" escrita.

ENCADEAMENTO — O QUE MAIS IMPORTA
A apresentação é uma corrente causal, não uma lista de tópicos. Cada slide deve
deixar no ar a pergunta que o slide seguinte responde. Essa é a estrutura da Parte I
do documento: cada seção termina numa pergunta explícita.

Teste que você deve aplicar antes de entregar: se dois slides consecutivos puderem
trocar de lugar sem que a apresentação fique estranha, o encadeamento falhou.
Refaça esse par.

ESTRUTURA OBRIGATÓRIA — TRÊS ATOS (está no §0.1 da fonte; respeite a ordem e o tempo)

ATO I — A DOR (slides 1 a 3, 2:30)
Função: estabelecer que o problema é econômico e assimétrico, não estatístico.
  1. Uma financeira perde dinheiro nos dois extremos da mesma decisão: aprova quem
     não paga e perde capital; nega quem pagaria e perde margem, e ainda exclui
     alguém do sistema financeiro.
  2. Os dois erros não custam a mesma coisa: dez para um. Premissa de negócio
     declarada, não descoberta no dado. Por isso acurácia não serve — numa base 92
     contra 8, aprovar todo mundo acerta 92% e é o modelo mais caro possível.
  3. O que se precisa não é um classificador: é uma probabilidade de verdade mais um
     corte derivado de custo. E o dado que melhor prevê inadimplência falta em 56,4%
     dos casos, porque o público é sub-bancarizado.
  Fecha no paradoxo: o sinal mais forte é o que menos está disponível.
  TRANSIÇÃO OBRIGATÓRIA (use esta frase no [GANCHO] do slide 3):
  "Este trabalho não começou num modelo. Começou em construir a informação que não
   vinha pronta."

ATO II — O DESENVOLVIMENTO (slides 4 a 10, 6:00)
Função: mostrar COMO se chega a um número em que se pode confiar. É o ato mais longo
e é onde a banca decide se o apresentador entende o que fez.
  4. O que os dados disseram: 307.511 clientes, 8,07% de inadimplência, risco
     desigual por escolaridade (1,8% a 10,9%), e dados que mentem (a sentinela de
     mil anos de emprego).
  5. A ABT e o bug: nove tabelas com granularidades diferentes viram uma linha por
     cliente — e a agregação descartava em silêncio todas as variáveis categóricas.
     473 colunas viraram 1.020. O maior ganho do projeto veio de corrigir dado, não
     de ajustar modelo.
  6. Os modelos testados: regressão logística 0,7776 contra LightGBM 0,7871. E POR
     QUE o LightGBM ganhou, que é o ponto: os nulos aqui são estruturais, não erro —
     "não tem score de bureau" é informação, e imputar apagaria o sinal do público
     que o trabalho quer atender. A logística é o piso de comparação; sem ela, "0,78"
     não significaria nada. Overfitting: o argumento é o empate validação 0,7835 ×
     teste 0,7871, não o gap contra o treino.
  7. O que o AUC mede: a chance de ordenar certo um par (inadimplente, adimplente).
     E o que ele NÃO mede: se o nível do risco está certo. Essa distinção volta no
     slide 10.
  8. O score não era probabilidade: o corte ótimo caía em 0,47 numa carteira com 8%
     de inadimplência. Calibração isotônica numa fatia reservada de 10% levou o Brier
     de 0,1668 para 0,0658 e o corte para 0,09. Calibrar não melhora o modelo —
     melhora o significado do número.
  9. As rodadas que foram rejeitadas: v1 linha de base, v2 ABT corrigida, v3
     calibrada (é a servida), e v4 com peso maior para jovens REJEITADA porque piorou
     o próprio alvo em 0,0032. Isso é o que prova método em vez de sorte.
 10. Onde o modelo falha: fomos medir e descobrimos que o NOSSO PRÓPRIO critério de
     fraqueza era inválido — 77,6% dos pares do AUC agregado são entre faixas
     etárias. Trocamos por bootstrap da diferença; a conclusão sobreviveu (-0,0514,
     p = 0,008); e a causa que declarávamos era falsa: uma coorte pareada de clientes
     maduros com o mesmo perfil de informação marca 0,7803 contra 0,7319. Seis
     modelos dedicados ficaram todos abaixo do geral. É teto de dado, medido.
  TRANSIÇÃO OBRIGATÓRIA (use esta frase no [GANCHO] do slide 10):
  "Com isso definido, o que sobrou foi entregar — e entregar significa que outra
   pessoa consiga operar isso sem mim."

ATO III — O QUE ESTÁ SENDO ENTREGUE (slides 11 a 14, 3:30)
Função: converter tudo em coisa que existe. É AQUI que os números de resultado
aparecem como resultado, e não como etapa.
 11. O modelo final: LightGBM calibrado, AUC 0,7868, KS 0,4342, Brier 0,0658, corte
     0,09 — que é probabilidade de verdade, porque a média do score na base é 0,081
     contra inadimplência real de 8,07%. Aprova 68,65% da carteira com 3,48% de
     inadimplência entre os aprovados, contra 8,07% sem modelo, e barra 70,4% dos
     maus pagadores.
 12. A régua de decisão: três faixas, não duas. APROVAR abaixo de 0,065, NEGAR acima
     de 0,115, REVISAR no meio. A faixa cinza dobra nos dois segmentos onde a fraqueza
     foi medida. O critério de risco é idêntico para todos; o que muda é quanto vai
     para revisão humana — 15,5% da carteira.
 13. A disponibilização: API com 27 endpoints servindo o mesmo artefato que o treino
     produziu, SHAP por cliente com verificação de consistência, e um endpoint que
     recalcula a régua de custo ao vivo para qualquer razão FN:FP sem re-treinar.
 14. O re-treino e o monitoramento: DAG do Airflow com 9 tasks a cada 7 dias, com um
     gate que FALHA a DAG se o AUC cair mais de 0,01 contra a última rodada aceita.
     PSI para drift. Prometheus e Grafana com 5 alertas, incluindo o que distingue
     "a API respondeu" de "a API consegue pontuar um cliente". E a prova de que nada
     disso é encenação: a rodada servida hoje foi produzida pelo próprio DAG.
     Fecha com os limites declarados: não está pronto para produção, e o trabalho diz
     onde não está.

REGRA DE ENCENAÇÃO — NÃO ANTECIPE OS NÚMEROS DO ATO III
Os valores 0,7868, corte 0,09, 68,65% e 3,48% são CONSTRUÍDOS no Ato II e
APRESENTADOS no Ato III. No Ato II, ao falar de calibração, cite o Brier
(0,1668 -> 0,0658) e diga que o corte deixou de ser arbitrário — mas guarde a taxa de
aprovação e a inadimplência da carteira aprovada para o slide 11. Anunciar 68,65% no
meio da apresentação queima o fecho.

Cada número tem UM lugar:
  Ato I:   10:1 | 8,07% | 56,4% | 307.511
  Ato II:  473->1.020 | 24,14% | 39,71% | 0,7776 x 0,7871 | 0,8753/0,7835/0,7871 |
           0,47 | 0,1668->0,0658 | 77,6% | -0,0514 (p 0,008) | 0,7803 x 0,7319 | v4
  Ato III: AUC 0,7868 | corte 0,09 | 68,65% | 3,48% | 15,5% | 27 endpoints |
           9 tasks / 7 dias | 5 alertas

ORÇAMENTO DE TEMPO — 14 slides, 12 minutos
| Ato | Slide | Bloco                           | Tempo |
|-----|-------|---------------------------------|-------|
| I   | 1     | A dor: perde-se dos dois lados  | 0:45  |
| I   | 2     | A assimetria 10:1               | 0:50  |
| I   | 3     | O paradoxo do dado que falta    | 0:55  |
| II  | 4     | O que os dados disseram         | 0:50  |
| II  | 5     | A ABT e o bug                   | 1:00  |
| II  | 6     | Os modelos testados             | 1:00  |
| II  | 7     | O que o AUC mede                | 0:40  |
| II  | 8     | Calibracao                      | 0:55  |
| II  | 9     | O que foi rejeitado             | 0:35  |
| II  | 10    | Onde o modelo falha             | 1:00  |
| III | 11    | O modelo final                  | 0:55  |
| III | 12    | A regua de decisao              | 0:50  |
| III | 13    | API e disponibilizacao          | 0:50  |
| III | 14    | Airflow, monitoramento, limites | 0:55  |

O Ato II e o mais longo de proposito: e o que a banca mais cobra. Nao o comprima para
ganhar tempo em outro lugar. Se precisar cortar, corte fala do Ato I.

NÚMEROS — use exatamente estes, sem arredondar e sem inventar outros
307.511 clientes | 8,07% de inadimplência | 56,4% de nulos no EXT_SOURCE_1
473 -> 1.020 colunas | EXT_SOURCE_MEAN com 24,14% da importância
39,71% da importância vem das tabelas relacionais
AUC 0,7776 (baseline) -> 0,7868 (servido) | treino 0,8753, validação 0,7835, teste 0,7871
Brier 0,1668 -> 0,0658 | corte 0,47 -> 0,09 | custo FN:FP = 10:1
aprovação 68,65% | inadimplência dos aprovados 3,48% | 15,5% em revisão humana
<25 anos: AUC 0,7319, diferença -0,0514, p = 0,008
coorte pareada 0,7803 contra 0,7319 | 77,6% dos pares do AUC geral são entre faixas
27 endpoints | 9 tasks | a cada 7 dias | 5 alertas | 159 testes

Se um número não estiver nesta lista nem no documento, NÃO o use.

IDENTIDADE VISUAL — tons claros, suaves e de credibilidade

Contexto: é uma banca acadêmica sobre risco de crédito. O visual precisa passar
sobriedade e confiança, sem parecer apresentação comercial. Nada de fundo escuro,
gradiente, sombra, 3D, ícone colorido ou imagem de banco de imagens.

PALETA (use exatamente estes valores)
  Fundo dos slides    #F7F8FA   branco levemente frio, nunca branco puro
  Fundo alternativo   #EEF1F4   só para separar seções ou destacar um bloco
  Texto principal     #1F2933   grafite, nunca preto puro — preto puro cansa a vista
  Texto secundário    #5A6672   legendas, fontes, notas de rodapé
  Linhas e bordas     #DDE2E8   divisórias e grade de gráfico, sempre discretas
  Cor de destaque     #2F5D7C   azul-petróleo dessaturado: títulos, números grandes,
                                barra de destaque. É a cor da credibilidade aqui
  Apoio               #6E8FA3   azul claro, SÓ para a segunda série de um gráfico —
                                não tem contraste para texto

CORES SEMÂNTICAS — use SÓ na régua de decisão e nos gráficos que a representam.
Fora disso, o slide é monocromático em azul-petróleo. Nenhuma delas é saturada de
propósito: são decisões de crédito, não alarme.

Cada uma tem DUAS versões, e trocá-las quebra a legibilidade:
                 preenchimento    texto e rótulo
  APROVAR        #6F9B7E          #456B52
  REVISAR        #C9A24B          #806418
  NEGAR          #B4695D          #8A473C

A versão clara é para área preenchida — barra, fatia, faixa de fundo. Ela NÃO tem
contraste suficiente para texto sobre o fundo claro (a âmbar fica em 2,3:1, contra o
mínimo de 4,5:1). Para qualquer palavra ou número colorido, use a coluna da direita,
que fica entre 4,7:1 e 6,5:1. Na dúvida, escreva em grafite #1F2933 e deixe a cor
só na forma.

TIPOGRAFIA
  Títulos            IBM Plex Serif, semibold
  Corpo e legendas   IBM Plex Sans, regular
  Números e métricas IBM Plex Mono, medium (algarismos alinham em coluna)
  Alternativa se IBM Plex não estiver disponível: Source Serif 4 para títulos e
  Source Sans 3 para o resto. Se só for possível uma família, use IBM Plex Sans
  em tudo, variando o peso.

  A serifa no título e a sans no corpo é o par que passa credibilidade sem ficar
  formal demais. Não use Arial, Calibri, Times New Roman nem fonte decorativa.

TAMANHOS (proporção, ajuste ao formato do slide)
  Título              40 pt
  Subtítulo           24 pt
  Corpo               22 pt   nunca abaixo de 20 pt
  Número em destaque  100 pt  em IBM Plex Mono, cor #2F5D7C
  Legenda e fonte     14 pt   em #5A6672

LAYOUT
  Margem generosa: pelo menos 8% da largura em cada lado. Espaço em branco é parte
  do design, não desperdício.
  Alinhamento à esquerda. Não centralize blocos de texto.
  Um único elemento visual por slide.
  Slide de número: o número ocupa o centro-esquerda, e abaixo dele uma linha de no
  máximo 8 palavras dizendo o que ele é.
  Rodapé discreto e constante: nome do trabalho à esquerda, número do slide à
  direita, em 12 pt e cor #5A6672.

GRÁFICOS
  Fundo transparente, sem borda externa.
  Grade horizontal apenas, em #DDE2E8. Sem grade vertical.
  Rotule as séries diretamente no gráfico; evite legenda separada.
  Série única em #2F5D7C; duas séries em #2F5D7C e #6E8FA3.
  Sem 3D, sem sombra, sem gradiente, sem rótulo em cada ponto.

FORMATO DA SAÍDA
Para cada slide, exatamente assim:

--- SLIDE 3 ---
TÍTULO: (máximo 6 palavras)
CORPO: (o que aparece na tela — fragmentos, no máximo 20 palavras)
VISUAL: (que gráfico, tabela ou número grande colocar, e qual cor da paleta usar)
[FALA]: (60 a 110 palavras — o roteiro que eu vou falar. É AQUI que mora o conteúdo:
a explicação, o porquê, a transição. Escrito em primeira pessoa do plural, em
português do Brasil, no tom de quem defende um trabalho, não de quem lê um relatório.)
[GANCHO]: (uma frase — a pergunta ou tensão que este slide deixa no ar e que o
próximo resolve)

Ao final de todos os slides, acrescente:

--- CRONOMETRAGEM ---
A soma do tempo de fala estimado, e um aviso se passar de 12 minutos.

--- TRÊS SLIDES DE RESERVA ---
Três slides extras que eu NÃO apresento, mas deixo no fim do arquivo para puxar se a
banca perguntar: (a) o quadro de desempenho por segmento, (b) a arquitetura em
containers, (c) as limitações declaradas do trabalho.

IDIOMA E TOM
Português do Brasil. Tom técnico e direto, sem adjetivo publicitário. Não escreva
"solução inovadora", "robusto", "poderoso". O trabalho se defende pelos números e
pela honestidade sobre os limites — o texto deve refletir isso.
```

---

## Depois de gerar

- **Confira os números contra a §B do `resumo-projeto.md`.** Modelo de linguagem
  arredonda; a §B é a referência.
- **Leia os `[FALA]` em voz alta com cronômetro.** É o único teste que vale — se passar de
  12 minutos, corte fala, não slide.
- **Teste o encadeamento**: leia só os `[GANCHO]` em sequência. Eles sozinhos têm de contar
  a história inteira. Se algum não puxar o próximo, peça ao NotebookLM para refazer aquele
  par.
- **Se o tema não vier aplicado**, monte o slide-mestre com os valores do bloco
  `IDENTIDADE VISUAL` e cole o conteúdo dentro. As fontes IBM Plex são gratuitas e estão no
  Google Fonts; se não puder instalar nada, o par de reserva (Source Serif 4 / Source Sans
  3) já vem no Google Slides.
