# Tarefas restantes

Estado do projeto em **23/08/2026**, ao fim do ciclo de aperfeiçoamento.
Este documento existe para que o trabalho possa ser retomado depois — por você
ou por outra pessoa — sem precisar reconstruir o contexto.

**Antes de mexer em qualquer coisa, leia:**
- **[Dossiê do Credit Scoring](https://claude.ai/code/artifact/40f858c1-2775-40a9-accc-e1ac97221284)**
  — como o projeto foi construído e por quê, com os números da rodada vigente.
  Fonte em `docs/dossie/index.html`; atualize com `python docs/dossie/build_data.py`
- `MLOps/app/README.md` — referência da API
- `python Model/run_summary.py --markdown` — os números oficiais vigentes

---

## Onde o projeto está

| Frente | Estado |
|---|---|
| Pipeline de dados | ✅ Reprodutível ponta a ponta, com Parquet e perfil de colunas |
| ABT | ✅ 1.020 colunas — categóricas das tabelas relacionais recuperadas |
| Modelo | ✅ LightGBM calibrado (servido), AUC 0,7868 · KS 0,4342 · Brier 0,0658 |
| Rodada canônica | ✅ Congelada em `artifacts/`, é fonte única de verdade |
| API | ✅ 27 endpoints, documentação própria, 113 testes passando |
| Dashboard | ✅ Executado e coberto por `tests/test_dashboard.py` |
| Docker | ✅ Build e stack verificados, healthcheck falha corretamente |
| Orquestração | ✅ Airflow 3.3.1 em Docker, DAG de 9 tasks a cada 7 dias |
| Documentação | ✅ README, TCC, pitch e guia reconciliados com os artefatos |
| Monitoramento de drift | ✅ `GET /model/psi` implementado e testado |
| Notebook de avaliação | ✅ Reescrito lendo de `artifacts/`, re-executado nesta máquina |
| Slides | ✅ `credit_scoring_deck_v3.pptx` — falta só um slide de calibração, à mão |
| Fraqueza em `<25 anos` | ❌ Medida, **um conserto testado e rejeitado**, não resolvida |

---

## 1. Verificações — ✅ todas concluídas em 23/08/2026

Eram coisas escritas mas nunca executadas. Ficam registradas abaixo com o que
foi verificado e o que cada uma revelou.

### 1.1 Subir a stack em Docker — ✅ **feito em 23/08/2026**

```bash
docker compose -f MLOps/docker-compose.yml up -d --build
docker ps --filter name=hc- --format "{{.Names}}  {{.Status}}"
```
Verificado:
- Build com `python:3.14-slim` **funcionou** — o fallback para 3.12 não foi
  necessário. Imagem final: 1,84 GB.
- `hc-api` fica `healthy`; `hc-dashboard` só sobe depois (a condição
  `service_healthy` no compose funciona).
- Endpoints respondem 200 de fora do container, com os dados reais montados
  por volume.
- **Caminho de falha testado**: com `HC_ABT_PARQUET` apontando para um arquivo
  inexistente, `/health` devolve **503** com `data_loaded: false`, e `/clients`
  devolve 503 com a mensagem do comando a rodar. É o que dá sentido ao
  healthcheck — a versão anterior respondia 200 sempre.

### 1.2 Rodar o dashboard Streamlit — ✅ **feito em 23/08/2026**

Executado e, mais importante, **coberto por teste automatizado**:
`tests/test_dashboard.py` usa o `AppTest` do Streamlit com as chamadas
`requests` redirecionadas ao `TestClient` da API — exercita o script de
verdade, sem servidor no ar nem dados reais.

Um problema encontrado e corrigido no caminho: o painel derrubava a aba
*Modelo* inteira quando uma dimensão de fairness não existia na rodada
(404 vira traceback). Agora usa `get_opcional()` e mostra um aviso.

```bash
uvicorn MLOps.app.api:app --port 8000 &
HC_API_URL=http://localhost:8000 streamlit run MLOps/app/streamlit_app.py
```

### 1.3 Re-executar `Model/evaluation.ipynb` — ✅ **feito em 23/08/2026**

Reescrito e executado nesta máquina, com 0 erros e 7 gráficos.

A mudança estrutural: o notebook **não recria mais o split nem re-pontua a
base**. Ele lê `artifacts/scores.parquet`, que já traz o rótulo de fatia e o
score de todos os 307.511 clientes. É essa dependência que impede notebook,
API e documentos de divergirem — a versão anterior recriava o
`train_test_split` por conta própria, e foi assim que dois conjuntos de
métricas passaram a circular.

Correções aplicadas:
- O gráfico rotulado "gain" mostrava na verdade `split` (o padrão do
  `LGBMClassifier`). Agora as duas aparecem, cada uma com o rótulo certo.
- Nova seção de calibração, com a curva de confiabilidade antes/depois.
- Fairness agora com **intervalo de confiança bootstrap** por segmento, e o
  gráfico de IC contra a faixa do modelo geral.
- Nova seção comparando as três rodadas (v1 → v3) por segmento.

**Uma ambiguidade que o notebook expôs, e que foi resolvida:** ele calculava o
KS sobre o score calibrado (0,4342) enquanto os documentos citavam o do modelo
cru (0,4354). Duas métricas para a mesma coisa é exatamente o problema que
este ciclo veio matar. A solução: `artifacts/metrics.json` ganhou um bloco
**`served`** — as métricas do modelo que a API de fato entrega. É de lá que
sai todo número de capa, e `GET /model/metrics` expõe os dois lado a lado,
rotulados.

### 1.4 Regerar os slides — ✅ **parcial, feito em 23/08/2026**

Gerado `docs/credit_scoring_deck_v3.pptx` a partir do v2.5, **preservando o
design** e corrigindo os números por slide (o valor `0,785` aparecia em quatro
slides com significados diferentes, então a substituição foi feita slide a
slide, não global).

O que mudou:
- Todas as métricas passam a ser as do modelo **servido** (`artifacts/metrics.json` → `served`)
- ABT: 473 → 1.020 colunas · features: 473 → 1.018 · early stopping: 783 → 507
- Slide 8 e 10 passam a citar o Brier e o corte de 0,09
- **Slide 9 reescrito**: a tabela de fairness ganhou a coluna de intervalo de
  confiança, e a conclusão sobre thin-file foi corrigida — passou de "fraqueza
  comprovada" para "dentro do ruído amostral"

**O que ainda falta, e precisa ser feito à mão:**
- **Um slide de calibração.** É a correção técnica mais defensável do ciclo
  (Brier 0,1668 → 0,0658, corte 0,47 → 0,09) e hoje aparece só de passagem nos
  slides 8 e 10. Inserir slide preservando o layout não é seguro por script —
  duplicar slide não é suportado nativamente pelo python-pptx e improvisar
  arrisca corromper o arquivo. Faça no PowerPoint, copiando o layout do slide 8.
  Conteúdo pronto na seção 08 do dossiê.
- **Conferir os gráficos embutidos.** Se algum slide tiver imagem de curva
  gerada na rodada v1, ela continua desatualizada — o script só mexeu em texto.
  As figuras novas estão em `Model/evaluation.ipynb`, já re-executado.
- Os decks v1, v2 e v2.5 foram mantidos como estão, para histórico.

---

## 2. A fraqueza que não foi resolvida

**O segmento de menores de 25 anos.** AUC 0,7319 [0,7012–0,7597] contra
0,7868 [0,7806–0,7935] do modelo geral — os intervalos não se sobrepõem, então
é fraqueza real e não ruído amostral.

As seis correções da ABT (categóricas, flags de presença, scores externos
combinados, comportamento de pagamento, janela recente, variáveis relativas à
idade) melhoraram o modelo geral e a maioria dos segmentos, mas esse grupo
praticamente não se moveu: **0,7364 → 0,7319**.

Mitigação vigente: régua de três faixas com faixa cinza dobrada, encaminhando
esses casos a análise humana (`GET /model/decision-policy`).

### Reponderação no treino — ❌ **testada e REJEITADA em 23/08/2026**

Rodada `v4-pesos-idade`: `sample_weight` de 3× para `<25` e 2× para `55-65`
(36.560 linhas de treino reponderadas). Implementação em
`Model/train.py::build_sample_weight`, ligável em
`Model/config.yaml → champion.sample_weight`.

| | v3 (vigente) | v4 (pesos) | Δ |
|---|---|---|---|
| AUC geral | 0,7871 | 0,7872 | +0,0001 |
| KS | 0,4354 | 0,4359 | +0,0006 |
| **AUC `<25`** (alvo) | **0,7319** | **0,7287** | **-0,0032** |
| AUC `55-65` (alvo) | 0,7465 | 0,7481 | +0,0015 |
| AUC thin-file | 0,7745 | 0,7731 | -0,0014 |

**Veredito: rejeitada.** O alvo principal — a faixa `<25` — **piorou**, e a
maioria dos demais segmentos caiu junto. O ganho de +0,0001 no AUC geral
não compensa: entra dentro do ruído e não era o objetivo.

**Por que provavelmente falhou** (hipótese, não medida): o modelo já usa
`is_unbalance=true`, que reponderá as classes. Empilhar `sample_weight` por
segmento distorce o objetivo duas vezes — o gradiente passa a perseguir uma
distribuição que não é nem a real nem a balanceada. E, mais importante: a
dificuldade nos jovens não parece ser de *alocação de capacidade*, e sim de
**ausência de sinal**. Dar mais peso a linhas que não carregam informação não
cria informação.

A rodada fica registrada em `artifacts/improvement_log.json` com
`status: "rejeitada"` e o motivo. `GET /model/improvements` a devolve no campo
`rejeitadas` — o que foi tentado e não funcionou faz parte do trabalho, e uma
banca que perguntar merece a resposta.

Para reproduzir: ligue `champion.sample_weight.enabled` em `Model/config.yaml`
e rode `python Model/train.py --tag v4-pesos-idade`.

### Caminhos que ainda não foram tentados
- **Modelo segmentado** para clientes de pouco histórico. Risco: com ~12 mil
  jovens na base inteira, um modelo dedicado pode ficar pior que o geral.
  Testar antes de adotar, com a mesma regra de aceite do ciclo. Dado o
  resultado da reponderação, a expectativa é baixa: se o problema é falta de
  sinal, treinar só naquele grupo não o cria.
- **Dados alternativos** — telecom, contas de consumo, comportamento
  transacional. É a recomendação técnica correta e a mais honesta:
  a fraqueza vem de ausência de histórico, e histórico é exatamente o que um
  cliente de 22 anos não tem. Não está na base do Kaggle; entraria como
  proposta de evolução, não como implementação.
- **Features de posição relativa dentro do segmento** (percentil de renda
  dentro da faixa etária, por exemplo). Não testado. É a hipótese mais
  promissora das que sobraram, porque cria contraste onde o valor absoluto
  não distingue.

---

## 2b. Orquestração — ✅ implementada em 24/08/2026

O treino era disparado à mão, rodando quatro arquivos em sequência. Não havia
agendamento, histórico nem como acompanhar uma rodada — e `OKR.md:46,124`
registra o Airflow como **requisito do enunciado**, enquanto
`MLOps/Readme.md:9` o desenhava no diagrama de arquitetura como se existisse.

Agora existe de fato: `MLOps/airflow/` sobe uma instância do Airflow 3.3.1 em
Docker (5 serviços, LocalExecutor) e `dags/treino_credit_scoring.py` define
**9 tasks**, agendadas a cada **7 dias**, com log por etapa em `localhost:8080`.

Três tasks não existiam no pipeline manual, e existem porque falhar cedo é mais
barato: `checar_fontes` (segundos, em vez de descobrir 11 min adiante),
`validar_abt` (granularidade e vazamento, antes de gastar 15 min treinando) e
**`validar_metricas`** — o gate: se o AUC cair além do limiar frente à última
rodada aceita, a execução falha e o modelo anterior continua servindo. Mais
`calcular_psi`, que cumpre o "PSI em job no Airflow" que o Readme prometia.

Modo demonstração: Variable `hc_sample=30000` faz o mesmo DAG rodar em ~1 min,
gravando em `artifacts/demo/`.

### O que ficou de fora
- **Notificação ativa** quando o gate barra uma rodada — hoje fica no log.
  E-mail ou Slack seria o passo seguinte.
- **Retenção de logs**: `MLOps/airflow/logs/` cresce sem limpeza automática.
- **Segredos no compose**: `FERNET_KEY` e `JWT_SECRET` estão fixos com valores
  de desenvolvimento. Em produção viriam de um cofre.
- **Uma instância só**: LocalExecutor não distribui. Suficiente para uma
  máquina, insuficiente para vários pipelines concorrentes.

---

## 3. Dívidas técnicas conhecidas

### 3.1 Baseline não é pontuado na fatia de treino — ⚠️ **aceita, com erro explicativo**
`artifacts/scores.parquet` tem `proba_baseline` nula para `split="train"`.
Motivo: rodar o `OneHotEncoder` + regressão logística em 154 mil linhas custa
tempo e nenhum endpoint usa.

Tratado em 23/08/2026: a combinação `model=baseline&split=train` agora devolve
404 com mensagem que **explica o porquê e aponta a alternativa**, em vez de um
"sem scores" genérico. A fixture de teste foi alinhada ao artefato real (antes
preenchia o baseline em todas as fatias, escondendo o caso), e há teste
cobrindo. Continua uma limitação consciente, não um bug.

### 3.2 O SHAP explica o modelo cru, não o calibrado
`/clients/{id}/explain` calcula SHAP sobre o LightGBM antes da isotônica.
Como a isotônica é monotônica, a **ordem e o sinal** das contribuições
continuam válidos, e a resposta traz `raw_probability` para deixar isso
explícito. Mas a soma das contribuições reconstrói a probabilidade **crua**,
não a calibrada. Explicar contribuições diretamente no espaço calibrado é
possível, e não foi feito.

### 3.3 `Model.predict.predict` arredonda em 4 casas
Herdado da versão original. Limita a precisão de
`/clients/{id}/score?recompute=true` (o `agreement_error` fica na casa de
1e-5, não 1e-9). Sem impacto prático, mas o teste correspondente usa
tolerância `1e-4` por causa disso.

### 3.4 A ABT em CSV tem 1,3 GB
O formato CSV é exigido pelo enunciado (`/Dados/abt.csv`). Todo o resto do
projeto usa o Parquet (310 MB). O CSV existe só para cumprir o formato.

### 3.5 Sem autenticação na API
Deliberado — projeto acadêmico servindo dados públicos do Kaggle. Em produção
entrariam autenticação, rate limit e CORS restrito. Vale ter a frase pronta:
está em `MLOps/app/README.md`, §9.

---

## 4. Regra de aceite — se for mexer no modelo

Foi ela que deu defensabilidade a este ciclo. Mantê-la:

1. **Mesmo split, mesma semente, mesmo conjunto de teste.** O teste (20%) não é
   tocado por nenhuma decisão de modelagem.
2. **Uma mudança por rodada**, com `--tag` descritivo. O
   `artifacts/improvement_log.json` acumula o histórico automaticamente.
3. **Aceite por segmento, não só pelo geral:** um conserto fica se melhorar o
   segmento-alvo sem piorar o global.
4. **O que não funcionar sai — e é registrado.** Foi assim que o trabalho pôde
   afirmar honestamente que a faixa `<25` não cedeu.
5. **Todo AUC de segmento acompanhado do intervalo de confiança.** Sem ele não
   se distingue fraqueza de ruído, e essa distinção mudou a conclusão do
   trabalho (thin-file saiu da lista de fraquezas).

Depois de qualquer re-treino:
```bash
python Model/run_summary.py --markdown   # números oficiais
python docs/dossie/build_data.py         # atualiza o dossiê
pytest -q                                # 78 testes
```
E reconciliar `README.md`, `docs/TCC.md`, `docs/pitch_demoday.md` e a cola de
números do `docs/guia_apresentacao_demoday.md`.

> ⚠️ **Nunca edite número à mão nesses documentos.** Foi exatamente essa cópia
> manual que produziu os dois conjuntos de métricas conflitantes que este ciclo
> veio corrigir (654/0,50/71,7% nos artefatos contra 783/0,47/69,1% nos decks).

---

## 5. Melhorias de API que ficaram fora do escopo

Nenhuma é necessária para a defesa. Estão aqui por serem as continuações
naturais:

- ~~`GET /model/psi`~~ — ✅ **implementado em 23/08/2026.** Population Stability
  Index entre duas fatias, com as faixas de leitura de mercado
  (< 0,10 estável · 0,10–0,25 atenção · > 0,25 mudança relevante). Sem
  `features`, compara o próprio score, que é o sinal de drift mais importante.
  Entre `train` e `test` dá ~0,0004 — o esperado para a mesma safra, e uma
  confirmação de que o split não introduziu viés. Em produção, apontar
  `comparado` para a safra nova transforma o mesmo cálculo no alerta de drift
  descrito em `MLOps/Readme.md`. Coberto por 9 testes.
- ~~`POST /predict` em lote via CSV~~ — ✅ **implementado em 23/08/2026** como
  `POST /predict/csv`. Recebe CSV, devolve CSV com `probability_default`,
  `decision` e `score_band`, preservando as colunas originais. Limite de 50 mil
  linhas / 20 MB. É o formato que uma mesa de crédito usa de fato. 4 testes.
- Cache de resposta em `/model/threshold-analysis` — hoje varre 99 pontos
  sobre 61 mil linhas a cada chamada (leva ~20 ms, então não é urgente).
- Paginação por cursor em `/clients` — o `OFFSET` degrada em páginas muito
  profundas. Irrelevante na escala atual.
- Um endpoint que devolva o dossiê e os documentos renderizados, para a
  apresentação inteira sair de um endereço só.

---

## 6. Ensaio — o teste que mais importa

Com a API no ar e `MLOps/app/README.md` aberto, responder em voz alta,
usando a API ao vivo. Meta: menos de um minuto por pergunta.

1. *"E se aprovar um mau pagador custasse 20× em vez de 10×?"*
   → `GET /model/threshold-analysis?cost_fn=20&cost_fp=1`
2. *"Vocês disseram que o modelo falha com jovens. O que fizeram a respeito?"*
   → `GET /model/improvements` + a seção 09 do dossiê
3. *"Mostra um cliente jovem sem bureau que foi negado, e por quê."*
   → `GET /clients?age_max=25&thin_file=true&decision=NEGAR&page_size=1`
   → `GET /clients/{id}/explain`
4. *"Muda o `num_leaves` para 60 e me mostra o efeito."*
   → editar `Model/config.yaml` → `python Model/train.py --sample 30000 --tag demo`
   (grava em `artifacts/demo/`, **não** toca na rodada oficial) → ~25 s
5. *"Como vocês detectariam que o modelo envelheceu em produção?"*
   → `GET /model/psi` — e a leitura de faixa vem junto na resposta
6. *"O que vocês tentaram e não funcionou?"*
   → `GET /model/improvements` → campo `rejeitadas`, com o motivo registrado
7. *"Como isso entra na operação diária?"*
   → `POST /predict/csv` com a fila do dia
8. *"Como o modelo é re-treinado? Alguém roda na mão?"*
   → `localhost:8080` — o DAG roda a cada 7 dias, e com a Variable
   `hc_sample=30000` a demonstração inteira leva ~1 min
9. *"E se o re-treino piorar o modelo?"*
   → a task `validar_metricas` falha a execução; os artefatos da rodada
   anterior continuam intactos e servindo

A pergunta 2 é a que não tinha resposta antes deste ciclo. As perguntas 5 a 7
não existiam como resposta executável até hoje.

---

## 7. Se algo quebrar

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| API responde 503 | Falta modelo ou Parquet | A mensagem do `/health` diz o comando a rodar |
| `/health` diz `model_not_loaded` | `artifacts/model.joblib` ausente ou de outra versão de lib | `python Model/train.py` |
| `/clients` devolve 400 | Coluna fora da whitelist | `GET /meta/columns` lista as válidas |
| Import quebra após `pip install` | Pins incompatíveis com a versão do Python | Recriar o `.venv`; se persistir, usar 3.12 e alinhar o Dockerfile |
| Números não batem entre documentos | Alguém editou à mão | `python Model/run_summary.py --markdown` e reconciliar |
| Artefatos perdidos | — | Cópia da rodada v1 em `artifacts_v1_baseline/` |
| API em Docker diverge da local | Imagem sem rebuild após mudar código | `docker compose -f MLOps/docker-compose.yml up -d --build` |
| `artifacts/` com `tag: demo` | Um `--sample` antigo sobrescreveu a rodada oficial | `python Model/train.py --tag v3-calibrado` (~15 min) |

---

## 8. Armadilhas já encontradas (não repita)

Cada uma destas custou tempo neste ciclo e voltaria a custar.

**`--sample` sobrescrevia a rodada oficial.** Corrigido: o modo demonstração
grava em `artifacts/demo/`. Mas confira `Model/run_summary.py` antes de
apresentar — se o campo `tag` disser `demo`, os artefatos estão errados e
todo número apresentado estará errado junto.

**`.replace(0, pd.NA)` converte `float64` em `object`.** E `select_dtypes("number")`
descarta colunas `object` **em silêncio**. Foi assim que `PAYMENT_RATIO` e
`UTILIZATION` sumiram da primeira ABT corrigida, com todas as outras features
presentes. Use `np.nan`. O `abt_transform.py` hoje emite um aviso alto quando
detecta esse caso.

**Regex em documento cheio de número é perigoso.** Uma substituição de `783`
transformou o AUC `0,783` em `0,507` no guia de apresentação. Reconcilie
documentos **gerando as tabelas** a partir de `artifacts/`, nunca com busca e
substituição.

**`cv="prefit"` foi removido no scikit-learn 1.9.** O substituto é
`CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")`.

**DuckDB não aceita parâmetro preparado em `read_csv`/`read_parquet`/`COPY`.**
Os caminhos precisam ser interpolados — com aspas escapadas, e vindos sempre
de configuração, nunca de entrada de usuário (ver `db._lit`).

**`width_bucket` não existe no DuckDB 1.5.** O histograma de
`/stats/distribution` calcula o bucket por aritmética.

**Um modelo Pydantic como query param só funciona sozinho no FastAPI 0.138.**
Basta acrescentar qualquer outro parâmetro e ele volta a exigir um campo
chamado `filters`. Por isso os filtros entram por `Depends()` — e por isso os
filtros multivalorados são texto separado por vírgula, não `list[str]`.

**Airflow 3 mudou muita coisa de lugar, e nenhuma dá erro óbvio.** Imports
saíram do core para `airflow.providers.standard`; o `dag-processor` virou
serviço separado (sem ele os DAGs simplesmente não aparecem); as tasks falam
com a Execution API e precisam de `EXECUTION_API_SERVER_URL` + `JWT_SECRET`
compartilhado, senão morrem antes de rodar a primeira linha, deixando o log
com uma linha só; e `{{ ds }}` não existe em DAG com `schedule=timedelta`
disparado manualmente. A lista completa está em `MLOps/airflow/README.md`,
seção "Armadilhas do Airflow 3".

**A imagem Docker não vê mudança de código sem rebuild.** Óbvio, mas pega:
os artefatos entram por volume (mudam sozinhos), o **código é assado na
imagem**. Depois de alterar qualquer arquivo em `MLOps/app/` ou `Model/`,
`docker compose up -d --build` — sem o `--build` o container serve a versão
antiga e a API passa a divergir do que roda local. Foi assim que o campo
`served` sumiu da resposta durante uma verificação.

**O score do treino precisa passar pela mesma calibração.** Concatenar score
cru (treino) com calibrado (validação/teste) numa coluna só produz duas
escalas misturadas, e qualquer filtro por faixa de score passa a mentir. O
sintoma foi a régua de três faixas mostrar 58% de negados com aprovação
declarada de 68%.
