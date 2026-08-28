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
| API | ✅ 27 endpoints, documentação própria, 148 testes passando (11 pulados: DAG sem Airflow instalado) |
| Dashboard | ✅ Executado e coberto por `tests/test_dashboard.py` |
| Docker | ✅ Build e stack verificados, healthcheck falha corretamente |
| Orquestração | ✅ Airflow 3.3.1 em Docker, DAG de 9 tasks a cada 7 dias |
| Documentação | ✅ README, TCC, pitch e guia reconciliados com os artefatos |
| Monitoramento de drift | ✅ `GET /model/psi` implementado e testado |
| Telemetria de serviço | ✅ `GET /metrics` + Prometheus/Grafana, com alertas de indisponibilidade |
| Notebook de avaliação | ✅ Reescrito lendo de `artifacts/`, re-executado nesta máquina |
| Slides | ✅ `credit_scoring_deck_v3.pptx` — falta só um slide de calibração, à mão |
| Fraqueza em `<25 anos` | ✅ **Diagnosticada e com veredito**: teto de dado, não defeito de modelo. Causa declarada anteriormente (falta de histórico) **refutada**. Dois consertos testados e rejeitados. Ver [`docs/diagnostico-faixa-etaria.md`](docs/diagnostico-faixa-etaria.md) |

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

## 2. A fraqueza do `<25` — ✅ diagnosticada em 28/08/2026

Veredito completo em [`docs/diagnostico-faixa-etaria.md`](docs/diagnostico-faixa-etaria.md).
Resumo do que mudou nesta rodada:

**O critério que declarava a fraqueza era inválido.** Comparava o AUC do segmento com o IC
do AUC **geral** — que contém o próprio segmento e é composto 77,6% por pares entre faixas
etárias, comparações que nenhum AUC intra-faixa faz. E estava implementado em duas versões
divergentes (`policy.py` unilateral, `routers/model.py` simétrica). Substituído pelo
bootstrap da **diferença** contra os demais grupos do eixo
(`Model/metrics_lib.py::auc_diff_bootstrap`), unificado nos quatro pontos.

**A conclusão sobrevive ao critério correto**, e encolhe pouco: de −0,0549 para
**−0,0514** (IC [−0,0827; −0,0205], p = 0,008). A classificação não muda: `<25` e `55-65`
seguem sendo as fraquezas do eixo, e a faixa cinza dobrada continua valendo para os mesmos
clientes.

**A causa que este arquivo declarava está errada.** O texto anterior dizia que a
dificuldade era "ausência de sinal" e que "a fraqueza vem de ausência de histórico". Foi
testado: uma coorte de 25–45 anos reamostrada até reproduzir o perfil de informação do
jovem — mesma taxa de thin-file, mesmo número de scores externos, mesmo tempo de emprego,
mesmo comprimento de histórico, 71 estratos, 99,9% de cobertura — atinge **AUC 0,7803**
contra 0,7319 do jovem, e é melhor em 99,8% das réplicas. **Escassez de informação não
explica o buraco.**

Pareando também pelo **nível** do score externo, o buraco cai de −0,0484 para −0,0308:
~36% do efeito é "estar na região baixa do score externo, onde o modelo separa pior em
qualquer idade", e não "ser jovem".

### Modelo segmentado — ❌ testado e REJEITADO em 28/08/2026

Seis variantes (`<25` e `<30`, três capacidades cada), mesma partição, mesmos 2.355
clientes de teste. **Nenhuma supera o modelo geral (0,7319).** A melhor,
`segmentado_<30` com `num_leaves=34`, fica em 0,7296. Registro em
`artifacts/experimentos/teto_idade.json`; script em `Model/experimento_teto_idade.py`.

Somado à reponderação já rejeitada (`v4-pesos-idade`), fecha o cerco: o modelo global já
está no teto do segmento para este conjunto de variáveis.

### Achado novo — viés de calibração no `<25`

`<25` é a **única** faixa com viés de nível estatisticamente significativo: prevê **13,4%**
onde ocorrem **11,8%** (+1,6 pp, IC [+0,4; +2,9]). Causa identificada: a isotônica é
ajustada globalmente. Reajustada só nos jovens, o gap vai a −0,0005.

**Mas não é um conserto gratuito:** o Brier não melhora (0,09584 → 0,09583) e a taxa de
aprovação do `<25` no corte 0,09 **cai** de 47,6% para 44,8%. Corrige um viés agregado —
relevante para perda esperada de carteira — sem melhorar decisão individual nem ampliar
acesso. **Decisão de negócio pendente**, não adotado.

### O que muda na recomendação de dados alternativos

Ela era justificada por ausência de histórico, que foi refutado. Continua sendo uma via
plausível, mas agora precisa de outro argumento: seria preciso mostrar que a fonte nova
discrimina **dentro** do grupo jovem, não apenas que preenche lacunas.

### A faixa `55-65` — ✅ diagnosticada em 28/08/2026

Fraqueza confirmada (−0,0415, p = 0,004, n = 12.166), com causa **diferente** da do `<25`.

**Não é aposentadoria.** A faixa é 68,4% aposentada contra 0,4–7,8% nas do meio, e a
sanitização mata o bloco de emprego desse público (`DAYS_EMPLOYED = 365243` → nulo). Mas
dentro da faixa, aposentado (0,7488) e ativo (0,7427) são indistinguíveis, e os
aposentados de `45-55` vão **acima** da média (0,8065). Sem controlar idade o eixo
aposentadoria parece explicativo (Δ −0,0313, p = 0,012); é confundimento.

**Não é perfil de informação.** Coorte pareada tirada de `35-45`/`45-55` atinge 0,7906
contra 0,7465 — e o `55-65` é pior em **100%** das réplicas.

**Não se concentra em recorte nenhum** — gênero, tipo de contrato, thin-file e número de
scores externos dão todos ~0,74.

**A causa:** os três scores externos rendem ali o **pior de todas as faixas** (0,6288 /
0,6301 / 0,6506 contra 0,6610 / 0,6733 / 0,6701 aos 35-45). O modelo agrega o normal em
cima disso (ganho 0,0744, em linha com as melhores faixas) — ele só parte de um sinal
pior. É deficiência **da fonte de dado**, não de modelagem.

Contraste com o `<25`, que falha de outro jeito: lá o sinal também é fraco **e** o modelo
agrega o mínimo de todas as faixas (0,0518). Detalhe em
[`docs/diagnostico-faixa-etaria.md §8b`](docs/diagnostico-faixa-etaria.md).

### O que ficou aberto
- **O resíduo de −0,0308** é medido e específico da idade; a causa mecânica segue desconhecida.
- **`AGE_YEARS` e `DAYS_BIRTH`** são ambos variáveis vivas e colineares — divide o ganho
  entre duas colunas e subestima a importância da idade. Higiene, não conserto.

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

### 3.5b `scores.parquet` não tem todos os clientes
`Model/train.py:493-502` concatena `id_tr`, `id_val` e `id_test` e **omite
`id_cal`**: 30.751 clientes da fatia de calibração ficam de fora. `MLOps/app/db.py`
faz LEFT JOIN, então eles aparecem em `GET /clients` com `split` e
`proba_champion` nulos. Travado em `tests/test_split.py`.

### 3.5c O gate do Airflow compara métricas diferentes
`Model/train.py` grava no `improvement_log` o AUC **cru** do campeão, enquanto
`dags/callables.py::_auc_servido` compara contra o AUC **calibrado**. O viés é de
+0,0003 contra um limiar de 0,01 — nunca disparou, mas está errado.

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
5. **Fraqueza de segmento se declara pelo IC da DIFERENÇA**, nunca comparando
   com o AUC geral — que contém o próprio segmento e é composto em sua maior
   parte por pares entre grupos. Use `metrics_lib.auc_diff_bootstrap`. O
   critério antigo (IC do grupo sem sobrepor o IC geral) era descalibrado; ver
   `docs/diagnostico-faixa-etaria.md §5`.
6. **Toda causa afirmada precisa de teste.** "A fraqueza vem de X" é hipótese
   até existir a coorte pareada que isole X. Foi assim que a causa declarada
   para o `<25` caiu.

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
