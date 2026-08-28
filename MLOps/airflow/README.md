# Orquestração do treino — Apache Airflow

O re-treino do modelo de risco de crédito, agendado e acompanhável.

Antes, o pipeline era disparado à mão rodando quatro arquivos em sequência.
Não havia agendamento, histórico de execuções, nem como acompanhar uma rodada
enquanto ela acontecia — o único registro era o que sobrava no terminal.

---

## Subir

```bash
cd MLOps/airflow
echo "AIRFLOW_UID=$(id -u)" > .env      # obrigatório — ver §Permissões
docker compose up -d --build
```

Interface: **<http://localhost:8080>** · usuário `admin`, senha `admin`.

A primeira subida constrói a imagem (~4 GB: Airflow + as dependências de
modelagem) e migra o banco. As seguintes levam segundos.

```bash
docker compose ps      # 5 serviços; apiserver e db precisam ficar healthy
docker compose down    # parar (o histórico de execuções sobrevive no volume)
```

---

## Os serviços

| Serviço | Papel |
|---|---|
| `hc-airflow-db` | Postgres com o histórico de execuções e os logs |
| `hc-airflow-web` | interface e Execution API — porta **8080** |
| `hc-airflow-scheduler` | dispara o DAG e executa as tasks (LocalExecutor) |
| `hc-airflow-dag-processor` | varre `dags/` — **sem ele os DAGs nunca aparecem** |
| `hc-airflow-init` | roda uma vez: migra o banco e cria as Variables |

**LocalExecutor**, e não CeleryExecutor: sem Redis e sem workers separados. O
compose oficial do Airflow tem 8 serviços; aqui são 5, e o executor roda as
tasks como subprocessos do próprio scheduler. Suficiente para uma máquina.

Não conflita com a API (8000) nem com o dashboard (8501) do
`MLOps/docker-compose.yml` — são stacks independentes, com ciclos de vida
diferentes: a API fica no ar servindo predições, o Airflow acorda a cada 7 dias.

---

## O DAG `treino_credit_scoring`

Agendamento: **a cada 7 dias**. `catchup=False` (ao subir, não reprocessa as
semanas passadas) e `max_active_runs=1` (as tasks escrevem em caminhos fixos;
duas rodadas juntas se sobrescreveriam).

| # | Task | O que faz | Tempo |
|---|---|---|---|
| 1 | `checar_fontes` | Os 7 CSVs do Kaggle existem e não estão vazios? | seg |
| 2 | `sanitizacao` | Sentinelas → nulo, clip de renda → `clean_data.csv` | ~2 min |
| 3 | `abt_transform` | Agrega as 9 tabelas em 1 linha por cliente | **~11 min** |
| 4 | `validar_abt` | Granularidade, colunas, ausência de vazamento | seg |
| 5 | `perfil_colunas` | Valida o Parquet e gera `abt_profile.json` | ~1 min |
| 6 | `treino` | Baseline + LightGBM + calibração isotônica | **~15 min** |
| 7 | `validar_metricas` | **Gate**: falha se o AUC caiu além do limiar | seg |
| 8 | `calcular_psi` | Drift do score contra a rodada anterior | seg |
| 9 | `resumo_da_rodada` | Números oficiais no log, prontos para colar | seg |

Cada etapa é uma task porque cada uma tem entrada, saída e custo próprios —
quando algo falha, o Airflow diz **onde**. O `stdout` de cada script vai
direto para o log da task.

### As tasks que não existiam antes

**`checar_fontes`** falha em segundos se um CSV sumiu, em vez de descobrir isso
11 minutos adiante no meio da agregação.

**`validar_abt`** confere 1 linha por `SK_ID_CURR`, o mínimo de colunas
esperado e a ausência de colunas com cara de vazamento. Descobrir que a ABT
está quebrada **depois** de 15 minutos treinando não serve para nada.

**`validar_metricas`** é o gate de qualidade — a regra de aceite do projeto
(`TODO.md` §4) virando verificação automática:

```
AUC anterior 0,7868 → atual 0,7791  (−0,0077)   limiar 0,0100  → passa, com aviso
AUC anterior 0,7868 → atual 0,7712  (−0,0156)   limiar 0,0100  → FALHA a DAG
```

Um re-treino automático que degrada o modelo não pode substituir em silêncio o
que estava servindo. Compara com a última rodada **aceita** do
`improvement_log.json` — ignora as rejeitadas e as de amostra.

**`calcular_psi`** cumpre o que `MLOps/Readme.md` prometia como *"PSI calculado
em job no Airflow"*. O cálculo já existia (`Model.metrics_lib.psi`); faltava o
agendamento. Grava `artifacts/psi_report.json`.

---

## Modo demonstração

A rodada completa leva ~30 minutos — inviável de mostrar ao vivo. A Variable
`hc_sample` resolve:

```bash
# amostra estratificada: o MESMO DAG roda em ~1 min
docker compose exec airflow-scheduler airflow variables set hc_sample 30000
docker compose exec airflow-scheduler airflow dags trigger treino_credit_scoring

# voltar para a rodada oficial
docker compose exec airflow-scheduler airflow variables set hc_sample ""
```

Em modo amostra o `train.py` grava em `artifacts/demo/` e **nunca** por cima da
rodada oficial.

---

## Variables

| Variable | Padrão | Para quê |
|---|---|---|
| `hc_sample` | `""` (vazio) | Nº de linhas do modo demonstração. Vazio = base completa |
| `hc_auc_max_drop` | `0.01` | Queda de AUC tolerada antes do gate falhar |

Alteráveis pela interface em **Admin → Variables**, sem reiniciar nada.

---

## Acompanhar uma execução

Na interface: **Dags → treino_credit_scoring → Graph**. Cada quadrado é uma
task; clicando nela, **Logs** mostra a saída do script em tempo real.

Pela linha de comando:

```bash
docker compose exec airflow-scheduler airflow dags list-runs treino_credit_scoring
docker compose exec airflow-scheduler \
  airflow tasks states-for-dag-run treino_credit_scoring <run_id>
```

Os logs também ficam em `MLOps/airflow/logs/`, organizados por DAG, execução e
tentativa.

---

## Permissões — a falha mais provável

O `.env` com `AIRFLOW_UID=$(id -u)` **não é opcional**. Sem ele o container
escreve em `Dados/` e `artifacts/` como root, e o `.venv` do host perde acesso
à própria rodada canônica.

O **GID fica em 0**, e não é escolha: a imagem do Airflow recusa qualquer outro
valor (`ERROR! You should run the image with GID=0`). Na prática os arquivos
saem com **dono correto e grupo root** — o dono é o que importa para leitura e
escrita.

Conferir depois da primeira execução:

```bash
ls -la ../../artifacts/ | head    # o dono precisa ser você, não root
```

Se já aconteceu: `sudo chown -R $(id -u):$(id -g) artifacts Dados`.

---

## Quando uma task falha

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| DAG não aparece na lista | `dag-processor` fora do ar | `docker compose ps`, `logs airflow-dag-processor` |
| Log da task tem **uma linha só** | Worker não alcança a Execution API | Conferir `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` no compose |
| `Can't decrypt _val for key=...` | `FERNET_KEY` mudou entre containers | A chave é fixa no compose; se recriou o volume do banco, recrie as Variables |
| `checar_fontes` falha | CSVs do Kaggle ausentes | Baixar para `data/risco_fraude/home-credit-default-risk/` |
| `validar_abt` falha | ABT com cliente duplicado ou poucas colunas | Ler o log: ele diz qual checagem quebrou |
| `validar_metricas` falha | **O modelo piorou** — é o gate funcionando | Investigar antes de promover. Os artefatos anteriores continuam intactos |
| `UndefinedError: 'ds' is undefined` | Template Jinja usando `{{ ds }}` | Com agendamento por `timedelta` e execução manual **não existe `logical_date`**. Use `{{ run_id }}`, que sempre existe |
| `Database migration required` | Banco não migrado | `docker compose up airflow-init` e reiniciar o apiserver |

---

## Testes

```bash
# callables (rodam no host, sem Airflow instalado)
pytest tests/test_dags.py

# estrutura do DAG — precisa do Airflow, então roda no container
cd MLOps/airflow
docker compose exec airflow-scheduler \
  bash -c "cd /opt/projeto && pytest tests/test_dags.py"
```

23 testes: 9 de estrutura (agendamento de 7 dias, ordem das tasks, imports do
Airflow 3) e 14 dos callables — incluindo o gate aprovando queda dentro do
limiar e **falhando** acima dele.

---

## Armadilhas do Airflow 3 já pagas neste projeto

Ficam registradas porque nenhuma delas dá erro óbvio:

1. **Imports mudaram de lugar.** `airflow.operators.bash` virou
   `airflow.providers.standard.operators.bash`. O código antigo quebra.
2. **O `dag-processor` é um serviço separado.** Sem ele os arquivos em `dags/`
   ficam montados e **nunca aparecem** — sem erro nenhum.
3. **As tasks falam com a Execution API**, não com o banco. Sem
   `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` o worker tenta `localhost` e morre
   com `Connection refused` **antes** de rodar qualquer linha: o log da task
   fica com uma linha só, o que despista completamente.
4. **`JWT_SECRET` e `FERNET_KEY` precisam ser as mesmas** em todos os
   containers. Cada um gerando a sua produz `Invalid auth token` e
   `Can't decrypt _val`.
5. **`{{ ds }}` não existe** em DAG com `schedule=timedelta(...)` disparado
   manualmente — não há `logical_date`. Use `{{ run_id }}`.
6. **Sobrescrever `entrypoint`** por `/bin/bash` quebra o `PATH` da imagem:
   `ModuleNotFoundError: No module named 'airflow'`. Passe o comando como
   `command: [bash, -c, ...]` e deixe o entrypoint oficial em paz.
7. **`DagBag(...)` não aceita mais `include_examples`** — os exemplos saem por
   `AIRFLOW__CORE__LOAD_EXAMPLES`.

---

## Por que a imagem não instala o `requirements.txt` inteiro

`MLOps/airflow/requirements-pipeline.txt` é um subconjunto deliberado. O
`requirements.txt` do projeto traz `fastapi==0.138.1`, e o
`apache-airflow-core 3.3.1` exige `fastapi<0.137.0` — instalar tudo sobrescreve
a versão do Airflow e deixa a interface numa combinação não testada.

O Airflow **orquestra** o pipeline; ele não serve a API. As dependências de
serving ficam na imagem `home-credit-scoring:latest`.

As versões de modelagem (pandas, numpy, scikit-learn, lightgbm) são idênticas
às do `requirements.txt`, e a imagem base usa o mesmo Python 3.14 — o
`model.joblib` treinado aqui é carregado pela API sem risco de pickle
incompatível.
