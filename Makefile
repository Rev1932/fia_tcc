# Makefile — atalhos para o que se repete no dia a dia deste projeto.
#
# Não cobre o pipeline de dados: essa é a responsabilidade do DAG
# `treino_credit_scoring` no Airflow. Aqui ficam infraestrutura, testes,
# diagnóstico e relatórios. Nenhum alvo apaga volume, artefato ou dado.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= .venv/bin/python
PYSYS  ?= python3.12
NBOOT  ?= 500
DAG    ?= treino_credit_scoring
SAMPLE ?= 30000

MLOPS   := docker compose -f MLOps/docker-compose.yml
AIRFLOW := cd MLOps/airflow && docker compose
OBS     := docker compose -f MLOps/monitoring/docker-compose.yml

##@ Ajuda

help: ## Lista os alvos disponíveis
	@awk 'BEGIN {FS = ":.*##"} \
	  /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
	  /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2 }' \
	  $(MAKEFILE_LIST)
	@echo ""

urls: ## Mostra os endereços e credenciais dos serviços
	@echo "  API .......... http://localhost:8000/docs"
	@echo "  Dashboard .... http://localhost:8501"
	@echo "  Airflow ...... http://localhost:8080   (admin/admin)"
	@echo "  Grafana ...... http://localhost:3000   (sem login)"
	@echo "  Prometheus ... http://localhost:9090"

##@ Ambiente

check-venv:
	@test -x $(PYTHON) || { \
	  echo "$(PYTHON) não existe. Rode: make venv"; exit 1; }

venv: ## Cria .venv e instala as dependências (não refaz se já existe)
	@if [ -d .venv ]; then \
	  echo ".venv já existe — use 'make deps' para reinstalar"; \
	else \
	  $(PYSYS) -m venv .venv && .venv/bin/pip install --upgrade pip && \
	  .venv/bin/pip install -r requirements.txt; \
	fi

deps: check-venv ## Reinstala as dependências no .venv existente
	@.venv/bin/pip install -r requirements.txt

##@ Testes

test: check-venv ## Suíte inteira (pyproject já aplica -q; não acrescente outro)
	@$(PYTHON) -m pytest -p no:warnings

test-metrics: check-venv ## Só as métricas e as primitivas de comparação
	@$(PYTHON) -m pytest tests/test_metrics_lib.py -p no:warnings

test-policy: check-venv ## Só a régua de decisão (quem vai para revisão humana)
	@$(PYTHON) -m pytest tests/test_policy.py -p no:warnings

test-split: check-venv ## Prova que a partição de teste não foi tocada
	@$(PYTHON) -m pytest tests/test_split.py -p no:warnings

test-api: check-venv ## Só os endpoints
	@$(PYTHON) -m pytest tests/test_api.py -p no:warnings

test-dags: check-venv ## Callables do DAG (roda no host, sem Airflow)
	@$(PYTHON) -m pytest tests/test_dags.py -p no:warnings

test-airflow: ## Testes de estrutura do DagBag (exige a stack do Airflow no ar)
	@$(AIRFLOW) exec airflow-scheduler bash -c "cd /opt/projeto && pytest tests/test_dags.py"

##@ Diagnóstico e relatórios (leem artefatos; não refazem o pipeline)

resumo: check-venv ## Números oficiais da rodada, em markdown
	@$(PYTHON) Model/run_summary.py --markdown

diagnostico: check-venv ## Diagnóstico por faixa etária (padrão 500 réplicas, ~6 min)
	@$(PYTHON) Model/diagnostico_idade.py --n-boot $(NBOOT) --markdown

teto: check-venv ## Experimento do teto do segmento (~20 min)
	@$(PYTHON) Model/experimento_teto_idade.py --n-boot $(NBOOT)

fairness: check-venv ## Recalcula artifacts/fairness.json sem re-treinar
	@$(PYTHON) scripts/regenerar_fairness.py

dossie: check-venv ## Regenera os dados embutidos em docs/dossie/index.html
	@$(PYTHON) docs/dossie/build_data.py

restaurar-log: check-venv ## Reconstrói improvement_log.json (rode ANTES do 1o treino)
	@$(PYTHON) scripts/restaurar_improvement_log.py

relatorios: resumo dossie test ## Checklist pós-retreino (TODO.md §4)

##@ Serviço local, sem Docker

api-local: check-venv ## Sobe a API em foreground na 8000
	@$(PYTHON) -m uvicorn MLOps.app.api:app --port 8000

dashboard-local: check-venv ## Sobe o Streamlit na 8501 (exige a API no ar)
	@HC_API_URL=http://localhost:8000 .venv/bin/streamlit run MLOps/app/streamlit_app.py

health: ## Consulta o /health da API
	@curl -s localhost:8000/health | $(PYTHON) -m json.tool || \
	  echo "API não respondeu em localhost:8000"

predict: check-venv ## Pontua um cliente: make predict JSON='{"AMT_CREDIT": 500000}'
	@test -n '$(JSON)' || { echo "uso: make predict JSON='{\"AMT_CREDIT\": 500000}'"; exit 1; }
	@$(PYTHON) Model/predict.py --json '$(JSON)'

##@ Stack de serving — API + dashboard (hc-api, hc-dashboard)

mlops-build: ## Constrói a imagem home-credit-scoring:latest
	@$(MLOPS) build

mlops-up: ## Sobe API + dashboard em background, reconstruindo a imagem
	@$(MLOPS) up -d --build

mlops-up-api: ## Sobe só a API
	@$(MLOPS) up -d --build api

mlops-stop: ## Para os containers, preservando-os
	@$(MLOPS) stop

mlops-start: ## Religa containers já parados
	@$(MLOPS) start

mlops-restart: ## Reinicia API + dashboard
	@$(MLOPS) restart

mlops-restart-api: ## Reinicia só a API
	@$(MLOPS) restart api

mlops-ps: ## Estado dos containers de serving
	@$(MLOPS) ps

mlops-logs: ## Segue o log das duas
	@$(MLOPS) logs -f

mlops-logs-api: ## Segue o log da API
	@$(MLOPS) logs -f api

mlops-logs-dashboard: mlops-build ## Segue o log do dashboard (depende da imagem)
	@$(MLOPS) logs -f dashboard

mlops-down: ## Derruba a stack de serving (não remove volume algum)
	@$(MLOPS) down

##@ Airflow — orquestração do re-treino (5 serviços)

airflow-env: ## Cria .env, logs/ e a senha fixa do Airflow (não sobrescreve nada)
	@mkdir -p MLOps/airflow/logs MLOps/airflow/auth
	@if [ -f MLOps/airflow/.env ]; then \
	  echo "MLOps/airflow/.env já existe — preservado"; \
	else \
	  echo "AIRFLOW_UID=$$(id -u)" > MLOps/airflow/.env; \
	  echo "MLOps/airflow/.env criado com AIRFLOW_UID=$$(id -u)"; \
	fi
	@if [ -s MLOps/airflow/auth/passwords.json ]; then \
	  echo "MLOps/airflow/auth/passwords.json já existe — preservado"; \
	else \
	  echo '{"admin": "admin"}' > MLOps/airflow/auth/passwords.json; \
	  echo "MLOps/airflow/auth/passwords.json semeado com admin/admin"; \
	fi

airflow-senha: ## Mostra a senha do admin efetivamente em uso
	@cat MLOps/airflow/auth/passwords.json 2>/dev/null \
	  || echo "ainda não existe — rode: make airflow-env"

airflow-build: airflow-env ## Constrói hc-airflow:3.3.1 (~4 GB na primeira vez)
	@$(AIRFLOW) build

airflow-up: airflow-env ## Sobe os 5 serviços do Airflow em background
	@$(AIRFLOW) up -d --build

airflow-init: ## Remedia "Database migration required"
	@$(AIRFLOW) up airflow-init

airflow-stop: ## Para os containers do Airflow
	@$(AIRFLOW) stop

airflow-start: ## Religa os containers do Airflow
	@$(AIRFLOW) start

airflow-restart: ## Reinicia todos os serviços do Airflow
	@$(AIRFLOW) restart

airflow-restart-scheduler: ## Reinicia só o scheduler
	@$(AIRFLOW) restart airflow-scheduler

airflow-restart-dag-processor: ## Reinicia o dag-processor (é ele que varre dags/)
	@$(AIRFLOW) restart airflow-dag-processor

airflow-ps: ## Estado dos 5 serviços
	@$(AIRFLOW) ps

airflow-logs: ## Segue o log de todos os serviços
	@$(AIRFLOW) logs -f

airflow-logs-scheduler: ## Segue o log do scheduler
	@$(AIRFLOW) logs -f airflow-scheduler

airflow-logs-web: ## Segue o log da interface
	@$(AIRFLOW) logs -f airflow-apiserver

airflow-shell: ## Abre um shell no scheduler
	@$(AIRFLOW) exec airflow-scheduler bash

airflow-down: ## Derruba o Airflow preservando o histórico no volume
	@$(AIRFLOW) down

##@ Airflow — operação do DAG

airflow-sample-on: ## Liga o modo demonstração (padrão hc_sample=30000)
	@$(AIRFLOW) exec airflow-scheduler airflow variables set hc_sample $(SAMPLE)

airflow-sample-off: ## Volta para a base completa (rodada oficial)
	@$(AIRFLOW) exec airflow-scheduler airflow variables set hc_sample ""

airflow-trigger-demo: airflow-sample-on ## Dispara o DAG em amostra (~1 min, grava em artifacts/demo/)
	@$(AIRFLOW) exec airflow-scheduler airflow dags trigger $(DAG)

airflow-trigger: ## Dispara o DAG OFICIAL (~30 min; SOBRESCREVE Dados/ e artifacts/)
	@$(AIRFLOW) exec airflow-scheduler airflow dags trigger $(DAG)

airflow-runs: ## Lista as execuções do DAG
	@$(AIRFLOW) exec airflow-scheduler airflow dags list-runs $(DAG)

airflow-states: ## Estado task a task: make airflow-states RUN=<run_id>
	@test -n '$(RUN)' || { echo "uso: make airflow-states RUN=<run_id>   (veja: make airflow-runs)"; exit 1; }
	@$(AIRFLOW) exec airflow-scheduler airflow tasks states-for-dag-run $(DAG) '$(RUN)'

##@ Observabilidade — Prometheus + Grafana

obs-build: ## Baixa as imagens de Prometheus e Grafana
	@$(OBS) pull

obs-up: mlops-up ## Sobe o monitoramento (exige a stack de serving, pela rede)
	@$(OBS) up -d
	@echo "  Grafana ...... http://localhost:3000   (sem login)"
	@echo "  Prometheus ... http://localhost:9090"

obs-stop: ## Para Prometheus e Grafana
	@$(OBS) stop

obs-start: ## Religa Prometheus e Grafana
	@$(OBS) start

obs-restart: ## Reinicia os dois
	@$(OBS) restart

obs-ps: ## Estado dos containers de observabilidade
	@$(OBS) ps

obs-logs: ## Segue o log dos dois
	@$(OBS) logs -f

obs-logs-prometheus: ## Segue o log do Prometheus
	@$(OBS) logs -f prometheus

obs-alvos: ## Estado do scrape da API (o Prometheus enxerga a API?)
	@curl -s localhost:9090/api/v1/targets \
	  | $(PYTHON) -c "import sys,json; [print('  %-14s %-22s %s' % (t['labels']['job'], t['scrapeUrl'], t['health'])) for t in json.load(sys.stdin)['data']['activeTargets']]" \
	  || echo "  Prometheus não respondeu — rode: make obs-up"

obs-alertas: ## Alertas disparando agora
	@curl -s localhost:9090/api/v1/alerts \
	  | $(PYTHON) -c "import sys,json; a=json.load(sys.stdin)['data']['alerts']; print('  nenhum alerta ativo') if not a else [print('  %-9s %-22s %s' % (x['state'].upper(), x['labels']['alertname'], x['annotations'].get('resumo',''))) for x in a]" \
	  || echo "  Prometheus não respondeu — rode: make obs-up"

obs-regras: ## Lista as regras de alerta carregadas
	@curl -s localhost:9090/api/v1/rules \
	  | $(PYTHON) -c "import sys,json; [print('  [%s] %-20s %s' % (g['name'], r['name'], r['state'])) for g in json.load(sys.stdin)['data']['groups'] for r in g['rules']]" \
	  || echo "  Prometheus não respondeu — rode: make obs-up"

obs-down: ## Derruba o monitoramento (não remove volume algum)
	@$(OBS) down

##@ As duas stacks de uma vez

up: mlops-up airflow-up obs-up urls ## Sobe serving + Airflow + monitoramento

down: obs-down mlops-down airflow-down ## Derruba as três (nenhum volume é removido)

stop: obs-stop mlops-stop airflow-stop ## Para as três

start: mlops-start airflow-start obs-start ## Religa as três

restart: mlops-restart airflow-restart obs-restart ## Reinicia as três

ps: ## Estado das três stacks
	@echo "--- serving ---"; $(MLOPS) ps
	@echo "--- airflow ---"; $(AIRFLOW) ps
	@echo "--- observabilidade ---"; $(OBS) ps

.PHONY: help urls check-venv venv deps \
        test test-metrics test-policy test-split test-api test-dags test-airflow \
        resumo diagnostico teto fairness dossie restaurar-log relatorios \
        api-local dashboard-local health predict \
        mlops-build mlops-up mlops-up-api mlops-stop mlops-start mlops-restart \
        mlops-restart-api mlops-ps mlops-logs mlops-logs-api mlops-logs-dashboard \
        mlops-down \
        airflow-env airflow-senha airflow-build airflow-up airflow-init airflow-stop airflow-start \
        airflow-restart airflow-restart-scheduler airflow-restart-dag-processor \
        airflow-ps airflow-logs airflow-logs-scheduler airflow-logs-web airflow-shell \
        airflow-down airflow-sample-on airflow-sample-off airflow-trigger-demo \
        airflow-trigger airflow-runs airflow-states \
        obs-build obs-up obs-stop obs-start obs-restart obs-ps obs-logs \
        obs-logs-prometheus obs-alvos obs-alertas obs-regras obs-down \
        up down stop start restart ps
