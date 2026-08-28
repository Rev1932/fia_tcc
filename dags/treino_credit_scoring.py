"""
Re-treino agendado do modelo de credit scoring.

Substitui a execução manual de arquivos soltos do repositório. Cada etapa do
pipeline vira uma task, com log próprio na interface — quando algo falha, o
Airflow diz ONDE. O `MLOps/pipeline_orchestration.py`, que rodava tudo num
laço de subprocess, continua existindo como caminho manual.

    checar_fontes ─► sanitizacao ─► abt_transform ─► validar_abt
                                                          │
      resumo ◄─ calcular_psi ◄─ validar_metricas ◄─ treino ◄─ perfil_colunas

Agendamento: a cada 7 dias.

Modo demonstração: definindo a Variable `hc_sample` (ex.: 30000) o mesmo DAG
roda em ~1 min sobre uma amostra estratificada. O train.py grava em
artifacts/demo/ e NUNCA por cima da rodada oficial.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.models.dag import DAG
from airflow.models import Variable
# Airflow 3: os operadores saíram do core para o provider `standard`.
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

# A pasta dos DAGs entra no sys.path explicitamente: o dag-processor a
# adiciona sozinho, mas um DagBag() avulso (testes, CLI) não — e aí o import
# falha com ModuleNotFoundError sem explicar por quê.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from callables import calcular_psi, checar_fontes, validar_abt, validar_metricas  # noqa: E402

PROJETO = "/opt/projeto"

# Vazia = base completa (a rodada oficial). Preenchida = modo demonstração.
AMOSTRA = Variable.get("hc_sample", default_var="").strip()
LIMIAR_AUC = Variable.get("hc_auc_max_drop", default_var="0.01")

MODO_AMOSTRA = bool(AMOSTRA)
ARG_AMOSTRA = f" --sample {AMOSTRA}" if MODO_AMOSTRA else ""

argumentos_padrao = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def _bash(task_id: str, comando: str, **kwargs) -> BashOperator:
    """BashOperator com cwd na raiz do projeto.

    O cwd importa: os scripts resolvem caminhos com
    `ROOT = Path(__file__).resolve().parents[1]` e leem os YAMLs de config a
    partir dali. O stdout/stderr vai direto para o log da task.
    """
    return BashOperator(
        task_id=task_id,
        bash_command=comando,
        cwd=PROJETO,
        env={"PYTHONUNBUFFERED": "1", "PYTHONPATH": PROJETO,
             "HC_PROJETO": PROJETO},
        append_env=True,
        **kwargs,
    )


with DAG(
    dag_id="treino_credit_scoring",
    description="Re-treino do modelo de risco de crédito, etapa a etapa",
    default_args=argumentos_padrao,
    schedule=timedelta(days=7),
    start_date=datetime(2026, 8, 1),
    catchup=False,          # ao subir, não reprocessa as semanas passadas
    max_active_runs=1,      # duas rodadas juntas se sobrescreveriam em artifacts/
    tags=["mlops", "credit-scoring", "treino"],
    doc_md=__doc__,
) as dag:

    # ---------------------------------------------------------------- dados
    checar = PythonOperator(
        task_id="checar_fontes",
        python_callable=checar_fontes,
        doc_md="Os 7 CSVs do Kaggle existem e não estão vazios? Falhar aqui em "
               "segundos é melhor que falhar 11 min adiante na agregação.",
    )

    sanitizacao = _bash(
        "sanitizacao",
        "python DataPipeline/data_sanitization.py",
        doc_md="Sentinela DAYS_EMPLOYED=365243 e CODE_GENDER='XNA' viram nulo; "
               "renda limitada no p99,9. → Dados/clean_data.csv",
    )

    abt = _bash(
        "abt_transform",
        "python DataPipeline/abt_transform.py",
        execution_timeout=timedelta(hours=3),
        doc_md="Agrega as 9 tabelas em 1 linha por cliente (~1.020 colunas). "
               "A etapa mais cara: ~11 min e alto pico de memória.",
    )

    validar_a_abt = PythonOperator(
        task_id="validar_abt",
        python_callable=validar_abt,
        doc_md="1 linha por SK_ID_CURR, colunas esperadas, sem coluna com cara "
               "de vazamento. Barra o treino se a ABT estiver quebrada.",
    )

    perfil = _bash(
        "perfil_colunas",
        "python DataPipeline/to_parquet.py",
        doc_md="Valida o Parquet contra o CSV e gera artifacts/abt_profile.json "
               "(nulos, min/max, cardinalidade) — é o que GET /stats/missing serve.",
    )

    # ---------------------------------------------------------------- modelo
    # A tag identifica a rodada no improvement_log.json.
    #
    # Usa `run_id` e não `{{ ds }}`: com agendamento por timedelta e execução
    # manual não existe logical_date, e `{{ ds }}` estoura com
    # "UndefinedError: 'ds' is undefined". O run_id sempre existe e já carrega
    # a data — vira, por exemplo, `airflow-manual-20260828T003000`.
    treino = _bash(
        "treino",
        "python Model/train.py --tag "
        "airflow-{{ run_id | replace('__', '-') | replace(':', '') "
        "| replace('.', '') | replace('+', '') | truncate(40, True, '') }}"
        + ARG_AMOSTRA,
        # Repetir um treino que falhou gasta 15 min para falhar igual.
        retries=0,
        execution_timeout=timedelta(hours=3),
        doc_md="Baseline + LightGBM + calibração isotônica. Grava a rodada "
               "canônica em artifacts/ (ou artifacts/demo/ em modo amostra).",
    )

    validar_as_metricas = PythonOperator(
        task_id="validar_metricas",
        python_callable=validar_metricas,
        op_kwargs={"limiar": float(LIMIAR_AUC)},
        doc_md="GATE: falha a DAG se o AUC caiu além do limiar frente à última "
               "rodada aceita. A regra de aceite do projeto, automatizada.",
    )

    psi = PythonOperator(
        task_id="calcular_psi",
        python_callable=calcular_psi,
        doc_md="Population Stability Index do score contra a rodada anterior. "
               "É o 'PSI em job no Airflow' que o MLOps/Readme.md prometia.",
    )

    resumo = _bash(
        "resumo_da_rodada",
        "python Model/run_summary.py --markdown",
        retries=0,
        doc_md="Imprime no log os números oficiais da rodada, prontos para "
               "colar em documento ou slide.",
    )

    (checar >> sanitizacao >> abt >> validar_a_abt >> perfil
     >> treino >> validar_as_metricas >> psi >> resumo)
