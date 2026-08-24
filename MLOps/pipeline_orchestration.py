"""
pipeline_orchestration.py — Orquestra o fluxo end-to-end:
    raw -> clean (sanitization) -> ABT (transform) -> parquet/perfil -> train

Uso standalone:
    python MLOps/pipeline_orchestration.py

Em produção, cada `step` vira uma task de um DAG do Airflow (ver MLOps/Readme.md).
O bloco final expõe um DAG opcional caso o Airflow esteja instalado.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STEPS = [
    ("sanitization", [PY, str(ROOT / "DataPipeline" / "data_sanitization.py")]),
    ("abt_transform", [PY, str(ROOT / "DataPipeline" / "abt_transform.py")]),
    # Perfil das colunas para a API (nulos, min/max, cardinalidade). O parquet
    # em si já sai do abt_transform; aqui garantimos o abt_profile.json.
    ("profile", [PY, str(ROOT / "DataPipeline" / "to_parquet.py")]),
    ("train", [PY, str(ROOT / "Model" / "train.py")]),
]


def run_pipeline() -> None:
    for name, cmd in STEPS:
        print(f"\n===== STEP: {name} =====", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
    print("\n[pipeline] concluído com sucesso.")


# ---- DAG opcional do Airflow (carregado apenas se o pacote existir) ----
try:
    from datetime import datetime

    from airflow import DAG
    from airflow.operators.bash import BashOperator

    with DAG(
        dag_id="home_credit_scoring_pipeline",
        schedule="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["mlops", "credit-scoring"],
    ) as dag:
        prev = None
        for name, cmd in STEPS:
            task = BashOperator(task_id=name, bash_command=" ".join(cmd))
            if prev:
                prev >> task
            prev = task
except ImportError:
    pass  # Airflow não instalado: roda em modo standalone


if __name__ == "__main__":
    run_pipeline()
