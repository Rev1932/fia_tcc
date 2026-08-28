"""
pipeline_orchestration.py — Roda o pipeline end-to-end de uma vez:
    raw -> clean (sanitization) -> ABT (transform) -> parquet/perfil -> train

    python MLOps/pipeline_orchestration.py

É o caminho MANUAL, para rodar o pipeline sem subir infraestrutura — útil em
desenvolvimento e para reproduzir a rodada num clone limpo.

O caminho AGENDADO é o DAG `dags/treino_credit_scoring.py`, que roda a cada 7
dias no Airflow com uma task por etapa, log por etapa, gate de qualidade e
cálculo de drift. Ver MLOps/airflow/README.md.

Este arquivo já teve um bloco de DAG embutido, protegido por
`try/except ImportError`. Ele nunca executou (o Airflow não estava instalado em
lugar nenhum) e usava a sintaxe do Airflow 2.x. Foi removido: um DAG de
verdade vive em dags/, que é a pasta que o Airflow varre.
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


if __name__ == "__main__":
    run_pipeline()
