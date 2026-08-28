"""
settings.py — Tudo que é caminho, limite ou whitelist mora aqui.

Regra: se a banca pedir "mostra também a coluna X" ou "deixa ordenar por Y",
a mudança é neste arquivo, não espalhada pelos routers.

Todos os caminhos são sobrescrevíveis por variável de ambiente (HC_*). Isso
não é conveniência de teste: é o mesmo mecanismo que o docker-compose usa
para apontar a API para os volumes montados.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _path(env: str, default: str) -> Path:
    return Path(os.getenv(env, ROOT / default))


# ------------------------------------------------------------------ caminhos
ABT_PARQUET = _path("HC_ABT_PARQUET", "Dados/abt.parquet")
SCORES_PARQUET = _path("HC_SCORES_PARQUET", "artifacts/scores.parquet")
MODEL_PATH = _path("HC_MODEL_PATH", "artifacts/model.joblib")
ARTIFACTS_DIR = _path("HC_ARTIFACTS_DIR", "artifacts")

# ------------------------------------------------------------------- runtime
DUCKDB_THREADS = int(os.getenv("HC_DUCKDB_THREADS", "4"))
DUCKDB_MEMORY_LIMIT = os.getenv("HC_DUCKDB_MEMORY_LIMIT", "2GB")
# Constrói o TreeExplainer no startup: evita 1-2s de cold start bem no meio
# da demonstração.
EXPLAINER_EAGER = os.getenv("HC_EXPLAINER_EAGER", "1") == "1"

# -------------------------------------------------------------------- limites
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500
MAX_SELECTED_COLUMNS = 60      # a ABT tem 473+ colunas; SELECT * seria absurdo
MAX_PREDICT_RECORDS = 500
MAX_EXPLAIN_TOP = 50

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
SCORE_COL = "proba_champion"

# --------------------------------------------------- colunas devolvidas por padrão
# O que aparece em GET /clients quando ?fields= não é passado.
DEFAULT_CLIENT_COLUMNS = [
    "SK_ID_CURR", "TARGET", "AGE_YEARS", "CODE_GENDER", "NAME_CONTRACT_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "OCCUPATION_TYPE",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "BUREAU_COUNT", "PREV_COUNT", "proba_champion", "split",
]

# Blocos da ficha de cliente (GET /clients/{id})
CLIENT_DETAIL_BLOCKS = {
    "identificacao": ["CODE_GENDER", "AGE_YEARS", "NAME_FAMILY_STATUS", "CNT_CHILDREN",
                      "CNT_FAM_MEMBERS", "NAME_EDUCATION_TYPE", "OCCUPATION_TYPE",
                      "ORGANIZATION_TYPE", "NAME_HOUSING_TYPE", "YEARS_EMPLOYED"],
    "financeiro": ["AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "AMT_CREDIT", "AMT_ANNUITY",
                   "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE", "CREDIT_INCOME_RATIO",
                   "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"],
    "historico": ["BUREAU_COUNT", "PREV_COUNT", "POS_COUNT", "CC_COUNT", "INST_COUNT"],
    "scores_externos": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
}

# ------------------------------------------------------------------ dimensões
# Expressões SQL CONSTANTES: `by=` só aceita chaves deste dicionário, então
# nenhum texto de usuário chega a ser interpolado em SQL. Resolve segurança e
# reuso de uma vez — e é o que permite "agrupa por faixa etária" na hora.
DIMENSIONS: dict[str, str] = {
    "gender": "CODE_GENDER",
    "contract_type": "NAME_CONTRACT_TYPE",
    "education": "NAME_EDUCATION_TYPE",
    "family_status": "NAME_FAMILY_STATUS",
    "income_type": "NAME_INCOME_TYPE",
    "housing_type": "NAME_HOUSING_TYPE",
    "occupation": "OCCUPATION_TYPE",
    "organization": "ORGANIZATION_TYPE",
    "split": "split",
    "target": "CAST(TARGET AS VARCHAR)",
    "children": "CAST(LEAST(CNT_CHILDREN, 5) AS VARCHAR)",
    "age_band": (
        "CASE WHEN AGE_YEARS < 25 THEN '<25'"
        " WHEN AGE_YEARS < 35 THEN '25-35'"
        " WHEN AGE_YEARS < 45 THEN '35-45'"
        " WHEN AGE_YEARS < 55 THEN '45-55'"
        " WHEN AGE_YEARS < 65 THEN '55-65'"
        " WHEN AGE_YEARS IS NULL THEN 'desconhecido' ELSE '65+' END"
    ),
    "income_band": (
        "CASE WHEN AMT_INCOME_TOTAL < 100000 THEN 'ate 100k'"
        " WHEN AMT_INCOME_TOTAL < 150000 THEN '100k-150k'"
        " WHEN AMT_INCOME_TOTAL < 225000 THEN '150k-225k'"
        " WHEN AMT_INCOME_TOTAL IS NULL THEN 'desconhecido' ELSE '225k+' END"
    ),
    "credit_band": (
        "CASE WHEN AMT_CREDIT < 250000 THEN 'ate 250k'"
        " WHEN AMT_CREDIT < 500000 THEN '250k-500k'"
        " WHEN AMT_CREDIT < 1000000 THEN '500k-1M'"
        " WHEN AMT_CREDIT IS NULL THEN 'desconhecido' ELSE '1M+' END"
    ),
    "score_band": (
        "CASE WHEN proba_champion IS NULL THEN 'sem score'"
        " WHEN proba_champion < 0.05 THEN 'A (<5%)'"
        " WHEN proba_champion < 0.10 THEN 'B (5-10%)'"
        " WHEN proba_champion < 0.20 THEN 'C (10-20%)'"
        " WHEN proba_champion < 0.35 THEN 'D (20-35%)' ELSE 'E (35%+)' END"
    ),
    # thin-file = sem nenhum registro no bureau. Mesmo critério do
    # evaluation.ipynb, para os números baterem com o do notebook.
    "thin_file": (
        "CASE WHEN BUREAU_COUNT IS NULL THEN 'thin-file (sem bureau)'"
        " ELSE 'com historico de bureau' END"
    ),
}

DIMENSION_LABELS = {
    "gender": "Gênero", "contract_type": "Tipo de contrato",
    "education": "Escolaridade", "family_status": "Estado civil",
    "income_type": "Tipo de renda", "housing_type": "Moradia",
    "occupation": "Ocupação", "organization": "Organização",
    "split": "Fatia (treino/validação/teste)", "target": "Inadimplente (real)",
    "children": "Nº de filhos", "age_band": "Faixa etária",
    "income_band": "Faixa de renda", "credit_band": "Faixa de crédito",
    "score_band": "Faixa de score", "thin_file": "Thin-file",
}
