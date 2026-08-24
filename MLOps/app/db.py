"""
db.py — Acesso aos dados via DuckDB sobre Parquet.

Por que DuckDB: a ABT tem 307.511 clientes x 473+ colunas (656 MB em CSV,
156 MB em Parquet). Carregar em memória custaria ~1,2 GB e um startup lento;
o DuckDB lê direto do disco e faz pushdown de projeção e filtro, então pedir
12 colunas custa 12 colunas.

Se a banca pedir "adiciona um filtro por X", o lugar é `where_from_filters`.
"""
from __future__ import annotations

import math
from typing import Any

import duckdb
from fastapi import HTTPException, Request

from MLOps.app import settings


def _lit(value) -> str:
    """Caminho como literal SQL. O DuckDB não aceita parâmetro preparado em
    read_parquet(), então interpolamos — com aspas escapadas. Estes caminhos
    vêm de settings/env, nunca de entrada de usuário."""
    return "'" + str(value).replace("'", "''") + "'"


def connect() -> duckdb.DuckDBPyConnection:
    """Conexão + views. Chamada uma vez no lifespan da aplicação."""
    con = duckdb.connect(database=":memory:", config={
        "threads": settings.DUCKDB_THREADS,
        # trava a memória: um GROUP BY sobre 307k linhas não pode derrubar
        # o container no meio da apresentação
        "memory_limit": settings.DUCKDB_MEMORY_LIMIT,
    })
    con.execute(f"CREATE VIEW abt AS SELECT * FROM read_parquet({_lit(settings.ABT_PARQUET)})")

    if settings.SCORES_PARQUET.exists():
        con.execute(
            f"CREATE VIEW scores AS SELECT * FROM read_parquet({_lit(settings.SCORES_PARQUET)})")
        # LEFT JOIN como VIEW, não merge materializado: o DuckDB resolve o join
        # só nas colunas efetivamente pedidas.
        con.execute(f"""
            CREATE VIEW clients AS
            SELECT a.*, s.split, s.y_true, s.proba_champion, s.proba_baseline
            FROM abt a LEFT JOIN scores s USING ({settings.ID_COL})
        """)
    else:
        # A API sobe mesmo sem scores (antes do primeiro treino): as colunas
        # existem como NULL e os endpoints de modelo avisam o que falta.
        con.execute("""
            CREATE VIEW clients AS
            SELECT a.*, NULL::VARCHAR AS split, NULL::TINYINT AS y_true,
                   NULL::DOUBLE AS proba_champion, NULL::DOUBLE AS proba_baseline
            FROM abt a
        """)
    return con


def describe_columns(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """{coluna: tipo} da view `clients`.

    Esta é a whitelist de identificadores: derivada do dado real, então não
    existe lista de 473 nomes para manter desatualizada.
    """
    return {r[0]: r[1] for r in con.execute("DESCRIBE clients").fetchall()}


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """Dependency do FastAPI.

    Devolve um `.cursor()` e não a conexão: DuckDBPyConnection NÃO é
    thread-safe, e endpoints síncronos do FastAPI rodam num threadpool.
    O cursor é uma conexão-filha isolada sobre o mesmo banco.
    """
    con = getattr(request.app.state, "db", None)
    if con is None:
        raise HTTPException(
            status_code=503,
            detail=("Dados não carregados. Gere a ABT em Parquet com "
                    "`python DataPipeline/to_parquet.py`."),
        )
    return con.cursor()


def get_columns(request: Request) -> dict[str, str]:
    cols = getattr(request.app.state, "columns", None)
    if not cols:
        raise HTTPException(status_code=503, detail="Esquema não carregado.")
    return cols


# --------------------------------------------------------------------------
# Identificadores: nunca parametrizáveis em SQL, então sempre por whitelist
# --------------------------------------------------------------------------


def quote_ident(name: str, allowed: dict[str, str]) -> str:
    if name not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Coluna desconhecida: {name!r}. Veja GET /meta/columns.",
        )
    return '"' + name.replace('"', '""') + '"'


def parse_fields(fields: str | None, allowed: dict[str, str]) -> list[str]:
    """Lista de colunas pedida em ?fields=, validada contra a whitelist."""
    if not fields:
        return [c for c in settings.DEFAULT_CLIENT_COLUMNS if c in allowed]
    names = [c.strip() for c in fields.split(",") if c.strip()]
    if not names:
        return [c for c in settings.DEFAULT_CLIENT_COLUMNS if c in allowed]
    if len(names) > settings.MAX_SELECTED_COLUMNS:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo de {settings.MAX_SELECTED_COLUMNS} colunas por requisição "
                   f"(pedidas: {len(names)}).",
        )
    for n in names:
        quote_ident(n, allowed)
    # SK_ID_CURR sempre presente: sem ele a linha não é identificável
    if settings.ID_COL not in names:
        names.insert(0, settings.ID_COL)
    return names


def dimension_expr(by: str) -> str:
    """Expressão SQL de uma dimensão de agrupamento.

    Só aceita chaves de settings.DIMENSIONS — as expressões são constantes do
    código, então a interpolação é segura.
    """
    if by not in settings.DIMENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Dimensão desconhecida: {by!r}. Opções: "
                   f"{', '.join(sorted(settings.DIMENSIONS))}.",
        )
    return settings.DIMENSIONS[by]


# --------------------------------------------------------------------------
# Filtros -> WHERE parametrizado (reusado por /clients e por todo /stats)
# --------------------------------------------------------------------------


def split_csv(value) -> list[str] | None:
    """'a,b' -> ['a', 'b'] (ver comentário em schemas.CsvList)."""
    if value is None:
        return None
    if isinstance(value, list):
        return value or None
    partes = [s.strip() for s in str(value).split(",") if s.strip()]
    return partes or None


def where_from_filters(f, threshold: float | None = None) -> tuple[str, list]:
    """Traduz os filtros para (cláusula WHERE, parâmetros).

    Uma lista de `if`, deliberadamente sem abstração: é o arquivo que você
    abre para acrescentar um filtro na frente da banca. Todos os VALORES vão
    como parâmetro `?`; nenhum texto de usuário entra na string SQL.
    """
    parts: list[str] = []
    params: list[Any] = []

    def add(sql: str, *vals):
        parts.append(sql)
        params.extend(vals)

    def rng(col: str, lo, hi):
        if lo is not None:
            add(f"{col} >= ?", lo)
        if hi is not None:
            add(f"{col} <= ?", hi)

    rng("AGE_YEARS", f.age_min, f.age_max)
    rng("AMT_INCOME_TOTAL", f.income_min, f.income_max)
    rng("AMT_CREDIT", f.credit_min, f.credit_max)
    rng("AMT_ANNUITY", f.annuity_min, f.annuity_max)
    rng("CNT_CHILDREN", f.children_min, f.children_max)
    rng("YEARS_EMPLOYED", f.employed_years_min, f.employed_years_max)
    rng("proba_champion", f.score_min, f.score_max)

    if f.gender:
        add("CODE_GENDER = ?", f.gender)
    for col, raw in (("NAME_CONTRACT_TYPE", f.contract_type),
                     ("NAME_EDUCATION_TYPE", f.education),
                     ("NAME_INCOME_TYPE", f.income_type),
                     ("NAME_FAMILY_STATUS", f.family_status),
                     ("NAME_HOUSING_TYPE", f.housing_type),
                     ("OCCUPATION_TYPE", f.occupation)):
        values = split_csv(raw)
        if values:
            add(f"{col} IN ({', '.join('?' * len(values))})", *values)

    if f.target is not None:
        add("TARGET = ?", f.target)
    if f.split:
        add("split = ?", f.split)

    # thin-file: sem NENHUM registro no bureau. Mesmo critério do
    # evaluation.ipynb, para os números da API baterem com os do notebook.
    if f.thin_file is True:
        add("BUREAU_COUNT IS NULL")
    elif f.thin_file is False:
        add("BUREAU_COUNT IS NOT NULL")

    if f.decision and threshold is not None:
        if f.decision == "APROVAR":
            add("proba_champion < ?", threshold)
        elif f.decision == "NEGAR":
            add("proba_champion >= ?", threshold)

    return (" AND ".join(parts) if parts else "TRUE"), params


# --------------------------------------------------------------------------
# Execução
# --------------------------------------------------------------------------


def _clean(value):
    """NaN/Inf não são JSON válido — e a ABT tem coluna com 56% de nulo.
    Sanitizar aqui, num ponto só, evita um 500 na primeira chamada."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rows_to_dicts(cursor) -> list[dict]:
    cols = [d[0] for d in cursor.description]
    return [{c: _clean(v) for c, v in zip(cols, row)} for row in cursor.fetchall()]


def fetch_all(con, sql: str, params: list | None = None) -> list[dict]:
    return rows_to_dicts(con.execute(sql, params or []))


def fetch_one(con, sql: str, params: list | None = None) -> dict | None:
    rows = fetch_all(con, sql, params)
    return rows[0] if rows else None


def count_where(con, where: str, params: list) -> int:
    return con.execute(f"SELECT count(*) FROM clients WHERE {where}", params).fetchone()[0]


def fetch_page(con, select_sql: str, where: str, params: list,
               order_by: str, limit: int, offset: int) -> tuple[list[dict], int]:
    """Página + total. Duas queries; sobre Parquet com filtro, o count é
    milissegundos e a clareza vale mais que economizar uma passada."""
    total = count_where(con, where, params)
    cur = con.execute(
        f"SELECT {select_sql} FROM clients WHERE {where} "
        f"ORDER BY {order_by} LIMIT ? OFFSET ?",
        [*params, limit, offset],
    )
    return rows_to_dicts(cur), total
