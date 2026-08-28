"""
stats.py — A análise exploratória servida por SQL.

Todos os endpoints aceitam os MESMOS filtros de /clients. É isso que permite
responder "taxa de inadimplência por escolaridade, só entre thin-file com
menos de 25 anos" numa única chamada, ao vivo.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from MLOps.app import artifacts, db, settings
from MLOps.app.schemas import (
    ClientFilters,
    CrosstabResponse,
    DefaultRateResponse,
    DistributionResponse,
    MissingResponse,
    OverviewResponse,
)

router = APIRouter(prefix="/stats", tags=["Estatísticas"])


def _aplicados(f: ClientFilters) -> dict:
    return {k: v for k, v in f.model_dump().items() if v is not None}


@router.get("/overview", response_model=OverviewResponse,
            summary="KPIs da carteira (respeita os filtros)")
def overview(request: Request,
             filters: ClientFilters = Depends(),
             con=Depends(db.get_db)):
    thr = artifacts.threshold(request.app.state)
    where, params = db.where_from_filters(filters, thr)
    row = db.fetch_one(con, f"""
        SELECT
            count(*)                                              AS n_clients,
            sum(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END)           AS n_defaults,
            avg(TARGET)                                           AS default_rate,
            avg(CASE WHEN BUREAU_COUNT IS NULL THEN 1.0 ELSE 0 END) AS thin_file_rate,
            avg(AGE_YEARS)                                        AS avg_age,
            median(AMT_INCOME_TOTAL)                              AS median_income,
            avg(AMT_CREDIT)                                       AS avg_credit,
            avg(CREDIT_INCOME_RATIO)                              AS avg_cir,
            avg(CASE WHEN EXT_SOURCE_1 IS NULL THEN 1.0 ELSE 0 END) AS miss_ext1,
            avg(CASE WHEN EXT_SOURCE_3 IS NULL THEN 1.0 ELSE 0 END) AS miss_ext3,
            count({settings.SCORE_COL})                           AS scored,
            avg({settings.SCORE_COL})                             AS avg_score,
            avg(CASE WHEN {settings.SCORE_COL} < ? THEN 1.0
                     WHEN {settings.SCORE_COL} IS NULL THEN NULL ELSE 0 END) AS approval_rate
        FROM clients WHERE {where}
    """, [thr, *params])

    return {
        "n_clients": row["n_clients"], "n_defaults": row["n_defaults"] or 0,
        "default_rate": row["default_rate"] or 0.0,
        "thin_file_rate": row["thin_file_rate"] or 0.0,
        "avg_age": row["avg_age"], "median_income": row["median_income"],
        "avg_credit": row["avg_credit"], "avg_credit_income_ratio": row["avg_cir"],
        "missing_ext_source_1_rate": row["miss_ext1"] or 0.0,
        "missing_ext_source_3_rate": row["miss_ext3"] or 0.0,
        "scored": row["scored"] or 0, "avg_score": row["avg_score"],
        "approval_rate": row["approval_rate"], "threshold": thr,
        "filters_applied": _aplicados(filters),
    }


@router.get("/default-rate", response_model=DefaultRateResponse,
            summary="Taxa de inadimplência por segmento",
            description="Reproduz ao vivo os cortes da EDA. Ex.: `?by=education` devolve "
                        "de 1,83% (doutorado) a 10,93% (fundamental incompleto).")
def default_rate(request: Request,
                     by: str = Query(..., description="Dimensão — ver /meta/dimensions"),
                 min_count: int = Query(30, ge=1, description="Ignora grupos menores"),
                 order_by: Literal["default_rate", "n", "value"] = "default_rate",
                 limit: int = Query(50, ge=1, le=200),
             filters: ClientFilters = Depends(),
             con=Depends(db.get_db)):
    expr = db.dimension_expr(by)
    thr = artifacts.threshold(request.app.state)
    where, params = db.where_from_filters(filters, thr)

    geral = con.execute(f"SELECT avg(TARGET), count(*) FROM clients WHERE {where}",
                        params).fetchone()
    ordem = {"default_rate": "default_rate DESC NULLS LAST",
             "n": "n DESC", "value": "value ASC"}[order_by]

    buckets = db.fetch_all(con, f"""
        SELECT CAST({expr} AS VARCHAR) AS value,
               count(*)                AS n,
               sum(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) AS defaults,
               avg(TARGET)             AS default_rate,
               avg({settings.SCORE_COL}) AS avg_score,
               avg(CASE WHEN {settings.SCORE_COL} < ? THEN 1.0
                        WHEN {settings.SCORE_COL} IS NULL THEN NULL ELSE 0 END) AS approval_rate
        FROM clients WHERE {where}
        GROUP BY 1 HAVING count(*) >= ?
        ORDER BY {ordem} LIMIT ?
    """, [thr, *params, min_count, limit])

    base = geral[0]
    for b in buckets:
        b["lift"] = (b["default_rate"] / base) if base and b["default_rate"] is not None else None

    return {"dimension": by, "label": settings.DIMENSION_LABELS.get(by, by),
            "expression": expr, "overall_default_rate": base,
            "n_total": geral[1], "buckets": buckets}


@router.get("/distribution", response_model=DistributionResponse,
            summary="Distribuição de uma variável (histograma ou contagem)")
def distribution(request: Request,
                     feature: str = Query(..., description="Coluna — ver /meta/columns"),
                 bins: int = Query(20, ge=2, le=100),
                 by_target: bool = Query(False, description="Inclui taxa de default por faixa"),
                 filters: ClientFilters = Depends(),
                 con=Depends(db.get_db),
                 columns=Depends(db.get_columns)):
    col = db.quote_ident(feature, columns)
    dtype = columns[feature].upper()
    numerica = any(k in dtype for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "BIGINT", "HUGEINT"))

    thr = artifacts.threshold(request.app.state)
    where, params = db.where_from_filters(filters, thr)
    n_total = db.count_where(con, where, params)

    if not numerica:
        cats = db.fetch_all(con, f"""
            SELECT CAST({col} AS VARCHAR) AS value, count(*) AS n,
                   avg(TARGET) AS default_rate,
                   avg({settings.SCORE_COL}) AS avg_score
            FROM clients WHERE {where}
            GROUP BY 1 ORDER BY n DESC LIMIT 100
        """, params)
        for c in cats:
            c["defaults"] = None
            c["lift"] = None
            c["approval_rate"] = None
        return {"feature": feature, "type": "categorical", "n_total": n_total,
                "stats": None, "bins": None, "categories": cats}

    s = db.fetch_one(con, f"""
        SELECT count({col}) AS count, avg({col}) AS mean, stddev_samp({col}) AS std,
               min({col}) AS min, max({col}) AS max,
               quantile_cont({col}, 0.25) AS p25,
               quantile_cont({col}, 0.50) AS p50,
               quantile_cont({col}, 0.75) AS p75
        FROM clients WHERE {where}
    """, params)
    stats = {**s, "missing": n_total - (s["count"] or 0),
             "missing_rate": (n_total - (s["count"] or 0)) / n_total if n_total else 0.0}

    if not s["count"] or s["min"] is None or s["min"] == s["max"]:
        return {"feature": feature, "type": "numeric", "n_total": n_total,
                "stats": stats, "bins": [], "categories": None}

    lo, hi = float(s["min"]), float(s["max"])
    largura = (hi - lo) / bins
    # Bucket por aritmética: o DuckDB 1.5 não tem width_bucket. LEAST() prende
    # o valor máximo no último bin em vez de criar um bin extra só para ele.
    rows = db.fetch_all(con, f"""
        SELECT LEAST(CAST(floor(({col} - ?) / ?) AS INTEGER), ?) AS b,
               count(*) AS n, avg(TARGET) AS default_rate
        FROM clients WHERE {where} AND {col} IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """, [*params, lo, largura, bins - 1])

    total_validos = sum(r["n"] for r in rows) or 1
    out = []
    for r in rows:
        i = max(int(r["b"]), 0)
        out.append({"lower": lo + i * largura, "upper": lo + (i + 1) * largura,
                    "count": r["n"], "pct": r["n"] / total_validos,
                    "default_rate": r["default_rate"] if by_target else None})
    return {"feature": feature, "type": "numeric", "n_total": n_total,
            "stats": stats, "bins": out, "categories": None}


@router.get("/missing", response_model=MissingResponse,
            summary="Nulos por coluna",
            description="Servido do perfil pré-computado (artifacts/abt_profile.json): "
                        "responder isto varrendo 1.000 colunas por request seria inviável.")
def missing(request: Request,
            top: int = Query(30, ge=1, le=500),
            prefix: str | None = Query(None, description="Ex.: BUREAU_, EXT_, PREV_"),
            min_rate: float = Query(0.0, ge=0, le=1),
            con=Depends(db.get_db)):
    prof = artifacts.require(request.app.state, "profile")
    itens = []
    for c in prof["columns"]:
        nome = c.get("column_name")
        if prefix and not nome.startswith(prefix):
            continue
        rate = float(c.get("null_percentage") or 0) / 100.0
        if rate < min_rate:
            continue
        itens.append({"column": nome, "dtype": c.get("column_type"),
                      "missing_rate": rate,
                      "n_unique": c.get("approx_unique")})
    itens.sort(key=lambda x: -x["missing_rate"])
    return {"n_rows": prof["n_rows"], "n_columns": prof["n_columns"],
            "columns_with_missing": sum(1 for i in itens if i["missing_rate"] > 0),
            "items": itens[:top]}


@router.get("/crosstab", response_model=CrosstabResponse,
            summary="Tabela cruzada entre duas dimensões")
def crosstab(request: Request,
             rows: str = Query(..., description="Dimensão nas linhas"),
             cols: str = Query(..., description="Dimensão nas colunas"),
             metric: Literal["count", "default_rate"] = "default_rate",
             min_count: int = Query(30, ge=1),
             filters: ClientFilters = Depends(),
             con=Depends(db.get_db)):
    if rows == cols:
        raise HTTPException(400, detail="'rows' e 'cols' precisam ser dimensões diferentes.")
    er, ec = db.dimension_expr(rows), db.dimension_expr(cols)
    thr = artifacts.threshold(request.app.state)
    where, params = db.where_from_filters(filters, thr)

    data = db.fetch_all(con, f"""
        SELECT CAST({er} AS VARCHAR) AS row_value, CAST({ec} AS VARCHAR) AS col_value,
               count(*) AS n, avg(TARGET) AS default_rate
        FROM clients WHERE {where}
        GROUP BY 1, 2 HAVING count(*) >= ?
        ORDER BY 1, 2
    """, [*params, min_count])

    for d in data:
        d["value"] = d["n"] if metric == "count" else d["default_rate"]
    return {"rows_dimension": rows, "cols_dimension": cols, "metric": metric,
            "n_total": sum(d["n"] for d in data),
            "row_values": sorted({d["row_value"] for d in data if d["row_value"]}),
            "col_values": sorted({d["col_value"] for d in data if d["col_value"]}),
            "cells": data}
