"""
clients.py — Consulta da carteira.

Substitui o "método único" da API anterior: em vez de um endpoint que
devolveria tudo, três endpoints especializados com filtros, paginação,
ordenação e seleção de colunas.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from MLOps.app import artifacts, db, policy, settings
from MLOps.app.explain import score_band
from MLOps.app.schemas import (
    ClientDetail,
    ClientFilters,
    ClientSummary,
    Page,
    ScoreResponse,
    make_page,
)

router = APIRouter(tags=["Clientes"])


def _decision(p: float | None, thr: float, band: float) -> str | None:
    return None if p is None else policy.decide(float(p), thr, band)


@router.get(
    "/clients",
    response_model=Page[ClientSummary],
    # Preserva os nulos reais vindos do banco (a chave existe no dict, logo
    # está "set") e omite os campos que o ?fields= não pediu.
    response_model_exclude_unset=True,
    summary="Lista clientes com filtros, ordenação e paginação",
    description=(
        "Consulta a carteira (307.511 clientes). Todos os filtros são combináveis; "
        "os mesmos filtros valem em `/stats/*`.\n\n"
        "Exemplos:\n"
        "- `/clients?age_min=20&age_max=25&thin_file=true` — jovens sem histórico de bureau\n"
        "- `/clients?decision=NEGAR&education=Lower secondary&page_size=10`\n"
        "- `/clients?fields=SK_ID_CURR,AMT_CREDIT,proba_champion&sort=proba_champion&order=desc`"
    ),
)
def list_clients(
    request: Request,
    filters: ClientFilters = Depends(),
    page: int = Query(1, ge=1),
    page_size: int = Query(settings.DEFAULT_PAGE_SIZE, ge=1, le=settings.MAX_PAGE_SIZE),
    sort: str = Query(settings.ID_COL, description="Coluna de ordenação (ver /meta/columns)"),
    order: Literal["asc", "desc"] = "asc",
    fields: str | None = Query(None, description="Colunas separadas por vírgula"),
    con=Depends(db.get_db),
    columns=Depends(db.get_columns),
):
    thr = artifacts.threshold(request.app.state)
    names = db.parse_fields(fields, columns)
    select_sql = ", ".join(db.quote_ident(n, columns) for n in names)
    order_by = f"{db.quote_ident(sort, columns)} {order.upper()} NULLS LAST"

    where, params = db.where_from_filters(filters, thr)
    rows, total = db.fetch_page(con, select_sql, where, params, order_by,
                                page_size, (page - 1) * page_size)

    fairness = (request.app.state.artifacts or {}).get("fairness") or {}
    for r in rows:
        if settings.SCORE_COL in r:
            band = policy.band_for(r, fairness) if fairness else 0.0
            r["decision"] = _decision(r.get(settings.SCORE_COL), thr, band)
    return make_page(rows, total, page, page_size)


@router.get(
    "/clients/{sk_id_curr}",
    response_model=ClientDetail,
    summary="Ficha completa de um cliente",
)
def get_client(
    request: Request,
    sk_id_curr: int,
    include: Literal["core", "all"] = Query(
        "core", description="'all' inclui as 471+ features do modelo"),
    con=Depends(db.get_db),
    columns=Depends(db.get_columns),
):
    row = db.fetch_one(con, f"SELECT * FROM clients WHERE {settings.ID_COL} = ?",
                       [sk_id_curr])
    if row is None:
        raise HTTPException(404, detail=f"Cliente {sk_id_curr} não encontrado.")

    thr = artifacts.threshold(request.app.state)
    fairness = (request.app.state.artifacts or {}).get("fairness") or {}
    band = policy.band_for(row, fairness) if fairness else 0.0

    def bloco(nome: str) -> dict:
        return {c: row.get(c) for c in settings.CLIENT_DETAIL_BLOCKS[nome] if c in row}

    p = row.get(settings.SCORE_COL)
    score = None
    if p is not None:
        score = {
            "sk_id_curr": sk_id_curr,
            "probability_default": float(p),
            "threshold": thr,
            "decision": _decision(p, thr, band),
            "score_band": score_band(float(p)),
            "source": "batch",
            "target": row.get(settings.TARGET_COL),
            "baseline_probability": row.get("proba_baseline"),
        }

    ext = bloco("scores_externos")
    ext["n_disponiveis"] = sum(1 for v in ext.values() if v is not None)

    reservado = {settings.ID_COL, settings.TARGET_COL, "split", "y_true",
                 "proba_champion", "proba_baseline"}
    return {
        "sk_id_curr": sk_id_curr,
        "thin_file": row.get("BUREAU_COUNT") is None,
        "identificacao": bloco("identificacao"),
        "financeiro": bloco("financeiro"),
        "historico": bloco("historico"),
        "scores_externos": ext,
        "score": score,
        "features": ({k: v for k, v in row.items() if k not in reservado}
                     if include == "all" else None),
    }


@router.get(
    "/clients/{sk_id_curr}/score",
    response_model=ScoreResponse,
    summary="Score e decisão de um cliente",
    description=(
        "Devolve o score da rodada congelada. Com `recompute=true`, recalcula pelo "
        "modelo carregado em memória e compara — prova ao vivo que o artefato em "
        "disco e o modelo servido concordam."
    ),
)
def get_score(
    request: Request,
    sk_id_curr: int,
    threshold: float | None = Query(None, ge=0, le=1),
    recompute: bool = Query(False, description="Recalcula pelo modelo em memória"),
    con=Depends(db.get_db),
):
    row = db.fetch_one(con, f"SELECT * FROM clients WHERE {settings.ID_COL} = ?",
                       [sk_id_curr])
    if row is None:
        raise HTTPException(404, detail=f"Cliente {sk_id_curr} não encontrado.")

    thr = threshold if threshold is not None else artifacts.threshold(request.app.state)
    p = row.get(settings.SCORE_COL)
    fairness = (request.app.state.artifacts or {}).get("fairness") or {}
    band = policy.band_for(row, fairness) if fairness else 0.0

    live = err = None
    if recompute or p is None:
        bundle = request.app.state.bundle
        if bundle is None:
            raise HTTPException(503, detail="Modelo não carregado.")
        from Model.predict import predict
        feats = {k: v for k, v in row.items()
                 if k in set(bundle["feature_names"])}
        # SEMPRE com o caminho do settings: sem isto, o predict cairia no
        # modelo default e pontuaria com um artefato diferente do servido.
        live = float(predict([feats], str(settings.MODEL_PATH))[0]["probability_default"])
        if p is None:
            p = live
        else:
            err = abs(live - float(p))

    percentil = con.execute(
        f"SELECT avg(CASE WHEN {settings.SCORE_COL} <= ? THEN 1.0 ELSE 0.0 END) "
        f"FROM clients WHERE {settings.SCORE_COL} IS NOT NULL", [p]).fetchone()[0]

    return {
        "sk_id_curr": sk_id_curr,
        "probability_default": float(p),
        "threshold": thr,
        "decision": _decision(p, thr, band),
        "score_band": score_band(float(p)),
        "percentile": float(percentil) if percentil is not None else None,
        "source": "live" if (live is not None and row.get(settings.SCORE_COL) is None)
                  else "batch",
        "target": row.get(settings.TARGET_COL),
        "baseline_probability": row.get("proba_baseline"),
        "live_probability": live,
        "agreement_error": err,
    }
