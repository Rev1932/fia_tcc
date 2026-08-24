"""
api.py — Serviço de credit scoring (FastAPI).

Execução local:
    uvicorn MLOps.app.api:app --host 0.0.0.0 --port 8000
Documentação interativa: http://localhost:8000/docs
Referência escrita:      MLOps/app/README.md

Decisão de projeto: a API NUNCA cai por falta de artefato. Ela sobe degradada,
/health devolve 503 dizendo exatamente o que falta, e cada endpoint que
depende do recurso ausente responde com a instrução do comando a rodar.
Container em crash-loop no meio de uma apresentação é o pior cenário possível.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from MLOps.app import artifacts, db, settings
from MLOps.app.routers import clients, model, scoring, stats
from MLOps.app.schemas import ColumnsResponse, DimensionInfo, HealthResponse

DESCRIPTION = """
API de **credit scoring** do TCC (Home Credit Default Risk · FIA/LABDATA).

Entrega os dados da análise por meio de endpoints especializados e filtráveis,
em vez de um único método que devolve tudo.

**Arquitetura**: DuckDB consultando Parquet direto do disco (a ABT tem 307.511
clientes), modelo LightGBM em bundle joblib, e os artefatos congelados da
rodada canônica em `artifacts/`.

**Fonte única de verdade**: nenhum número exibido em slide, documento ou
notebook pode divergir de `GET /model/metrics`.

**Famílias de endpoints**
| Família | Para quê |
|---|---|
| `/clients` | consultar a carteira com filtros, ordenação e paginação |
| `/stats` | a análise exploratória servida por SQL |
| `/model` | métricas congeladas e recálculos ao vivo (custo, threshold, fairness) |
| `/predict`, `/simulate`, `/explain` | pontuar, simular cenários e explicar decisões |
"""

TAGS = [
    {"name": "Saúde", "description": "Status do serviço, do modelo e dos dados."},
    {"name": "Metadados", "description": "Colunas e dimensões disponíveis para filtrar e agrupar."},
    {"name": "Clientes", "description": "Consulta da carteira: filtros, ficha e score."},
    {"name": "Estatísticas", "description": "EDA por SQL: KPIs, inadimplência por segmento, distribuições, nulos."},
    {"name": "Modelo", "description": "Métricas da rodada congelada e recálculos ao vivo."},
    {"name": "Score e Explicabilidade", "description": "Predição, simulação what-if e SHAP por cliente."},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = time.time()
    app.state.errors = {}

    try:
        app.state.db = db.connect()
        app.state.columns = db.describe_columns(app.state.db)
    except Exception as e:
        app.state.db, app.state.columns = None, {}
        app.state.errors["data"] = f"{type(e).__name__}: {e}"

    try:
        from Model.predict import load_bundle
        app.state.bundle = load_bundle(str(settings.MODEL_PATH))
    except Exception as e:
        app.state.bundle = None
        app.state.errors["model"] = f"{type(e).__name__}: {e}"

    try:
        app.state.artifacts = artifacts.load_all()
        if not app.state.artifacts:
            app.state.errors["artifacts"] = "nenhum artefato encontrado em artifacts/"
    except Exception as e:
        app.state.artifacts = {}
        app.state.errors["artifacts"] = f"{type(e).__name__}: {e}"

    # Constrói o TreeExplainer agora: evita 1-2s de espera na primeira
    # explicação, que cairia bem no meio de uma demonstração.
    app.state.explainer_loaded = False
    if settings.EXPLAINER_EAGER and app.state.bundle is not None:
        try:
            from MLOps.app.explain import get_explainer
            get_explainer(str(settings.MODEL_PATH))
            app.state.explainer_loaded = True
        except Exception as e:
            app.state.errors["explainer"] = f"{type(e).__name__}: {e}"

    yield

    if getattr(app.state, "db", None) is not None:
        app.state.db.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Home Credit — Credit Scoring API",
        version="2.0.0",
        description=DESCRIPTION,
        openapi_tags=TAGS,
        lifespan=lifespan,
    )

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        codes = {400: "BAD_REQUEST", 404: "NOT_FOUND", 422: "VALIDATION_ERROR",
                 503: "SERVICE_UNAVAILABLE"}
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": codes.get(exc.status_code, "ERROR"),
                               "message": str(exc.detail), "detail": None}},
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "VALIDATION_ERROR",
                               "message": "Parâmetros inválidos.",
                               "detail": str(exc.errors())}},
        )

    @app.get("/", include_in_schema=False)
    def raiz():
        return RedirectResponse("/docs")

    @app.get("/health", response_model=HealthResponse, tags=["Saúde"],
             summary="Status do serviço",
             description="Devolve **503** quando o modelo ou os dados não carregaram. "
                         "É o que faz o healthcheck do docker-compose ter significado: "
                         "antes, respondia 200 mesmo sem modelo.")
    def health(request: Request):
        st = request.app.state
        m = (st.artifacts or {}).get("metrics") or {}
        run = m.get("run") or {}

        n_clients = None
        if st.db is not None:
            try:
                n_clients = st.db.execute("SELECT count(*) FROM clients").fetchone()[0]
            except Exception as e:
                st.errors["query"] = str(e)

        ok = st.bundle is not None and st.db is not None
        corpo = {
            "status": "ok" if ok else "degraded",
            "model_loaded": st.bundle is not None,
            "data_loaded": st.db is not None,
            "artifacts_loaded": bool(st.artifacts),
            "explainer_loaded": bool(getattr(st, "explainer_loaded", False)),
            "run_id": run.get("run_id"),
            "trained_at": run.get("trained_at"),
            "threshold": st.bundle["threshold"] if st.bundle else None,
            "n_features": run.get("n_features"),
            "n_clients": n_clients,
            "uptime_seconds": round(time.time() - st.started_at, 1),
            "errors": st.errors,
        }
        return JSONResponse(status_code=200 if ok else 503, content=corpo)

    @app.get("/meta/columns", response_model=ColumnsResponse, tags=["Metadados"],
             summary="Colunas disponíveis para fields, sort e feature")
    def meta_columns(request: Request, search: str | None = None,
                     prefix: str | None = None, limit: int = 200):
        cols = getattr(request.app.state, "columns", {}) or {}
        if not cols:
            raise HTTPException(503, detail="Esquema não carregado.")
        perfil = {c.get("column_name"): c
                  for c in ((request.app.state.artifacts or {})
                            .get("profile") or {}).get("columns", [])}
        itens = []
        for nome, tipo in cols.items():
            if search and search.lower() not in nome.lower():
                continue
            if prefix and not nome.startswith(prefix):
                continue
            p = perfil.get(nome, {})
            itens.append({"name": nome, "type": tipo,
                          "missing_rate": (float(p["null_percentage"]) / 100.0
                                           if p.get("null_percentage") is not None else None),
                          "n_unique": p.get("approx_unique")})
        return {"n_columns": len(cols), "returned": len(itens[:limit]),
                "columns": itens[:limit]}

    @app.get("/meta/dimensions", response_model=list[DimensionInfo], tags=["Metadados"],
             summary="Dimensões aceitas em ?by=")
    def meta_dimensions():
        return [{"key": k, "label": settings.DIMENSION_LABELS.get(k, k), "expression": v}
                for k, v in settings.DIMENSIONS.items()]

    @app.post("/admin/reload", tags=["Metadados"],
              summary="Recarrega modelo e artefatos sem reiniciar",
              description="Útil depois de um re-treino feito ao vivo.")
    def reload(request: Request):
        st = request.app.state
        st.errors = {}
        try:
            from Model.predict import load_bundle
            load_bundle.cache_clear()
            st.bundle = load_bundle(str(settings.MODEL_PATH))
        except Exception as e:
            st.bundle = None
            st.errors["model"] = str(e)
        try:
            if st.db is not None:
                st.db.close()
            st.db = db.connect()
            st.columns = db.describe_columns(st.db)
        except Exception as e:
            st.db = None
            st.errors["data"] = str(e)
        artifacts.clear_caches()
        from MLOps.app.explain import clear_cache
        clear_cache()
        st.artifacts = artifacts.load_all()
        return {"reloaded": True, "run_id": artifacts.run_id(st), "errors": st.errors}

    app.include_router(clients.router)
    app.include_router(stats.router)
    app.include_router(model.router)
    app.include_router(scoring.router)
    return app


app = create_app()
