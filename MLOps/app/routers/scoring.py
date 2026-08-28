"""
scoring.py — Predição, simulação what-if e explicabilidade.

/simulate é o endpoint que responde "e se a renda dele fosse o dobro?" numa
chamada — o tipo de pergunta que, sem isto, exige abrir um notebook.
"""
from __future__ import annotations

import numpy as np
import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from MLOps.app import artifacts, db, explain as ex, policy, settings
from Model.derived import apply_changes, recompute
from MLOps.app.schemas import (
    ExplainResponse,
    PredictRequest,
    PredictResponse,
    SimulateRequest,
    SimulateResponse,
)

router = APIRouter(tags=["Score e Explicabilidade"])


def _bundle(request: Request) -> dict:
    b = getattr(request.app.state, "bundle", None)
    if b is None:
        raise HTTPException(503, detail="Modelo não carregado. Rode `python Model/train.py`.")
    return b


def _band(request: Request, client: dict | None = None) -> float:
    f = (request.app.state.artifacts or {}).get("fairness") or {}
    if not f:
        return 0.0
    return policy.band_for(client or {}, f)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Pontua um lote de clientes",
    description=(
        "Aceita features parciais: o que faltar vira nulo, que o LightGBM trata "
        "nativamente. A resposta informa `coverage` e `unknown_features` — antes, "
        "campos desconhecidos eram descartados em silêncio."
    ),
)
def predict_endpoint(request: Request, req: PredictRequest):
    from Model.predict import predict

    bundle = _bundle(request)
    thr = req.threshold if req.threshold is not None else artifacts.threshold(request.app.state)
    esperadas = set(bundle["feature_names"])

    # Recalcula as derivadas a partir do que veio no payload: sem isto, quem
    # manda AMT_CREDIT e AMT_INCOME_TOTAL mas não a razão entre eles perde uma
    # das variáveis mais informativas do modelo.
    registros = [{**r, **recompute(r, only_if_present=False)} for r in req.records]
    resultados = predict(registros, str(settings.MODEL_PATH))
    saida = []
    for i, (rec, res) in enumerate(zip(registros, resultados)):
        informadas = [k for k in req.records[i] if k in esperadas
                      and req.records[i][k] is not None]
        p = float(res["probability_default"])
        item = {
            "index": i,
            "probability_default": p,
            "threshold": thr,
            "decision": policy.decide(p, thr, _band(request, rec)),
            "score_band": ex.score_band(p),
            "features_informed": len(informadas),
            "features_expected": len(esperadas),
            "coverage": len(informadas) / len(esperadas) if esperadas else 0.0,
            "unknown_features": sorted(k for k in req.records[i] if k not in esperadas),
        }
        if req.explain:
            r = ex.explain_record(rec, bundle, str(settings.MODEL_PATH), req.explain_top)
            item["contributions"] = (r["top_risk_drivers"] + r["top_protective_factors"])
        saida.append(item)

    return {"run_id": artifacts.run_id(request.app.state), "threshold": thr,
            "n_records": len(saida), "predictions": saida}


@router.get(
    "/clients/{sk_id_curr}/explain",
    response_model=ExplainResponse,
    summary="Por que este cliente recebeu este score (SHAP)",
    description=(
        "Contribuição de cada variável em log-odds, separada em fatores de risco e "
        "fatores favoráveis. `consistency_check` mostra que "
        "`base_value + Σ shap` reconstrói exatamente a probabilidade do modelo — "
        "é a prova de que a explicação é fiel, e não uma aproximação."
    ),
)
def explain_client(request: Request, sk_id_curr: int,
                   top: int = Query(10, ge=1, le=settings.MAX_EXPLAIN_TOP),
                   threshold: float | None = Query(None, ge=0, le=1),
                   con=Depends(db.get_db)):
    bundle = _bundle(request)
    row = db.fetch_one(con, f"SELECT * FROM clients WHERE {settings.ID_COL} = ?",
                       [sk_id_curr])
    if row is None:
        raise HTTPException(404, detail=f"Cliente {sk_id_curr} não encontrado.")

    thr = threshold if threshold is not None else artifacts.threshold(request.app.state)
    feats = {k: v for k, v in row.items() if k in set(bundle["feature_names"])}
    r = ex.explain_record(feats, bundle, str(settings.MODEL_PATH), top)
    decisao = policy.decide(r["probability_default"], thr, _band(request, row))

    return {"sk_id_curr": sk_id_curr, "threshold": thr, "decision": decisao,
            "narrative": ex.narrate(r, thr, decisao, sk_id_curr), **r}


@router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="What-if: muda uma variável e vê o score mudar",
    description=(
        "Parte de um cliente real (`sk_id_curr`) ou de um payload livre (`record`), "
        "aplica `changes` e devolve antes/depois. Com `sweep`, varre uma variável "
        "inteira e mostra onde a decisão vira."
    ),
)
def simulate(request: Request, req: SimulateRequest, con=Depends(db.get_db)):
    from Model.predict import predict

    bundle = _bundle(request)
    esperadas = set(bundle["feature_names"])
    thr = req.threshold if req.threshold is not None else artifacts.threshold(request.app.state)

    if req.sk_id_curr is not None:
        row = db.fetch_one(con, f"SELECT * FROM clients WHERE {settings.ID_COL} = ?",
                           [req.sk_id_curr])
        if row is None:
            raise HTTPException(404, detail=f"Cliente {req.sk_id_curr} não encontrado.")
        base_rec = {k: v for k, v in row.items() if k in esperadas}
    else:
        row = req.record or {}
        base_rec = {k: v for k, v in row.items() if k in esperadas}

    band = _band(request, row)

    def pontuar(rec: dict) -> dict:
        p = float(predict([rec], str(settings.MODEL_PATH))[0]["probability_default"])
        return {"probability_default": p,
                "decision": policy.decide(p, thr, band),
                "score_band": ex.score_band(p)}

    base = pontuar(base_rec)

    aplicadas = {k: v for k, v in req.changes.items() if k in esperadas}
    ignoradas = sorted(k for k in req.changes if k not in esperadas)

    simulado = delta = None
    drivers = None
    if aplicadas:
        # apply_changes propaga: mudar AMT_CREDIT move CREDIT_INCOME_RATIO e
        # CREDIT_TERM, como aconteceria numa proposta de verdade.
        novo = apply_changes(base_rec, aplicadas)
        simulado = pontuar(novo)
        delta = simulado["probability_default"] - base["probability_default"]
        r_base = ex.explain_record(base_rec, bundle, str(settings.MODEL_PATH), 50)
        r_novo = ex.explain_record(novo, bundle, str(settings.MODEL_PATH), 50)
        antes = {c["feature"]: c["shap_value"]
                 for c in r_base["top_risk_drivers"] + r_base["top_protective_factors"]}
        depois = {c["feature"]: c["shap_value"]
                  for c in r_novo["top_risk_drivers"] + r_novo["top_protective_factors"]}
        difs = [{"feature": f, "value": novo.get(f),
                 "shap_value": depois.get(f, 0.0) - antes.get(f, 0.0),
                 "abs_shap": abs(depois.get(f, 0.0) - antes.get(f, 0.0)),
                 "effect": ("aumenta risco"
                            if depois.get(f, 0.0) - antes.get(f, 0.0) > 0 else "reduz risco"),
                 "source_table": ex.source_table(f)}
                for f in set(antes) | set(depois)]
        difs = [d for d in difs if d["abs_shap"] > 1e-9]
        difs.sort(key=lambda d: -d["abs_shap"])
        drivers = [{**d, "rank": i} for i, d in enumerate(difs[:req.explain_top], 1)]

    varredura = None
    if req.sweep:
        feat = req.sweep.get("feature")
        if feat not in esperadas:
            raise HTTPException(400, detail=f"'{feat}' não é uma feature do modelo.")
        valores = req.sweep.get("values")
        if valores is None:
            try:
                valores = np.linspace(float(req.sweep["start"]), float(req.sweep["stop"]),
                                      int(req.sweep.get("steps", 11))).tolist()
            except KeyError as e:
                raise HTTPException(400, detail=f"sweep precisa de 'values' ou de "
                                                f"'start'/'stop'/'steps' (faltou {e}).")
        if len(valores) > 100:
            raise HTTPException(400, detail="sweep limitado a 100 pontos.")
        varredura = []
        for v in valores:
            s = pontuar(apply_changes(base_rec, {**aplicadas, feat: v}))
            varredura.append({"value": v, "probability_default": s["probability_default"],
                              "decision": s["decision"]})

    return {"base": base, "simulated": simulado, "delta_probability": delta,
            "decision_changed": bool(simulado and simulado["decision"] != base["decision"]),
            "threshold": thr, "changes_applied": aplicadas, "ignored_changes": ignoradas,
            "sweep": varredura, "top_drivers": drivers}


MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_CSV_LINHAS = 50_000


@router.post(
    "/predict/csv",
    summary="Pontua um lote enviado em CSV e devolve CSV",
    description=(
        "Recebe um arquivo CSV com uma linha por cliente e as colunas nomeadas como "
        "as features do modelo (as que faltarem viram nulo, que o LightGBM trata "
        "nativamente). Devolve o mesmo CSV acrescido de `probability_default`, "
        "`decision` e `score_band`.\n\n"
        "É o formato que uma mesa de crédito usa de fato: exporta a fila do dia, "
        "pontua, reimporta. Limite de 50 mil linhas e 20 MB por envio.\n\n"
        "```bash\n"
        "curl -X POST localhost:8000/predict/csv -F 'arquivo=@fila.csv' -o pontuado.csv\n"
        "```"
    ),
    response_class=StreamingResponse,
)
async def predict_csv(request: Request,
                      arquivo: UploadFile = File(..., description="CSV com uma linha por cliente"),
                      threshold: float | None = Query(None, ge=0, le=1)):
    from Model.predict import predict

    bundle = _bundle(request)
    thr = threshold if threshold is not None else artifacts.threshold(request.app.state)
    esperadas = set(bundle["feature_names"])

    bruto = await arquivo.read()
    if len(bruto) > MAX_CSV_BYTES:
        raise HTTPException(413, detail=f"Arquivo acima de {MAX_CSV_BYTES // 1024 // 1024} MB.")
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, detail="O arquivo precisa estar em UTF-8.")

    leitor = csv.DictReader(io.StringIO(texto))
    if not leitor.fieldnames:
        raise HTTPException(400, detail="CSV sem cabeçalho.")
    conhecidas = [c for c in leitor.fieldnames if c in esperadas]
    if not conhecidas:
        raise HTTPException(
            400,
            detail=("Nenhuma coluna do CSV corresponde a uma feature do modelo. "
                    "Veja as esperadas em GET /meta/columns."))

    linhas = list(leitor)
    if not linhas:
        raise HTTPException(400, detail="CSV sem linhas de dados.")
    if len(linhas) > MAX_CSV_LINHAS:
        raise HTTPException(413, detail=f"Máximo de {MAX_CSV_LINHAS:,} linhas por envio."
                                        .replace(",", "."))

    registros = []
    for linha in linhas:
        rec = {k: (v if v not in ("", None) else None) for k, v in linha.items()
               if k in esperadas}
        registros.append({**rec, **recompute(rec, only_if_present=False)})

    resultados = predict(registros, str(settings.MODEL_PATH))

    saida = io.StringIO()
    campos = list(leitor.fieldnames) + ["probability_default", "decision", "score_band"]
    escritor = csv.DictWriter(saida, fieldnames=campos, extrasaction="ignore")
    escritor.writeheader()
    for original, res in zip(linhas, resultados):
        p = float(res["probability_default"])
        escritor.writerow({**original,
                           "probability_default": f"{p:.6f}",
                           "decision": policy.decide(p, thr, _band(request, original)),
                           "score_band": ex.score_band(p)})
    saida.seek(0)

    nome = (arquivo.filename or "lote").rsplit(".", 1)[0]
    return StreamingResponse(
        iter([saida.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{nome}_pontuado.csv"',
            "X-Linhas-Pontuadas": str(len(linhas)),
            "X-Features-Reconhecidas": f"{len(conhecidas)}/{len(esperadas)}",
            "X-Threshold": f"{thr:.4f}",
        },
    )
