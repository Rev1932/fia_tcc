"""
model.py — Métricas congeladas e recálculos ao vivo.

Duas naturezas de endpoint aqui:
  - os que SERVEM a rodada canônica (/metrics, /roc, /ks, /feature-importance,
    /fairness, /improvements) — leem artifacts/ e nada mais;
  - os que RECALCULAM sobre artifacts/scores.parquet (/threshold-analysis,
    /confusion-matrix) — respondem "e se o custo fosse outro?" em
    milissegundos, sem re-treinar nada.

O recálculo usa Model/metrics_lib, o MESMO módulo que gerou os números do
treino. Por isso a frase "é o mesmo código que produziu o número do slide"
é literal, e não força de expressão.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from MLOps.app import artifacts, db, policy, settings
from MLOps.app.schemas import (
    ConfusionMatrixResponse,
    PsiResponse,
    FairnessResponse,
    FeatureImportanceResponse,
    ImprovementsResponse,
    KsResponse,
    MetricsResponse,
    RocResponse,
    ThresholdAnalysisResponse,
)
from Model.metrics_lib import (
    confusion_at,
    cost_at,
    derived_metrics,
    psi,
    threshold_sweep,
)

router = APIRouter(prefix="/model", tags=["Modelo"])

Split = Literal["train", "valid", "test"]
ModelName = Literal["champion", "baseline"]


@router.get("/metrics", response_model=MetricsResponse,
            summary="Métricas oficiais da rodada congelada",
            description="Fonte única de verdade. Qualquer número em slide, documento "
                        "ou notebook precisa bater com o que sai daqui.")
def metrics(request: Request):
    m = artifacts.require(request.app.state, "metrics")
    if "run" not in m:
        raise HTTPException(503, detail="metrics.json antigo, sem bloco 'run'. "
                                        "Rode `python Model/train.py` novamente.")
    return {**m, "lift_vs_baseline": m["champion"]["auc"] - m["baseline"]["auc"]}


@router.get("/roc", response_model=RocResponse, summary="Curva ROC")
def roc(request: Request, model: ModelName = "champion", split: Literal["valid", "test"] = "test"):
    c = artifacts.require(request.app.state, "curves")
    node = (c.get(model) or {}).get(split, {}).get("roc")
    if not node:
        raise HTTPException(404, detail=f"Sem curva ROC para {model}/{split}.")
    return {"model": model, "split": split, **node}


@router.get("/ks", response_model=KsResponse,
            summary="Curva KS e tabela de decis",
            description="A tabela de decis é a linguagem que uma mesa de crédito "
                        "reconhece: risco ordenado do pior para o melhor decil.")
def ks(request: Request, model: ModelName = "champion", split: Literal["valid", "test"] = "test"):
    c = artifacts.require(request.app.state, "curves")
    node = (c.get(model) or {}).get(split, {})
    if not node.get("ks"):
        raise HTTPException(404, detail=f"Sem curva KS para {model}/{split}.")
    k = node["ks"]
    return {"model": model, "split": split, "ks": k["ks"],
            "ks_threshold": k.get("ks_threshold"), "points": k["points"],
            "deciles": node.get("deciles")}


@router.get("/calibration", summary="Curva de confiabilidade e Brier",
            description="Um modelo calibrado fica na diagonal. Com `is_unbalance=true` "
                        "e sem calibração, o score ordena bem mas NÃO é P(default) real — "
                        "é por isso que o threshold ótimo não fica perto de 0,09.")
def calibration(request: Request, model: ModelName = "champion",
                split: Literal["valid", "test"] = "test"):
    c = artifacts.require(request.app.state, "curves")
    pts = (c.get(model) or {}).get(split, {}).get("calibration")
    if not pts:
        raise HTTPException(404, detail=f"Sem curva de calibração para {model}/{split}.")
    m = artifacts.require(request.app.state, "metrics")
    cal = m.get("calibrated") or {}
    return {"model": model, "split": split,
            "calibrated": bool(cal),
            "method": cal.get("method"),
            "brier": (cal.get("brier") if model == "champion" and cal
                      else (m.get(model) or {}).get("brier")),
            "brier_before_calibration": cal.get("brier_before"),
            "points": pts,
            "points_raw": (c.get(model) or {}).get(split, {}).get("calibration_raw"),
            "leitura": ("gap > 0 significa que o modelo prevê risco MAIOR do que o "
                        "observado naquele decil de score")}


@router.get("/feature-importance", response_model=FeatureImportanceResponse,
            summary="Importância das variáveis",
            description="`by_source` responde com número se a ABT das 9 tabelas valeu "
                        "a pena: é a fração da importância que vem de cada tabela.")
def feature_importance(request: Request,
                       top: int = Query(20, ge=1, le=200),
                       kind: Literal["gain", "split"] = "gain",
                       prefix: str | None = Query(None, description="Ex.: BUREAU_, PREV_")):
    fi = artifacts.require(request.app.state, "feature_importance")
    itens = fi.get(kind) or []
    if prefix:
        itens = [i for i in itens if i["feature"].startswith(prefix)]
    return {"kind": kind, "n_features_total": fi.get("n_features", len(fi.get(kind, []))),
            "items": itens[:top], "by_source": fi.get("by_source", {})}


@router.get(
    "/threshold-analysis",
    response_model=ThresholdAnalysisResponse,
    summary="Recalcula a régua de custo para qualquer par de custos",
    description=(
        "O endpoint mais útil numa arguição. Varre os thresholds sobre "
        "`artifacts/scores.parquet` e devolve custo, taxa de aprovação e matriz de "
        "confusão em cada ponto, apontando o ótimo — **sem re-treinar nada**.\n\n"
        "- `?cost_fn=1&cost_fp=0.1` reproduz o threshold congelado (razão 10:1)\n"
        "- `?cost_fn=20&cost_fp=1` responde 'e se aprovar um mau pagador custasse 20x?'"
    ),
)
def threshold_analysis(request: Request,
                       cost_fn: float = Query(1.0, gt=0, description="Custo de aprovar um mau pagador"),
                       cost_fp: float = Query(0.1, gt=0, description="Custo de negar um bom pagador"),
                       split: Literal["valid", "test"] = Query(
                           "valid", description="'valid' reproduz o que o treino usou"),
                       model: ModelName = "champion",
                       n_points: int = Query(99, ge=10, le=500)):
    y, p = artifacts.load_scores(split, model)
    pontos = threshold_sweep(y, p, cost_fn, cost_fp, n_points=n_points)
    melhor = min(pontos, key=lambda r: r["cost"])

    congelado = artifacts.threshold(request.app.state)
    cm = confusion_at(y, p, congelado)
    atual = {**cm, "cost": cost_at(cm, cost_fn, cost_fp), **derived_metrics(cm)}

    return {"cost_fn": cost_fn, "cost_fp": cost_fp, "cost_ratio": cost_fn / cost_fp,
            "split": split, "model": model, "n": int(y.size),
            "frozen_threshold": congelado, "best": melhor, "current": atual,
            "delta_cost_vs_current": melhor["cost"] - atual["cost"],
            "points": pontos}


@router.get("/confusion-matrix", response_model=ConfusionMatrixResponse,
            summary="Matriz de confusão em qualquer threshold")
def confusion_matrix(request: Request,
                     threshold: float | None = Query(None, ge=0, le=1),
                     split: Split = "test",
                     model: ModelName = "champion",
                     cost_fn: float = Query(1.0, gt=0),
                     cost_fp: float = Query(0.1, gt=0)):
    y, p = artifacts.load_scores(split, model)
    thr = threshold if threshold is not None else artifacts.threshold(request.app.state)
    cm = confusion_at(y, p, thr)
    return {**cm, "split": split, **derived_metrics(cm),
            "cost": cost_at(cm, cost_fn, cost_fp)}


@router.get(
    "/fairness",
    response_model=FairnessResponse,
    summary="Desempenho por segmento sensível, com intervalo de confiança",
    description=(
        "Cada grupo vem com IC bootstrap do AUC. Isso é o que separa fraqueza real "
        "de ruído amostral: `overlaps_overall=false` significa que a diferença NÃO é "
        "explicável por tamanho de amostra."
    ),
)
def fairness(request: Request,
             by: str = Query("gender", description="gender | age_band | thin_file"),
             threshold: float | None = Query(None, ge=0, le=1)):
    f = artifacts.require(request.app.state, "fairness")
    grupos = (f.get("dimensions") or {}).get(by)
    if grupos is None:
        raise HTTPException(
            404, detail=f"Dimensão {by!r} não calculada. Disponíveis: "
                        f"{', '.join(sorted((f.get('dimensions') or {}))) or 'nenhuma'}.")

    geral = f.get("overall") or {}
    lo, hi = geral.get("ci_low"), geral.get("ci_high")
    saida = []
    for g in grupos:
        g = dict(g)
        if None not in (lo, hi, g.get("ci_low"), g.get("ci_high")):
            g["overlaps_overall"] = not (g["ci_high"] < lo or g["ci_low"] > hi)
        saida.append(g)

    return {"dimension": by,
            "threshold": threshold if threshold is not None else f.get("threshold"),
            "overall": geral, "groups": saida}


@router.get(
    "/improvements",
    response_model=ImprovementsResponse,
    summary="Antes e depois das correções do modelo",
    description=(
        "Compara as rodadas registradas em `artifacts/improvement_log.json`, por "
        "segmento. É a resposta executável para 'vocês encontraram onde o modelo "
        "falha — e o que fizeram a respeito?'"
    ),
)
def improvements(request: Request):
    log = artifacts.require(request.app.state, "improvements")
    runs = [r for r in log.get("runs", []) if not r.get("sample")]
    if not runs:
        raise HTTPException(404, detail="Nenhuma rodada oficial registrada.")

    # Rodadas REJEITADAS continuam no log de propósito — o que foi tentado e
    # não funcionou faz parte do trabalho. Mas o "depois" da comparação é a
    # última rodada ACEITA, que é a que está servindo.
    aceitas = [r for r in runs if r.get("status", "aceita") == "aceita"]
    if not aceitas:
        raise HTTPException(404, detail="Nenhuma rodada aceita registrada.")

    primeira, ultima = aceitas[0], aceitas[-1]
    deltas = {}
    if len(runs) > 1:
        for k in ("auc", "ks", "brier", "approval_rate"):
            a, b = primeira.get(k), ultima.get(k)
            if a is not None and b is not None:
                deltas[k] = b - a
        seg = {}
        for dim, grupos in (ultima.get("segments") or {}).items():
            linhas = {}
            for nome, novo in grupos.items():
                velho = ((primeira.get("segments") or {}).get(dim) or {}).get(nome)
                if velho and velho.get("auc") is not None and novo.get("auc") is not None:
                    linhas[nome] = {"antes": velho["auc"], "depois": novo["auc"],
                                    "delta": novo["auc"] - velho["auc"], "n": novo.get("n")}
            if linhas:
                seg[dim] = linhas
        deltas["segments"] = seg

    rejeitadas = [{"tag": r.get("tag"), "auc": r.get("auc"),
                   "motivo": r.get("motivo")}
                  for r in runs if r.get("status") == "rejeitada"]
    return {"runs": runs, "baseline_tag": primeira.get("tag"),
            "latest_tag": ultima.get("tag"), "deltas": deltas,
            "rejeitadas": rejeitadas}


@router.get("/decision-policy", summary="Régua de decisão em três faixas",
            description="Mostra quais segmentos entram em faixa cinza ampliada e "
                        "quantos clientes caem em cada faixa.")
def decision_policy(request: Request,
                    base_band: float = Query(policy.DEFAULT_BAND, ge=0, le=0.5),
                    con=Depends(db.get_db)):
    f = artifacts.require(request.app.state, "fairness")
    thr = artifacts.threshold(request.app.state)
    resumo = policy.policy_summary(f, base_band)

    half = base_band / 2
    dist = db.fetch_one(con, f"""
        SELECT count({settings.SCORE_COL}) AS scored,
               sum(CASE WHEN {settings.SCORE_COL} <  ? THEN 1 ELSE 0 END) AS aprovar,
               sum(CASE WHEN {settings.SCORE_COL} >= ? AND {settings.SCORE_COL} < ? THEN 1 ELSE 0 END) AS revisar,
               sum(CASE WHEN {settings.SCORE_COL} >= ? THEN 1 ELSE 0 END) AS negar
        FROM clients WHERE {settings.SCORE_COL} IS NOT NULL
    """, [thr - half, thr - half, thr + half, thr + half])

    total = dist["scored"] or 1
    return {"threshold": thr, **resumo,
            "faixas": {"APROVAR": f"p < {thr - half:.4f}",
                       "REVISAR": f"{thr - half:.4f} <= p < {thr + half:.4f}",
                       "NEGAR": f"p >= {thr + half:.4f}"},
            "distribuicao": {k: {"n": dist[k], "pct": dist[k] / total}
                             for k in ("aprovar", "revisar", "negar")},
            "observacao": ("A faixa alarga por disponibilidade de informação e por "
                           "segmento de baixa confiança MEDIDA. O critério de risco é "
                           "o mesmo para todos: o que muda é quanto vai a revisão humana.")}


@router.get(
    "/psi",
    response_model=PsiResponse,
    summary="Estabilidade populacional (PSI) entre duas fatias",
    description=(
        "Mede o quanto a distribuição de cada variável mudou em relação à fatia de "
        "referência. É a métrica padrão de monitoramento de crédito, com leitura fixa "
        "de mercado: **< 0,10** estável · **0,10–0,25** atenção · **> 0,25** mudança "
        "relevante.\n\n"
        "Aqui as fatias são `train`/`valid`/`test`, então o PSI deve dar baixo — é o "
        "mesmo período. O valor da implementação é operacional: em produção, basta "
        "apontar `comparado` para a safra nova e o mesmo cálculo vira o alerta de "
        "drift descrito em `MLOps/Readme.md`.\n\n"
        "Sem `features`, compara o próprio score — que é o sinal de drift mais "
        "importante, por resumir todas as variáveis de uma vez."
    ),
)
def psi_endpoint(request: Request,
                 referencia: Split = Query("train", description="Fatia de referência"),
                 comparado: Split = Query("test", description="Fatia a comparar"),
                 features: str | None = Query(
                     None, description="Colunas separadas por vírgula. "
                                       "Omitido = compara o score do modelo"),
                 n_bins: int = Query(10, ge=3, le=50),
                 top: int = Query(30, ge=1, le=200),
                 con=Depends(db.get_db),
                 columns=Depends(db.get_columns)):
    if referencia == comparado:
        raise HTTPException(400, detail="'referencia' e 'comparado' precisam ser fatias diferentes.")

    nomes = db.split_csv(features) or [settings.SCORE_COL]
    for nome in nomes:
        db.quote_ident(nome, columns)
    if len(nomes) > 200:
        raise HTTPException(400, detail="Máximo de 200 variáveis por requisição.")

    sel = ", ".join(db.quote_ident(nome, columns) for nome in nomes)
    ref = db.fetch_all(con, f"SELECT {sel} FROM clients WHERE split = ?", [referencia])
    obs = db.fetch_all(con, f"SELECT {sel} FROM clients WHERE split = ?", [comparado])
    if not ref or not obs:
        raise HTTPException(404, detail=f"Sem linhas em '{referencia}' ou '{comparado}'.")

    itens = []
    for nome in nomes:
        a = [r[nome] for r in ref if isinstance(r.get(nome), (int, float))]
        b = [r[nome] for r in obs if isinstance(r.get(nome), (int, float))]
        out = psi(a, b, n_bins=n_bins)
        itens.append({"feature": nome, **out})
    itens.sort(key=lambda x: -(x.get("psi") or -1))

    resumo = {"estável": 0, "atenção": 0, "mudança relevante": 0, "sem_calculo": 0}
    for i in itens:
        resumo[i.get("faixa") or "sem_calculo"] += 1

    return {"referencia": referencia, "comparado": comparado, "n_features": len(itens),
            "limiares": {"estável": "< 0,10", "atenção": "0,10 a 0,25",
                         "mudança relevante": "> 0,25"},
            "resumo": resumo, "items": itens[:top]}
