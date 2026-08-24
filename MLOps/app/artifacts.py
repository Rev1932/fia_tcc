"""
artifacts.py — Leitura dos artefatos canônicos da rodada congelada.

Princípio do projeto: NENHUM número de slide, documento ou endpoint pode vir
de outro lugar que não `artifacts/`. Foi a ausência disso que fez dois
conjuntos de métricas circularem ao mesmo tempo (654/0,50/71,7% no artefato
e 783/0,47/69,1% nos decks).
"""
from __future__ import annotations

import json
from functools import lru_cache

import duckdb
import numpy as np
from fastapi import HTTPException

from MLOps.app import settings
from MLOps.app.db import _lit

FILES = {
    "metrics": "metrics.json",
    "curves": "curves.json",
    "feature_importance": "feature_importance.json",
    "fairness": "fairness.json",
    "improvements": "improvement_log.json",
    "profile": "abt_profile.json",
    "feature_metadata": "feature_metadata.json",
}


def load_all() -> dict:
    """Carrega o que existir. Artefato ausente não derruba a API — o endpoint
    que depende dele é que devolve 503 com a instrução do que rodar."""
    out: dict = {}
    for key, fname in FILES.items():
        path = settings.ARTIFACTS_DIR / fname
        if path.exists():
            try:
                out[key] = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                out[key] = {"_error": f"JSON inválido em {fname}: {e}"}
    return out


def require(state, key: str) -> dict:
    data = (getattr(state, "artifacts", None) or {}).get(key)
    if not data or "_error" in data:
        raise HTTPException(
            status_code=503,
            detail=f"Artefato '{FILES.get(key, key)}' indisponível. "
                   f"Rode `python Model/train.py` para gerar a rodada canônica.",
        )
    return data


def run_id(state) -> str | None:
    metrics = (getattr(state, "artifacts", None) or {}).get("metrics") or {}
    return (metrics.get("run") or {}).get("run_id")


def threshold(state) -> float:
    """Threshold vigente: o do bundle do modelo; na falta dele, o das métricas."""
    bundle = getattr(state, "bundle", None)
    if bundle is not None:
        return float(bundle["threshold"])
    metrics = (getattr(state, "artifacts", None) or {}).get("metrics") or {}
    thr = (metrics.get("business") or {}).get("threshold")
    if thr is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado: sem threshold.")
    return float(thr)


@lru_cache(maxsize=16)
def load_scores(split: str = "test", model: str = "champion") -> tuple[np.ndarray, np.ndarray]:
    """(y_true, score) de uma fatia, direto do scores.parquet.

    É o que permite recalcular matriz de confusão, custo e threshold ótimo
    ao vivo para QUALQUER par de custos, sem re-treinar e sem carregar o
    modelo — a resposta para "e se o falso negativo custasse 20x?".
    """
    if not settings.SCORES_PARQUET.exists():
        raise HTTPException(
            status_code=503,
            detail="artifacts/scores.parquet não existe. Rode `python Model/train.py`.")
    col = "proba_champion" if model == "champion" else "proba_baseline"
    con = duckdb.connect()
    df = con.execute(
        f"SELECT y_true, {col} AS p FROM read_parquet({_lit(settings.SCORES_PARQUET)}) "
        f"WHERE split = ? AND {col} IS NOT NULL", [split]).df()
    con.close()
    if df.empty:
        if model == "baseline" and split == "train":
            raise HTTPException(
                status_code=404,
                detail="O baseline não é pontuado na fatia de treino — rodar o "
                       "OneHotEncoder + regressão logística em 154 mil linhas "
                       "custa tempo e nenhum endpoint usa. Use split=valid ou "
                       "split=test, ou model=champion.")
        raise HTTPException(
            status_code=404,
            detail=f"Sem scores para split={split!r}, model={model!r} em "
                   f"artifacts/scores.parquet.")
    return df["y_true"].to_numpy(np.int8), df["p"].to_numpy(np.float64)


def clear_caches() -> None:
    load_scores.cache_clear()
