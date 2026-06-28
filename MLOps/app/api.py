"""
api.py — Serviço de predição via FastAPI.

Endpoints:
  GET  /health   -> status + métricas do modelo
  POST /predict  -> recebe features de 1+ clientes e retorna risco + decisão

Execução local:
  uvicorn MLOps.app.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Garante import de Model.predict tanto local quanto no container
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Model.predict import load_bundle, predict  # noqa: E402

app = FastAPI(title="Home Credit — Credit Scoring API", version="1.0.0")


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., description="Lista de clientes (features -> valor)")


@app.get("/health")
def health() -> dict:
    try:
        bundle = load_bundle()
        return {"status": "ok", "threshold": bundle["threshold"],
                "metrics": bundle["metrics"].get("champion", {})}
    except Exception as e:  # modelo ainda não treinado
        return {"status": "model_not_loaded", "detail": str(e)}


@app.post("/predict")
def predict_endpoint(req: PredictRequest) -> dict:
    return {"predictions": predict(req.records)}
