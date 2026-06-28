"""
predict.py — Serviço de predição (carrega o bundle e retorna risco + decisão).

Pode ser usado como módulo (import) ou via CLI:
    python Model/predict.py --json '{"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000}'

Retorna P(default) e a decisão (APROVAR / NEGAR) com base no threshold de negócio.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "model.joblib"


@lru_cache(maxsize=1)
def load_bundle(model_path: str = str(DEFAULT_MODEL)) -> dict:
    return joblib.load(model_path)


def _prepare(records: list[dict], bundle: dict) -> pd.DataFrame:
    """Alinha o payload às features do modelo (faltantes -> NaN) e recria categóricas."""
    X = pd.DataFrame(records)
    X = X.reindex(columns=bundle["feature_names"])
    for col in bundle["categorical_features"]:
        X[col] = pd.Categorical(X[col], categories=bundle["cat_categories"][col])
    num_cols = [c for c in X.columns if c not in bundle["categorical_features"]]
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")
    return X


def predict(records: list[dict] | dict, model_path: str = str(DEFAULT_MODEL)) -> list[dict]:
    if isinstance(records, dict):
        records = [records]
    bundle = load_bundle(model_path)
    X = _prepare(records, bundle)
    proba = bundle["model"].predict_proba(X)[:, 1]
    thr = bundle["threshold"]
    return [
        {
            "probability_default": round(float(p), 4),
            "threshold": round(float(thr), 4),
            "decision": "NEGAR" if p >= thr else "APROVAR",
        }
        for p in proba
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON com as features de um cliente")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    payload = json.loads(args.json)
    print(json.dumps(predict(payload, args.model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
