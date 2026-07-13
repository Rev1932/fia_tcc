"""
train.py — Treino do modelo de credit scoring.

Pipeline:
  1. Carrega a ABT (Dados/abt.csv)
  2. Split estratificado treino/validação/teste
  3. Baseline interpretável: Regressão Logística
  4. Modelo campeão: LightGBM (categóricas nativas + early stopping)
  5. Avalia (AUC, KS) e define o threshold de negócio por matriz de custo
  6. Serializa o bundle de serving em artifacts/model.joblib + métricas

Uso: python Model/train.py [--config Model/config.yaml]
"""
from __future__ import annotations

import argparse
import gc
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def business_threshold(y_true, y_score, cost_fn: float, cost_fp: float):
    """Threshold que minimiza o custo esperado (FN caro, FP barato)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    y_true = np.asarray(y_true)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    approval_rate = float((y_score < best_t).mean())  # aprovado = baixo risco
    return best_t, approval_rate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Model" / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    target, idc = cfg["target"], cfg["id_col"]

    print("[train] carregando ABT")
    df = pd.read_csv(ROOT / cfg["paths"]["abt"])
    y = df[target].astype(int)
    X = df.drop(columns=[target, idc])

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"[train] features: {X.shape[1]} ({len(num_cols)} num, {len(cat_cols)} cat)")

    # float32 em vez de float64: metade da memória em todas as cópias
    # (splits, ColumnTransformer, imputer) sem impacto relevante nas métricas
    X[num_cols] = X[num_cols].astype("float32")

    # categóricas como 'category' (LightGBM trata nativamente)
    cat_categories = {c: list(X[c].astype("category").cat.categories) for c in cat_cols}
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_categories[c])

    sp = cfg["split"]
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=sp["test_size"], random_state=sp["random_state"],
        stratify=y if sp["stratify"] else None)
    val_rel = sp["valid_size"] / (1 - sp["test_size"])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=sp["random_state"],
        stratify=y_tmp if sp["stratify"] else None)
    print(f"[train] treino={len(X_tr)} valid={len(X_val)} teste={len(X_test)}")

    # libera as cópias grandes que não são mais usadas (df/X/X_tmp ficavam
    # ociosas na memória durante todo o treino, duplicando os splits)
    del df, X, X_tmp, y_tmp
    gc.collect()

    metrics: dict = {}

    # ---------------- Baseline: Regressão Logística ----------------
    print("[train] treinando baseline (LogisticRegression)")
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=20,
                                               sparse_output=True))]), cat_cols),
    ])
    base = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(**cfg["baseline"]["params"]))])
    # OneHot precisa de string, não Categorical
    base.fit(X_tr.assign(**{c: X_tr[c].astype("object") for c in cat_cols}), y_tr)
    base_score = base.predict_proba(
        X_test.assign(**{c: X_test[c].astype("object") for c in cat_cols}))[:, 1]
    metrics["baseline"] = {"auc": float(roc_auc_score(y_test, base_score)),
                           "ks": ks_statistic(y_test, base_score)}
    print(f"[train] baseline AUC={metrics['baseline']['auc']:.4f} "
          f"KS={metrics['baseline']['ks']:.4f}")

    # ---------------- Campeão: LightGBM ----------------
    print("[train] treinando campeão (LightGBM)")
    champ = LGBMClassifier(**cfg["champion"]["params"])
    champ.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="auc",
              callbacks=[early_stopping(cfg["champion"]["early_stopping_rounds"]),
                         log_evaluation(0)])
    champ_score_val = champ.predict_proba(X_val)[:, 1]
    champ_score_test = champ.predict_proba(X_test)[:, 1]
    train_score = champ.predict_proba(X_tr)[:, 1]
    metrics["champion"] = {
        "auc": float(roc_auc_score(y_test, champ_score_test)),
        "ks": ks_statistic(y_test, champ_score_test),
        "auc_train": float(roc_auc_score(y_tr, train_score)),
        "auc_valid": float(roc_auc_score(y_val, champ_score_val)),
        "best_iteration": int(champ.best_iteration_ or champ.n_estimators),
    }
    print(f"[train] champion AUC(test)={metrics['champion']['auc']:.4f} "
          f"KS={metrics['champion']['ks']:.4f} | "
          f"AUC train={metrics['champion']['auc_train']:.4f} "
          f"valid={metrics['champion']['auc_valid']:.4f}")

    # ---------------- Threshold de negócio ----------------
    b = cfg["business"]
    thr, approval = business_threshold(y_val, champ_score_val,
                                       b["cost_false_negative"], b["cost_false_positive"])
    metrics["business"] = {"threshold": thr, "approval_rate": approval,
                           "cost_false_negative": b["cost_false_negative"],
                           "cost_false_positive": b["cost_false_positive"]}
    print(f"[train] threshold de negócio={thr:.2f} | taxa de aprovação={approval:.1%}")

    # ---------------- Serialização ----------------
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    bundle = {
        "model": champ,
        "feature_names": list(X_tr.columns),
        "categorical_features": cat_cols,
        "cat_categories": cat_categories,
        "threshold": thr,
        "metrics": metrics,
    }
    joblib.dump(bundle, ROOT / cfg["paths"]["model_out"])
    with open(ROOT / cfg["paths"]["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)
    feat_meta = {"feature_names": list(X_tr.columns), "numeric": num_cols,
                 "categorical": cat_cols, "n_features": X_tr.shape[1]}
    with open(ROOT / cfg["paths"]["feature_meta_out"], "w") as f:
        json.dump(feat_meta, f, indent=2)
    print(f"[train] modelo salvo em {cfg['paths']['model_out']}")


if __name__ == "__main__":
    main()
