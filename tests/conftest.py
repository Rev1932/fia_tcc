"""
Fixtures da suíte da API.

A suíte NÃO depende dos 1,3 GB da ABT real: monta uma base sintética de 300
clientes com os nomes de coluna verdadeiros, treina um LightGBM minúsculo e
escreve artefatos de brinquedo no mesmo formato dos reais.

Isso é possível porque `settings` lê todos os caminhos de variável de
ambiente — o mesmo mecanismo que o docker-compose usa. Não é conveniência de
teste, é requisito de projeto.
"""
from __future__ import annotations

import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest

CATEGORICAS = {
    "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
    "CODE_GENDER": ["M", "F"],
    "NAME_EDUCATION_TYPE": ["Higher education", "Secondary / secondary special",
                            "Lower secondary"],
    "NAME_FAMILY_STATUS": ["Married", "Single / not married"],
    "NAME_INCOME_TYPE": ["Working", "Pensioner", "State servant"],
    "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment"],
    "OCCUPATION_TYPE": ["Laborers", "Core staff", "Managers"],
    "ORGANIZATION_TYPE": ["Business Entity Type 3", "Self-employed", "School"],
}


def _base_sintetica(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idade = rng.uniform(21, 68, n).round(1)
    ext2 = rng.beta(2, 2, n)
    ext3 = np.where(rng.random(n) < 0.2, np.nan, rng.beta(2, 2, n))
    renda = rng.lognormal(11.8, 0.4, n).round(-2)
    credito = renda * rng.uniform(1.5, 6.0, n)
    anuidade = credito * rng.uniform(0.03, 0.09, n)

    # target correlacionado com idade e score externo, ~8% de eventos
    logito = -2.6 - 2.2 * ext2 - 0.035 * (idade - 40) + 0.25 * (credito / renda)
    alvo = (rng.random(n) < 1 / (1 + np.exp(-logito))).astype(int)

    # 15% thin-file: sem nenhum registro no bureau
    thin = rng.random(n) < 0.15
    bureau = np.where(thin, np.nan, rng.integers(1, 12, n).astype(float))

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100001, 100001 + n, dtype="int64"),
        "TARGET": alvo,
        "AGE_YEARS": idade,
        "DAYS_BIRTH": (-idade * 365.25).round(),
        "YEARS_EMPLOYED": rng.uniform(0, 25, n).round(1),
        "DAYS_EMPLOYED": (-rng.uniform(0, 25, n) * 365.25).round(),
        "DAYS_REGISTRATION": -rng.uniform(100, 8000, n).round(),
        "DAYS_ID_PUBLISH": -rng.uniform(100, 6000, n).round(),
        "DAYS_LAST_PHONE_CHANGE": -rng.uniform(0, 3000, n).round(),
        "CNT_CHILDREN": rng.integers(0, 4, n),
        "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
        "AMT_INCOME_TOTAL": renda,
        "AMT_CREDIT": credito.round(-2),
        "AMT_ANNUITY": anuidade.round(-1),
        "AMT_GOODS_PRICE": (credito * 0.9).round(-2),
        "EXT_SOURCE_1": np.where(rng.random(n) < 0.56, np.nan, rng.beta(2, 2, n)),
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "BUREAU_COUNT": bureau,
        "PREV_COUNT": rng.integers(0, 9, n).astype(float),
        "POS_COUNT": rng.integers(0, 30, n).astype(float),
        "CC_COUNT": rng.integers(0, 20, n).astype(float),
        "INST_COUNT": rng.integers(0, 40, n).astype(float),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
    })
    for col, vals in CATEGORICAS.items():
        df[col] = rng.choice(vals, n)

    df["CREDIT_INCOME_RATIO"] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL
    df["ANNUITY_INCOME_RATIO"] = df.AMT_ANNUITY / df.AMT_INCOME_TOTAL
    df["CREDIT_TERM"] = df.AMT_ANNUITY / df.AMT_CREDIT
    df["EMPLOYED_AGE_RATIO"] = df.DAYS_EMPLOYED / df.DAYS_BIRTH
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MAX"] = ext.max(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["N_EXT_SOURCE_PRESENT"] = ext.notna().sum(axis=1).astype("int8")
    df["HAS_BUREAU"] = (~thin).astype("int8")
    return df


@pytest.fixture(scope="session")
def ambiente(tmp_path_factory):
    """Monta ABT + modelo + artefatos sintéticos e aponta o settings para lá."""
    import joblib
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    raiz = tmp_path_factory.mktemp("hc")
    art = raiz / "artifacts"
    art.mkdir()
    dados = raiz / "Dados"
    dados.mkdir()

    df = _base_sintetica()
    abt = dados / "abt.parquet"
    df.to_parquet(abt, index=False)

    y = df.TARGET
    X = df.drop(columns=["SK_ID_CURR", "TARGET"])
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    cat_categories = {c: sorted(X[c].dropna().unique().tolist()) for c in cat_cols}
    for c in cat_cols:
        X[c] = pd.Categorical(X[c], categories=cat_categories[c])

    modelo = LGBMClassifier(n_estimators=40, num_leaves=6, min_child_samples=10,
                            learning_rate=0.1, verbose=-1, random_state=1)
    modelo.fit(X, y)
    score = modelo.predict_proba(X)[:, 1]
    thr = 0.5

    joblib.dump({"model": modelo, "calibrator": None,
                 "feature_names": list(X.columns), "categorical_features": cat_cols,
                 "cat_categories": cat_categories, "threshold": thr,
                 "metrics": {}, "run_id": "teste-0001"},
                art / "model.joblib")

    split = np.where(np.arange(len(df)) % 5 == 0, "test",
                     np.where(np.arange(len(df)) % 5 == 1, "valid", "train"))
    # proba_baseline nula no treino, como no artefato real: pontuar a regressão
    # logística com OneHot em 154 mil linhas custa tempo e nenhum endpoint usa.
    base_proba = np.where(split == "train", np.nan, score * 0.95)
    pd.DataFrame({"SK_ID_CURR": df.SK_ID_CURR, "split": split, "y_true": y,
                  "proba_champion": score,
                  "proba_champion_raw": np.clip(score * 1.05, 0, 1),
                  "proba_baseline": base_proba}
                 ).to_parquet(art / "scores.parquet", index=False)

    auc = float(roc_auc_score(y, score))
    (art / "metrics.json").write_text(json.dumps({
        "run": {"run_id": "teste-0001", "tag": "teste",
                "trained_at": "2026-01-01T00:00:00-03:00", "n_features": X.shape[1],
                "n_rows": len(df), "n_train": 180, "n_valid": 60, "n_test": 60,
                "versions": {"python": "3.14"}},
        "baseline": {"auc": auc - 0.02, "ks": 0.3, "brier": 0.08},
        "champion": {"auc": auc, "ks": 0.35, "auc_train": auc + 0.05,
                     "auc_valid": auc, "best_iteration": 40, "brier": 0.07,
                     "overfit_gap": 0.05},
        "business": {"threshold": thr, "approval_rate": float((score < thr).mean()),
                     "cost_false_negative": 1.0, "cost_false_positive": 0.1},
        "served": {"model": "champion", "auc": auc, "ks": 0.35, "brier": 0.07,
                   "threshold": thr, "approval_rate": float((score < thr).mean())},
    }))

    from Model.metrics_lib import (auc_bootstrap_ci, auc_diff_bootstrap,
                                   auc_within_between, calibration_points,
                                   decile_table, ks_curve, roc_points, threshold_sweep)
    (art / "curves.json").write_text(json.dumps({
        "run_id": "teste-0001",
        "champion": {
            "test": {"roc": roc_points(y, score, 50), "ks": ks_curve(y, score, 50),
                     "deciles": decile_table(y, score),
                     "calibration": calibration_points(y, score)},
            "valid": {"roc": roc_points(y, score, 50), "ks": ks_curve(y, score, 50),
                      "sweep": threshold_sweep(y, score, 1.0, 0.1, 99)},
        },
        "baseline": {"test": {"roc": roc_points(y, score * 0.95, 50),
                              "ks": ks_curve(y, score * 0.95, 50),
                              "calibration": calibration_points(y, score * 0.95)}},
    }))

    imp = modelo.booster_.feature_importance(importance_type="gain")
    tot = float(imp.sum()) or 1.0
    linhas = sorted(({"feature": n, "importance": float(v), "importance_pct": float(v) / tot,
                      "source_table": "application"} for n, v in zip(X.columns, imp)),
                    key=lambda r: -r["importance"])
    for i, r in enumerate(linhas, 1):
        r["rank"] = i
    (art / "feature_importance.json").write_text(json.dumps(
        {"run_id": "teste-0001", "n_features": X.shape[1], "gain": linhas,
         "split": linhas, "by_source": {"application": 1.0}}))

    geral = auc_bootstrap_ci(y, score, n_boot=60)
    def grupos(serie):
        out = []
        for g in sorted(serie.dropna().unique()):
            m = (serie == g).to_numpy()
            if m.sum() < 30:
                continue
            out.append({"group": str(g), **auc_bootstrap_ci(y[m], score[m], n_boot=60),
                        "pct_da_base": float(m.mean()),
                        "approval_rate": float((score[m] < thr).mean()),
                        "default_rate": float(y[m].mean()),
                        "avg_score": float(score[m].mean()), "brier": 0.07})
        return out
    def _faixa(v):
        for hi, rot in ((25, "<25"), (35, "25-35"), (45, "35-45"),
                        (55, "45-55"), (65, "55-65")):
            if v < hi:
                return rot
        return "65+"

    def faixas_com_diferenca():
        """Dimensão no formato NOVO, para exercitar o critério da diferença.

        `gender` fica no formato antigo de propósito: o mesmo fixture cobre os
        dois caminhos de `low_confidence_groups`.
        """
        rot = np.array([_faixa(v) for v in df.AGE_YEARS], dtype=object)
        out = []
        for g in sorted(set(rot)):
            m = rot == g
            if m.sum() < 30:
                continue
            dif = auc_diff_bootstrap(y, score, rot, g, n_boot=40)
            out.append({"group": g, **auc_bootstrap_ci(y[m], score[m], n_boot=40),
                        "pct_da_base": float(m.mean()),
                        "approval_rate": float((score[m] < thr).mean()),
                        "default_rate": float(y[m].mean()),
                        "avg_score": float(score[m].mean()), "brier": 0.07,
                        "vs_referencia": dif,
                        "fraqueza_confirmada": bool(dif.get("pior_que_referencia")),
                        "calibracao": {"previsto": float(score[m].mean()),
                                       "observado": float(y[m].mean())}})
        return out

    (art / "fairness.json").write_text(json.dumps({
        "run_id": "teste-0001", "threshold": thr,
        "overall": {**geral, "approval_rate": float((score < thr).mean()),
                    "default_rate": float(y.mean()), "brier": 0.07},
        "dimensions": {
            "gender": grupos(df.CODE_GENDER),
            "thin_file": grupos(df.BUREAU_COUNT.isna().map(
                {True: "thin-file (sem bureau)", False: "com histórico de bureau"})),
            "age_band": faixas_com_diferenca(),
        },
        "criterio": {"nome": "bootstrap_da_diferenca_intra_eixo",
                     "descricao": "IC da diferença contra os demais grupos do eixo"},
        "decomposicao": {"age_band": auc_within_between(
            y, score, [_faixa(v) for v in df.AGE_YEARS])},
    }))

    (art / "improvement_log.json").write_text(json.dumps({"runs": [
        {"run_id": "teste-0000", "tag": "v1", "n_features": 10, "auc": auc - 0.01,
         "ks": 0.34, "brier": 0.075, "threshold": thr, "approval_rate": 0.7,
         "segments": {"gender": {"F": {"auc": auc - 0.01, "n": 150}}}},
        {"run_id": "teste-0001", "tag": "v2", "n_features": X.shape[1], "auc": auc,
         "ks": 0.35, "brier": 0.07, "threshold": thr, "approval_rate": 0.72,
         "segments": {"gender": {"F": {"auc": auc, "n": 150}}}},
    ]}))

    perfil = [{"column_name": c, "column_type": str(df[c].dtype),
               "null_percentage": float(df[c].isna().mean() * 100),
               "approx_unique": int(df[c].nunique())} for c in df.columns]
    (art / "abt_profile.json").write_text(json.dumps(
        {"source": "abt.parquet", "n_rows": len(df), "n_columns": df.shape[1],
         "columns": perfil}))
    (art / "feature_metadata.json").write_text(json.dumps(
        {"feature_names": list(X.columns), "numeric": [], "categorical": cat_cols,
         "n_features": X.shape[1], "run_id": "teste-0001"}))

    return {"root": raiz, "artifacts": art, "abt": abt, "df": df, "threshold": thr}


@pytest.fixture(scope="session")
def app_teste(ambiente):
    os.environ.update({
        "HC_ABT_PARQUET": str(ambiente["abt"]),
        "HC_SCORES_PARQUET": str(ambiente["artifacts"] / "scores.parquet"),
        "HC_MODEL_PATH": str(ambiente["artifacts"] / "model.joblib"),
        "HC_ARTIFACTS_DIR": str(ambiente["artifacts"]),
        "HC_EXPLAINER_EAGER": "0",
    })
    from MLOps.app import settings
    importlib.reload(settings)
    for mod in ("MLOps.app.db", "MLOps.app.artifacts", "MLOps.app.explain",
                "MLOps.app.routers.clients", "MLOps.app.routers.stats",
                "MLOps.app.routers.model", "MLOps.app.routers.scoring",
                "MLOps.app.api"):
        importlib.reload(importlib.import_module(mod))
    from MLOps.app.api import create_app
    return create_app()


@pytest.fixture(scope="session")
def client(app_teste):
    from fastapi.testclient import TestClient
    with TestClient(app_teste) as c:
        yield c


@pytest.fixture(autouse=True)
def _limpa_caches():
    """load_bundle e o TreeExplainer são @lru_cache: sem limpar, o segundo
    módulo de teste recebe o bundle carregado pelo primeiro."""
    from Model.predict import load_bundle
    from MLOps.app import artifacts
    from MLOps.app.explain import clear_cache
    yield
    load_bundle.cache_clear()
    artifacts.clear_caches()
    clear_cache()
