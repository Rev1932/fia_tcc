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
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Métricas vivem num módulo só: o mesmo código gera o número do treino,
# do notebook e da API (ver Model/metrics_lib.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Model.metrics_lib import (  # noqa: E402
    auc_bootstrap_ci,
    auc_diff_all_groups,
    auc_within_between,
    brier_score,
    business_threshold,
    calibration_points,
    decile_table,
    ks_curve,
    ks_statistic,
    roc_points,
    threshold_sweep,
)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------
# Metadados da rodada — é o que impede dois conjuntos de números de circularem
# --------------------------------------------------------------------------


def git_sha() -> str | None:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, cwd=ROOT, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def lib_versions() -> dict:
    import platform
    import sklearn
    import lightgbm
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


SOURCE_PREFIXES = ("BUREAU_BB_", "BUREAU_", "PREV_", "POS_", "CC_", "INST_")


def feature_source(name: str) -> str:
    """De qual tabela a feature veio — responde 'a ABT das 9 tabelas valeu a pena?'."""
    for p in SOURCE_PREFIXES:
        if name.startswith(p):
            return "BUREAU_BALANCE" if p == "BUREAU_BB_" else p.rstrip("_")
    return "application"


def build_sample_weight(X, cfg: dict):
    """Peso por linha para reforçar segmentos de baixa confiança medida.

    Hipótese a testar: o modelo enxerga pior os extremos de idade porque eles
    são poucos (<25 são ~4% da base) e o gradiente é dominado pelo miolo.
    Dar mais peso a essas linhas força o modelo a gastar capacidade nelas.

    Risco conhecido: melhora o segmento às custas do global. A regra de aceite
    (ver TODO.md §4) é que só fica se o alvo melhorar SEM piorar o geral.
    """
    w_cfg = (cfg.get("champion") or {}).get("sample_weight") or {}
    if not w_cfg.get("enabled"):
        return None, {}

    pesos = np.ones(len(X), dtype="float64")
    aplicado: dict = {}
    faixas = (w_cfg.get("segments") or {}).get("age_band") or {}
    if faixas and "AGE_YEARS" in X.columns:
        rotulos = np.array([age_band(v) for v in X["AGE_YEARS"].to_numpy()])
        for faixa, peso in faixas.items():
            m = rotulos == faixa
            if m.any():
                pesos[m] = float(peso)
                aplicado[faixa] = {"peso": float(peso), "n": int(m.sum())}
    if not aplicado:
        return None, {}
    print(f"[train] sample_weight ativo: {aplicado}")
    return pesos, aplicado


def age_band(years: float) -> str:
    if pd.isna(years):
        return "desconhecido"
    for hi, label in ((25, "<25"), (35, "25-35"), (45, "35-45"),
                      (55, "45-55"), (65, "55-65")):
        if years < hi:
            return label
    return "65+"


CRITERIO_FRAQUEZA = {
    "nome": "bootstrap_da_diferenca_intra_eixo",
    "descricao": (
        "Um grupo conta como fraqueza quando o IC 95% da DIFERENÇA entre o seu AUC "
        "e o AUC de referência — a média, ponderada por pares, do AUC medido DENTRO "
        "de cada um dos demais grupos do mesmo eixo — fica inteiramente abaixo de "
        "zero. Substitui o critério anterior (IC do grupo sem sobrepor o IC geral), "
        "que era inválido: o grupo é subconjunto do geral, e o AUC geral inclui "
        "pares entre grupos que nenhum AUC intra-grupo tem."),
    "criterio_anterior": "ci_high(grupo) < ci_low(geral)",
}


def segment_report(X, y, score, threshold: float, n_boot: int) -> dict:
    """AUC (com IC bootstrap), aprovação e default real por segmento sensível.

    O IC é o que separa fraqueza real de ruído amostral: o grupo <25 tem
    ~2,4 mil clientes no teste, então um AUC menor pode ser só tamanho de
    amostra. Sem esse número, a conclusão do trabalho fica indefensável.

    A comparação entre grupos usa o IC da DIFERENÇA contra os demais grupos do
    mesmo eixo (`vs_referencia`), e não o AUC geral — ver CRITERIO_FRAQUEZA.
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)

    segments = {}
    if "CODE_GENDER" in X.columns:
        segments["gender"] = pd.Series(X["CODE_GENDER"].astype("object").to_numpy())
    if "AGE_YEARS" in X.columns:
        segments["age_band"] = pd.Series([age_band(v) for v in X["AGE_YEARS"].to_numpy()])
    if "BUREAU_COUNT" in X.columns:
        thin = pd.isna(X["BUREAU_COUNT"].to_numpy())
        segments["thin_file"] = pd.Series(
            np.where(thin, "thin-file (sem bureau)", "com histórico de bureau"))

    out = {
        "threshold": float(threshold),
        "criterio": CRITERIO_FRAQUEZA,
        "overall": {
            **auc_bootstrap_ci(y, score, n_boot=n_boot),
            "approval_rate": float((score < threshold).mean()),
            "default_rate": float(y.mean()),
            "brier": brier_score(y, score),
        },
        "dimensions": {},
        "decomposicao": {},
    }

    for dim, values in segments.items():
        rotulos = values.to_numpy()
        # Uma passada de bootstrap por eixo, compartilhada por todos os grupos.
        difs = auc_diff_all_groups(y, score, rotulos, n_boot=n_boot)
        groups = []
        for g in sorted(v for v in values.dropna().unique()):
            m = (values == g).to_numpy()
            if m.sum() < 30:
                continue
            vs = difs.get(g)
            groups.append({
                "group": str(g),
                **auc_bootstrap_ci(y[m], score[m], n_boot=n_boot),
                "pct_da_base": float(m.mean()),
                "approval_rate": float((score[m] < threshold).mean()),
                "default_rate": float(y[m].mean()),
                "avg_score": float(score[m].mean()),
                "brier": brier_score(y[m], score[m]),
                "vs_referencia": vs,
                "fraqueza_confirmada": bool((vs or {}).get("pior_que_referencia")),
                # Ordenar mal (AUC) e errar o NÍVEL de risco são coisas
                # diferentes; sem este bloco a segunda não é medida.
                "calibracao": {
                    "previsto": float(score[m].mean()),
                    "observado": float(y[m].mean()),
                    "gap": float(score[m].mean() - y[m].mean()),
                },
            })
        out["dimensions"][dim] = groups
        out["decomposicao"][dim] = auc_within_between(y, score, rotulos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "Model" / "config.yaml"))
    ap.add_argument("--sample", type=int, default=None,
                    help="Treina numa amostra estratificada da ABT (modo demo, <90s). "
                         "A rodada oficial roda sem esta flag.")
    ap.add_argument("--tag", default=None,
                    help="Nome desta rodada no artifacts/improvement_log.json (ex.: v2.1-categoricas)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    target, idc = cfg["target"], cfg["id_col"]
    paths = cfg["paths"]
    art_cfg = cfg.get("artifacts", {})

    # Parquet quando existir: mesma ABT, leitura muito mais rápida
    pq = ROOT / paths.get("abt_parquet", "Dados/abt.parquet")
    if pq.exists():
        print(f"[train] carregando ABT (parquet) {pq.name}")
        df = pd.read_parquet(pq)
    else:
        print("[train] carregando ABT (csv)")
        df = pd.read_csv(ROOT / paths["abt"])

    if args.sample and args.sample < len(df):
        df, _ = train_test_split(df, train_size=args.sample,
                                 random_state=cfg["split"]["random_state"],
                                 stratify=df[target])
        df = df.reset_index(drop=True)
        print(f"[train] MODO AMOSTRA: {len(df)} linhas (não é a rodada oficial)")

    y = df[target].astype(int)
    ids = df[idc].to_numpy()
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

    # Os ids acompanham o split como terceiro array: a partição depende só de
    # (n_samples, stratify, random_state), então isto NÃO muda quem cai onde —
    # só permite saber, depois, qual cliente ficou em qual fatia.
    sp = cfg["split"]
    X_tmp, X_test, y_tmp, y_test, id_tmp, id_test = train_test_split(
        X, y, ids, test_size=sp["test_size"], random_state=sp["random_state"],
        stratify=y if sp["stratify"] else None)
    val_rel = sp["valid_size"] / (1 - sp["test_size"])
    X_tr, X_val, y_tr, y_val, id_tr, id_val = train_test_split(
        X_tmp, y_tmp, id_tmp, test_size=val_rel, random_state=sp["random_state"],
        stratify=y_tmp if sp["stratify"] else None)

    # Fatia de CALIBRAÇÃO (Fix 7): recortada do treino, nunca usada nem no
    # ajuste nem no early stopping. Calibrar na mesma fatia do early stopping
    # daria uma calibração otimista. calib_size=0 reproduz o split original.
    X_cal = y_cal = id_cal = None
    calib_size = sp.get("calib_size", 0) or 0
    if calib_size:
        cal_rel = calib_size / (1 - sp["test_size"] - sp["valid_size"])
        X_tr, X_cal, y_tr, y_cal, id_tr, id_cal = train_test_split(
            X_tr, y_tr, id_tr, test_size=cal_rel, random_state=sp["random_state"],
            stratify=y_tr if sp["stratify"] else None)

    print(f"[train] treino={len(X_tr)} valid={len(X_val)} "
          f"calib={len(X_cal) if X_cal is not None else 0} teste={len(X_test)}")

    n_rows, n_features = len(df), X_tr.shape[1]
    del df, X, X_tmp, y_tmp, id_tmp
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
    base_score_val = base.predict_proba(
        X_val.assign(**{c: X_val[c].astype("object") for c in cat_cols}))[:, 1]
    metrics["baseline"] = {"auc": float(roc_auc_score(y_test, base_score)),
                           "ks": ks_statistic(y_test, base_score),
                           "brier": brier_score(y_test, base_score)}
    print(f"[train] baseline AUC={metrics['baseline']['auc']:.4f} "
          f"KS={metrics['baseline']['ks']:.4f}")

    # ---------------- Campeão: LightGBM ----------------
    print("[train] treinando campeão (LightGBM)")
    champ = LGBMClassifier(**cfg["champion"]["params"])
    pesos, pesos_aplicados = build_sample_weight(X_tr, cfg)
    champ.fit(X_tr, y_tr, sample_weight=pesos,
              eval_set=[(X_val, y_val)], eval_metric="auc",
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
        "brier": brier_score(y_test, champ_score_test),
    }
    metrics["champion"]["overfit_gap"] = (
        metrics["champion"]["auc_train"] - metrics["champion"]["auc"])
    print(f"[train] champion AUC(test)={metrics['champion']['auc']:.4f} "
          f"KS={metrics['champion']['ks']:.4f} | "
          f"AUC train={metrics['champion']['auc_train']:.4f} "
          f"valid={metrics['champion']['auc_valid']:.4f}")

    # ---------------- Calibração de probabilidade (Fix 7) ----------------
    # Com is_unbalance=true o LightGBM produz um score que ORDENA bem mas não
    # é P(default) real — por isso o threshold ótimo cai perto de 0,5 quando,
    # para uma razão de custo 10:1 com score calibrado, o esperado é ~0,09.
    # A isotônica é monotônica: corrige a probabilidade sem mexer em AUC/KS.
    calibrator = None
    score_val, score_test = champ_score_val, champ_score_test
    score_train = train_score
    if X_cal is not None:
        print("[train] calibrando (isotônica, fatia exclusiva de calibração)")
        # FrozenEstimator: no scikit-learn 1.9 substitui o antigo cv="prefit".
        # Congela o campeão já treinado e ajusta só a isotônica por cima.
        calibrator = CalibratedClassifierCV(FrozenEstimator(champ), method="isotonic")
        calibrator.fit(X_cal, y_cal)
        cal_val = calibrator.predict_proba(X_val)[:, 1]
        cal_test = calibrator.predict_proba(X_test)[:, 1]
        metrics["calibrated"] = {
            "method": "isotonic",
            "auc": float(roc_auc_score(y_test, cal_test)),
            "ks": ks_statistic(y_test, cal_test),
            "brier": brier_score(y_test, cal_test),
            "brier_before": metrics["champion"]["brier"],
            "n_calib": int(len(X_cal)),
        }
        print(f"[train] calibrado: AUC={metrics['calibrated']['auc']:.4f} "
              f"(AUC não muda: isotônica é monotônica) | "
              f"Brier {metrics['champion']['brier']:.4f} -> "
              f"{metrics['calibrated']['brier']:.4f}")
        # O treino também precisa passar pela isotônica: sem isto a coluna
        # proba_champion misturaria escala crua (treino) com calibrada
        # (validação/teste), e qualquer filtro por faixa de score mentiria.
        score_train = calibrator.predict_proba(X_tr)[:, 1]
        score_val, score_test = cal_val, cal_test

    # ---------------- Threshold de negócio ----------------
    b = cfg["business"]
    thr, approval = business_threshold(y_val, score_val,
                                       b["cost_false_negative"], b["cost_false_positive"])
    metrics["business"] = {"threshold": thr, "approval_rate": approval,
                           "cost_false_negative": b["cost_false_negative"],
                           "cost_false_positive": b["cost_false_positive"]}
    print(f"[train] threshold de negócio={thr:.2f} | taxa de aprovação={approval:.1%}")

    # ---------------- Métricas do modelo SERVIDO ----------------
    # Qual é "o número" do trabalho? O do modelo que a API de fato serve.
    # Com calibração ativa, a isotônica introduz empates no score e move o KS
    # na terceira casa — pequeno, mas suficiente para duas métricas diferentes
    # circularem. Este bloco elimina a ambiguidade: é daqui que sai todo
    # número de capa em slide, documento e dossiê.
    servido = metrics.get("calibrated") or metrics["champion"]
    metrics["served"] = {
        "model": "champion+isotonic" if calibrator else "champion",
        "auc": servido["auc"],
        "ks": servido["ks"],
        "brier": servido["brier"],
        "threshold": thr,
        "approval_rate": approval,
    }

    # ---------------- Identidade da rodada ----------------
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}"
    if (sha := git_sha()):
        run_id += f"-{sha}"
    metrics["run"] = {
        "run_id": run_id,
        "tag": args.tag or ("amostra" if args.sample else "oficial"),
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": sha,
        "sample": args.sample,
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "n_train": int(len(X_tr)), "n_valid": int(len(X_val)), "n_test": int(len(X_test)),
        "versions": lib_versions(),
        "sample_weight": pesos_aplicados or None,
    }

    # ---------------- Serialização ----------------
    # Rodada de demonstração NUNCA escreve por cima da oficial: sem isto, mexer
    # num hiperparâmetro ao vivo destruiria a rodada canônica que todos os
    # documentos citam — e só se descobriria depois.
    art = ROOT / "artifacts"
    if args.sample:
        art = art / "demo"
        print(f"[train] MODO AMOSTRA: artefatos vão para {art.relative_to(ROOT)}/ "
              f"(a rodada oficial em artifacts/ fica intacta)")
    art.mkdir(parents=True, exist_ok=True)
    bundle = {
        # Campeão cru: é o que o TreeExplainer sabe explicar.
        "model": champ,
        # Calibrador (ou None): é o que define o score servido.
        "calibrator": calibrator,
        "feature_names": list(X_tr.columns),
        "categorical_features": cat_cols,
        "cat_categories": cat_categories,
        "threshold": thr,
        "metrics": metrics,
        "run_id": run_id,
    }
    joblib.dump(bundle, art / Path(paths["model_out"]).name)
    with open(art / Path(paths["metrics_out"]).name, "w") as f:
        json.dump(metrics, f, indent=2)
    feat_meta = {"feature_names": list(X_tr.columns), "numeric": num_cols,
                 "categorical": cat_cols, "n_features": int(n_features),
                 "run_id": run_id}
    with open(art / Path(paths["feature_meta_out"]).name, "w") as f:
        json.dump(feat_meta, f, indent=2)
    print(f"[train] modelo salvo em {(art / Path(paths['model_out']).name).relative_to(ROOT)}")

    # ---------------- Curvas (ROC/KS/decis/calibração/sweep) ----------------
    print("[train] gravando curvas")
    max_pts = art_cfg.get("roc_max_points", 300)
    sweep_pts = art_cfg.get("sweep_points", 99)
    curves = {
        "run_id": run_id,
        "champion": {
            "test": {"roc": roc_points(y_test, score_test, max_pts),
                     "ks": ks_curve(y_test, score_test, max_pts),
                     "deciles": decile_table(y_test, score_test),
                     "calibration": calibration_points(y_test, score_test),
                     "calibration_raw": calibration_points(y_test, champ_score_test)},
            "valid": {"roc": roc_points(y_val, score_val, max_pts),
                      "ks": ks_curve(y_val, score_val, max_pts),
                      "sweep": threshold_sweep(y_val, score_val,
                                               b["cost_false_negative"],
                                               b["cost_false_positive"], sweep_pts)},
        },
        "baseline": {
            "test": {"roc": roc_points(y_test, base_score, max_pts),
                     "ks": ks_curve(y_test, base_score, max_pts),
                     "calibration": calibration_points(y_test, base_score)},
        },
    }
    with open(art / "curves.json", "w") as f:
        json.dump(curves, f)

    # ---------------- Importância de variáveis ----------------
    print("[train] gravando importância de variáveis")
    names = list(X_tr.columns)
    booster = champ.booster_
    gain = booster.feature_importance(importance_type="gain")
    split_imp = booster.feature_importance(importance_type="split")

    def ranked(values):
        tot = float(values.sum()) or 1.0
        rows = sorted(({"feature": n, "importance": float(v), "importance_pct": float(v) / tot,
                        "source_table": feature_source(n)}
                       for n, v in zip(names, values)),
                      key=lambda r: -r["importance"])
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows

    gain_rows = ranked(gain)
    by_source: dict[str, float] = {}
    for r in gain_rows:
        by_source[r["source_table"]] = by_source.get(r["source_table"], 0.0) + r["importance_pct"]
    with open(art / "feature_importance.json", "w") as f:
        json.dump({"run_id": run_id, "n_features": int(n_features),
                   "gain": gain_rows, "split": ranked(split_imp),
                   "by_source": dict(sorted(by_source.items(), key=lambda kv: -kv[1]))}, f)

    # ---------------- Fairness / segmentos ----------------
    print("[train] calculando fairness por segmento (bootstrap)")
    n_boot = art_cfg.get("bootstrap_n", 500)
    fairness = segment_report(X_test, y_test, score_test, thr, n_boot)
    fairness["run_id"] = run_id
    with open(art / "fairness.json", "w") as f:
        json.dump(fairness, f, indent=2)

    # ---------------- Scores de todos os clientes ----------------
    print("[train] gravando scores.parquet")
    nan_tr = np.full(len(id_tr), np.nan)
    scores = pd.DataFrame({
        idc: np.concatenate([id_tr, id_val, id_test]).astype("int64"),
        "split": np.repeat(["train", "valid", "test"],
                           [len(id_tr), len(id_val), len(id_test)]),
        "y_true": np.concatenate([y_tr.to_numpy(), y_val.to_numpy(),
                                  y_test.to_numpy()]).astype("int8"),
        "proba_champion": np.concatenate([score_train, score_val, score_test]),
        "proba_champion_raw": np.concatenate([train_score, champ_score_val, champ_score_test]),
        "proba_baseline": np.concatenate([nan_tr, base_score_val, base_score]),
    })
    scores.to_parquet(art / Path(paths.get("scores_out", "scores.parquet")).name,
                      index=False)

    # ---------------- Log de melhoria (v1 -> v2.x) ----------------
    # O histórico de rodadas OFICIAIS fica em artifacts/. Rodadas de
    # demonstração têm o próprio log em artifacts/demo/, para poder comparar
    # dois experimentos ao vivo sem sujar o registro de evolução do projeto.
    log_path = art / "improvement_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"runs": []}
    entry = {
        "run_id": run_id,
        "tag": metrics["run"]["tag"],
        # "aceita" | "rejeitada" — decidido pela regra de aceite (TODO.md §4)
        # depois de comparar com a rodada anterior. Rodadas rejeitadas ficam
        # no log de propósito: o que foi tentado e não funcionou faz parte do
        # trabalho, e uma banca que perguntar merece a resposta.
        "status": "aceita",
        "trained_at": metrics["run"]["trained_at"],
        "n_features": int(n_features),
        "sample": args.sample,
        "auc": metrics["champion"]["auc"],
        "ks": metrics["champion"]["ks"],
        "brier": (metrics.get("calibrated") or metrics["champion"])["brier"],
        "calibrated": bool(calibrator),
        "auc_baseline": metrics["baseline"]["auc"],
        "threshold": thr,
        "approval_rate": approval,
        "segments": {
            dim: {g["group"]: {"auc": g["auc"], "ci_low": g["ci_low"],
                               "ci_high": g["ci_high"], "n": g["n"],
                               # o veredito do segmento, e não só o AUC: é o que
                               # permite ver se uma rodada MUDOU a classificação
                               "diff": (g.get("vs_referencia") or {}).get("diff"),
                               "fraqueza_confirmada": g.get("fraqueza_confirmada")}
                  for g in groups}
            for dim, groups in fairness["dimensions"].items()
        },
    }
    anteriores = {r.get("tag"): r for r in log["runs"]}
    if entry["tag"] in anteriores:
        entry["status"] = anteriores[entry["tag"]].get("status", "aceita")
    log["runs"] = [r for r in log["runs"] if r.get("tag") != entry["tag"]] + [entry]
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[train] rodada {run_id} ({entry['tag']}) concluída — artefatos em artifacts/")


if __name__ == "__main__":
    main()
