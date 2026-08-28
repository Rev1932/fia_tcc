"""
experimento_teto_idade.py — Qual é o melhor AUC alcançável dentro do `<25`?

Decide entre conserto e veredito. Se um modelo treinado SÓ nos jovens não
supera o modelo geral naqueles mesmos clientes de teste, o geral já extrai o
sinal que existe, e a fraqueza é teto de dado — não falta de capacidade.

A partição NÃO é refeita: ela é lida de `artifacts/scores.parquet`, que grava
em qual fatia cada cliente caiu na rodada congelada. Os clientes ausentes do
parquet são a fatia de calibração (train.py não a inclui na gravação).

Uso:
    python Model/experimento_teto_idade.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model.metrics_lib import auc_bootstrap_ci, brier_score, ks_statistic  # noqa: E402

ART = ROOT / "artifacts"

# (rótulo, idade máxima de treino). O teste é SEMPRE o mesmo `<25`.
VARIANTES = [("segmentado_<25", 25.0), ("segmentado_<30", 30.0)]
GRADE = [{"num_leaves": 34, "min_child_samples": 70},
         {"num_leaves": 16, "min_child_samples": 40},
         {"num_leaves": 8, "min_child_samples": 20}]


def fatias_da_rodada(ids: np.ndarray) -> dict[str, np.ndarray]:
    """Máscaras booleanas por fatia, lidas da rodada congelada."""
    sc = pd.read_parquet(ART / "scores.parquet", columns=["SK_ID_CURR", "split"])
    mapa = dict(zip(sc["SK_ID_CURR"].to_numpy(), sc["split"].to_numpy()))
    onde = np.array([mapa.get(i, "calib") for i in ids], dtype=object)
    return {f: onde == f for f in ("train", "valid", "calib", "test")}


def avalia(y, s, n_boot: int) -> dict:
    return {**auc_bootstrap_ci(y, s, n_boot=n_boot),
            "ks": ks_statistic(y, s), "brier": brier_score(y, s)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "Model" / "config.yaml").read_text())
    alvo, idc = cfg["target"], cfg["id_col"]

    print("[teto] carregando ABT")
    df = pd.read_parquet(ROOT / "Dados" / "abt.parquet")
    y = df[alvo].astype(int).to_numpy()
    ids = df[idc].to_numpy()
    X = df.drop(columns=[alvo, idc])
    idade = X["AGE_YEARS"].to_numpy(dtype="float64")

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    X[num_cols] = X[num_cols].astype("float32")
    for c in cat_cols:
        X[c] = pd.Categorical(X[c])

    fat = fatias_da_rodada(ids)
    print("[teto] fatias: " + "  ".join(f"{k}={v.sum():,}" for k, v in fat.items()))

    jovem = idade < 25.0
    m_test = fat["test"] & jovem
    print(f"[teto] teste <25: {m_test.sum():,} clientes, {y[m_test].sum():,} eventos")

    base_params = dict(cfg["champion"]["params"])
    resultados = []

    # Referência: o modelo GERAL da rodada congelada, nos mesmos clientes.
    sc = pd.read_parquet(ART / "scores.parquet")
    sc = sc[sc["split"] == "test"]
    ref = pd.DataFrame({idc: ids[m_test], "y": y[m_test]}).merge(sc, on=idc, how="left")
    resultados.append({
        "variante": "geral (rodada congelada)", "n_treino": int(fat["train"].sum()),
        "params": {k: base_params[k] for k in ("num_leaves", "min_child_samples")},
        "calibrado": avalia(ref["y"].to_numpy(), ref["proba_champion"].to_numpy(), args.n_boot),
        "cru": avalia(ref["y"].to_numpy(), ref["proba_champion_raw"].to_numpy(), args.n_boot),
    })

    for rotulo, lim in VARIANTES:
        seg = idade < lim
        m_tr, m_val, m_cal = fat["train"] & seg, fat["valid"] & seg, fat["calib"] & seg
        if m_tr.sum() < 500 or y[m_tr].sum() < 50:
            print(f"[teto] {rotulo}: treino pequeno demais, pulando")
            continue
        for g in GRADE:
            params = {**base_params, **g}
            mdl = LGBMClassifier(**params)
            mdl.fit(X[m_tr], y[m_tr], eval_set=[(X[m_val], y[m_val])], eval_metric="auc",
                    callbacks=[early_stopping(cfg["champion"]["early_stopping_rounds"],
                                              verbose=False), log_evaluation(0)])
            cru = mdl.predict_proba(X[m_test])[:, 1]
            linha = {
                "variante": rotulo, "n_treino": int(m_tr.sum()),
                "n_eventos_treino": int(y[m_tr].sum()),
                "best_iteration": int(mdl.best_iteration_ or params["n_estimators"]),
                "params": g,
                "cru": avalia(y[m_test], cru, args.n_boot),
            }
            # Calibra na fatia de calibração do próprio segmento, para comparar
            # com o número servido (que é o calibrado).
            if m_cal.sum() >= 200 and len(set(y[m_cal])) == 2:
                iso = CalibratedClassifierCV(FrozenEstimator(mdl), method="isotonic")
                iso.fit(X[m_cal], y[m_cal])
                linha["calibrado"] = avalia(
                    y[m_test], iso.predict_proba(X[m_test])[:, 1], args.n_boot)
                linha["n_calib"] = int(m_cal.sum())
            resultados.append(linha)
            a = linha.get("calibrado", linha["cru"])
            print(f"[teto] {rotulo:18s} {g} -> AUC {a['auc']:.4f} "
                  f"[{a['ci_low']:.4f}-{a['ci_high']:.4f}]  (n_treino={m_tr.sum():,})")

    melhor = max((r for r in resultados if r["variante"] != "geral (rodada congelada)"),
                 key=lambda r: r.get("calibrado", r["cru"])["auc"], default=None)
    geral = resultados[0]
    saida = {
        "run_id": json.loads((ART / "metrics.json").read_text())["run"]["run_id"],
        "n_teste_jovem": int(m_test.sum()),
        "resultados": resultados,
        "veredito": None,
    }
    if melhor:
        a_g = geral.get("calibrado", geral["cru"])["auc"]
        a_m = melhor.get("calibrado", melhor["cru"])["auc"]
        saida["veredito"] = {
            "auc_geral_no_segmento": a_g,
            "melhor_auc_segmentado": a_m,
            "delta": a_m - a_g,
            "modelo_dedicado_supera": bool(a_m > a_g),
        }

    (ART / "experimentos").mkdir(exist_ok=True)
    (ART / "experimentos" / "teto_idade.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=float))
    print("\n[teto] artifacts/experimentos/teto_idade.json")
    if saida["veredito"]:
        v = saida["veredito"]
        print(f"[teto] geral no segmento {v['auc_geral_no_segmento']:.4f} | "
              f"melhor dedicado {v['melhor_auc_segmentado']:.4f} | "
              f"delta {v['delta']:+.4f}")


if __name__ == "__main__":
    main()
