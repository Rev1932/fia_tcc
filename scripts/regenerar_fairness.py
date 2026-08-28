"""
regenerar_fairness.py — Recalcula artifacts/fairness.json sem re-treinar.

Chama `Model.train.segment_report` com exatamente os mesmos insumos que o
treino usaria: os scores servidos da fatia de teste (`artifacts/scores.parquet`)
e as colunas de segmento da ABT. Serve para trazer uma rodada já treinada para
o critério novo sem gastar 15 minutos de treino.

Uso:
    python scripts/regenerar_fairness.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model.train import segment_report  # noqa: E402

ART = ROOT / "artifacts"
SEGMENTO = ["SK_ID_CURR", "CODE_GENDER", "AGE_YEARS", "BUREAU_COUNT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=None,
                    help="padrão: o mesmo de Model/config.yaml")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "Model" / "config.yaml").read_text())
    n_boot = args.n_boot or cfg["artifacts"]["bootstrap_n"]

    metrics = json.loads((ART / "metrics.json").read_text())
    thr = metrics["business"]["threshold"]

    sc = pd.read_parquet(ART / "scores.parquet")
    sc = sc[sc["split"] == "test"]
    abt = pd.read_parquet(ROOT / "Dados" / "abt.parquet", columns=SEGMENTO)
    df = sc.merge(abt, on="SK_ID_CURR", how="inner")
    if len(df) != len(sc):
        raise SystemExit(f"[fairness] join perdeu linhas: {len(sc)} -> {len(df)}")

    fair = segment_report(df[SEGMENTO[1:]], df["y_true"].to_numpy(),
                          df["proba_champion"].to_numpy(), thr, n_boot)
    fair["run_id"] = metrics["run"]["run_id"]

    destino = ART / "fairness.json"
    if destino.exists():
        shutil.copy(destino, ART / "fairness_criterio_anterior.json")
    destino.write_text(json.dumps(fair, indent=2, ensure_ascii=False, default=float))

    print(f"[fairness] {len(df):,} clientes de teste · bootstrap {n_boot}"
          .replace(",", "."))
    for dim, grupos in fair["dimensions"].items():
        print(f"\n  {dim}")
        for g in grupos:
            v = g.get("vs_referencia") or {}
            marca = "FRAQUEZA" if g["fraqueza_confirmada"] else "        "
            print(f"    {marca} {g['group']:26s} n={g['n']:6,} AUC={g['auc']:.4f} "
                  f"diff={v.get('diff', 0):+.4f} "
                  f"IC[{v.get('diff_ci_low', 0):+.4f},{v.get('diff_ci_high', 0):+.4f}] "
                  f"p={v.get('p_value', 1):.3f}".replace(",", "."))
        d = fair["decomposicao"][dim]
        print(f"    decomposição: {d['w_within']:.1%} dos pares DENTRO do grupo "
              f"(AUC {d['auc_within']:.4f}) · {d['w_between']:.1%} ENTRE "
              f"(AUC {d['auc_between']:.4f})")


if __name__ == "__main__":
    main()
