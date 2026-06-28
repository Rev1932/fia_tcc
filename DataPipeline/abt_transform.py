"""
abt_transform.py — Constrói a Analytical Base Table (1 linha por cliente).

Entrada : Dados/clean_data.csv (application limpa) + tabelas relacionais brutas
Saída   : Dados/abt.csv

Estratégia (incremental):
  base = application limpa
  + agregações numéricas por SK_ID_CURR de:
      bureau (+ bureau_balance via SK_ID_BUREAU)
      previous_application
      POS_CASH_balance
      credit_card_balance
      installments_payments
  + ratios financeiros derivados
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def aggregate_numeric(df: pd.DataFrame, group_col: str, prefix: str,
                      agg_funcs: list[str], keep_ids: set | None = None) -> pd.DataFrame:
    """Agrega colunas numéricas por group_col e adiciona contagem de registros."""
    if keep_ids is not None:
        df = df[df[group_col].isin(keep_ids)]

    num_cols = [c for c in df.select_dtypes("number").columns
                if c not in {group_col, "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}]
    grouped = df.groupby(group_col)
    agg = grouped[num_cols].agg(agg_funcs)
    agg.columns = [f"{prefix}_{col}_{func}".upper() for col, func in agg.columns]
    agg[f"{prefix}_COUNT".upper()] = grouped.size()
    return agg.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "DataPipeline" / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    p = cfg["paths"]
    raw_dir = ROOT / p["raw_dir"]
    agg_funcs = cfg["abt"]["agg_funcs"]
    idc = cfg["id_col"]

    print("[abt] lendo clean_data.csv")
    base = pd.read_csv(ROOT / p["clean_data"])
    keep_ids = set(base[idc])
    print(f"[abt] base (application limpa): {base.shape}")

    # --- bureau (+ bureau_balance) ---
    print("[abt] agregando bureau + bureau_balance")
    bureau = pd.read_csv(raw_dir / p["bureau"])
    bureau = bureau[bureau[idc].isin(keep_ids)]
    bb = pd.read_csv(raw_dir / p["bureau_balance"])
    bb_agg = aggregate_numeric(bb, "SK_ID_BUREAU", "BB", agg_funcs,
                               keep_ids=set(bureau["SK_ID_BUREAU"]))
    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau_agg = aggregate_numeric(bureau, idc, "BUREAU", agg_funcs)
    base = base.merge(bureau_agg, on=idc, how="left")
    del bureau, bb, bb_agg, bureau_agg

    # --- demais tabelas por SK_ID_CURR ---
    tables = [
        ("previous_application", "PREV"),
        ("pos_cash", "POS"),
        ("credit_card", "CC"),
        ("installments", "INST"),
    ]
    for key, prefix in tables:
        print(f"[abt] agregando {key}")
        t = pd.read_csv(raw_dir / p[key])
        agg = aggregate_numeric(t, idc, prefix, agg_funcs, keep_ids=keep_ids)
        base = base.merge(agg, on=idc, how="left")
        del t, agg

    # --- ratios financeiros derivados ---
    print("[abt] criando ratios derivados")
    for name, (num, den) in cfg["abt"]["ratios"].items():
        if num in base.columns and den in base.columns:
            base[name] = base[num] / base[den].replace(0, pd.NA)

    out = ROOT / p["abt"]
    base.to_csv(out, index=False)
    print(f"[abt] abt.csv salvo em {out} | shape: {base.shape}")

    # Verificação: 1 linha por cliente
    assert base[idc].is_unique, "ABT tem SK_ID_CURR duplicado!"
    print(f"[abt] OK: {base[idc].nunique()} clientes únicos, {base.shape[1]} colunas")


if __name__ == "__main__":
    main()
