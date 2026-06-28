"""
data_sanitization.py — Limpeza e padronização da tabela principal (application).

Entrada : data/risco_fraude/home-credit-default-risk/application_train.csv
Saídas  : Dados/raw_data.csv   (amostra reprodutível da base bruta)
          Dados/clean_data.csv (base limpa, 1 linha por cliente)

Tratamentos aplicados (ver DataPipeline/config.yaml):
  - DAYS_EMPLOYED sentinela 365243 (~1000 anos) -> NaN
  - CODE_GENDER == 'XNA' -> NaN
  - Clip de outliers de AMT_INCOME_TOTAL no quantil superior
  - Conversão de DAYS_* (negativos) para anos positivos auxiliares
  - Padronização de strings categóricas (strip)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sanitize(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["sanitization"]
    df = df.copy()

    # 1) Sentinela de emprego -> NaN
    sentinel = s["days_employed_sentinel"]
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(sentinel, np.nan)

    # 2) Gênero indefinido -> NaN
    if s.get("drop_gender_xna") and "CODE_GENDER" in df.columns:
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # 3) Clip de renda
    q = s.get("income_clip_quantile")
    if q and "AMT_INCOME_TOTAL" in df.columns:
        cap = df["AMT_INCOME_TOTAL"].quantile(q)
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=cap)

    # 4) Variáveis derivadas legíveis a partir de DAYS_* (mantém as originais)
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(1)
    if "DAYS_EMPLOYED" in df.columns:
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED"] / 365.25).round(1)

    # 5) Padroniza strings categóricas
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].str.strip()

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "DataPipeline" / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    p = cfg["paths"]
    raw_path = ROOT / p["raw_dir"] / p["application_train"]

    print(f"[sanitization] lendo {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[sanitization] shape bruto: {df.shape}")

    # Amostragem opcional (reprodutível) para execução rápida
    n = cfg.get("sample_n")
    if n:
        df = df.sample(n=int(n), random_state=cfg["random_state"]).reset_index(drop=True)
        print(f"[sanitization] amostra aplicada: {df.shape}")

    # Salva amostra bruta (entregável /Dados/raw_data.csv)
    out_raw = ROOT / p["raw_sample"]
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_raw, index=False)
    print(f"[sanitization] raw_data.csv salvo em {out_raw}")

    clean = sanitize(df, cfg)
    out_clean = ROOT / p["clean_data"]
    clean.to_csv(out_clean, index=False)
    print(f"[sanitization] clean_data.csv salvo em {out_clean} | shape: {clean.shape}")

    # Relatório rápido
    na_top = clean.isna().mean().sort_values(ascending=False).head(5)
    print("[sanitization] top 5 colunas com nulos (%):")
    for col, frac in na_top.items():
        print(f"    {col}: {100 * frac:.1f}%")


if __name__ == "__main__":
    main()
