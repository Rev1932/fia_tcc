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

Correções desta versão (ver plano, Parte 1):
  Fix 1  agregação das CATEGÓRICAS das tabelas relacionais. A versão anterior
         só agregava colunas numéricas, descartando em silêncio o sinal mais
         forte da base: bureau_balance.STATUS (histórico mês a mês de atraso)
         e previous_application.NAME_CONTRACT_STATUS (histórico de recusa).
  Fix 2  flags explícitas de ausência de fonte (HAS_BUREAU, HAS_PREV, ...),
         para que "cliente sem histórico" seja um fato aprendível e não algo
         implícito em centenas de NaN.  -> alvo: thin-file
  Fix 3  combinação dos scores externos disponíveis (EXT_SOURCE_MEAN/MAX/...),
         em vez de tratar cada um isoladamente.  -> alvo: thin-file
  Fix 4  comportamento de pagamento como razão (PAYMENT_RATIO, DPD,
         utilização de limite), que é o que um analista de crédito olha e
         está disponível mesmo para quem não tem bureau.  -> alvo: thin-file
  Fix 5  janela recente (últimos 12 meses) em paralelo às agregações
         vitalícias.  -> alvo: geral e jovens
  Fix 6  variáveis relativas à idade: 3 anos de registro significam coisas
         diferentes aos 22 e aos 55.  -> alvo: jovens
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
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

    ids = {group_col, "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"}
    num_cols = [c for c in df.select_dtypes("number").columns if c not in ids]
    # Descartar coluna em silêncio foi o bug que custou o sinal mais forte da
    # base (ver docstring). Se sobrar algo numérico-mas-object, avisa alto.
    suspeitas = [c for c in df.columns
                 if c not in ids and c not in num_cols
                 and df[c].dtype == "object"
                 and pd.api.types.is_numeric_dtype(
                     pd.to_numeric(df[c].head(1000), errors="coerce"))
                 and pd.to_numeric(df[c].head(1000), errors="coerce").notna().any()]
    if suspeitas:
        print(f"[abt]   AVISO {prefix}: colunas numéricas em dtype object, "
              f"não agregadas: {suspeitas[:8]}")
    grouped = df.groupby(group_col)
    agg = grouped[num_cols].agg(agg_funcs)
    agg.columns = [f"{prefix}_{col}_{func}".upper() for col, func in agg.columns]
    agg[f"{prefix}_COUNT".upper()] = grouped.size()
    return agg.reset_index()


def aggregate_categorical(df: pd.DataFrame, group_col: str, prefix: str,
                          cols: list[str], max_categories: int = 15,
                          keep_ids: set | None = None) -> pd.DataFrame:
    """Fix 1 — agrega colunas CATEGÓRICAS por cliente.

    Para cada categoria gera duas features: a contagem (`_SUM`) e a proporção
    (`_MEAN`) dos registros daquele cliente. Ex.: quantos créditos do bureau
    estão ativos, e que fração do histórico de pedidos anteriores foi recusada.

    Sem isto, `aggregate_numeric` descarta essas colunas silenciosamente —
    era a maior perda de sinal da ABT anterior.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame({group_col: []})
    if keep_ids is not None:
        df = df[df[group_col].isin(keep_ids)]

    parts = []
    for col in cols:
        s = df[col].astype("object").where(df[col].notna(), None)
        # cauda longa (PRODUCT_COMBINATION tem dezenas de valores) vira "OUTROS"
        top = s.value_counts().head(max_categories).index
        s = s.where(s.isin(top), "OUTROS")
        d = pd.get_dummies(s, prefix=f"{prefix}_{col}", prefix_sep="_", dtype="float32")
        d[group_col] = df[group_col].to_numpy()
        g = d.groupby(group_col).agg(["sum", "mean"])
        g.columns = [_clean_name(f"{c}_{f}") for c, f in g.columns]
        parts.append(g)

    out = pd.concat(parts, axis=1).reset_index()
    return out


def _clean_name(name: str) -> str:
    """Nome de coluna seguro para Parquet/DuckDB/LightGBM."""
    out = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip().upper())
    return re.sub(r"_+", "_", out).strip("_")


def add_payment_behaviour(inst: pd.DataFrame) -> pd.DataFrame:
    """Fix 4 — razões de comportamento de pagamento em `installments_payments`.

    Dado interno da financeira: existe mesmo para quem não tem bureau.
    """
    inst = inst.copy()
    inst["PAYMENT_RATIO"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].replace(0, np.nan)
    inst["PAYMENT_DIFF"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    # dias de atraso: pagou depois do vencimento => positivo
    dpd = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["DPD"] = dpd.clip(lower=0)
    inst["DBD"] = (-dpd).clip(lower=0)          # dias de antecedência
    inst["LATE_FLAG"] = (dpd > 0).astype("float32")
    return inst


def add_card_behaviour(cc: pd.DataFrame) -> pd.DataFrame:
    """Fix 4 — utilização de limite no cartão."""
    cc = cc.copy()
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
    cc["DRAWING_RATIO"] = (cc["AMT_DRAWINGS_CURRENT"]
                           / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan))
    return cc


def add_presence_flags(base: pd.DataFrame) -> pd.DataFrame:
    """Fix 2 — "não ter histórico" vira feature explícita, não 450 NaN."""
    sources = {"HAS_BUREAU": "BUREAU_COUNT", "HAS_PREV": "PREV_COUNT",
               "HAS_POS": "POS_COUNT", "HAS_CC": "CC_COUNT", "HAS_INST": "INST_COUNT"}
    for flag, col in sources.items():
        base[flag] = (base[col].notna() & (base[col].fillna(0) > 0)).astype("int8") \
            if col in base.columns else 0

    internas = [f for f in ("HAS_PREV", "HAS_POS", "HAS_CC", "HAS_INST") if f in base.columns]
    base["N_INTERNAL_SOURCES"] = base[internas].sum(axis=1).astype("int8")
    base["N_SOURCES"] = (base["N_INTERNAL_SOURCES"] + base["HAS_BUREAU"]).astype("int8")
    # sem bureau E sem histórico interno: o caso realmente cego
    base["NO_HISTORY"] = ((base["N_SOURCES"] == 0)).astype("int8")
    return base


def add_ext_source_features(base: pd.DataFrame) -> pd.DataFrame:
    """Fix 3 — usa os scores externos que EXISTEM, em vez de penalizar o cliente
    por cada um que falta. Ataca de frente a tese do trabalho: o melhor dado é
    o que menos existe (EXT_SOURCE_1 falta em 56% dos casos)."""
    cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in base.columns]
    if not cols:
        return base
    ext = base[cols]
    base["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    base["EXT_SOURCE_MAX"] = ext.max(axis=1)
    base["EXT_SOURCE_MIN"] = ext.min(axis=1)
    base["EXT_SOURCE_STD"] = ext.std(axis=1)
    base["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=len(cols))
    base["N_EXT_SOURCE_PRESENT"] = ext.notna().sum(axis=1).astype("int8")
    return base


def add_age_relative_features(base: pd.DataFrame) -> pd.DataFrame:
    """Fix 6 — tempo relativo à idade. "Registrado há 3 anos" significa coisas
    diferentes aos 22 e aos 55; hoje o modelo só via o valor absoluto."""
    if "DAYS_BIRTH" not in base.columns:
        return base
    birth = base["DAYS_BIRTH"].replace(0, np.nan)
    for col, name in (("DAYS_REGISTRATION", "REGISTRATION_AGE_RATIO"),
                      ("DAYS_ID_PUBLISH", "ID_PUBLISH_AGE_RATIO"),
                      ("DAYS_LAST_PHONE_CHANGE", "PHONE_CHANGE_AGE_RATIO")):
        if col in base.columns:
            base[name] = base[col] / birth
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"} <= set(base.columns):
        base["INCOME_PER_PERSON"] = (base["AMT_INCOME_TOTAL"]
                                     / base["CNT_FAM_MEMBERS"].replace(0, np.nan))
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"} <= set(base.columns):
        base["CREDIT_GOODS_RATIO"] = (base["AMT_CREDIT"]
                                      / base["AMT_GOODS_PRICE"].replace(0, np.nan))
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "DataPipeline" / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    p = cfg["paths"]
    raw_dir = ROOT / p["raw_dir"]
    abt_cfg = cfg["abt"]
    agg_funcs = abt_cfg["agg_funcs"]
    cat_cfg = abt_cfg.get("categorical_agg", {})
    max_cats = cat_cfg.get("max_categories", 15)
    recent = abt_cfg.get("recent_window", {})
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
    bb_ids = set(bureau["SK_ID_BUREAU"])
    bb_agg = aggregate_numeric(bb, "SK_ID_BUREAU", "BB", agg_funcs, keep_ids=bb_ids)

    # Fix 1: STATUS do bureau_balance é o histórico mês a mês de atraso
    # (0..5 = faixas de DPD, C = quitado, X = sem informação). Agregado por
    # SK_ID_BUREAU ANTES do merge — a tabela tem 27M linhas.
    if cat_cfg.get("bureau_balance"):
        bb_cat = aggregate_categorical(bb, "SK_ID_BUREAU", "BB",
                                       cat_cfg["bureau_balance"], max_cats,
                                       keep_ids=bb_ids)
        bb_agg = bb_agg.merge(bb_cat, on="SK_ID_BUREAU", how="left")
        del bb_cat

    bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau_agg = aggregate_numeric(bureau, idc, "BUREAU", agg_funcs)
    if cat_cfg.get("bureau"):
        bureau_agg = bureau_agg.merge(
            aggregate_categorical(bureau, idc, "BUREAU", cat_cfg["bureau"], max_cats),
            on=idc, how="left")
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
        t = t[t[idc].isin(keep_ids)]

        # Fix 4: razões de comportamento antes de agregar
        if key == "installments":
            t = add_payment_behaviour(t)
        elif key == "credit_card":
            t = add_card_behaviour(t)

        agg = aggregate_numeric(t, idc, prefix, agg_funcs)
        base = base.merge(agg, on=idc, how="left")
        del agg

        # Fix 1: categóricas desta tabela
        if cat_cfg.get(key):
            base = base.merge(
                aggregate_categorical(t, idc, prefix, cat_cfg[key], max_cats),
                on=idc, how="left")

        # Fix 5: janela recente, em paralelo às agregações vitalícias
        win = recent.get(key)
        if win:
            col, months = win["column"], win["months"]
            if col in t.columns:
                cut = -abs(months) * (30 if col.startswith("DAYS_") else 1)
                rec = t[t[col] >= cut]
                if len(rec):
                    print(f"[abt]   janela recente ({months}m): {len(rec)} linhas")
                    base = base.merge(
                        aggregate_numeric(rec, idc, f"{prefix}_R{months}", agg_funcs),
                        on=idc, how="left")
                del rec
        del t

    # --- features derivadas ---
    print("[abt] criando ratios derivados")
    for name, (num, den) in abt_cfg["ratios"].items():
        if num in base.columns and den in base.columns:
            base[name] = base[num] / base[den].replace(0, np.nan)

    print("[abt] flags de presença de fonte (Fix 2)")
    base = add_presence_flags(base)
    print("[abt] combinação dos scores externos (Fix 3)")
    base = add_ext_source_features(base)
    print("[abt] variáveis relativas à idade (Fix 6)")
    base = add_age_relative_features(base)

    # Verificação: 1 linha por cliente (antes de gravar 1 GB)
    assert base[idc].is_unique, "ABT tem SK_ID_CURR duplicado!"

    out = ROOT / p["abt"]
    base.to_csv(out, index=False)
    print(f"[abt] abt.csv salvo em {out} | shape: {base.shape}")

    if abt_cfg.get("write_parquet", True):
        pq = ROOT / p.get("abt_parquet", "Dados/abt.parquet")
        base.to_parquet(pq, index=False)
        print(f"[abt] abt.parquet salvo em {pq}")

    print(f"[abt] OK: {base[idc].nunique()} clientes únicos, {base.shape[1]} colunas")


if __name__ == "__main__":
    main()
