"""
streamlit_app.py — Demo interativa do credit scoring.

Execução local:
  streamlit run MLOps/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Model.predict import load_bundle, predict  # noqa: E402

st.set_page_config(page_title="Credit Scoring — Home Credit", page_icon="💳")
st.title("💳 Credit Scoring — Home Credit Default Risk")
st.caption("Estima a probabilidade de inadimplência e sugere a decisão de crédito.")

try:
    bundle = load_bundle()
    m = bundle["metrics"].get("champion", {})
    st.success(f"Modelo carregado · AUC teste {m.get('auc', 0):.3f} · "
               f"KS {m.get('ks', 0):.3f} · threshold {bundle['threshold']:.2f}")
except Exception as e:
    st.error(f"Modelo não encontrado. Rode `python Model/train.py`. ({e})")
    st.stop()

st.subheader("Dados do cliente")
col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Tipo de contrato", ["Cash loans", "Revolving loans"])
    income = st.number_input("Renda anual (AMT_INCOME_TOTAL)", 10000, 2_000_000, 150_000, step=10000)
    credit = st.number_input("Valor do crédito (AMT_CREDIT)", 10000, 4_000_000, 500_000, step=10000)
    annuity = st.number_input("Anuidade (AMT_ANNUITY)", 1000, 300_000, 25_000, step=1000)
with col2:
    age = st.slider("Idade (anos)", 18, 90, 35)
    ext2 = st.slider("Score externo 2 (EXT_SOURCE_2)", 0.0, 1.0, 0.5)
    ext3 = st.slider("Score externo 3 (EXT_SOURCE_3)", 0.0, 1.0, 0.4)
    children = st.number_input("Filhos (CNT_CHILDREN)", 0, 20, 0)

if st.button("Avaliar risco", type="primary"):
    record = {
        "NAME_CONTRACT_TYPE": contract,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "DAYS_BIRTH": -int(age * 365.25),
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "CNT_CHILDREN": children,
        "CREDIT_INCOME_RATIO": credit / income if income else None,
        "ANNUITY_INCOME_RATIO": annuity / income if income else None,
    }
    res = predict(record)[0]
    prob = res["probability_default"]
    st.metric("Probabilidade de inadimplência", f"{prob:.1%}")
    if res["decision"] == "APROVAR":
        st.success(f"✅ Decisão sugerida: **APROVAR** (risco {prob:.1%} < threshold {res['threshold']:.0%})")
    else:
        st.error(f"⛔ Decisão sugerida: **NEGAR** (risco {prob:.1%} ≥ threshold {res['threshold']:.0%})")
    st.caption("Demo: as demais features são preenchidas como ausentes (NaN), tratadas pelo modelo.")
