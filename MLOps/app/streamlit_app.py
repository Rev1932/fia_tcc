"""
streamlit_app.py — Painel de credit scoring.

Consome a API por HTTP, em vez de importar `Model.predict` diretamente. A
versão anterior chamava o modelo no próprio processo, o que contradizia o
diagrama de arquitetura apresentado: existia uma API que nada consumia.
Agora o painel é um cliente como qualquer outro sistema seria.

Execução:
    HC_API_URL=http://localhost:8000 streamlit run MLOps/app/streamlit_app.py
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("HC_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 30

st.set_page_config(page_title="Credit Scoring — Home Credit", page_icon="💳",
                   layout="wide")


def get(caminho: str, **params):
    r = requests.get(f"{API}{caminho}", params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_opcional(caminho: str, **params):
    """Como `get`, mas devolve None em 404/503 em vez de derrubar a página.

    Usado onde o recurso pode não existir nesta rodada — por exemplo, uma
    dimensão de fairness que não foi calculada. Uma aba a menos é melhor que
    um traceback no meio da apresentação.
    """
    try:
        r = requests.get(f"{API}{caminho}", params=params, timeout=TIMEOUT)
        if r.status_code in (404, 503):
            return None
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def post(caminho: str, payload: dict):
    r = requests.post(f"{API}{caminho}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


CORES = {"APROVAR": "🟢", "REVISAR": "🟡", "NEGAR": "🔴"}

st.title("💳 Credit Scoring — Home Credit")
st.caption(f"Cliente HTTP da API em `{API}`")

# ---------------------------------------------------------------- saúde
try:
    saude = get("/health")
except Exception as e:
    st.error(f"API indisponível em {API}. Suba com "
             f"`uvicorn MLOps.app.api:app --port 8000`.\n\n{e}")
    st.stop()

if saude["status"] != "ok":
    st.error(f"API degradada: {saude.get('errors')}")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clientes", f"{saude['n_clients']:,}".replace(",", "."))
c2.metric("Features", saude["n_features"])
c3.metric("Threshold", f"{saude['threshold']:.3f}")
c4.metric("Rodada", saude["run_id"].split("-")[0])

aba_carteira, aba_cliente, aba_simular, aba_modelo = st.tabs(
    ["📊 Carteira", "🔍 Cliente", "🧪 Simulação", "📈 Modelo"])

# ------------------------------------------------------------- carteira
with aba_carteira:
    st.subheader("Inadimplência por segmento")
    dims = {d["label"]: d["key"] for d in get("/meta/dimensions")}
    col_a, col_b = st.columns([1, 2])
    with col_a:
        rotulo = st.selectbox("Agrupar por", list(dims), index=list(dims).index("Escolaridade"))
        so_thin = st.checkbox("Apenas thin-file (sem bureau)")
        idade_max = st.slider("Idade máxima", 20, 70, 70)
    filtros = {"by": dims[rotulo], "min_count": 30}
    if so_thin:
        filtros["thin_file"] = "true"
    if idade_max < 70:
        filtros["age_max"] = idade_max

    dados = get("/stats/default-rate", **filtros)
    df = pd.DataFrame(dados["buckets"])
    with col_b:
        if df.empty:
            st.info("Nenhum grupo atinge o mínimo de 30 clientes com esses filtros.")
        else:
            st.bar_chart(df.set_index("value")["default_rate"])

    if not df.empty:
        st.caption(f"Inadimplência geral do recorte: {dados['overall_default_rate']:.2%} "
                   f"· {dados['n_total']:,} clientes".replace(",", "."))
        st.dataframe(
            df[["value", "n", "default_rate", "lift", "approval_rate"]]
            .rename(columns={"value": rotulo, "n": "clientes",
                             "default_rate": "inadimplência",
                             "lift": "lift vs geral", "approval_rate": "aprovação"}),
            use_container_width=True, hide_index=True)

# -------------------------------------------------------------- cliente
with aba_cliente:
    sk = st.number_input("SK_ID_CURR", min_value=100001, value=100002, step=1)
    if st.button("Consultar", type="primary"):
        try:
            ficha = get(f"/clients/{sk}")
        except requests.HTTPError:
            st.error(f"Cliente {sk} não encontrado.")
            st.stop()

        score = ficha.get("score")
        if score:
            a, b, c = st.columns(3)
            a.metric("P(inadimplência)", f"{score['probability_default']:.2%}")
            b.metric("Decisão", f"{CORES.get(score['decision'],'')} {score['decision']}")
            c.metric("Faixa de risco", score["score_band"])
        if ficha["thin_file"]:
            st.warning("Thin-file: sem registro em bureau. O modelo tem menos "
                       "informação sobre este cliente.")

        e1, e2 = st.columns(2)
        e1.json(ficha["identificacao"])
        e2.json(ficha["financeiro"])

        st.subheader("Por que este score")
        exp = get(f"/clients/{sk}/explain", top=8)
        st.info(exp["narrative"])
        contrib = pd.DataFrame(exp["top_risk_drivers"] + exp["top_protective_factors"])
        if not contrib.empty:
            st.bar_chart(contrib.set_index("feature")["shap_value"])
        st.caption(
            "Verificação de fidelidade: `base + Σ SHAP` reconstrói a probabilidade "
            f"do modelo com erro de {exp['consistency_check']['max_abs_error']:.2e}.")

# ------------------------------------------------------------ simulação
with aba_simular:
    st.subheader("What-if")
    st.caption("As variáveis derivadas (comprometimento de renda, prazo, média "
               "dos scores externos) são recalculadas junto — como numa proposta real.")
    sk_sim = st.number_input("Cliente base", min_value=100001, value=100002,
                             step=1, key="sim")
    coluna = st.selectbox("Variável a variar",
                          ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY",
                           "EXT_SOURCE_2", "EXT_SOURCE_3"])
    lo, hi = (0.0, 1.0) if coluna.startswith("EXT_") else (50_000.0, 2_000_000.0)
    faixa = st.slider("Faixa da varredura", lo, hi, (lo, hi))
    if st.button("Simular", type="primary"):
        r = post("/simulate", {"sk_id_curr": int(sk_sim),
                               "sweep": {"feature": coluna, "start": faixa[0],
                                         "stop": faixa[1], "steps": 15}})
        base = r["base"]
        st.metric("Score atual", f"{base['probability_default']:.2%}",
                  help=f"Decisão: {base['decision']}")
        sw = pd.DataFrame(r["sweep"])
        st.line_chart(sw.set_index("value")["probability_default"])
        viradas = sw[sw.decision != sw.decision.shift()].iloc[1:]
        if not viradas.empty:
            st.success("A decisão muda em: " +
                       ", ".join(f"{v:.4g} → {d}" for v, d in
                                 zip(viradas.value, viradas.decision)))
        else:
            st.info(f"A decisão permanece **{sw.decision.iloc[0]}** em toda a faixa.")

# --------------------------------------------------------------- modelo
with aba_modelo:
    m = get("/model/metrics")
    a, b, c, d = st.columns(4)
    a.metric("AUC (teste)", f"{m['champion']['auc']:.4f}",
             delta=f"{m['lift_vs_baseline']:+.4f} vs baseline")
    b.metric("KS", f"{m['champion']['ks']:.4f}")
    c.metric("Aprovação", f"{m['business']['approval_rate']:.1%}")
    d.metric("Threshold", f"{m['business']['threshold']:.3f}")

    st.subheader("Régua de custo — recalculada ao vivo")
    st.caption("Quanto custa aprovar um mau pagador, em relação a negar um bom?")
    razao = st.slider("Razão de custo (FN : FP)", 1.0, 30.0, 10.0, 0.5)
    ta = get("/model/threshold-analysis", cost_fn=razao, cost_fp=1.0)
    x, y, z = st.columns(3)
    x.metric("Threshold ótimo", f"{ta['best']['threshold']:.3f}")
    y.metric("Aprovação", f"{ta['best']['approval_rate']:.1%}")
    z.metric("Inadimplência da carteira aprovada",
             f"{ta['best']['default_rate_approved']:.2%}")
    pts = pd.DataFrame(ta["points"])
    st.line_chart(pts.set_index("threshold")[["cost"]])

    st.subheader("Onde o modelo é confiável")
    st.caption("Um grupo é fraqueza quando o IC da DIFERENÇA entre o AUC dele e o dos "
               "demais grupos do mesmo eixo exclui o zero (`fraqueza confirmada`). "
               "`sobrepõe o geral` compara com o AUC geral e está deprecado: o grupo é "
               "subconjunto do geral.")
    dim = st.radio("Segmento", ["age_band", "gender", "thin_file"], horizontal=True)
    fr = get_opcional("/model/fairness", by=dim)
    if fr is None:
        st.info(f"A dimensão `{dim}` não foi calculada nesta rodada. "
                "Rode `python Model/train.py` para gerar `artifacts/fairness.json`.")
        tabela = pd.DataFrame()
    else:
        tabela = pd.DataFrame(fr["groups"])
    if not tabela.empty:
        if "vs_referencia" in tabela:
            vs = tabela["vs_referencia"].apply(lambda v: v or {})
            tabela["diff"] = vs.apply(lambda v: v.get("diff"))
            tabela["p_value"] = vs.apply(lambda v: v.get("p_value"))
        cols = ["group", "n", "auc", "ci_low", "ci_high", "diff", "p_value",
                "fraqueza_confirmada", "overlaps_overall",
                "approval_rate", "default_rate"]
        st.dataframe(
            tabela[[c for c in cols if c in tabela]]
            .rename(columns={"group": "grupo", "auc": "AUC", "ci_low": "IC inf",
                             "ci_high": "IC sup", "diff": "Δ vs. demais",
                             "p_value": "p", "fraqueza_confirmada": "fraqueza",
                             "overlaps_overall": "sobrepõe o geral (depr.)",
                             "approval_rate": "aprovação",
                             "default_rate": "inadimplência real"}),
            use_container_width=True, hide_index=True)

    st.subheader("O que foi consertado")
    imp = get_opcional("/model/improvements")
    if imp is None:
        st.info("Nenhuma rodada oficial registrada ainda em "
                "`artifacts/improvement_log.json`.")
        st.stop()
    st.dataframe(
        pd.DataFrame(imp["runs"])[["tag", "n_features", "auc", "ks", "brier",
                                   "threshold", "approval_rate"]]
        .rename(columns={"tag": "rodada", "n_features": "features",
                         "threshold": "corte", "approval_rate": "aprovação"}),
        use_container_width=True, hide_index=True)
