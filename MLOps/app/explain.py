"""
explain.py — Explicabilidade por cliente com SHAP.

Por que TreeExplainer com `tree_path_dependent`: não precisa de dataset de
background (guarda só a estrutura das árvores, poucos MB) e explicar uma
linha é uma passagem pelas árvores — poucos milissegundos. A alternativa
(`interventional`) exigiria milhares de linhas na memória do container.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from Model.predict import _prepare

RISK_LABEL = "aumenta risco"
SAFE_LABEL = "reduz risco"

SOURCE_PREFIXES = ("BUREAU_BB_", "BUREAU_", "PREV_", "POS_", "CC_", "INST_")


def source_table(name: str) -> str:
    for p in SOURCE_PREFIXES:
        if name.startswith(p):
            return "BUREAU_BALANCE" if p == "BUREAU_BB_" else p.rstrip("_")
    return "application"


@lru_cache(maxsize=1)
def _build_explainer(model_path: str):
    import shap
    from Model.predict import load_bundle
    bundle = load_bundle(model_path)
    return shap.TreeExplainer(bundle["model"], feature_perturbation="tree_path_dependent")


def get_explainer(model_path: str):
    return _build_explainer(model_path)


def clear_cache() -> None:
    _build_explainer.cache_clear()


def _binary_shap(raw, base) -> tuple[np.ndarray, float]:
    """Normaliza a saída do SHAP para (matriz (n, f), base escalar).

    O shap 0.52 com LightGBM binário já devolveu lista de 2 arrays, array
    (n, f) e array (n, f, 2) em versões diferentes — o próprio
    evaluation.ipynb do projeto tem esse warning capturado. Sem normalizar,
    o endpoint quebra dependendo da versão instalada.
    """
    if isinstance(raw, list):
        raw = raw[1] if len(raw) == 2 else raw[0]
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[:, :, -1]
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    if isinstance(base, (list, tuple, np.ndarray)):
        flat = np.ravel(np.asarray(base))
        base = float(flat[-1] if flat.size >= 2 else flat[0])
    return raw, float(base)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def score_band(p: float) -> str:
    for hi, label in ((0.05, "A (<5%)"), (0.10, "B (5-10%)"),
                      (0.20, "C (10-20%)"), (0.35, "D (20-35%)")):
        if p < hi:
            return label
    return "E (35%+)"


def explain_record(record: dict, bundle: dict, model_path: str, top: int = 10) -> dict:
    """Contribuições SHAP de UM cliente, em log-odds."""
    X = _prepare([record], bundle)
    explainer = get_explainer(model_path)
    shap_matrix, base_value = _binary_shap(
        explainer.shap_values(X), explainer.expected_value)
    row = np.asarray(shap_matrix[0], dtype=float)

    names = bundle["feature_names"]
    values = X.iloc[0]

    contribs = []
    for i, (name, sv) in enumerate(zip(names, row)):
        if not np.isfinite(sv) or sv == 0:
            continue
        v = values.iloc[i]
        if pd.isna(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        contribs.append({
            "feature": name,
            "value": v,
            "shap_value": float(sv),
            "abs_shap": abs(float(sv)),
            "effect": RISK_LABEL if sv > 0 else SAFE_LABEL,
            "source_table": source_table(name),
        })
    contribs.sort(key=lambda c: -c["abs_shap"])

    risco = [c for c in contribs if c["shap_value"] > 0][:top]
    protetor = [c for c in contribs if c["shap_value"] < 0][:top]
    for rank, c in enumerate(risco, 1):
        c["rank"] = rank
    for rank, c in enumerate(protetor, 1):
        c["rank"] = rank

    mostrados = {id(c) for c in risco + protetor}
    resto = float(sum(c["shap_value"] for c in contribs if id(c) not in mostrados))

    # base + soma dos SHAP reconstrói a log-odds do modelo. Devolver isso
    # explicitamente responde "essa explicação é fiel ao modelo?" com número.
    margin = base_value + float(row.sum())
    reconstruida = _sigmoid(margin)
    # O SHAP explica o modelo CRU. Quando há calibração, o score servido passa
    # pela isotônica — que é monotônica, então a ordem e o sinal das
    # contribuições continuam valendo, mas a probabilidade final difere.
    prob_crua = float(bundle["model"].predict_proba(X)[:, 1][0])
    from Model.predict import score_matrix
    prob = float(score_matrix(X, bundle)[0])

    return {
        "probability_default": prob,
        "raw_probability": prob_crua,
        "base_value": base_value,
        "base_probability": _sigmoid(base_value),
        "top_risk_drivers": risco,
        "top_protective_factors": protetor,
        "sum_other_features": resto,
        "consistency_check": {
            # Confere contra o modelo CRU, que é o que o SHAP explica.
            "reconstructed_probability": reconstruida,
            "model_probability": prob_crua,
            "max_abs_error": abs(reconstruida - prob_crua),
        },
    }


def _fmt(value) -> str:
    if value is None:
        return "ausente"
    if isinstance(value, float):
        return f"{value:,.2f}".replace(",", "·").replace(".", ",").replace("·", ".")
    return str(value)


def narrate(result: dict, threshold: float, decision: str,
            sk_id: int | None = None) -> str:
    """Resumo em português do caso — é o que um analista de crédito receberia
    junto com a decisão (a exigência de governança citada no MLOps/Readme)."""
    quem = f"Cliente {sk_id}" if sk_id is not None else "Cliente informado"
    p = result["probability_default"]
    frase = (f"{quem}: risco estimado de {p:.1%}, "
             f"{'acima' if p >= threshold else 'abaixo'} do corte de {threshold:.1%} "
             f"→ {decision}.")

    def lista(cs):
        return "; ".join(f"{c['feature']} = {_fmt(c['value'])} "
                         f"({c['shap_value']:+.3f})" for c in cs[:3])

    if result["top_risk_drivers"]:
        frase += " Principais fatores de risco: " + lista(result["top_risk_drivers"]) + "."
    if result["top_protective_factors"]:
        frase += " Fatores favoráveis: " + lista(result["top_protective_factors"]) + "."
    return frase
