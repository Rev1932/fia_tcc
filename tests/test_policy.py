"""
Trava a régua de decisão: quem vai para revisão humana, e por quê.

`low_confidence_groups` decide a largura da faixa cinza em produção. A troca do
critério (IC contra o AUC geral -> IC da diferença contra os demais grupos do
eixo) não pode nem passar a mandar gente nova para revisão sem medida, nem
deixar de mandar quem já ia.
"""
from __future__ import annotations

import json

import pytest

from MLOps.app.policy import (DEFAULT_BAND, LOW_CONFIDENCE_FACTOR, band_for,
                              decide, low_confidence_groups, policy_summary)

ANTIGO = {
    "overall": {"auc": 0.78, "ci_low": 0.775, "ci_high": 0.785},
    "dimensions": {"age_band": [
        {"group": "<25", "auc": 0.73, "ci_low": 0.70, "ci_high": 0.760},
        {"group": "25-35", "auc": 0.78, "ci_low": 0.772, "ci_high": 0.790},
    ]},
}

NOVO = {
    "overall": {"auc": 0.78, "ci_low": 0.775, "ci_high": 0.785},
    "criterio": {"descricao": "IC da diferença contra os demais grupos do eixo"},
    "dimensions": {"age_band": [
        {"group": "<25", "auc": 0.73, "ci_low": 0.70, "ci_high": 0.760,
         "vs_referencia": {"diff": -0.05, "diff_ci_low": -0.08, "diff_ci_high": -0.02},
         "fraqueza_confirmada": True},
        # IC do grupo abaixo do IC geral (o critério ANTIGO acusaria), mas a
        # diferença contra os demais cruza o zero: não é fraqueza.
        {"group": "65+", "auc": 0.74, "ci_low": 0.69, "ci_high": 0.770,
         "vs_referencia": {"diff": -0.02, "diff_ci_low": -0.07, "diff_ci_high": 0.03},
         "fraqueza_confirmada": False},
    ]},
}


def test_criterio_novo_usa_a_diferenca_e_nao_o_auc_geral():
    assert low_confidence_groups(NOVO) == {"age_band": ["<25"]}


def test_criterio_novo_absolve_grupo_que_o_antigo_condenaria():
    """`65+` tem IC abaixo do geral, mas não é pior que os demais do eixo."""
    assert "65+" in low_confidence_groups(ANTIGO | {"dimensions": {"age_band": [
        {"group": "65+", "auc": 0.74, "ci_low": 0.69, "ci_high": 0.770}]}})["age_band"]
    assert "65+" not in low_confidence_groups(NOVO)["age_band"]


def test_artefato_antigo_cai_no_criterio_anterior():
    """Rodada gravada antes da mudança não pode deixar a API sem régua."""
    assert low_confidence_groups(ANTIGO) == {"age_band": ["<25"]}


def test_faixa_cinza_dobra_para_quem_esta_confirmado():
    jovem = {"AGE_YEARS": 22.0, "BUREAU_COUNT": 3}
    maduro = {"AGE_YEARS": 40.0, "BUREAU_COUNT": 3}
    assert band_for(jovem, NOVO) == pytest.approx(DEFAULT_BAND * LOW_CONFIDENCE_FACTOR)
    assert band_for(maduro, NOVO) == pytest.approx(DEFAULT_BAND)


def test_faixa_cinza_do_jovem_nao_muda_com_a_troca_de_criterio():
    """A mitigação vigente é re-fundamentada, não removida."""
    jovem = {"AGE_YEARS": 22.0, "BUREAU_COUNT": 3}
    assert band_for(jovem, ANTIGO) == band_for(jovem, NOVO)


def test_resumo_descreve_a_regua_que_de_fato_aplica():
    assert "diferença" in policy_summary(NOVO)["criterio"]
    assert "IC geral" in policy_summary(ANTIGO)["criterio"]


def test_tres_faixas_continuam_particionando():
    thr, banda = 0.09, 0.10
    assert decide(0.01, thr, banda) == "APROVAR"
    assert decide(0.09, thr, banda) == "REVISAR"
    assert decide(0.50, thr, banda) == "NEGAR"
    assert decide(0.50, thr, 0.0) == "NEGAR"
