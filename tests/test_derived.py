"""
Garante que Model/derived.py reproduz EXATAMENTE o que o pipeline gravou na ABT.

Sem este teste, as fórmulas do what-if e as do DataPipeline divergiriam em
silêncio na primeira alteração — e /simulate passaria a mostrar um cenário
que o modelo nunca veria em produção.

Pulado quando a ABT não está no disco (ambiente sem os dados).
"""
from __future__ import annotations

import math

import pytest

from Model.derived import DERIVED_NAMES, apply_changes, recompute

duckdb = pytest.importorskip("duckdb")

# Caminho fixo, e NÃO MLOps.app.settings: a suíte da API repõe as variáveis
# HC_* para apontar a uma base sintética de 300 linhas. Este teste precisa da
# ABT de verdade, que é o que o pipeline gravou.
from pathlib import Path  # noqa: E402

ABT = Path(__file__).resolve().parents[1] / "Dados" / "abt.parquet"

pytestmark = pytest.mark.skipif(
    not ABT.exists(),
    reason="Dados/abt.parquet ausente (rode DataPipeline/to_parquet.py)",
)


@pytest.fixture(scope="module")
def amostra():
    con = duckdb.connect()
    # Amostra determinística por módulo (~310 clientes espalhados por toda a
    # base). `USING SAMPLE` sem semente muda a cada execução e deixaria o teste
    # intermitente: some hoje, volta amanhã com outro cliente.
    df = con.execute(
        f"SELECT * FROM read_parquet('{ABT}') "
        f"WHERE SK_ID_CURR % 997 = 0 ORDER BY SK_ID_CURR"
    ).df()
    con.close()
    return df.to_dict("records")


def _perto(a, b, tol=1e-6) -> bool:
    a_nulo = a is None or (isinstance(a, float) and math.isnan(a))
    b_nulo = b is None or (isinstance(b, float) and math.isnan(b))
    if a_nulo or b_nulo:
        return a_nulo and b_nulo
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


@pytest.mark.parametrize("nome", DERIVED_NAMES)
def test_derivada_bate_com_a_abt(amostra, nome):
    """Cada derivada recalculada tem de bater com a coluna gravada na ABT."""
    conferidas = divergentes = 0
    for rec in amostra:
        if nome not in rec:
            pytest.skip(f"{nome} não existe na ABT desta rodada")
        esperado = rec[nome]
        obtido = recompute(rec).get(nome)
        conferidas += 1
        if not _perto(obtido, esperado):
            divergentes += 1
            if divergentes == 1:
                primeiro = (rec.get("SK_ID_CURR"), esperado, obtido)
    assert conferidas > 0
    assert divergentes == 0, (
        f"{nome}: {divergentes}/{conferidas} divergiram. "
        f"Primeiro caso (id, ABT, recalculado): {primeiro}"
    )


def test_apply_changes_propaga_para_as_derivadas(amostra):
    """Mudar o crédito precisa mover comprometimento de renda e prazo."""
    rec = next(r for r in amostra
               if r.get("AMT_INCOME_TOTAL") and r.get("AMT_CREDIT") and r.get("AMT_ANNUITY"))
    novo = apply_changes(rec, {"AMT_CREDIT": float(rec["AMT_CREDIT"]) * 3})

    assert _perto(novo["CREDIT_INCOME_RATIO"],
                  novo["AMT_CREDIT"] / novo["AMT_INCOME_TOTAL"])
    assert novo["CREDIT_INCOME_RATIO"] > rec["CREDIT_INCOME_RATIO"]
    assert novo["CREDIT_TERM"] < rec["CREDIT_TERM"]


def test_mudar_ext_source_move_a_media(amostra):
    """EXT_SOURCE_MEAN é a feature mais importante do modelo: precisa
    acompanhar qualquer mudança nos scores externos individuais."""
    rec = next(r for r in amostra if r.get("EXT_SOURCE_2") is not None)
    novo = apply_changes(rec, {"EXT_SOURCE_2": 0.99})
    assert novo["EXT_SOURCE_MAX"] == pytest.approx(0.99)
    assert novo["EXT_SOURCE_MEAN"] > rec["EXT_SOURCE_MEAN"]


def test_denominador_zero_vira_nulo():
    """`replace(0, np.nan)` no pipeline: renda zero não pode virar infinito."""
    out = recompute({"AMT_CREDIT": 1000.0, "AMT_INCOME_TOTAL": 0.0,
                     "CREDIT_INCOME_RATIO": None}, only_if_present=False)
    assert out["CREDIT_INCOME_RATIO"] is None
