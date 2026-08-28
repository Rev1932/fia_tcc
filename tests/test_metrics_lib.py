"""
Trava a semântica das métricas ANTES de qualquer retreino.

Motivo: o threshold de negócio congelado em artifacts/ depende de detalhes
de implementação (grid de 99 pontos, desempate estrito). Se alguém
"melhorar" business_threshold, o número do slide muda e a divergência que
este ciclo está matando volta a existir. Estes testes impedem isso.
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, roc_auc_score

from Model.metrics_lib import (
    auc_bootstrap_ci,
    auc_diff_bootstrap,
    auc_within_between,
    brier_score,
    business_threshold,
    calibration_points,
    confusion_at,
    cost_at,
    decile_table,
    derived_metrics,
    ks_statistic,
    roc_points,
    threshold_sweep,
)


@pytest.fixture(scope="module")
def toy():
    """8% de eventos, score correlacionado com o target — imita a base real."""
    rng = np.random.default_rng(42)
    n = 4000
    y = (rng.random(n) < 0.08).astype(int)
    score = np.clip(rng.beta(2, 8, n) + 0.25 * y, 0, 1)
    return y, score


# ---------------------------------------------------------------- semântica

def test_confusion_at_bate_com_sklearn(toy):
    y, s = toy
    cm = confusion_at(y, s, 0.3)
    tn, fp, fn, tp = confusion_matrix(y, (s >= 0.3).astype(int)).ravel()
    assert (cm["tn"], cm["fp"], cm["fn"], cm["tp"]) == (tn, fp, fn, tp)


def test_aprovado_e_score_abaixo_do_threshold(toy):
    """A convenção do projeto: aprovado = risco baixo = score < threshold."""
    y, s = toy
    cm = confusion_at(y, s, 0.4)
    aprovados = cm["tn"] + cm["fn"]
    assert aprovados == int((s < 0.4).sum())
    assert derived_metrics(cm)["approval_rate"] == pytest.approx(float((s < 0.4).mean()))


def test_fn_e_aprovar_mau_pagador(toy):
    """FN é o erro caro: aprovou quem deu default."""
    y, s = toy
    cm = confusion_at(y, s, 0.4)
    assert cm["fn"] == int(((s < 0.4) & (y == 1)).sum())
    assert cm["fp"] == int(((s >= 0.4) & (y == 0)).sum())


def test_ks_statistic_confere_com_calculo_direto(toy):
    y, s = toy
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    assert ks_statistic(y, s) == pytest.approx(float(np.max(tpr - fpr)))


# ------------------------------------------------- equivalência com o sweep

def test_business_threshold_equivale_ao_minimo_do_sweep(toy):
    """A API usa threshold_sweep; o treino usa business_threshold.
    Os dois PRECISAM concordar, senão o número da API diverge do slide."""
    y, s = toy
    for cfn, cfp in [(1.0, 0.1), (1.0, 1.0), (10.0, 1.0), (1.0, 0.5)]:
        thr, approval = business_threshold(y, s, cfn, cfp)
        pts = threshold_sweep(y, s, cfn, cfp, n_points=99, lo=0.01, hi=0.99)
        best = min(pts, key=lambda r: r["cost"])   # min() também pega o primeiro
        assert best["threshold"] == pytest.approx(thr), f"divergiu em {cfn}/{cfp}"
        assert best["approval_rate"] == pytest.approx(approval)


def test_business_threshold_mantem_grid_de_99_pontos():
    """Congelado de propósito: mudar o grid muda o threshold oficial."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    s = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.7, 0.3, 0.05])
    thr, _ = business_threshold(y, s, 1.0, 0.1)
    grid = np.linspace(0.01, 0.99, 99)
    assert np.isclose(grid, thr).any(), "threshold fora do grid de 99 pontos"


def test_desempate_pega_o_primeiro_minimo():
    """Score constante: todos os thresholds acima dele têm o mesmo custo.
    O primeiro (menor) tem de vencer — é o desempate estrito `<`."""
    y = np.array([0, 0, 0, 1])
    s = np.full(4, 0.5)
    thr, _ = business_threshold(y, s, 1.0, 0.1)
    # abaixo de 0.5 todos são negados (fp=3, custo 0.3); a partir de 0.51
    # todos são aprovados (fn=1, custo 1.0). O mínimo é o primeiro do grid.
    assert thr == pytest.approx(0.01)


# ------------------------------------------------------- custo e monotonia

def test_custo_usa_fn_e_fp_com_os_pesos_certos(toy):
    y, s = toy
    cm = confusion_at(y, s, 0.3)
    assert cost_at(cm, 1.0, 0.1) == pytest.approx(cm["fn"] + 0.1 * cm["fp"])


def test_fn_mais_caro_nao_aumenta_o_threshold(toy):
    """Monotonicidade econômica: se aprovar um mau pagador fica mais caro,
    o corte tem de ficar mais rigoroso (menor), nunca mais frouxo."""
    y, s = toy
    thr_barato, _ = business_threshold(y, s, 1.0, 1.0)
    thr_caro, _ = business_threshold(y, s, 20.0, 1.0)
    assert thr_caro <= thr_barato


def test_threshold_extremos(toy):
    y, s = toy
    assert derived_metrics(confusion_at(y, s, 0.0))["approval_rate"] == 0.0
    assert derived_metrics(confusion_at(y, s, 1.01))["approval_rate"] == 1.0


# ------------------------------------------------------------------ curvas

def test_roc_points_reduz_e_preserva_auc(toy):
    y, s = toy
    r = roc_points(y, s, max_points=50)
    assert r["auc"] == pytest.approx(float(roc_auc_score(y, s)))
    assert r["gini"] == pytest.approx(2 * r["auc"] - 1)
    assert len(r["points"]) <= 50
    fprs = [p["fpr"] for p in r["points"]]
    assert fprs == sorted(fprs), "ROC precisa sair ordenada por fpr"


def test_decile_table_particiona_a_base(toy):
    y, s = toy
    rows = decile_table(y, s, q=10)
    assert len(rows) == 10
    assert sum(r["n"] for r in rows) == len(y)
    assert sum(r["events"] for r in rows) == int(y.sum())
    assert rows[-1]["cum_event_pct"] == pytest.approx(1.0)
    # decil 1 = maior risco; a taxa de evento deve cair ao longo dos decis
    assert rows[0]["event_rate"] > rows[-1]["event_rate"]


# -------------------------------------------------------------- calibração

def test_brier_de_previsao_perfeita_e_zero():
    y = np.array([0, 1, 0, 1])
    assert brier_score(y, y.astype(float)) == 0.0


def test_calibration_points_cobrem_toda_a_base(toy):
    y, s = toy
    pts = calibration_points(y, s, n_bins=10)
    assert sum(p["n"] for p in pts) == len(y)
    assert all(0 <= p["mean_predicted"] <= 1 for p in pts)


# ---------------------------------------------------------------- bootstrap

def test_auc_ci_contem_o_ponto_e_encolhe_com_n(toy):
    y, s = toy
    grande = auc_bootstrap_ci(y, s, n_boot=200)
    assert grande["ci_low"] <= grande["auc"] <= grande["ci_high"]
    pequeno = auc_bootstrap_ci(y[:300], s[:300], n_boot=200)
    assert pequeno["ci_width"] > grande["ci_width"], \
        "amostra menor tem de ter IC mais largo — é a base do argumento sobre <25 anos"


def test_auc_ci_com_classe_unica_nao_quebra():
    y = np.zeros(50, dtype=int)
    out = auc_bootstrap_ci(y, np.random.default_rng(0).random(50), n_boot=20)
    assert out["auc"] is None and "note" in out


# ---------------------------------------------------------------------- PSI

def test_psi_de_distribuicoes_identicas_e_zero():
    from Model.metrics_lib import psi
    rng = np.random.default_rng(1)
    x = rng.normal(size=5000)
    out = psi(x, x.copy())
    assert out["psi"] == pytest.approx(0.0, abs=1e-9)
    assert out["faixa"] == "estável"


def test_psi_cresce_com_o_deslocamento():
    """É a propriedade que torna o PSI útil: quanto mais a distribuição anda,
    maior o índice."""
    from Model.metrics_lib import psi
    rng = np.random.default_rng(2)
    ref = rng.normal(size=8000)
    valores = [psi(ref, rng.normal(loc=d, size=8000))["psi"] for d in (0.0, 0.3, 0.8, 1.5)]
    assert valores == sorted(valores)
    assert valores[0] < 0.10 < valores[-1]


def test_psi_classifica_nas_faixas_de_mercado():
    from Model.metrics_lib import psi
    rng = np.random.default_rng(3)
    ref = rng.normal(size=8000)
    assert psi(ref, rng.normal(size=8000))["faixa"] == "estável"
    assert psi(ref, rng.normal(loc=2.0, size=8000))["faixa"] == "mudança relevante"


def test_psi_com_variavel_constante_nao_quebra():
    from Model.metrics_lib import psi
    out = psi(np.ones(500), np.ones(500))
    assert out["psi"] is None and "note" in out


def test_psi_bins_somam_um():
    from Model.metrics_lib import psi
    rng = np.random.default_rng(4)
    out = psi(rng.normal(size=3000), rng.normal(size=3000))
    assert sum(b["pct_esperado"] for b in out["bins"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(b["pct_observado"] for b in out["bins"]) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------- comparação entre segmentos

@pytest.fixture(scope="module")
def segmentado():
    """Seis grupos com o mesmo sinal, menos o grupo 'A', degradado de propósito."""
    rng = np.random.default_rng(11)
    n = 6000
    grupos = rng.choice(list("ABCDEF"), n)
    y = (rng.random(n) < 0.09).astype(int)
    s = rng.normal(0, 1, n) + 1.2 * y
    m = grupos == "A"
    s[m] = rng.normal(0, 1, m.sum()) + 0.45 * y[m]
    return y, s, grupos


def test_decomposicao_reconstroi_o_auc_agregado(segmentado):
    """A decomposição é identidade, não aproximação: se não fecha, está errada."""
    y, s, g = segmentado
    d = auc_within_between(y, s, g)
    assert d["auc_overall"] == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    recomposto = d["w_within"] * d["auc_within"] + d["w_between"] * d["auc_between"]
    assert recomposto == pytest.approx(d["auc_overall"], abs=1e-9)
    assert d["w_within"] + d["w_between"] == pytest.approx(1.0)


def test_decomposicao_sem_sinal_entre_grupos():
    """Grupos com distribuições idênticas não geram vantagem entre faixas."""
    rng = np.random.default_rng(3)
    n = 4000
    g = rng.choice(list("XYZ"), n)
    y = (rng.random(n) < 0.3).astype(int)
    s = rng.normal(0, 1, n) + 0.8 * y
    d = auc_within_between(y, s, g)
    assert d["auc_between"] == pytest.approx(d["auc_within"], abs=0.03)


def test_diferenca_detecta_grupo_degradado(segmentado):
    y, s, g = segmentado
    r = auc_diff_bootstrap(y, s, g, "A", n_boot=300)
    assert r["diff"] < 0 and r["pior_que_referencia"] is True
    assert r["diff_ci_low"] <= r["diff"] <= r["diff_ci_high"]
    assert r["p_value"] < 0.05


def test_diferenca_nao_acusa_grupo_normal(segmentado):
    y, s, g = segmentado
    r = auc_diff_bootstrap(y, s, g, "C", n_boot=300)
    assert r["pior_que_referencia"] is False
    assert r["diff_ci_low"] <= 0 <= r["diff_ci_high"]


def test_diferenca_pesos_somam_um_e_excluem_o_alvo(segmentado):
    y, s, g = segmentado
    r = auc_diff_bootstrap(y, s, g, "A", n_boot=100)
    assert "A" not in r["reference_weights"]
    assert sum(r["reference_weights"].values()) == pytest.approx(1.0)


def test_diferenca_e_deterministica(segmentado):
    y, s, g = segmentado
    a = auc_diff_bootstrap(y, s, g, "B", n_boot=100, random_state=7)
    b = auc_diff_bootstrap(y, s, g, "B", n_boot=100, random_state=7)
    assert a["diff_ci_low"] == b["diff_ci_low"]


def test_diferenca_com_grupo_degenerado_nao_quebra():
    """Grupo de uma classe só devolve nota, como auc_bootstrap_ci."""
    y = np.array([0] * 40 + [1] * 40 + [0] * 40)
    s = np.random.default_rng(0).random(120)
    g = np.array(["A"] * 40 + ["A"] * 40 + ["B"] * 40, dtype=object)
    out = auc_diff_bootstrap(y, s, g, "B", n_boot=20)
    assert out.get("diff") is None or "note" in out
