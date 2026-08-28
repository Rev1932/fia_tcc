"""
Testes da API. Não tocam nos dados reais — ver tests/conftest.py.
"""
from __future__ import annotations

import pytest


# ------------------------------------------------------------------- saúde

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok"
    assert j["model_loaded"] and j["data_loaded"] and j["artifacts_loaded"]
    assert j["run_id"] == "teste-0001"


MODULOS = ("MLOps.app.db", "MLOps.app.artifacts", "MLOps.app.explain",
           "MLOps.app.routers.clients", "MLOps.app.routers.stats",
           "MLOps.app.routers.model", "MLOps.app.routers.scoring", "MLOps.app.api")


def _recarrega():
    import importlib
    from MLOps.app import settings
    importlib.reload(settings)
    for m in MODULOS:
        importlib.reload(importlib.import_module(m))


def test_health_devolve_503_sem_modelo(ambiente, monkeypatch):
    """A regressão que quebrava o healthcheck do compose: a API antiga
    respondia 200 mesmo sem modelo, então o container ficava 'healthy'
    sem conseguir pontuar nada."""
    import os
    from fastapi.testclient import TestClient

    original = os.environ["HC_MODEL_PATH"]
    monkeypatch.setenv("HC_MODEL_PATH", str(ambiente["root"] / "nao_existe.joblib"))
    try:
        _recarrega()
        from MLOps.app.api import create_app
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            r = c.get("/health")
            assert r.status_code == 503
            assert r.json()["status"] == "degraded"
            assert "model" in r.json()["errors"]
    finally:
        # Sem restaurar, os módulos ficariam apontando para um modelo
        # inexistente e TODOS os testes seguintes falhariam.
        os.environ["HC_MODEL_PATH"] = original
        _recarrega()


# ----------------------------------------------------------------- clientes

def test_clients_pagina(client):
    r = client.get("/clients", params={"page_size": 10})
    assert r.status_code == 200
    j = r.json()
    assert j["meta"]["total"] == 300
    assert len(j["items"]) == 10
    assert j["meta"]["has_next"] and not j["meta"]["has_prev"]


def test_paginas_sao_disjuntas_e_total_estavel(client):
    p1 = client.get("/clients", params={"page": 1, "page_size": 25, "sort": "SK_ID_CURR"}).json()
    p2 = client.get("/clients", params={"page": 2, "page_size": 25, "sort": "SK_ID_CURR"}).json()
    ids1 = {i["SK_ID_CURR"] for i in p1["items"]}
    ids2 = {i["SK_ID_CURR"] for i in p2["items"]}
    assert not (ids1 & ids2)
    assert p1["meta"]["total"] == p2["meta"]["total"]


def test_filtro_de_idade_e_respeitado(client):
    j = client.get("/clients", params={"age_min": 40, "page_size": 100}).json()
    assert j["items"], "filtro não devia zerar a base"
    assert all(i["AGE_YEARS"] >= 40 for i in j["items"])


def test_thin_file_significa_sem_bureau(client):
    j = client.get("/clients", params={"thin_file": True, "page_size": 100}).json()
    assert j["items"]
    assert all(i["BUREAU_COUNT"] is None for i in j["items"])
    outros = client.get("/clients", params={"thin_file": False, "page_size": 100}).json()
    assert all(i["BUREAU_COUNT"] is not None for i in outros["items"])


def test_thin_file_particiona_a_base(client):
    total = client.get("/clients", params={"page_size": 1}).json()["meta"]["total"]
    a = client.get("/clients", params={"thin_file": True, "page_size": 1}).json()["meta"]["total"]
    b = client.get("/clients", params={"thin_file": False, "page_size": 1}).json()["meta"]["total"]
    assert a + b == total


def test_fields_limita_as_colunas(client):
    j = client.get("/clients", params={"fields": "AMT_CREDIT", "page_size": 3}).json()
    assert j["items"]
    for item in j["items"]:
        assert set(item) <= {"SK_ID_CURR", "AMT_CREDIT", "decision"}, \
            "?fields= não pode devolver as demais colunas como null"


def test_coluna_desconhecida_da_400_com_dica(client):
    r = client.get("/clients", params={"fields": "NAO_EXISTE"})
    assert r.status_code == 400
    assert "/meta/columns" in r.json()["error"]["message"]


def test_cliente_inexistente_404(client):
    assert client.get("/clients/999999").status_code == 404


def test_ficha_do_cliente(client):
    j = client.get("/clients/100001").json()
    assert j["sk_id_curr"] == 100001
    assert "AMT_CREDIT" in j["financeiro"]
    assert j["features"] is None
    completo = client.get("/clients/100001", params={"include": "all"}).json()
    assert len(completo["features"]) > 20


def test_decision_bate_com_o_threshold(client, ambiente):
    thr = ambiente["threshold"]
    j = client.get("/clients", params={"decision": "NEGAR", "page_size": 50}).json()
    assert all(i["proba_champion"] >= thr for i in j["items"])


def test_score_recompute_concorda_com_o_batch(client):
    """Prova que o artefato em disco e o modelo servido são o mesmo modelo.

    A tolerância é 1e-4 porque `Model.predict.predict` arredonda a
    probabilidade em 4 casas antes de devolver.
    """
    j = client.get("/clients/100001/score", params={"recompute": True}).json()
    assert j["live_probability"] is not None
    assert j["agreement_error"] == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------- estatísticas

def test_overview_bate_com_a_soma_dos_grupos(client):
    o = client.get("/stats/overview").json()
    d = client.get("/stats/default-rate", params={"by": "gender", "min_count": 1}).json()
    assert sum(b["n"] for b in d["buckets"]) == o["n_clients"]
    assert d["overall_default_rate"] == pytest.approx(o["default_rate"])


def test_filtros_valem_tambem_no_stats(client):
    todos = client.get("/stats/overview").json()["n_clients"]
    jovens = client.get("/stats/overview", params={"age_max": 30}).json()["n_clients"]
    assert 0 < jovens < todos


def test_dimensao_invalida_da_400(client):
    r = client.get("/stats/default-rate", params={"by": "nao_existe"})
    assert r.status_code == 400
    assert "Dimensão desconhecida" in r.json()["error"]["message"]


def test_histograma_cobre_todos_os_nao_nulos(client):
    j = client.get("/stats/distribution", params={"feature": "AGE_YEARS", "bins": 8}).json()
    assert j["type"] == "numeric"
    assert sum(b["count"] for b in j["bins"]) == j["stats"]["count"]


def test_distribuicao_categorica(client):
    j = client.get("/stats/distribution", params={"feature": "CODE_GENDER"}).json()
    assert j["type"] == "categorical"
    assert sum(c["n"] for c in j["categories"]) == j["n_total"]


def test_missing_reporta_coluna_com_nulo_conhecido(client):
    j = client.get("/stats/missing", params={"top": 50}).json()
    taxas = {i["column"]: i["missing_rate"] for i in j["items"]}
    assert taxas.get("EXT_SOURCE_1", 0) > 0.3      # ~56% nulo por construção


# --------------------------------------------------------------------- modelo

def test_metrics_traz_lift_e_run(client):
    j = client.get("/model/metrics").json()
    assert j["run"]["run_id"] == "teste-0001"
    assert j["lift_vs_baseline"] == pytest.approx(
        j["champion"]["auc"] - j["baseline"]["auc"])


def test_metrics_expoe_o_modelo_servido(client):
    """O bloco `served` é a fonte dos números de capa. Sem ele, volta a haver
    ambiguidade entre o modelo cru e o calibrado."""
    j = client.get("/model/metrics").json()
    assert j["served"] is not None
    assert j["served"]["model"]
    assert 0 < j["served"]["auc"] <= 1
    assert j["served"]["threshold"] == j["business"]["threshold"]


def test_roc_e_monotonica(client):
    pts = client.get("/model/roc").json()["points"]
    fprs = [p["fpr"] for p in pts]
    assert fprs == sorted(fprs)
    assert pts[0]["fpr"] == 0.0 and pts[-1]["fpr"] == pytest.approx(1.0)


def test_threshold_zero_nega_tudo_e_um_aprova_tudo(client):
    assert client.get("/model/confusion-matrix",
                      params={"threshold": 0.0}).json()["approval_rate"] == 0.0
    assert client.get("/model/confusion-matrix",
                      params={"threshold": 1.0}).json()["approval_rate"] == 1.0


def test_best_e_de_fato_o_minimo_do_sweep(client):
    j = client.get("/model/threshold-analysis").json()
    assert j["best"]["cost"] == min(p["cost"] for p in j["points"])


def test_fn_mais_caro_nao_afrouxa_o_corte(client):
    """Monotonicidade econômica: se aprovar um mau pagador ficar mais caro,
    o corte tem de ficar mais rigoroso — nunca mais frouxo."""
    barato = client.get("/model/threshold-analysis",
                        params={"cost_fn": 1, "cost_fp": 1}).json()
    caro = client.get("/model/threshold-analysis",
                      params={"cost_fn": 20, "cost_fp": 1}).json()
    assert caro["best"]["threshold"] <= barato["best"]["threshold"]
    assert caro["cost_ratio"] == 20.0


def test_fairness_marca_sobreposicao_de_intervalo(client):
    j = client.get("/model/fairness", params={"by": "gender"}).json()
    assert j["groups"]
    for g in j["groups"]:
        assert g["ci_low"] <= g["auc"] <= g["ci_high"]
        assert isinstance(g["overlaps_overall"], bool)


def test_fairness_dimensao_ausente_404(client):
    assert client.get("/model/fairness", params={"by": "renda"}).status_code == 404


def test_improvements_compara_as_rodadas(client):
    j = client.get("/model/improvements").json()
    assert [r["tag"] for r in j["runs"]] == ["v1", "v2"]
    assert j["deltas"]["auc"] == pytest.approx(j["runs"][1]["auc"] - j["runs"][0]["auc"])


def test_decision_policy_particiona_a_carteira(client):
    j = client.get("/model/decision-policy").json()
    pcts = sum(v["pct"] for v in j["distribuicao"].values())
    assert pcts == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------- predição e simulação

def test_predict_aceita_payload_parcial(client):
    r = client.post("/predict", json={"records": [
        {"AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000, "EXT_SOURCE_2": 0.5}]})
    assert r.status_code == 200
    p = r.json()["predictions"][0]
    assert 0 < p["probability_default"] < 1
    assert p["decision"] in {"APROVAR", "REVISAR", "NEGAR"}


def test_predict_sinaliza_campo_desconhecido(client):
    p = client.post("/predict", json={"records": [{"AMT_CREDIT": 1, "XPTO": 9}]}).json()
    assert p["predictions"][0]["unknown_features"] == ["XPTO"]


def test_predict_rejeita_lista_vazia(client):
    assert client.post("/predict", json={"records": []}).status_code == 422


def test_predict_rejeita_lote_grande_demais(client):
    from MLOps.app import settings
    grande = [{"AMT_CREDIT": 1}] * (settings.MAX_PREDICT_RECORDS + 1)
    assert client.post("/predict", json={"records": grande}).status_code == 422


def test_simulate_propaga_para_as_derivadas(client):
    """Triplicar o crédito precisa mover o score: se não mover, é sinal de que
    CREDIT_INCOME_RATIO e CREDIT_TERM não foram recalculados."""
    j = client.post("/simulate", json={
        "sk_id_curr": 100001, "changes": {"AMT_CREDIT": 3_000_000}}).json()
    assert j["simulated"] is not None
    assert abs(j["delta_probability"]) > 1e-6


def test_simulate_exige_exatamente_uma_origem(client):
    assert client.post("/simulate", json={"changes": {}}).status_code == 422
    assert client.post("/simulate", json={
        "sk_id_curr": 100001, "record": {}, "changes": {}}).status_code == 422


def test_sweep_devolve_um_ponto_por_valor(client):
    j = client.post("/simulate", json={
        "sk_id_curr": 100001,
        "sweep": {"feature": "EXT_SOURCE_2", "start": 0.0, "stop": 1.0, "steps": 6}}).json()
    assert len(j["sweep"]) == 6
    assert all(0 <= p["probability_default"] <= 1 for p in j["sweep"])


def test_explain_e_fiel_ao_modelo(client):
    """base_value + soma dos SHAP tem de reconstruir a probabilidade."""
    j = client.get("/clients/100001/explain", params={"top": 5}).json()
    c = j["consistency_check"]
    assert c["max_abs_error"] < 1e-6
    assert j["narrative"].startswith("Cliente 100001")
    for d in j["top_risk_drivers"]:
        assert d["shap_value"] > 0
    for d in j["top_protective_factors"]:
        assert d["shap_value"] < 0


# ------------------------------------------------------------------- drift

def test_psi_compara_o_score_entre_fatias(client):
    j = client.get("/model/psi", params={"referencia": "train", "comparado": "test"}).json()
    assert j["n_features"] == 1
    item = j["items"][0]
    assert item["feature"] == "proba_champion"
    assert item["psi"] is not None and item["psi"] >= 0
    assert item["faixa"] in {"estável", "atenção", "mudança relevante"}


def test_psi_aceita_lista_de_variaveis(client):
    j = client.get("/model/psi", params={
        "features": "AGE_YEARS,AMT_CREDIT,EXT_SOURCE_2", "top": 10}).json()
    assert {i["feature"] for i in j["items"]} == {"AGE_YEARS", "AMT_CREDIT", "EXT_SOURCE_2"}
    assert sum(j["resumo"].values()) == j["n_features"]


def test_psi_rejeita_fatias_iguais(client):
    r = client.get("/model/psi", params={"referencia": "test", "comparado": "test"})
    assert r.status_code == 400


def test_psi_rejeita_coluna_desconhecida(client):
    assert client.get("/model/psi", params={"features": "NAO_EXISTE"}).status_code == 400


def test_improvements_ignora_rodada_rejeitada_no_latest(client, ambiente):
    """Uma rodada rejeitada fica no log — mas não pode virar o 'depois' da
    comparação, senão a API apresentaria como atual algo que foi descartado."""
    import json
    lp = ambiente["artifacts"] / "improvement_log.json"
    original = lp.read_text()
    log = json.loads(original)
    log["runs"].append({**log["runs"][-1], "run_id": "teste-9999", "tag": "v9-ruim",
                        "auc": 0.99, "status": "rejeitada", "motivo": "piorou o alvo"})
    lp.write_text(json.dumps(log))
    try:
        client.post("/admin/reload")
        j = client.get("/model/improvements").json()
        assert j["latest_tag"] == "v2", "o 'depois' tem de ser a última ACEITA"
        assert any(r["tag"] == "v9-ruim" for r in j["rejeitadas"])
    finally:
        lp.write_text(original)
        client.post("/admin/reload")


def test_baseline_no_treino_da_404_explicativo(client):
    """Limitação conhecida e aceita: o baseline não é pontuado na fatia de
    treino. O erro precisa dizer isso e apontar a alternativa."""
    r = client.get("/model/confusion-matrix",
                   params={"model": "baseline", "split": "train"})
    assert r.status_code == 404
    msg = r.json()["error"]["message"]
    assert "split=valid" in msg or "split=test" in msg


# ---------------------------------------------------------------- lote CSV

def _csv(linhas: list[dict]) -> bytes:
    import csv, io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(linhas[0]))
    w.writeheader()
    w.writerows(linhas)
    return buf.getvalue().encode("utf-8")


def test_predict_csv_devolve_csv_pontuado(client):
    dados = _csv([
        {"SK_ID_CURR": 1, "AMT_CREDIT": 500000, "AMT_INCOME_TOTAL": 150000,
         "EXT_SOURCE_2": 0.5, "AGE_YEARS": 35},
        {"SK_ID_CURR": 2, "AMT_CREDIT": 900000, "AMT_INCOME_TOTAL": 90000,
         "EXT_SOURCE_2": 0.1, "AGE_YEARS": 23},
    ])
    r = client.post("/predict/csv", files={"arquivo": ("fila.csv", dados, "text/csv")})
    assert r.status_code == 200
    assert r.headers["X-Linhas-Pontuadas"] == "2"
    assert "attachment" in r.headers["content-disposition"]

    import csv, io
    linhas = list(csv.DictReader(io.StringIO(r.text)))
    assert len(linhas) == 2
    for ln in linhas:
        assert 0 <= float(ln["probability_default"]) <= 1
        assert ln["decision"] in {"APROVAR", "REVISAR", "NEGAR"}
        assert ln["SK_ID_CURR"] in {"1", "2"}, "as colunas originais têm de ser preservadas"


def test_predict_csv_rejeita_cabecalho_sem_feature(client):
    dados = _csv([{"coluna_qualquer": 1, "outra": 2}])
    r = client.post("/predict/csv", files={"arquivo": ("x.csv", dados, "text/csv")})
    assert r.status_code == 400
    assert "/meta/columns" in r.json()["error"]["message"]


def test_predict_csv_rejeita_arquivo_vazio(client):
    r = client.post("/predict/csv",
                    files={"arquivo": ("x.csv", b"AMT_CREDIT\n", "text/csv")})
    assert r.status_code == 400


def test_predict_csv_recalcula_derivadas(client):
    """Quem manda crédito e renda mas não a razão entre eles não pode perder
    uma das variáveis mais informativas do modelo."""
    import csv, io
    a = _csv([{"AMT_CREDIT": 300000, "AMT_INCOME_TOTAL": 150000, "AMT_ANNUITY": 15000}])
    b = _csv([{"AMT_CREDIT": 1500000, "AMT_INCOME_TOTAL": 150000, "AMT_ANNUITY": 15000}])
    pa, pb = (float(list(csv.DictReader(io.StringIO(
        client.post("/predict/csv", files={"arquivo": ("x.csv", d, "text/csv")}).text
    )))[0]["probability_default"]) for d in (a, b))
    assert pa != pb, "quintuplicar o crédito tem de mover o score"
