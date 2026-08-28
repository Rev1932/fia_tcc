"""
Trava a telemetria de serviço da API.

O teste que mais importa aqui é `test_duas_apps_nao_colidem_no_registry`: o
`MLOps/app/api.py` termina com `app = create_app()` e o conftest recarrega esse
módulo, então `create_app()` roda cinco vezes por sessão. Se as métricas forem
parar no REGISTRY global do prometheus_client, a segunda chamada levanta
`Duplicated timeseries in CollectorRegistry` e derruba a suíte inteira.
"""
from __future__ import annotations

import pytest

PREFIXO_PROM = "text/plain"


def _corpo(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    return r.text


# ------------------------------------------------------------ registry por app

def test_duas_apps_nao_colidem_no_registry():
    """Cria duas instâncias seguidas — é o que a suíte faz na prática."""
    from MLOps.app.api import create_app

    a, b = create_app(), create_app()
    assert a.state.metrics_registry is not b.state.metrics_registry


def test_instrumentar_e_idempotente():
    """Chamar duas vezes no mesmo app não recria nem duplica série."""
    from MLOps.app import metrics
    from MLOps.app.api import create_app

    app = create_app()
    antes = app.state.metrics_registry
    metrics.instrumentar(app)
    assert app.state.metrics_registry is antes


# ------------------------------------------------------------------- o endpoint

def test_metrics_responde_no_formato_prometheus(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert PREFIXO_PROM in r.headers["content-type"]


def test_metrics_fica_fora_do_openapi(client):
    """A contagem de endpoints citada em quatro documentos não pode mudar por
    causa de uma rota de infraestrutura."""
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


# --------------------------------------------------------------- saúde exposta

@pytest.mark.parametrize("serie", [
    "hc_model_loaded", "hc_data_loaded", "hc_artifacts_loaded",
    "hc_explainer_loaded", "hc_api_pronta", "hc_uptime_seconds",
    "hc_erros_ativos", "hc_erro_componente",
])
def test_gauges_de_saude_presentes(client, serie):
    assert serie in _corpo(client)


def test_api_pronta_reflete_modelo_e_dados(client):
    """No fixture o modelo e o banco carregam, então a API está pronta."""
    corpo = _corpo(client)
    assert "hc_api_pronta 1.0" in corpo
    assert "hc_model_loaded 1.0" in corpo


def test_processo_exposto(client):
    """CPU e memória vêm dos coletores default, que um registry novo NÃO traz —
    se este teste cair, o ProcessCollector deixou de ser registrado."""
    corpo = _corpo(client)
    assert "process_cpu_seconds_total" in corpo
    assert "process_resident_memory_bytes" in corpo


# ------------------------------------------------------------------ HTTP e negócio

def test_requisicoes_sao_contadas(client):
    client.get("/health")
    corpo = _corpo(client)
    assert "http_requests_total" in corpo
    assert "http_request_duration_seconds" in corpo


def test_predicao_incrementa_contador_por_decisao(client):
    r = client.post("/predict", json={"records": [
        {"NAME_CONTRACT_TYPE": "Cash loans", "AMT_INCOME_TOTAL": 150000,
         "AMT_CREDIT": 500000, "AMT_ANNUITY": 25000, "EXT_SOURCE_2": 0.55}]})
    assert r.status_code == 200
    decisao = r.json()["predictions"][0]["decision"]

    corpo = _corpo(client)
    assert f'hc_predicoes_total{{decision="{decisao}",endpoint="/predict"}}' in corpo
    assert "hc_score_previsto_bucket" in corpo


def test_metrics_nao_se_mede(client):
    """`/metrics` está em excluded_handlers: o scrape não pode inflar o volume."""
    for _ in range(3):
        client.get("/metrics")
    assert 'handler="/metrics"' not in _corpo(client)
