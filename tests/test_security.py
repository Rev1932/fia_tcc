"""
Injeção de SQL pelos parâmetros que viram identificadores.

Valores sempre vão parametrizados; nomes de coluna e dimensões passam por
whitelist. Estes testes garantem que continue assim.
"""
from __future__ import annotations


PAYLOADS = [
    "SK_ID_CURR; DROP TABLE abt",
    "1); DROP TABLE abt --",
    "SK_ID_CURR' OR '1'='1",
    '" OR 1=1 --',
]


def test_sort_malicioso_e_rejeitado(client):
    for p in PAYLOADS:
        r = client.get("/clients", params={"sort": p})
        assert r.status_code == 400, p


def test_fields_malicioso_e_rejeitado(client):
    for p in PAYLOADS:
        assert client.get("/clients", params={"fields": p}).status_code == 400, p


def test_dimensao_maliciosa_e_rejeitada(client):
    for p in PAYLOADS + ["gender'--"]:
        assert client.get("/stats/default-rate", params={"by": p}).status_code == 400, p


def test_feature_maliciosa_e_rejeitada(client):
    for p in PAYLOADS:
        assert client.get("/stats/distribution", params={"feature": p}).status_code == 400, p


def test_valor_de_filtro_e_parametrizado_nao_interpolado(client):
    """Texto malicioso num VALOR não é SQL: é só um valor que não casa."""
    r = client.get("/clients", params={"education": "' OR 1=1 --", "page_size": 5})
    assert r.status_code == 200
    assert r.json()["meta"]["total"] == 0


def test_a_api_continua_de_pe_depois_das_tentativas(client):
    r = client.get("/clients", params={"page_size": 1})
    assert r.status_code == 200
    assert r.json()["meta"]["total"] == 300
