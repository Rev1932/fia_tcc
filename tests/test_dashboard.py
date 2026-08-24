"""
Testes do painel Streamlit.

O painel consome a API por HTTP. Aqui as chamadas `requests` são redirecionadas
para o `TestClient` da própria API sobre a base sintética — então o teste
exercita o script de verdade, sem depender de servidor no ar nem dos dados
reais.

Motivo de existir: o painel foi reescrito de importar `Model.predict` para
consumir HTTP, e passou um ciclo inteiro sem nunca ter sido aberto. Um erro
aqui só apareceria na demonstração.
"""
from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = "MLOps/app/streamlit_app.py"


@pytest.fixture
def painel(client, monkeypatch):
    """Aponta o `requests` do painel para o TestClient da API."""
    import requests

    class Resposta:
        def __init__(self, r):
            self._r = r
            self.status_code = r.status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")
        def json(self):
            return self._r.json()

    def get(url, params=None, timeout=None, **kw):
        return Resposta(client.get(url.replace("http://testserver", ""), params=params))

    def post(url, json=None, timeout=None, **kw):
        return Resposta(client.post(url.replace("http://testserver", ""), json=json))

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setenv("HC_API_URL", "http://testserver")
    return AppTest.from_file(APP, default_timeout=120)


def test_painel_carrega_sem_excecao(painel):
    at = painel.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_cabecalho_mostra_a_rodada_vigente(painel):
    at = painel.run()
    rotulos = {m.label: m.value for m in at.metric}
    assert "Clientes" in rotulos and "Threshold" in rotulos
    assert rotulos["Clientes"] == "300"        # base sintética do conftest


def test_tem_as_quatro_abas(painel):
    at = painel.run()
    assert len(at.tabs) == 4


def test_aba_carteira_renderiza_a_tabela_de_segmentos(painel):
    at = painel.run()
    assert at.dataframe, "a aba Carteira deveria renderizar ao menos uma tabela"


def test_painel_avisa_quando_a_api_esta_fora(monkeypatch):
    """Se a API cair, o painel precisa dizer isso — não estourar traceback."""
    import requests

    def falha(*a, **kw):
        raise requests.ConnectionError("recusada")

    monkeypatch.setattr(requests, "get", falha)
    monkeypatch.setenv("HC_API_URL", "http://localhost:1")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.error, "deveria mostrar erro na tela em vez de quebrar"
    assert "API indisponível" in at.error[0].value
