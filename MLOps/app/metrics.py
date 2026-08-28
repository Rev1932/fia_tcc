"""
metrics.py — Telemetria de serviço da API, no formato Prometheus.

Fecha a linha "Performance do serviço" que `MLOps/Readme.md` listava como
proposta: latência, taxa de erro e throughput passam a ser medidos por request,
e não inferidos do `/health`, que é pull-only e some com os erros quando alguém
chama `POST /admin/reload`.

Não confundir com o PSI (`GET /model/psi`), que mede drift de distribuição em
batch a cada 7 dias. São coisas ortogonais: aqui é o serviço, lá é o modelo.

Três decisões que o formato do arquivo obriga:

1. **Registry por app, nunca o global.** `MLOps/app/api.py` termina com
   `app = create_app()`, e `tests/conftest.py` recarrega esse módulo — logo
   `create_app()` roda cinco vezes por sessão de pytest. Métrica no REGISTRY
   global levantaria `Duplicated timeseries` na segunda.
2. **O estado é lido no scrape, não capturado.** `POST /admin/reload` substitui
   `app.state.db`, `bundle` e `artifacts` por objetos novos. Guardar referência
   a eles congelaria a métrica na versão antiga.
3. **Nada aqui toca o DuckDB.** O `/health` faz `count(*)` sobre 307 mil linhas;
   repetir isso a cada 10 segundos de scrape sairia caro.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter,
                               GCCollector, Histogram, PlatformCollector,
                               ProcessCollector, generate_latest)
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from prometheus_fastapi_instrumentator import Instrumentator

# Faixas de probabilidade de inadimplência. Concentradas embaixo porque o corte
# de negócio é 0,09 e é ali que a distribuição precisa de resolução.
FAIXAS_SCORE = (0.01, 0.025, 0.05, 0.09, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0)


class _EstadoCollector:
    """Lê `app.state` no momento do scrape e devolve os gauges de saúde.

    É a peça que responde "está no ar, mas consegue pontuar?" — o `up` do
    Prometheus diz só que o processo respondeu.
    """

    def __init__(self, app):
        self._app = app

    def collect(self):
        st = self._app.state
        componentes = {
            "hc_model_loaded": getattr(st, "bundle", None) is not None,
            "hc_data_loaded": getattr(st, "db", None) is not None,
            "hc_artifacts_loaded": bool(getattr(st, "artifacts", None)),
            "hc_explainer_loaded": bool(getattr(st, "explainer_loaded", False)),
        }
        for nome, ligado in componentes.items():
            yield GaugeMetricFamily(nome, f"1 quando {nome[3:]} está disponível",
                                    value=float(ligado))

        # `ok` replica a condição de 200 vs 503 do /health (api.py): só modelo e
        # banco derrubam o serviço; artefato e explainer o degradam.
        yield GaugeMetricFamily(
            "hc_api_pronta",
            "1 quando a API consegue pontuar (mesma regra do 200 vs 503 do /health)",
            value=float(componentes["hc_model_loaded"] and componentes["hc_data_loaded"]))

        iniciada = getattr(st, "started_at", None)
        if iniciada:
            yield GaugeMetricFamily("hc_uptime_seconds", "Segundos desde o startup",
                                    value=time.time() - iniciada)

        erros = getattr(st, "errors", {}) or {}
        yield GaugeMetricFamily("hc_erros_ativos",
                                "Componentes que falharam ao carregar",
                                value=float(len(erros)))
        por_componente = GaugeMetricFamily(
            "hc_erro_componente", "1 quando o componente registrou erro no startup",
            labels=["componente"])
        for componente in ("data", "model", "artifacts", "explainer", "query"):
            por_componente.add_metric([componente], float(componente in erros))
        yield por_componente

        yield from self._modelo(st)

    def _modelo(self, st):
        """Identidade da rodada servida. Silencioso quando o artefato falta:
        `artifacts.threshold` levanta 503, e exceção no collect quebra o scrape."""
        from MLOps.app import artifacts

        try:
            yield GaugeMetricFamily("hc_threshold",
                                    "Corte de decisão da rodada servida",
                                    value=float(artifacts.threshold(st)))
        except Exception:
            pass

        try:
            run = (st.artifacts["metrics"] or {}).get("run") or {}
            servido = (st.artifacts["metrics"] or {}).get("served") or {}
            if run.get("n_features"):
                yield GaugeMetricFamily("hc_n_features", "Variáveis do modelo servido",
                                        value=float(run["n_features"]))
            if servido.get("auc"):
                yield GaugeMetricFamily("hc_auc_servido",
                                        "AUC da rodada em teste (referência, não ao vivo)",
                                        value=float(servido["auc"]))
            if run.get("run_id"):
                yield InfoMetricFamily("hc_rodada", "Rodada servida",
                                       value={"run_id": str(run["run_id"]),
                                              "tag": str(run.get("tag") or "")})
        except Exception:
            pass


def _criar_metricas(registry: CollectorRegistry) -> dict[str, Any]:
    return {
        "predicoes": Counter(
            "hc_predicoes_total", "Predições servidas, por decisão e endpoint",
            ["decision", "endpoint"], registry=registry),
        "score": Histogram(
            "hc_score_previsto", "Distribuição da probabilidade de inadimplência prevista",
            buckets=FAIXAS_SCORE, registry=registry),
    }


def instrumentar(app) -> None:
    """Liga a telemetria num app recém-criado. Idempotente por app."""
    if getattr(app.state, "metrics_registry", None) is not None:
        return

    registry = CollectorRegistry()
    # Um registry novo não traz os coletores default: sem isto não há CPU nem
    # memória do processo, que é metade da pergunta "performance".
    ProcessCollector(registry=registry)
    PlatformCollector(registry=registry)
    GCCollector(registry=registry)

    app.state.metrics_registry = registry
    app.state.metrics = _criar_metricas(registry)
    registry.register(_EstadoCollector(app))

    Instrumentator(
        registry=registry,
        should_group_status_codes=True,      # 2xx/4xx/5xx em vez de 200/201/404/...
        should_ignore_untemplated=True,      # rota não casada não vira série nova
        excluded_handlers=["/metrics"],      # o scrape não se mede
    ).instrument(app)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


def registrar_predicao(request, decision: str | None, probabilidade: float | None,
                       endpoint: str) -> None:
    """Contabiliza uma predição servida. Silenciosa se a app não foi instrumentada.

    Chamada só dos caminhos que de fato pontuam. A listagem de `/clients` lê
    score pré-computado e ficaria de fora de propósito: contar ali inflaria o
    volume de "predições" com paginação de tela.
    """
    m = getattr(request.app.state, "metrics", None)
    if not m:
        return
    if decision:
        m["predicoes"].labels(decision=decision, endpoint=endpoint).inc()
    if probabilidade is not None:
        m["score"].observe(float(probabilidade))
