"""
Testes do DAG de treino e dos seus callables.

O `MLOps/pipeline_orchestration.py` tem 0% de cobertura — o DAG que o substitui
não pode repetir isso. Os testes de estrutura pegam justamente as classes de
erro que já apareceram durante a implementação:

  - import do Airflow 2.x (`airflow.operators.bash`) que quebra no 3
  - `from callables import ...` sem a pasta no sys.path
  - agendamento diferente dos 7 dias pedidos
  - `max_active_runs` > 1, que deixaria duas rodadas escrevendo em artifacts/

Os callables são funções puras (não importam Airflow), então são testados
sempre. Os testes de estrutura precisam do pacote instalado e são pulados
quando ele não está — nesse caso rode dentro do container:
    docker compose exec airflow-scheduler pytest /opt/projeto/tests/test_dags.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
DAGS = RAIZ / "dags"


# ==========================================================================
# Estrutura do DAG (exige apache-airflow)
# ==========================================================================

# Marca só a classe de testes de estrutura — os callables rodam sempre.
precisa_airflow = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("airflow") is None,
    reason="apache-airflow não instalado no host; rode dentro do container",
)


@pytest.fixture(scope="module")
def dagbag():
    import sys
    sys.path.insert(0, str(DAGS))
    from airflow.models import DagBag
    # `include_examples` foi removido do DagBag no Airflow 3; os exemplos são
    # desligados por AIRFLOW__CORE__LOAD_EXAMPLES no compose.
    return DagBag(str(DAGS))


@pytest.fixture(scope="module")
def dag(dagbag):
    d = dagbag.dags.get("treino_credit_scoring")
    assert d is not None, f"DAG não carregou. Erros: {dagbag.import_errors}"
    return d


@precisa_airflow
def test_dag_carrega_sem_erro_de_import(dagbag):
    """Pega o import do Airflow 2.x (`airflow.operators.bash`), que no 3 vive
    em `airflow.providers.standard.operators.bash`."""
    assert not dagbag.import_errors, dagbag.import_errors


@precisa_airflow
def test_agendamento_e_de_sete_dias(dag):
    from datetime import timedelta
    assert dag.schedule == timedelta(days=7)


@precisa_airflow
def test_uma_rodada_por_vez(dag):
    """As tasks escrevem em Dados/ e artifacts/, que são caminhos fixos. Duas
    execuções simultâneas se sobrescreveriam."""
    assert dag.max_active_runs == 1


@precisa_airflow
def test_nao_reprocessa_o_passado(dag):
    """catchup=True dispararia uma execução por semana desde start_date no
    primeiro `unpause` — dezenas de treinos de 15 min em fila."""
    assert dag.catchup is False


@precisa_airflow
def test_tem_as_nove_tasks(dag):
    esperadas = {
        "checar_fontes", "sanitizacao", "abt_transform", "validar_abt",
        "perfil_colunas", "treino", "validar_metricas", "calcular_psi",
        "resumo_da_rodada",
    }
    assert {t.task_id for t in dag.tasks} == esperadas


@precisa_airflow
def test_ordem_das_dependencias(dag):
    esperada = ["checar_fontes", "sanitizacao", "abt_transform", "validar_abt",
                "perfil_colunas", "treino", "validar_metricas", "calcular_psi",
                "resumo_da_rodada"]
    for anterior, seguinte in zip(esperada, esperada[1:]):
        filhos = {t.task_id for t in dag.get_task(anterior).downstream_list}
        assert seguinte in filhos, f"{anterior} deveria preceder {seguinte}"


@precisa_airflow
def test_validacao_barra_o_treino(dag):
    """validar_abt precisa vir ANTES do treino: descobrir que a ABT está
    quebrada depois de 15 min treinando não serve para nada."""
    posterior = dag.get_task("validar_abt").get_flat_relatives(upstream=False)
    assert "treino" in {t.task_id for t in posterior}


@precisa_airflow
def test_treino_nao_repete_em_caso_de_falha(dag):
    """Repetir um treino que falhou gasta mais 15 min para falhar igual."""
    assert dag.get_task("treino").retries == 0


@precisa_airflow
def test_templates_nao_usam_ds(dag):
    """`{{ ds }}` e `{{ logical_date }}` não existem em DAG com
    schedule=timedelta disparado manualmente — estouram com UndefinedError
    depois de o pipeline já ter gasto 15 minutos nas etapas anteriores."""
    proibidos = ("{{ ds ", "{{ds", "{{ logical_date", "{{ execution_date")
    for task in dag.tasks:
        cmd = getattr(task, "bash_command", "") or ""
        for p in proibidos:
            assert p not in cmd, (
                f"{task.task_id} usa {p!r}: sem logical_date isso quebra. "
                f"Use {{{{ run_id }}}}.")


@precisa_airflow
def test_tag_do_treino_identifica_a_rodada(dag):
    """A tag entra no improvement_log.json — precisa distinguir uma rodada da
    outra, senão o histórico de rodadas se sobrescreve."""
    cmd = dag.get_task("treino").bash_command
    assert "--tag" in cmd and "run_id" in cmd


@precisa_airflow
def test_tasks_rodam_na_raiz_do_projeto(dag):
    """Os scripts resolvem caminhos relativos à raiz do repositório."""
    for tid in ("sanitizacao", "abt_transform", "perfil_colunas", "treino"):
        assert dag.get_task(tid).cwd == "/opt/projeto"


# ==========================================================================
# Callables (funções puras — rodam sempre)
# ==========================================================================

from dags.callables import calcular_psi, validar_abt, validar_metricas  # noqa: E402


def _metrics(auc: float, tag: str = "atual", run_id: str = "r2") -> dict:
    return {"run": {"run_id": run_id, "tag": tag},
            "served": {"model": "champion+isotonic", "auc": auc, "ks": 0.43,
                       "brier": 0.066, "threshold": 0.09, "approval_rate": 0.69},
            "champion": {"auc": auc, "ks": 0.43},
            "business": {"threshold": 0.09, "approval_rate": 0.69}}


def _log(auc_anterior: float) -> dict:
    return {"runs": [{"run_id": "r1", "tag": "anterior", "auc": auc_anterior,
                      "status": "aceita"}]}


@pytest.fixture
def artefatos(tmp_path):
    return tmp_path


def test_gate_aprova_queda_dentro_do_limiar(artefatos):
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7800)))
    (artefatos / "improvement_log.json").write_text(json.dumps(_log(0.7868)))
    r = validar_metricas(artefatos=artefatos, limiar=0.01)
    assert r["delta"] == pytest.approx(-0.0068, abs=1e-6)
    assert r["status"] == "piorou dentro do limiar"


def test_gate_falha_com_queda_acima_do_limiar(artefatos):
    """O teste que mais importa: um re-treino que degrada o modelo não pode
    substituir em silêncio o que estava servindo."""
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7700)))
    (artefatos / "improvement_log.json").write_text(json.dumps(_log(0.7868)))
    with pytest.raises(ValueError, match="REGRESSÃO"):
        validar_metricas(artefatos=artefatos, limiar=0.01)


def test_gate_aceita_melhora(artefatos):
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7950)))
    (artefatos / "improvement_log.json").write_text(json.dumps(_log(0.7868)))
    assert validar_metricas(artefatos=artefatos, limiar=0.01)["status"] == "melhorou"


def test_gate_sem_rodada_anterior_vira_linha_de_base(artefatos):
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7868)))
    r = validar_metricas(artefatos=artefatos, limiar=0.01)
    assert r["status"] == "linha_de_base" and r["referencia"] is None


def test_gate_ignora_rodada_rejeitada(artefatos):
    """Uma rodada rejeitada fica no log de propósito, mas não pode virar a
    referência de comparação."""
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7860)))
    (artefatos / "improvement_log.json").write_text(json.dumps({"runs": [
        {"run_id": "r1", "tag": "boa", "auc": 0.7868, "status": "aceita"},
        {"run_id": "rx", "tag": "ruim", "auc": 0.9000, "status": "rejeitada"},
    ]}))
    assert validar_metricas(artefatos=artefatos, limiar=0.01)["referencia"] == 0.7868


def test_gate_ignora_rodada_de_amostra(artefatos):
    (artefatos / "metrics.json").write_text(json.dumps(_metrics(0.7860)))
    (artefatos / "improvement_log.json").write_text(json.dumps({"runs": [
        {"run_id": "r1", "tag": "oficial", "auc": 0.7868, "status": "aceita"},
        {"run_id": "rd", "tag": "demo", "auc": 0.70, "status": "aceita",
         "sample": 30000},
    ]}))
    assert validar_metricas(artefatos=artefatos, limiar=0.01)["referencia"] == 0.7868


def test_gate_sem_metrics_falha_claramente(artefatos):
    with pytest.raises(FileNotFoundError, match="não existe"):
        validar_metricas(artefatos=artefatos, limiar=0.01)


# ---------------------------------------------------------------- validar_abt

def _abt(caminho: Path, n=2000, duplicar=False, colunas=500, sem_target=False):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(3)
    ids = np.arange(1, n + 1)
    if duplicar:
        ids[-1] = ids[0]
    d = {"SK_ID_CURR": ids}
    if not sem_target:
        d["TARGET"] = rng.integers(0, 2, n)
    for i in range(colunas):
        d[f"F_{i}"] = rng.normal(size=n)
    pd.DataFrame(d).to_parquet(caminho, index=False)
    return caminho


def test_abt_valida_passa(tmp_path):
    r = validar_abt(parquet=_abt(tmp_path / "abt.parquet"))
    assert r["linhas"] == r["clientes"] == 2000


def test_abt_com_cliente_duplicado_falha(tmp_path):
    """A granularidade é a garantia central da ABT: 1 linha por cliente."""
    with pytest.raises(ValueError, match="granularidade"):
        validar_abt(parquet=_abt(tmp_path / "dup.parquet", duplicar=True))


def test_abt_sem_target_falha(tmp_path):
    with pytest.raises(ValueError, match="TARGET"):
        validar_abt(parquet=_abt(tmp_path / "sem.parquet", sem_target=True))


def test_abt_com_poucas_colunas_falha(tmp_path):
    """Barra o caso de uma agregação ter sido pulada em silêncio."""
    with pytest.raises(ValueError, match="colunas"):
        validar_abt(parquet=_abt(tmp_path / "magra.parquet", colunas=50))


def test_abt_ausente_falha_claramente(tmp_path):
    with pytest.raises(FileNotFoundError, match="abt_transform"):
        validar_abt(parquet=tmp_path / "nao_existe.parquet")


# ---------------------------------------------------------------- calcular_psi

def test_psi_gera_relatorio(tmp_path):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 4000
    pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "split": np.where(np.arange(n) % 4 == 0, "test", "train"),
        "y_true": rng.integers(0, 2, n),
        "proba_champion": rng.beta(2, 20, n),
    }).to_parquet(tmp_path / "scores.parquet", index=False)

    r = calcular_psi(artefatos=tmp_path)
    assert r["psi"] is not None and r["faixa"] in {
        "estável", "atenção", "mudança relevante"}
    assert json.loads((tmp_path / "psi_report.json").read_text())["psi"] == r["psi"]
    assert (tmp_path / "scores_anterior.parquet").exists(), \
        "precisa guardar os scores para servirem de referência na próxima rodada"


def test_psi_sem_scores_falha_claramente(tmp_path):
    with pytest.raises(FileNotFoundError, match="treino"):
        calcular_psi(artefatos=tmp_path)
