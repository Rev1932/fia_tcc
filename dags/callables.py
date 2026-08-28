"""
callables.py — As tarefas de validação e monitoramento do DAG de treino.

São funções Python puras: recebem caminhos, devolvem dicionários e levantam
exceção quando algo está errado. Nenhuma delas importa Airflow, o que permite
testá-las com `pytest` sem instância rodando (ver tests/test_dags.py).

Elas reusam o que o projeto já tem em vez de reimplementar:
  - Model.metrics_lib.psi        (coberta por 9 testes)
  - artifacts/metrics.json       (bloco `served`, a fonte dos números de capa)
  - artifacts/improvement_log.json  (histórico de rodadas aceitas/rejeitadas)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PROJETO = Path(os.getenv("HC_PROJETO", Path(__file__).resolve().parents[1]))

# Nomes que denunciariam vazamento: informação do desfecho do próprio contrato
# avaliado, que não existiria no momento da decisão de crédito.
PADROES_DE_VAZAMENTO = ("PROBA", "PREDIC", "SCORE_MODEL", "Y_TRUE", "Y_PRED")

# Piso de sanidade. A ABT v1 tinha 473 colunas; a atual tem 1.020. Muito abaixo
# disso significa que alguma etapa da agregação foi pulada em silêncio.
MIN_COLUNAS_ABT = 400
MIN_LINHAS_ABT = 1000


def _artefato(nome: str, base: Path | None = None) -> Path:
    return (base or PROJETO / "artifacts") / nome


def _ler_json(caminho: Path) -> dict:
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe — a etapa anterior falhou?")
    return json.loads(caminho.read_text())


# --------------------------------------------------------------------------
# 1. Fontes de dados
# --------------------------------------------------------------------------


def checar_fontes(projeto: Path | None = None, **_) -> dict:
    """Confere que os CSVs brutos existem e não estão vazios.

    Falhar aqui, em segundos, é muito melhor que falhar 11 minutos adiante no
    meio da agregação da ABT.
    """
    import yaml

    raiz = projeto or PROJETO
    cfg = yaml.safe_load((raiz / "DataPipeline" / "config.yaml").read_text())
    p = cfg["paths"]
    bruto = raiz / p["raw_dir"]

    tabelas = ["application_train", "bureau", "bureau_balance",
               "previous_application", "pos_cash", "credit_card", "installments"]

    achados, faltando = {}, []
    for chave in tabelas:
        nome = p.get(chave)
        if not nome:
            continue
        caminho = bruto / nome
        if not caminho.exists():
            faltando.append(str(caminho))
            continue
        mb = caminho.stat().st_size / 1e6
        if mb < 1:
            faltando.append(f"{caminho} (vazio: {mb:.2f} MB)")
            continue
        achados[nome] = round(mb, 1)
        print(f"  OK  {nome:32s} {mb:8.1f} MB")

    if faltando:
        raise FileNotFoundError(
            "Fontes ausentes ou vazias:\n  " + "\n  ".join(faltando) +
            f"\n\nBaixe a base do Kaggle para {bruto}/")

    total = sum(achados.values())
    print(f"\n{len(achados)} tabelas, {total:.0f} MB no total.")
    return {"tabelas": achados, "total_mb": round(total, 1)}


# --------------------------------------------------------------------------
# 2. A ABT
# --------------------------------------------------------------------------


def validar_abt(parquet: Path | None = None, **_) -> dict:
    """Confere a granularidade e a sanidade da ABT antes de gastar 15 min treinando.

    O abt_transform.py já tem `assert base[idc].is_unique`, mas um assert
    dentro do script morre no meio da execução. Aqui a checagem é uma task
    visível, com o resultado no log.
    """
    import duckdb

    caminho = parquet or (PROJETO / "Dados" / "abt.parquet")
    if not caminho.exists():
        raise FileNotFoundError(f"{caminho} não existe — abt_transform falhou?")

    con = duckdb.connect()
    origem = f"read_parquet('{str(caminho).replace(chr(39), chr(39) * 2)}')"

    linhas, ids = con.execute(
        f"SELECT count(*), count(DISTINCT SK_ID_CURR) FROM {origem}").fetchone()
    colunas = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {origem}").fetchall()]
    con.close()

    problemas = []
    if linhas != ids:
        problemas.append(
            f"granularidade quebrada: {linhas} linhas para {ids} clientes "
            f"({linhas - ids} duplicados)")
    if linhas < MIN_LINHAS_ABT:
        problemas.append(f"só {linhas} linhas (mínimo {MIN_LINHAS_ABT})")
    if len(colunas) < MIN_COLUNAS_ABT:
        problemas.append(
            f"só {len(colunas)} colunas (mínimo {MIN_COLUNAS_ABT}) — alguma "
            f"agregação foi pulada?")
    if "TARGET" not in colunas:
        problemas.append("coluna TARGET ausente")
    if "SK_ID_CURR" not in colunas:
        problemas.append("coluna SK_ID_CURR ausente")

    suspeitas = [c for c in colunas
                 if any(p in c.upper() for p in PADROES_DE_VAZAMENTO)]
    if suspeitas:
        problemas.append(f"colunas com cara de vazamento: {suspeitas[:5]}")

    print(f"  linhas   : {linhas:,}".replace(",", "."))
    print(f"  clientes : {ids:,}  (1 linha por cliente: "
          f"{'sim' if linhas == ids else 'NÃO'})".replace(",", "."))
    print(f"  colunas  : {len(colunas)}")
    print(f"  tamanho  : {caminho.stat().st_size / 1e6:.0f} MB")

    if problemas:
        raise ValueError("ABT inválida:\n  - " + "\n  - ".join(problemas))

    print("\nABT válida.")
    return {"linhas": linhas, "clientes": ids, "colunas": len(colunas)}


# --------------------------------------------------------------------------
# 3. O gate de qualidade
# --------------------------------------------------------------------------


def _auc_servido(metrics: dict) -> float:
    """AUC do modelo que a API entrega — o número de capa do projeto."""
    servido = metrics.get("served") or metrics.get("champion") or {}
    auc = servido.get("auc")
    if auc is None:
        raise KeyError("metrics.json sem bloco 'served' nem 'champion'")
    return float(auc)


def validar_metricas(artefatos: Path | None = None,
                     limiar: float | None = None, **_) -> dict:
    """Falha a DAG se o re-treino degradou o modelo além do limiar.

    É a regra de aceite do projeto (TODO.md §4) virando gate automático: um
    re-treino automático que piora o modelo não pode substituir em silêncio o
    que estava servindo.

    Sem rodada anterior, apenas registra a linha de base.
    """
    art = artefatos or (PROJETO / "artifacts")
    if limiar is None:
        limiar = float(os.getenv("HC_AUC_MAX_DROP", "0.01"))

    metrics = _ler_json(_artefato("metrics.json", art))
    atual = _auc_servido(metrics)
    tag = (metrics.get("run") or {}).get("tag", "?")
    run_id = (metrics.get("run") or {}).get("run_id", "?")

    log_path = _artefato("improvement_log.json", art)
    anteriores = []
    if log_path.exists():
        anteriores = [r for r in json.loads(log_path.read_text()).get("runs", [])
                      if not r.get("sample")
                      and r.get("status", "aceita") == "aceita"
                      and r.get("run_id") != run_id]

    print(f"rodada   : {run_id}  (tag: {tag})")
    print(f"AUC      : {atual:.4f}")

    if not anteriores:
        print("\nSem rodada anterior aceita — esta vira a linha de base.")
        return {"auc": atual, "referencia": None, "delta": None, "status": "linha_de_base"}

    ref = anteriores[-1]
    ref_auc = float(ref.get("auc"))
    delta = atual - ref_auc

    print(f"anterior : {ref_auc:.4f}  ({ref.get('tag')})")
    print(f"delta    : {delta:+.4f}   (limiar de queda: {limiar:.4f})")

    if delta < -limiar:
        raise ValueError(
            f"REGRESSÃO: o AUC caiu {abs(delta):.4f}, acima do limiar de "
            f"{limiar:.4f}.\n"
            f"  {ref.get('tag')}: {ref_auc:.4f}  ->  {tag}: {atual:.4f}\n\n"
            f"Os artefatos desta rodada estão em disco, mas a rodada NÃO deve "
            f"ser promovida. Investigue antes de servir.")

    status = "melhorou" if delta >= 0 else "piorou dentro do limiar"
    print(f"\n{status}.")
    return {"auc": atual, "referencia": ref_auc, "delta": delta, "status": status}


# --------------------------------------------------------------------------
# 4. Drift
# --------------------------------------------------------------------------


def calcular_psi(artefatos: Path | None = None,
                 anterior: Path | None = None, **_) -> dict:
    """PSI entre o score desta rodada e o da anterior.

    Cumpre o que MLOps/Readme.md prometia como "PSI calculado em job no
    Airflow" — o cálculo já existia (Model.metrics_lib.psi, 9 testes), faltava
    o agendamento.

    Compara com a cópia da rodada anterior quando existe; na falta dela,
    compara treino contra teste da rodada atual, que é o mesmo cálculo aplicado
    às fatias internas.
    """
    import pandas as pd
    import sys
    sys.path.insert(0, str(PROJETO))
    from Model.metrics_lib import psi

    art = artefatos or (PROJETO / "artifacts")
    scores = _artefato("scores.parquet", art)
    if not scores.exists():
        raise FileNotFoundError(f"{scores} não existe — o treino falhou?")

    atual = pd.read_parquet(scores)
    ref_path = anterior or (art / "scores_anterior.parquet")

    if ref_path.exists():
        base = pd.read_parquet(ref_path)["proba_champion"].dropna()
        novo = atual["proba_champion"].dropna()
        comparacao = "rodada anterior vs. atual"
    else:
        base = atual[atual.split == "train"]["proba_champion"].dropna()
        novo = atual[atual.split == "test"]["proba_champion"].dropna()
        comparacao = "treino vs. teste (sem rodada anterior para comparar)"

    resultado = psi(base, novo)
    print(f"comparação : {comparacao}")
    print(f"PSI        : {resultado['psi']:.5f}   [{resultado['faixa']}]")
    print(f"n          : {resultado['n_esperado']:,} vs {resultado['n_observado']:,}"
          .replace(",", "."))
    print("\nleitura: < 0,10 estável · 0,10–0,25 atenção · > 0,25 mudança relevante")

    relatorio = {"comparacao": comparacao, "psi": resultado["psi"],
                 "faixa": resultado["faixa"],
                 "n_esperado": resultado["n_esperado"],
                 "n_observado": resultado["n_observado"]}
    (art / "psi_report.json").write_text(json.dumps(relatorio, indent=2, ensure_ascii=False))

    # Guarda os scores desta rodada para servirem de referência na próxima.
    atual.to_parquet(art / "scores_anterior.parquet", index=False)

    if resultado["faixa"] != "estável":
        print(f"\nATENÇÃO: distribuição do score em '{resultado['faixa']}'.")
    return relatorio
