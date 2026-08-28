"""
Prova que a partição de teste não foi tocada.

É a base da regra de aceite (TODO.md §4.1): toda comparação entre rodadas só
vale se o conjunto de teste for exatamente o mesmo. Este teste re-deriva o
split a partir da ABT e confere contra o que a rodada congelada gravou em
artifacts/scores.parquet.

Pula quando a ABT ou os artefatos não estão na árvore (ambos são gitignored).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
ABT = ROOT / "Dados" / "abt.parquet"
SCORES = ROOT / "artifacts" / "scores.parquet"

pytestmark = pytest.mark.skipif(
    not (ABT.exists() and SCORES.exists()),
    reason="precisa da ABT real e de artifacts/scores.parquet")


@pytest.fixture(scope="module")
def particao():
    import pandas as pd
    cfg = yaml.safe_load((ROOT / "Model" / "config.yaml").read_text())
    alvo, idc, sp = cfg["target"], cfg["id_col"], cfg["split"]

    df = pd.read_parquet(ABT, columns=[idc, alvo])
    y = df[alvo].astype(int).to_numpy()
    ids = df[idc].to_numpy()

    # Mesma sequência de train.py:243-264. train_test_split deriva a permutação
    # de (n_samples, stratify, random_state), então indexar não muda a partição.
    idx = np.arange(len(y))
    idx_tmp, idx_test = train_test_split(
        idx, test_size=sp["test_size"], random_state=sp["random_state"],
        stratify=y if sp["stratify"] else None)
    val_rel = sp["valid_size"] / (1 - sp["test_size"])
    idx_tr, idx_val = train_test_split(
        idx_tmp, test_size=val_rel, random_state=sp["random_state"],
        stratify=y[idx_tmp] if sp["stratify"] else None)
    idx_cal = np.array([], dtype=int)
    if (calib := sp.get("calib_size", 0) or 0):
        cal_rel = calib / (1 - sp["test_size"] - sp["valid_size"])
        idx_tr, idx_cal = train_test_split(
            idx_tr, test_size=cal_rel, random_state=sp["random_state"],
            stratify=y[idx_tr] if sp["stratify"] else None)

    sc = pd.read_parquet(SCORES, columns=[idc, "split"])
    return {"ids": ids, "idx": {"train": idx_tr, "valid": idx_val,
                                "calib": idx_cal, "test": idx_test}, "scores": sc, "id_col": idc}


@pytest.mark.parametrize("fatia", ["train", "valid", "test"])
def test_split_rederivado_bate_com_a_rodada_congelada(particao, fatia):
    ids, sc, idc = particao["ids"], particao["scores"], particao["id_col"]
    esperado = set(sc.loc[sc["split"] == fatia, idc].to_numpy().tolist())
    obtido = set(ids[particao["idx"][fatia]].tolist())
    assert obtido == esperado, (
        f"a fatia {fatia!r} mudou: {len(obtido ^ esperado)} clientes divergem. "
        "Nenhuma comparação entre rodadas vale enquanto isso não fechar.")


def test_scores_parquet_nao_inclui_a_fatia_de_calibracao(particao):
    """Limitação real de train.py:493-502 — id_cal fica de fora da gravação.

    Fica travado como teste para o número não ser reclamado como 'todos os
    clientes' em documento nenhum.
    """
    ids, sc, idc = particao["ids"], particao["scores"], particao["id_col"]
    gravados = set(sc[idc].to_numpy().tolist())
    calib = set(ids[particao["idx"]["calib"]].tolist())
    assert calib and not (calib & gravados)
    assert len(gravados) == len(ids) - len(calib)
