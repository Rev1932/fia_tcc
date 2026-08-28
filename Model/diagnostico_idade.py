"""
diagnostico_idade.py — Por que o AUC cai nos extremos de idade.

Reanalisa a pendência do segmento `<25 anos` sem re-treinar: lê os scores da
rodada congelada (`artifacts/scores.parquet`), junta com a ABT e responde, com
número, às hipóteses que o `TODO.md §2` levanta mas não mede.

Uso:
    python Model/diagnostico_idade.py              # tabelas legíveis
    python Model/diagnostico_idade.py --markdown   # pronto para colar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Model.metrics_lib import (auc_bootstrap_ci, auc_diff_all_groups,  # noqa: E402
                               auc_diff_bootstrap, auc_within_between,
                               brier_score, calibration_points)
from Model.train import age_band  # noqa: E402  (mesma definição do fairness.json)

ART = ROOT / "artifacts"
N_BOOT = 1000
TOP_N = 20


def carregar(nome: str) -> dict:
    p = ART / nome
    if not p.exists():
        raise SystemExit(f"[diag] {p} não existe. Rode `python Model/train.py`.")
    return json.loads(p.read_text())


def faixas_sextis(idade: np.ndarray) -> tuple[np.ndarray, dict]:
    """Corte por mesma frequência: tira o tamanho de amostra da equação."""
    q = np.quantile(idade[~np.isnan(idade)], np.linspace(0, 1, 7))
    q[0], q[-1] = -np.inf, np.inf
    rotulos = [f"S{i+1} {q[i]:.0f}-{q[i+1]:.0f}" for i in range(6)]
    rotulos[0] = f"S1 ate {q[1]:.0f}"
    rotulos[-1] = f"S6 {q[-2]:.0f}+"
    return np.asarray(pd.cut(idade, q, labels=rotulos, include_lowest=True), dtype=object), \
        {"cortes": [float(x) for x in q[1:-1]]}


def faixas_5anos(idade: np.ndarray) -> np.ndarray:
    """Janelas de largura fixa: tira a largura da faixa da equação."""
    base = np.floor(idade / 5) * 5
    return np.asarray([np.nan if np.isnan(v) else f"{int(v)}-{int(v)+5}" for v in base],
                      dtype=object)


def avaliar_corte(y, score, rotulos, n_boot: int) -> dict:
    """AUC + IC por grupo, teste da diferença contra os demais, e decomposição."""
    # Uma passada de bootstrap para o eixo inteiro, não uma por grupo.
    difs = auc_diff_all_groups(y, score, rotulos, n_boot=n_boot)
    grupos = []
    for g in sorted({r for r in rotulos if isinstance(r, str)}):
        m = rotulos == g
        if m.sum() < 30:
            continue
        ci = auc_bootstrap_ci(y[m], score[m], n_boot=n_boot)
        dif = difs.get(g) or {}
        grupos.append({
            "grupo": g, "n": int(m.sum()),
            "auc": ci.get("auc"), "ci_low": ci.get("ci_low"), "ci_high": ci.get("ci_high"),
            "default_rate": float(y[m].mean()),
            "avg_score": float(score[m].mean()),
            "diff": dif.get("diff"),
            "diff_ci_low": dif.get("diff_ci_low"), "diff_ci_high": dif.get("diff_ci_high"),
            "p_value": dif.get("p_value"),
            "pior_que_referencia": dif.get("pior_que_referencia"),
        })
    return {"grupos": grupos, "decomposicao": auc_within_between(y, score, rotulos)}


def perfil_de_features(df, y, rotulos, cols) -> list[dict]:
    """Disponibilidade e poder de cada feature DENTRO de cada faixa.

    Testa a hipótese central do TODO.md §2: se o AUC univariado das features
    dominantes cai dentro do `<25`, o teto está no dado e não no modelo.
    """
    out = []
    for c in cols:
        v = df[c].to_numpy(dtype="float64")
        presente = ~np.isnan(v)
        std_geral = float(np.nanstd(v)) if presente.any() else np.nan
        linha = {"feature": c, "cobertura_geral": float(presente.mean()), "por_faixa": {}}
        for g in sorted({r for r in rotulos if isinstance(r, str)}):
            m = (rotulos == g)
            mp = m & presente
            if mp.sum() < 30 or len({*y[mp]}) < 2:
                continue
            # AUC univariado orientado: |AUC-0,5| mede poder, o sinal não importa
            a = float(roc_auc_score(y[mp], v[mp]))
            # A ausência da feature pode ela mesma predizer: se sim, o LightGBM
            # já a explora pelo tratamento nativo de NaN.
            a_aus = None
            if 0 < presente[m].mean() < 1 and len({*y[m]}) == 2:
                a_aus = float(roc_auc_score(y[m], (~presente[m]).astype(int)))
            sd = float(np.nanstd(v[mp]))
            linha["por_faixa"][g] = {
                "cobertura": float(presente[m].mean()),
                "auc_univariada": max(a, 1 - a),
                "auc_da_ausencia": None if a_aus is None else max(a_aus, 1 - a_aus),
                "std_ratio": None if not std_geral else sd / std_geral,
                "n_usavel": int(mp.sum()),
            }
        out.append(linha)
    return out


def calibracao_por_faixa(y, score, rotulos, n_boot: int) -> list[dict]:
    """O modelo acerta o NÍVEL de risco de cada faixa? (pergunta diferente do AUC)"""
    rng = np.random.default_rng(42)
    out = []
    for g in sorted({r for r in rotulos if isinstance(r, str)}):
        m = rotulos == g
        if m.sum() < 30:
            continue
        yg, sg = y[m], score[m]
        gaps = [float(sg[i].mean() - yg[i].mean())
                for i in (rng.integers(0, yg.size, yg.size) for _ in range(n_boot))]
        lo, hi = np.quantile(gaps, [0.025, 0.975])
        out.append({
            "grupo": g, "n": int(m.sum()),
            "previsto": float(sg.mean()), "observado": float(yg.mean()),
            "gap": float(sg.mean() - yg.mean()),
            "gap_ci_low": float(lo), "gap_ci_high": float(hi),
            "gap_significativo": bool(hi < 0 or lo > 0),
            "brier": brier_score(yg, sg),
            "curva": calibration_points(yg, sg, n_bins=5, strategy="quantile"),
        })
    return out


def isotonica_por_faixa(y, cal, cru, rotulos) -> list[dict]:
    """Custo dos empates da calibração isotônica, medido DENTRO da mesma rodada.

    Compara os dois scores da mesma rodada: v2->v3 não isola isso, porque
    aquela mudança também recortou a fatia de calibração do treino.
    """
    out = []
    for g in sorted({r for r in rotulos if isinstance(r, str)}):
        m = rotulos == g
        if m.sum() < 30 or len({*y[m]}) < 2:
            continue
        a_cal, a_cru = float(roc_auc_score(y[m], cal[m])), float(roc_auc_score(y[m], cru[m]))
        out.append({
            "grupo": g, "n": int(m.sum()),
            "auc_calibrado": a_cal, "auc_cru": a_cru, "delta": a_cal - a_cru,
            "valores_distintos_calibrado": int(np.unique(cal[m]).size),
            "valores_distintos_cru": int(np.unique(cru[m]).size),
        })
    return out


def placebo_coorte(df, y, score, rotulos, alvo: str, referencia: list[str],
                   n_boot: int) -> dict:
    """Idade ou perfil? O teste que separa as duas para uma faixa qualquer.

    Monta uma coorte tirada de `referencia` e reamostrada estrato a estrato até
    reproduzir o perfil do grupo `alvo` — mesma taxa de thin-file, mesmo número
    de scores externos, mesma faixa de tempo de emprego (com o NaN do
    aposentado como faixa própria) e mesmo comprimento de histórico. Se essa
    coorte for ranqueada bem, o perfil não explica a fraqueza do alvo.

    A segunda variante acrescenta ao estrato o decil do score externo, medindo
    quanto do buraco vem de o grupo estar concentrado na região baixa do score,
    onde o modelo separa pior em qualquer idade.
    """
    m_alvo = (rotulos == alvo)
    m_ref = np.isin(rotulos, referencia)
    if m_alvo.sum() < 100 or m_ref.sum() < 500:
        return {"note": f"amostra insuficiente para {alvo}"}

    def _estratos(com_ext: bool):
        # YEARS_EMPLOYED nulo é a sentinela do aposentado: vira faixa própria,
        # senão a coorte pareada nunca reproduz um grupo majoritariamente inativo.
        k = (df["BUREAU_COUNT"].isna().astype(int).astype(str) + "|"
             + df["N_EXT_SOURCE_PRESENT"].fillna(0).clip(0, 3).astype(int).astype(str) + "|"
             + pd.cut(df["YEARS_EMPLOYED"].fillna(-1), [-2, 0, 2, 5, 10, 100],
                      labels=False).astype(str) + "|"
             + pd.cut(df["INST_COUNT"].fillna(0), [-1, 5, 15, 30, 60, 10 ** 9],
                      labels=False).astype(str))
        if com_ext:
            k = k + "|" + pd.qcut(df["EXT_SOURCE_MEAN"].rank(method="first"), 8,
                                  labels=False).astype(str)
        return k.to_numpy()

    auc_alvo = float(roc_auc_score(y[m_alvo], score[m_alvo]))
    saida = {"alvo": alvo, "referencia": referencia, "auc_alvo": auc_alvo,
             "n_alvo": int(m_alvo.sum()), "variantes": {}}
    rng = np.random.default_rng(42)

    for rotulo, com_ext in (("perfil_de_informacao", False),
                            ("perfil_de_informacao_e_nivel_do_score", True)):
        est = _estratos(com_ext)
        contagem = pd.Series(est[m_alvo]).value_counts()
        pool = {e: np.flatnonzero(m_ref & (est == e)) for e in contagem.index}
        cobertos = sum(n for e, n in contagem.items() if pool[e].size)
        aucs = []
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(pool[e], n, replace=True)
                                  for e, n in contagem.items() if pool[e].size])
            if len({*y[idx]}) == 2:
                aucs.append(float(roc_auc_score(y[idx], score[idx])))
        if not aucs:
            saida["variantes"][rotulo] = {"note": "sem coorte utilizável"}
            continue
        a = np.asarray(aucs)
        lo, hi = np.quantile(a, [0.025, 0.975])
        saida["variantes"][rotulo] = {
            "n_estratos": int(len(contagem)),
            "cobertura": float(cobertos / m_alvo.sum()),
            "auc_coorte": float(a.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "diff_alvo_menos_coorte": float(auc_alvo - a.mean()),
            "p_alvo_pior": float((auc_alvo < a).mean()),
        }
    return saida


def recorte_por_ocupacao(df, y, score, rotulos, n_boot: int) -> dict:
    """Idade ou aposentadoria? `55-65` é 68% aposentado; as faixas do meio, ~1%.

    Se a fraqueza viesse da aposentadoria — que zera o bloco de emprego pela
    sentinela DAYS_EMPLOYED=365243 — o aposentado seria pior que o ativo DENTRO
    da mesma faixa. Medir separado é o que impede confundir estado com idade.
    """
    if "NAME_INCOME_TYPE" not in df.columns:
        return {"note": "NAME_INCOME_TYPE ausente da ABT"}
    apos = (df["NAME_INCOME_TYPE"] == "Pensioner").to_numpy()
    linhas = []
    for faixa in sorted({r for r in rotulos if isinstance(r, str)}):
        mf = (rotulos == faixa)
        linha = {"faixa": faixa, "pct_aposentado": float(apos[mf].mean()),
                 "n": int(mf.sum())}
        for rot, m in (("aposentado", mf & apos), ("ativo", mf & ~apos)):
            if m.sum() < 50 or len({*y[m]}) < 2:
                linha[rot] = None
                continue
            linha[rot] = {**auc_bootstrap_ci(y[m], score[m], n_boot=n_boot),
                          "default_rate": float(y[m].mean())}
        linhas.append(linha)
    rot_ocup = np.where(apos, "aposentado", "ativo")
    return {"por_faixa": linhas,
            "eixo_aposentadoria": auc_diff_all_groups(y, score, rot_ocup, n_boot=n_boot)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    metrics = carregar("metrics.json")
    imp = carregar("feature_importance.json")

    scores = pd.read_parquet(ART / "scores.parquet")
    scores = scores[scores["split"] == "test"].reset_index(drop=True)

    # Só as colunas necessárias: a ABT tem 1.020 e não cabe em memória à toa.
    top = [f["feature"] for f in imp["gain"][:TOP_N]]
    disponiveis = set(pq.ParquetFile(ROOT / "Dados" / "abt.parquet").schema.names)
    extras = ["SK_ID_CURR", "AGE_YEARS", "BUREAU_COUNT", "N_EXT_SOURCE_PRESENT",
              "YEARS_EMPLOYED", "INST_COUNT", "EXT_SOURCE_MEAN", "NAME_INCOME_TYPE"]
    cols = [c for c in dict.fromkeys(extras + top) if c in disponiveis]
    abt = pd.read_parquet(ROOT / "Dados" / "abt.parquet", columns=cols)

    df = scores.merge(abt, on="SK_ID_CURR", how="inner")
    y = df["y_true"].to_numpy(dtype=int)
    cal = df["proba_champion"].to_numpy(dtype="float64")
    cru = df["proba_champion_raw"].to_numpy(dtype="float64")
    idade = df["AGE_YEARS"].to_numpy(dtype="float64")

    canonicas = np.asarray([age_band(v) for v in idade], dtype=object)
    sextis, meta_sextis = faixas_sextis(idade)
    janelas = faixas_5anos(idade)

    # Features numéricas do top-N (categóricas entram só como nota)
    num_top = [c for c in top if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])]
    cat_top = [c for c in top if c not in num_top]

    out = {
        "run_id": metrics.get("run", {}).get("run_id"),
        "tag": metrics.get("run", {}).get("tag"),
        "n_teste": int(len(df)),
        "n_boot": args.n_boot,
        "cortes": {
            "canonicas": avaliar_corte(y, cal, canonicas, args.n_boot),
            "sextis": {**avaliar_corte(y, cal, sextis, args.n_boot), **meta_sextis},
            "janelas_5anos": avaliar_corte(y, cal, janelas, args.n_boot),
        },
        "features": perfil_de_features(df, y, canonicas, num_top),
        "features_categoricas_ignoradas": cat_top,
        "calibracao": calibracao_por_faixa(y, cal, canonicas, args.n_boot),
        "isotonica": isotonica_por_faixa(y, cal, cru, canonicas),
        # Uma coorte pareada por faixa fraca. A referência de cada uma são as
        # faixas vizinhas que NÃO são fraqueza, para não pescar o defeito de volta.
        "placebo": {
            "<25": placebo_coorte(df, y, cal, canonicas, "<25",
                                  ["25-35", "35-45"], args.n_boot),
            "55-65": placebo_coorte(df, y, cal, canonicas, "55-65",
                                    ["35-45", "45-55"], args.n_boot),
        },
        "ocupacao": recorte_por_ocupacao(df, y, cal, canonicas, args.n_boot),
        "cobertura_scores": {
            "n_scores_parquet": int(len(scores)),
            "n_apos_join_abt": int(len(df)),
        },
    }

    (ART / "diagnostico_idade.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float))
    print(f"[diag] artifacts/diagnostico_idade.json — rodada {out['tag']}")

    if args.markdown:
        print()
        print("\n".join(relatorio(out)))


def _n(v, casas=4):
    return "—" if v is None else f"{v:.{casas}f}"


def _p(v):
    return "—" if v is None else f"{v:.1%}"


def relatorio(d: dict) -> list[str]:
    L: list[str] = []
    add = L.append

    add(f"## Diagnóstico da faixa etária — rodada `{d['tag']}`")
    add("")
    add(f"Teste: {d['n_teste']:,} clientes · bootstrap de {d['n_boot']} réplicas."
        .replace(",", "."))
    add("")

    for nome, titulo in (("canonicas", "Faixas do projeto"),
                         ("sextis", "Sextis (mesma frequência)"),
                         ("janelas_5anos", "Janelas de 5 anos (mesma largura)")):
        c = d["cortes"][nome]
        dec = c["decomposicao"]
        add(f"### {titulo}")
        add("")
        add("| Grupo | n | AUC | IC 95% | vs. demais | IC da diferença | p | Pior? |")
        add("|---|---|---|---|---|---|---|---|")
        for g in c["grupos"]:
            add(f"| {g['grupo']} | {g['n']:,} | {_n(g['auc'])} | "
                f"[{_n(g['ci_low'])} – {_n(g['ci_high'])}] | {_n(g['diff'])} | "
                f"[{_n(g['diff_ci_low'])} – {_n(g['diff_ci_high'])}] | "
                f"{_n(g['p_value'], 3)} | {'**sim**' if g['pior_que_referencia'] else 'não'} |"
                .replace(",", "."))
        add("")
        add(f"Decomposição: AUC agregado {_n(dec.get('auc_overall'))} = "
            f"{_p(dec.get('w_within'))} de pares DENTRO da faixa "
            f"(AUC {_n(dec.get('auc_within'))}) + "
            f"{_p(dec.get('w_between'))} ENTRE faixas (AUC {_n(dec.get('auc_between'))}).")
        add("")

    add("### O modelo acerta o NÍVEL de risco de cada faixa?")
    add("")
    add("| Faixa | n | Previsto | Observado | Gap | IC 95% do gap | Enviesado? | Brier |")
    add("|---|---|---|---|---|---|---|---|")
    for g in d["calibracao"]:
        add(f"| {g['grupo']} | {g['n']:,} | {_p(g['previsto'])} | {_p(g['observado'])} | "
            f"{g['gap']:+.4f} | [{_n(g['gap_ci_low'])} – {_n(g['gap_ci_high'])}] | "
            f"{'**sim**' if g['gap_significativo'] else 'não'} | {_n(g['brier'])} |"
            .replace(",", "."))
    add("")

    add("### Custo dos empates da calibração isotônica")
    add("")
    add("| Faixa | n | AUC cru | AUC calibrado | Δ | Valores distintos (cru → cal) |")
    add("|---|---|---|---|---|---|")
    for g in d["isotonica"]:
        add(f"| {g['grupo']} | {g['n']:,} | {_n(g['auc_cru'])} | {_n(g['auc_calibrado'])} | "
            f"{g['delta']:+.4f} | {g['valores_distintos_cru']:,} → "
            f"{g['valores_distintos_calibrado']:,} |".replace(",", "."))
    add("")

    add("### Disponibilidade e poder das features dominantes, por faixa")
    add("")
    faixas = [g["grupo"] for g in d["cortes"]["canonicas"]["grupos"]]
    add("| Feature | " + " | ".join(faixas) + " |")
    add("|---" * (len(faixas) + 1) + "|")
    for bloco, rotulo in (("auc_univariada", "AUC univariado"),
                          ("cobertura", "cobertura")):
        add(f"| **{rotulo}** | " + " | ".join("" for _ in faixas) + " |")
        for f in d["features"][:10]:
            cel = []
            for fx in faixas:
                v = (f["por_faixa"].get(fx) or {}).get(bloco)
                cel.append("—" if v is None else
                           (f"{v:.3f}" if bloco == "auc_univariada" else f"{v:.0%}"))
            add(f"| {f['feature']} | " + " | ".join(cel) + " |")
    add("")

    add("### Idade ou perfil? (coorte pareada, por faixa fraca)")
    add("")
    add("Coorte tirada das faixas vizinhas saudáveis e reamostrada até reproduzir o "
        "perfil do grupo fraco. Se ela for ranqueada bem, o perfil não explica a fraqueza.")
    add("")
    add("| Faixa | Coorte pareada por | Estratos | Cobertura | AUC da coorte | IC 95% | "
        "AUC da faixa | Δ | Réplicas em que a faixa é pior |")
    add("|---|---|---|---|---|---|---|---|---|")
    for faixa, pl in (d.get("placebo") or {}).items():
        for rot, v in (pl.get("variantes") or {}).items():
            if "auc_coorte" not in v:
                continue
            add(f"| `{faixa}` | {rot.replace('_', ' ')} | {v['n_estratos']} | "
                f"{_p(v['cobertura'])} | {_n(v['auc_coorte'])} | "
                f"[{_n(v['ci_low'])} – {_n(v['ci_high'])}] | {_n(pl['auc_alvo'])} | "
                f"{v['diff_alvo_menos_coorte']:+.4f} | {_p(v['p_alvo_pior'])} |")
    add("")

    oc = d.get("ocupacao") or {}
    if oc.get("por_faixa"):
        add("### Idade ou aposentadoria?")
        add("")
        add("| Faixa | % aposentado | AUC aposentado | AUC ativo | Δ (apos − ativo) |")
        add("|---|---|---|---|---|")
        for l in oc["por_faixa"]:
            a, b = l.get("aposentado"), l.get("ativo")
            fa = "—" if not a or a.get("auc") is None else \
                f"{_n(a['auc'])} (n={a['n']:,})".replace(",", ".")
            fb = "—" if not b or b.get("auc") is None else \
                f"{_n(b['auc'])} (n={b['n']:,})".replace(",", ".")
            dl = "—" if not (a and b and a.get("auc") and b.get("auc")) else \
                f"{a['auc'] - b['auc']:+.4f}"
            add(f"| {l['faixa']} | {_p(l['pct_aposentado'])} | {fa} | {fb} | {dl} |")
        add("")
        for g, v in (oc.get("eixo_aposentadoria") or {}).items():
            if v.get("diff") is None:
                continue
            add(f"- Eixo aposentadoria (sem controlar idade) — `{g}`: "
                f"Δ {v['diff']:+.4f} [{_n(v['diff_ci_low'])} – {_n(v['diff_ci_high'])}], "
                f"p = {_n(v['p_value'], 3)}")
        add("")
    return L


if __name__ == "__main__":
    main()
