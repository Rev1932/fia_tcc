"""
run_summary.py — Imprime os números oficiais da rodada congelada.

Existe por um motivo específico: antes deste ciclo havia DOIS conjuntos de
métricas circulando (654/0,50/71,7% nos artefatos e 783/0,47/69,1% nos decks),
porque cada número era copiado à mão de um lugar diferente.

Agora há um comando só. O que ele imprime é o que vai para o slide.

Uso:
    python Model/run_summary.py              # tabela legível
    python Model/run_summary.py --markdown   # pronto para colar no documento
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def carregar(nome: str) -> dict:
    p = ART / nome
    if not p.exists():
        raise SystemExit(f"[resumo] {p} não existe. Rode `python Model/train.py`.")
    return json.loads(p.read_text())


def pct(v) -> str:
    return "—" if v is None else f"{v:.1%}"


def num(v, casas: int = 4) -> str:
    return "—" if v is None else f"{v:.{casas}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    m = carregar("metrics.json")
    run, base, champ, biz = m["run"], m["baseline"], m["champion"], m["business"]
    cal = m.get("calibrated") or {}
    fair = carregar("fairness.json")
    imp = carregar("feature_importance.json")

    L: list[str] = []
    add = L.append

    add(f"# Rodada canônica `{run['run_id']}`")
    add("")
    add(f"- **Tag:** {run.get('tag')}")
    add(f"- **Treinada em:** {run.get('trained_at')}")
    add(f"- **Commit:** `{run.get('git_sha')}`")
    add(f"- **Base:** {run['n_rows']:,} clientes × {run['n_features']:,} features"
        .replace(",", "."))
    add(f"- **Split:** treino {run['n_train']:,} · validação {run['n_valid']:,} · "
        f"teste {run['n_test']:,}".replace(",", "."))
    add("")

    servido = m.get("served") or {**champ, "model": "champion"}
    add("## Desempenho (conjunto de teste)")
    add("")
    add(f"O modelo **servido** é `{servido.get('model')}`. São os números dele que vão "
        "para slide, documento e dossiê — não os do modelo cru.")
    add("")
    add("| Métrica | Baseline (Reg. Logística) | **Servido** | Campeão cru (pré-calibração) |")
    add("|---|---|---|---|")
    add(f"| AUC | {num(base['auc'])} | **{num(servido['auc'])}** | {num(champ['auc'])} |")
    add(f"| KS | {num(base['ks'])} | **{num(servido['ks'])}** | {num(champ['ks'])} |")
    add(f"| Brier | {num(base.get('brier'))} | **{num(servido.get('brier'))}** | "
        f"{num(champ.get('brier'))} |")
    add("")
    add(f"Ganho sobre o baseline: **{servido['auc'] - base['auc']:+.4f}** de AUC, "
        f"**{servido['ks'] - base['ks']:+.4f}** de KS.")
    add("")
    add("> A diferença entre servido e cru no AUC/KS é de terceira casa: a isotônica "
        "é monotônica, mas cria empates no score. O que ela realmente muda é o Brier.")
    add("")

    add("## Controle de overfitting")
    add("")
    add(f"AUC treino **{num(champ['auc_train'])}** → validação "
        f"**{num(champ['auc_valid'])}** → teste **{num(champ['auc'])}**.")
    add("")
    add("Validação e teste praticamente empatam — é esse empate, e não o gap "
        "treino→validação, que mostra generalização. Early stopping parou na "
        f"iteração **{champ['best_iteration']}**.")
    add("")

    if cal:
        add("## Calibração")
        add("")
        add(f"Método: **{cal.get('method')}**, ajustado em {cal.get('n_calib'):,} "
            f"clientes reservados exclusivamente para isso.".replace(",", "."))
        add("")
        add(f"Brier **{num(cal.get('brier_before'))} → {num(cal.get('brier'))}**. "
            f"O AUC não muda ({num(cal.get('auc'))}) porque a isotônica é "
            "monotônica: ela corrige a probabilidade, não a ordenação.")
        add("")
        add("Consequência prática: o score passa a ser lido como P(inadimplência) "
            f"real, e o corte ótimo cai para **{biz['threshold']:.2f}**.")
        add("")

    add("## Régua de decisão")
    add("")
    add(f"- Custo do falso negativo: **{biz['cost_false_negative']}** · "
        f"falso positivo: **{biz['cost_false_positive']}** "
        f"(razão {biz['cost_false_negative'] / biz['cost_false_positive']:.0f}:1)")
    add(f"- Threshold: **{biz['threshold']:.2f}**")
    add(f"- Taxa de aprovação: **{pct(biz['approval_rate'])}**")
    add("")

    add("## Origem do sinal")
    add("")
    add("| Tabela | % da importância (gain) |")
    add("|---|---|")
    for tabela, frac in imp.get("by_source", {}).items():
        add(f"| {tabela} | {frac:.1%} |")
    relacional = sum(v for k, v in imp.get("by_source", {}).items() if k != "application")
    add("")
    add(f"As tabelas relacionais agregadas na ABT respondem por **{relacional:.1%}** "
        "da importância — é a resposta numérica para \"a ABT valeu a pena?\".")
    add("")
    add("**Top 10 variáveis:**")
    add("")
    add("| # | Variável | % | Origem |")
    add("|---|---|---|---|")
    for i in imp.get("gain", [])[:10]:
        add(f"| {i['rank']} | `{i['feature']}` | {i['importance_pct']:.2%} | {i['source_table']} |")
    add("")

    add("## Onde o modelo é (e não é) confiável")
    add("")
    geral = fair.get("overall", {})
    add(f"AUC geral: **{num(geral.get('auc'))}** "
        f"[{num(geral.get('ci_low'))} – {num(geral.get('ci_high'))}]")
    add("")
    add((fair.get("criterio") or {}).get("descricao")
        or ("O intervalo de confiança é bootstrap. Um grupo só conta como fraqueza "
            "real quando seu IC **não sobrepõe** o geral (critério anterior)."))
    add("")
    for dim, grupos in (fair.get("dimensions") or {}).items():
        dec = (fair.get("decomposicao") or {}).get(dim) or {}
        add(f"### {dim}")
        add("")
        novo_criterio = any("fraqueza_confirmada" in g for g in grupos)
        if novo_criterio:
            add("| Grupo | n | AUC | IC 95% | Δ vs. demais | IC da diferença | p | "
                "Fraqueza? | Aprovação | Inadimplência real |")
            add("|---|---|---|---|---|---|---|---|---|---|")
        else:
            add("| Grupo | n | AUC | IC 95% | Fraqueza real? | Aprovação | "
                "Inadimplência real |")
            add("|---|---|---|---|---|---|---|")
        for g in grupos:
            if novo_criterio:
                v = g.get("vs_referencia") or {}
                add(f"| {g['group']} | {g['n']:,} | {num(g.get('auc'))} | "
                    f"[{num(g.get('ci_low'))} – {num(g.get('ci_high'))}] | "
                    f"{num(v.get('diff'))} | "
                    f"[{num(v.get('diff_ci_low'))} – {num(v.get('diff_ci_high'))}] | "
                    f"{num(v.get('p_value'), 3)} | "
                    f"{'**sim**' if g.get('fraqueza_confirmada') else 'não'} | "
                    f"{pct(g.get('approval_rate'))} | "
                    f"{pct(g.get('default_rate'))} |".replace(",", "."))
            else:
                lo = geral.get("ci_low")
                real = (g.get("ci_high") is not None and lo is not None
                        and g["ci_high"] < lo)
                add(f"| {g['group']} | {g['n']:,} | {num(g.get('auc'))} | "
                    f"[{num(g.get('ci_low'))} – {num(g.get('ci_high'))}] | "
                    f"{'**sim**' if real else 'não'} | {pct(g.get('approval_rate'))} | "
                    f"{pct(g.get('default_rate'))} |".replace(",", "."))
        add("")
        if dec.get("auc_within") is not None:
            add(f"Decomposição do AUC agregado: {pct(dec.get('w_within'))} dos pares são "
                f"DENTRO do mesmo grupo (AUC {num(dec.get('auc_within'))}) e "
                f"{pct(dec.get('w_between'))} são ENTRE grupos "
                f"(AUC {num(dec.get('auc_between'))}).")
            add("")

    log_path = ART / "improvement_log.json"
    if log_path.exists():
        runs = [r for r in json.loads(log_path.read_text()).get("runs", [])
                if not r.get("sample")]
        if len(runs) > 1:
            add("## Evolução entre as rodadas")
            add("")
            add("| Rodada | Status | Features | AUC | KS | Brier | Corte | Aprovação |")
            add("|---|---|---|---|---|---|---|---|")
            for r in runs:
                # Rodada reconstruída do dossiê não tem todos os campos: o log
                # original é gitignored e só o AUC por segmento sobreviveu.
                add(f"| {r.get('tag')} | {r.get('status', 'aceita')} | "
                    f"{'—' if r.get('n_features') is None else format(r['n_features'], ',')} | "
                    f"{num(r.get('auc'))} | {num(r.get('ks'))} | {num(r.get('brier'))} | "
                    f"{'—' if r.get('threshold') is None else format(r['threshold'], '.2f')} | "
                    f"{pct(r.get('approval_rate'))} |".replace(",", "."))
            add("")

    texto = "\n".join(L)
    if args.markdown:
        print(texto)
    else:
        print(texto.replace("**", "").replace("`", ""))


if __name__ == "__main__":
    main()
