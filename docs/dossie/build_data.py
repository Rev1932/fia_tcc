"""
build_data.py — Injeta os números da rodada canônica no dossiê.

Lê `artifacts/*.json` e reescreve o bloco de dados dentro de
`docs/dossie/index.html`, entre os marcadores DADOS-INICIO / DADOS-FIM.

É o que garante que o dossiê nunca tenha número digitado à mão: re-treinar e
rodar este script atualiza a página inteira. Foi a cópia manual entre
documentos que produziu os dois conjuntos de métricas conflitantes que este
ciclo veio corrigir.

Uso:
    python docs/dossie/build_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artifacts"
HTML = Path(__file__).resolve().parent / "index.html"

INICIO = "// DADOS-INICIO"
FIM = "// DADOS-FIM"


def ler(nome: str) -> dict:
    p = ART / nome
    if not p.exists():
        raise SystemExit(f"[dossie] {p} não existe. Rode `python Model/train.py`.")
    return json.loads(p.read_text())


def fraqueza_real(grupo: dict, geral: dict) -> bool:
    """Pior que os DEMAIS grupos do mesmo eixo, pelo IC bootstrap da diferença.

    Artefato antigo não tem o campo; nesse caso vale o critério anterior
    (IC do grupo abaixo do IC geral), que era inválido mas é o que existe.
    """
    if "fraqueza_confirmada" in grupo:
        return bool(grupo["fraqueza_confirmada"])
    return (grupo.get("ci_high") is not None
            and geral.get("ci_low") is not None
            and grupo["ci_high"] < geral["ci_low"])


def montar() -> dict:
    m = ler("metrics.json")
    fair = ler("fairness.json")
    imp = ler("feature_importance.json")
    curvas = ler("curves.json")
    log = ler("improvement_log.json")
    perfil = ler("abt_profile.json")

    geral = fair.get("overall", {})
    dims = {}
    for dim, grupos in (fair.get("dimensions") or {}).items():
        dims[dim] = [{
            "grupo": g["group"], "n": g["n"], "auc": g.get("auc"),
            "ci_low": g.get("ci_low"), "ci_high": g.get("ci_high"),
            "vs_referencia": g.get("vs_referencia"),
            "aprovacao": g.get("approval_rate"), "default": g.get("default_rate"),
            "fraqueza_real": fraqueza_real(g, geral),
        } for g in grupos]

    todas = [r for r in log.get("runs", []) if not r.get("sample")]
    rodadas = [r for r in todas if r.get("status", "aceita") == "aceita"]
    rejeitadas = [{"tag": r.get("tag"), "auc": r.get("auc"),
                   "motivo": r.get("motivo"),
                   "segmentos": {d: {k: v.get("auc") for k, v in g.items()}
                                 for d, g in (r.get("segments") or {}).items()}}
                  for r in todas if r.get("status") == "rejeitada"]
    evolucao = []
    for r in rodadas:
        seg = {}
        for d, grupos in (r.get("segments") or {}).items():
            seg[d] = {k: v.get("auc") for k, v in grupos.items()}
        evolucao.append({
            "tag": r.get("tag"), "features": r.get("n_features"), "auc": r.get("auc"),
            "ks": r.get("ks"), "brier": r.get("brier"), "threshold": r.get("threshold"),
            "aprovacao": r.get("approval_rate"), "segmentos": seg,
        })

    nulos = sorted(
        ({"coluna": c["column_name"], "taxa": float(c.get("null_percentage") or 0) / 100}
         for c in perfil["columns"]),
        key=lambda x: -x["taxa"])[:8]

    sweep = (curvas.get("champion", {}).get("valid", {}) or {}).get("sweep", [])
    cal_pts = (curvas.get("champion", {}).get("test", {}) or {}).get("calibration", [])
    cal_raw = (curvas.get("champion", {}).get("test", {}) or {}).get("calibration_raw", [])
    roc = (curvas.get("champion", {}).get("test", {}) or {}).get("roc", {})
    roc_base = (curvas.get("baseline", {}).get("test", {}) or {}).get("roc", {})

    return {
        "run": m["run"],
        "served": m.get("served") or {**m["champion"], "model": "champion"},
        "baseline": m["baseline"],
        "champion": m["champion"],
        "calibrated": m.get("calibrated"),
        "business": m["business"],
        "fairness": {"geral": geral, "dimensoes": dims},
        "importancia": {
            "por_origem": imp.get("by_source", {}),
            "top": imp.get("gain", [])[:12],
        },
        "curvas": {
            "roc": roc.get("points", []),
            "roc_auc": roc.get("auc"),
            "roc_baseline": roc_base.get("points", []),
            "roc_baseline_auc": roc_base.get("auc"),
            "sweep": [{"t": p["threshold"], "custo": p["cost"],
                       "aprovacao": p["approval_rate"],
                       "inad_aprovada": p["default_rate_approved"]}
                      for p in sweep],
            "calibracao": cal_pts,
            "calibracao_crua": cal_raw,
        },
        "decomposicao": fair.get("decomposicao"),
        "criterio": fair.get("criterio"),
        "evolucao": evolucao,
        "rejeitadas": rejeitadas,
        "nulos": nulos,
        "abt": {"linhas": perfil["n_rows"], "colunas": perfil["n_columns"]},
    }


def main() -> None:
    dados = montar()
    bloco = json.dumps(dados, ensure_ascii=False, separators=(",", ":"))

    if not HTML.exists():
        raise SystemExit(f"[dossie] {HTML} não existe.")
    texto = HTML.read_text()
    if INICIO not in texto or FIM not in texto:
        raise SystemExit(f"[dossie] marcadores {INICIO}/{FIM} não encontrados em index.html")

    antes = texto.split(INICIO)[0]
    depois = texto.split(FIM, 1)[1]
    novo = f"{antes}{INICIO}\nconst D = {bloco};\n{FIM}{depois}"
    HTML.write_text(novo)

    print(f"[dossie] index.html atualizado com a rodada {dados['run']['run_id']}")
    print(f"[dossie] {len(bloco) / 1024:.0f} KB de dados injetados")


if __name__ == "__main__":
    main()
