"""
restaurar_improvement_log.py — Reconstrói artifacts/improvement_log.json.

Uso único. O histórico das rodadas v1..v4 vive só no JSON embutido em
docs/dossie/index.html, porque artifacts/ é gitignored. Model/train.py cria o
log do zero quando o arquivo não existe, então o primeiro treino numa árvore
limpa apagaria esse histórico. Rode isto ANTES do primeiro treino.

O dossiê guarda menos campos que o log original: v1, v2 e v4 recuperam AUC por
segmento, mas não n nem intervalo de confiança. Cada entrada reconstruída leva
`reconstruido_de` para não se passar por registro original.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOSSIE = ROOT / "docs" / "dossie" / "index.html"
SAIDA = ROOT / "artifacts" / "improvement_log.json"

# Ordem cronológica das rodadas (TODO.md §1 e §2). O dossiê não grava data.
ORDEM = ["v1-baseline", "v2-abt-corrigida", "v3-calibrado", "v4-pesos-idade"]


def ler_dossie() -> dict:
    txt = DOSSIE.read_text(encoding="utf-8")
    ini = txt.index("{", txt.index("const D = "))
    nivel = 0
    for i in range(ini, len(txt)):
        if txt[i] == "{":
            nivel += 1
        elif txt[i] == "}":
            nivel -= 1
            if nivel == 0:
                return json.loads(txt[ini:i + 1])
    raise SystemExit("[restaurar] bloco `const D` não fechou em index.html")


def segmentos_com_ci(fair: dict) -> dict:
    """Segmentos da rodada vigente: o dossiê guarda n e IC só para ela."""
    return {dim: {g["grupo"]: {"auc": g["auc"], "ci_low": g["ci_low"],
                               "ci_high": g["ci_high"], "n": g["n"]}
                  for g in grupos}
            for dim, grupos in (fair.get("dimensoes") or {}).items()}


def segmentos_sem_ci(seg: dict) -> dict:
    """Rodadas antigas: o dossiê achatou o segmento para só o AUC."""
    return {dim: {nome: {"auc": auc, "ci_low": None, "ci_high": None, "n": None}
                  for nome, auc in grupos.items()}
            for dim, grupos in (seg or {}).items()}


def montar(D: dict) -> dict:
    run, served, champ = D["run"], D["served"], D["champion"]
    vigente = run["tag"]
    seg_vigente = segmentos_com_ci(D["fairness"])

    entradas: dict[str, dict] = {}

    for r in D["evolucao"]:
        tag = r["tag"]
        # A rodada vigente é a única com run_id, data e IC por segmento.
        e_vigente = tag == vigente
        entradas[tag] = {
            "run_id": run["run_id"] if e_vigente else None,
            "tag": tag,
            "status": "aceita",
            "trained_at": run["trained_at"] if e_vigente else None,
            "n_features": r.get("features"),
            "sample": None,
            "auc": r.get("auc"),
            "ks": r.get("ks"),
            "brier": r.get("brier"),
            "calibrated": r.get("brier") is not None and r["brier"] < 0.10,
            "auc_baseline": D["baseline"]["auc"] if e_vigente else None,
            "threshold": r.get("threshold"),
            "approval_rate": r.get("aprovacao"),
            "segments": seg_vigente if e_vigente else segmentos_sem_ci(r.get("segmentos")),
            "reconstruido_de": "docs/dossie/index.html",
        }

    for r in D.get("rejeitadas") or []:
        entradas[r["tag"]] = {
            "run_id": None, "tag": r["tag"], "status": "rejeitada",
            "trained_at": None, "n_features": None, "sample": None,
            "auc": r.get("auc"), "ks": None, "brier": None,
            "calibrated": None, "auc_baseline": None,
            "threshold": None, "approval_rate": None,
            "motivo": r.get("motivo"),
            "segments": segmentos_sem_ci(r.get("segmentos")),
            "reconstruido_de": "docs/dossie/index.html",
        }

    ordenadas = [entradas.pop(t) for t in ORDEM if t in entradas]
    ordenadas += [entradas[t] for t in entradas]        # a rodada do Airflow por último
    return {"runs": ordenadas}


def main() -> None:
    if SAIDA.exists():
        raise SystemExit(f"[restaurar] {SAIDA} já existe — nada foi tocado.\n"
                         "Apague-o à mão se quiser mesmo reconstruir.")
    D = ler_dossie()
    log = montar(D)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(log, indent=2, ensure_ascii=False))

    print(f"[restaurar] {SAIDA.relative_to(ROOT)} — {len(log['runs'])} rodadas")
    for r in log["runs"]:
        auc = "—" if r["auc"] is None else f"{r['auc']:.4f}"
        j25 = ((r.get("segments") or {}).get("age_band") or {}).get("<25", {}).get("auc")
        print(f"  {r['status']:10s} {r['tag']:45s} AUC {auc}"
              f"  <25 {'—' if j25 is None else f'{j25:.4f}'}")


if __name__ == "__main__":
    main()
