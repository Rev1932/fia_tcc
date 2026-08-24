"""
policy.py — Régua de decisão consciente da incerteza (Fix 8 do plano).

O trabalho v1 dizia "human-in-the-loop nos segmentos fracos" como frase de
slide, sem implementação. Aqui isso vira regra executável.

A ideia: um corte único trata como igual uma probabilidade de 46,9% e uma de
47,1%, quando o modelo não tem essa precisão. Em vez disso, três faixas:

    APROVAR   p <  thr - w/2
    REVISAR   thr - w/2 <= p < thr + w/2      -> vai para análise humana
    NEGAR     p >= thr + w/2

E a largura `w` não é fixa: cresce nos segmentos onde o AUC medido é pior
que o geral. Ou seja, quanto menos o modelo enxerga aquele perfil, mais
casos ele manda para um humano — em vez de fingir a mesma confiança.

Importante: a faixa alarga por *disponibilidade de informação* (thin-file) e
por faixa etária SOMENTE porque a fraqueza foi medida ali. Diferenciar o
CORTE por gênero ou idade seria discriminação direta e não é feito: o que
muda é quanto do caso vai para revisão humana, nunca o critério de risco.
"""
from __future__ import annotations

from typing import Literal

Decision = Literal["APROVAR", "REVISAR", "NEGAR"]

# Largura base da faixa cinza, em pontos de probabilidade.
DEFAULT_BAND = 0.05
# Multiplicador aplicado quando o segmento do cliente tem AUC medido abaixo
# do geral (ver artifacts/fairness.json).
LOW_CONFIDENCE_FACTOR = 2.0


def decide(p: float, threshold: float, band: float = 0.0) -> Decision:
    """Decisão em três faixas. `band=0` reproduz o corte único do v1."""
    if band <= 0:
        return "NEGAR" if p >= threshold else "APROVAR"
    half = band / 2
    if p < threshold - half:
        return "APROVAR"
    if p >= threshold + half:
        return "NEGAR"
    return "REVISAR"


def low_confidence_groups(fairness: dict) -> dict[str, list[str]]:
    """Segmentos cujo IC de AUC fica INTEIRAMENTE abaixo do IC geral.

    Usar o intervalo, e não o ponto, é o que separa fraqueza real de ruído
    amostral — sem isso, um grupo pequeno com AUC menor por acaso entraria
    na lista e mandaria gente para revisão humana sem motivo.
    """
    overall = fairness.get("overall") or {}
    ref = overall.get("ci_low")
    if ref is None:
        return {}

    out: dict[str, list[str]] = {}
    for dim, groups in (fairness.get("dimensions") or {}).items():
        fracos = [g["group"] for g in groups
                  if g.get("ci_high") is not None and g["ci_high"] < ref]
        if fracos:
            out[dim] = fracos
    return out


def band_for(client: dict, fairness: dict, base_band: float = DEFAULT_BAND) -> float:
    """Largura da faixa cinza para um cliente, dado o que foi medido."""
    fracos = low_confidence_groups(fairness)
    if not fracos:
        return base_band

    from MLOps.app.settings import DIMENSIONS  # noqa: F401  (documenta a origem)

    idade = client.get("AGE_YEARS")
    thin = client.get("BUREAU_COUNT") is None
    rotulos = []
    if idade is not None:
        for hi, label in ((25, "<25"), (35, "25-35"), (45, "35-45"),
                          (55, "45-55"), (65, "55-65")):
            if idade < hi:
                rotulos.append(("age_band", label))
                break
        else:
            rotulos.append(("age_band", "65+"))
    rotulos.append(("thin_file",
                    "thin-file (sem bureau)" if thin else "com histórico de bureau"))

    for dim, label in rotulos:
        if label in fracos.get(dim, []):
            return base_band * LOW_CONFIDENCE_FACTOR
    return base_band


def policy_summary(fairness: dict, base_band: float = DEFAULT_BAND) -> dict:
    return {
        "base_band": base_band,
        "low_confidence_factor": LOW_CONFIDENCE_FACTOR,
        "low_confidence_groups": low_confidence_groups(fairness),
        "criterio": ("um grupo entra na lista quando o limite SUPERIOR do seu IC de "
                     "AUC fica abaixo do limite INFERIOR do IC geral — ou seja, "
                     "quando a diferença não é explicável por ruído amostral"),
    }
