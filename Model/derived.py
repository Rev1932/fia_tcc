"""
derived.py — Recalcula as variáveis derivadas de um cliente.

Por que existe: várias features do modelo não são medidas, são calculadas a
partir de outras (razão crédito/renda, média dos scores externos, prazo...).
Numa simulação what-if, mudar `AMT_CREDIT` sem recalcular `CREDIT_TERM` e
`CREDIT_INCOME_RATIO` produz um cenário que não existe — e o score quase não
se move, dando a impressão errada de que a variável não importa.

As fórmulas espelham `DataPipeline/abt_transform.py`. `tests/test_derived.py`
compara este cálculo com os valores realmente gravados na ABT para clientes
reais: é esse teste que impede os dois lados de divergirem em silêncio.
"""
from __future__ import annotations

import math

# nome -> (numerador, denominador). Espelha `abt.ratios` do DataPipeline/config.yaml.
RATIOS: dict[str, tuple[str, str]] = {
    "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
    "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    "CREDIT_TERM": ("AMT_ANNUITY", "AMT_CREDIT"),
    "EMPLOYED_AGE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
    "REGISTRATION_AGE_RATIO": ("DAYS_REGISTRATION", "DAYS_BIRTH"),
    "ID_PUBLISH_AGE_RATIO": ("DAYS_ID_PUBLISH", "DAYS_BIRTH"),
    "PHONE_CHANGE_AGE_RATIO": ("DAYS_LAST_PHONE_CHANGE", "DAYS_BIRTH"),
    "INCOME_PER_PERSON": ("AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    "CREDIT_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
}

EXT_COLS = ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")

# Derivadas de DAYS_* criadas em data_sanitization.py
DAYS_TO_YEARS = {"AGE_YEARS": "DAYS_BIRTH", "YEARS_EMPLOYED": "DAYS_EMPLOYED"}

DERIVED_NAMES = (
    tuple(RATIOS)
    + ("EXT_SOURCE_MEAN", "EXT_SOURCE_MAX", "EXT_SOURCE_MIN", "EXT_SOURCE_STD",
       "EXT_SOURCE_PROD", "N_EXT_SOURCE_PRESENT")
    + tuple(DAYS_TO_YEARS)
)


def _num(v):
    """Valor numérico utilizável, ou None."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def recompute(record: dict, only_if_present: bool = True) -> dict:
    """Devolve as derivadas recalculadas a partir das colunas-base de `record`.

    `only_if_present=True` recalcula apenas as derivadas que já existem no
    registro — assim um payload parcial em /predict não ganha colunas que o
    cliente não mandou.
    """
    out: dict = {}

    def guardar(nome, valor):
        if only_if_present and nome not in record:
            return
        out[nome] = valor

    for nome, (num_col, den_col) in RATIOS.items():
        num, den = _num(record.get(num_col)), _num(record.get(den_col))
        # `.replace(0, np.nan)` no pipeline: denominador zero vira nulo
        guardar(nome, None if num is None or not den else num / den)

    for nome, origem in DAYS_TO_YEARS.items():
        d = _num(record.get(origem))
        # `+ 0.0` normaliza -0.0 para 0.0 (o pipeline grava -0.0 quando DAYS=0)
        guardar(nome, None if d is None else round(-d / 365.25, 1) + 0.0)

    presentes = [v for v in (_num(record.get(c)) for c in EXT_COLS) if v is not None]
    n = len(presentes)
    guardar("N_EXT_SOURCE_PRESENT", n)
    guardar("EXT_SOURCE_MEAN", sum(presentes) / n if n else None)
    guardar("EXT_SOURCE_MAX", max(presentes) if n else None)
    guardar("EXT_SOURCE_MIN", min(presentes) if n else None)
    # pandas Series.std() usa ddof=1; com 1 só valor o resultado é NaN
    if n >= 2:
        media = sum(presentes) / n
        var = sum((x - media) ** 2 for x in presentes) / (n - 1)
        guardar("EXT_SOURCE_STD", math.sqrt(var))
    else:
        guardar("EXT_SOURCE_STD", None)
    # `prod(min_count=3)`: só existe quando os três estão presentes
    guardar("EXT_SOURCE_PROD",
            math.prod(presentes) if n == len(EXT_COLS) else None)

    return out


def apply_changes(record: dict, changes: dict) -> dict:
    """Aplica `changes` e propaga o efeito para as variáveis derivadas.

    É o que faz uma simulação what-if ser coerente: mudar o valor do crédito
    move também o comprometimento de renda e o prazo, como aconteceria numa
    proposta real.
    """
    novo = {**record, **changes}
    novo.update(recompute(novo, only_if_present=True))
    return novo
