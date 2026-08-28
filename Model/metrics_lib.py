"""
metrics_lib.py — Métricas compartilhadas entre treino, avaliação e API.

Fonte única de verdade: `Model/train.py` (que gera os números oficiais) e
`MLOps/app` (que os recalcula ao vivo) importam daqui. Assim, a resposta
"é o mesmo código que gerou o número do slide" é literalmente verdadeira.

Convenção de decisão usada em todo o projeto:
    aprovado  <=>  score <  threshold   (risco baixo)
    negado    <=>  score >= threshold   (risco alto)

Logo, na matriz de confusão com `pred = (score >= threshold)`:
    TN = aprovou bom pagador     (acerto)
    FP = negou bom pagador       (custa margem/receita)
    FN = aprovou mau pagador     (custa o crédito inteiro)
    TP = negou mau pagador       (acerto)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# --------------------------------------------------------------------------
# Métricas básicas — copiadas de train.py SEM alteração de semântica.
# Mudar qualquer detalhe aqui (grid, desempate) muda o threshold oficial.
# --------------------------------------------------------------------------


def ks_statistic(y_true, y_score) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def business_threshold(y_true, y_score, cost_fn: float, cost_fp: float):
    """Threshold que minimiza o custo esperado (FN caro, FP barato).

    Grid de 99 pontos e desempate estrito (`<`): o PRIMEIRO mínimo vence.
    Não "otimizar" — o valor congelado em artifacts/ depende disso.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf
    y_true = np.asarray(y_true)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    approval_rate = float((y_score < best_t).mean())  # aprovado = baixo risco
    return best_t, approval_rate


# --------------------------------------------------------------------------
# Matriz de confusão, custo e derivadas
# --------------------------------------------------------------------------


def confusion_at(y_true, y_score, threshold: float) -> dict:
    """Confusão no threshold dado. Aprovado = score < threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    denied = y_score >= threshold
    good = y_true == 0
    return {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "tn": int((~denied & good).sum()),
        "fp": int((denied & good).sum()),
        "fn": int((~denied & ~good).sum()),
        "tp": int((denied & ~good).sum()),
    }


def cost_at(cm: dict, cost_fn: float, cost_fp: float) -> float:
    return float(cost_fn * cm["fn"] + cost_fp * cm["fp"])


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def derived_metrics(cm: dict) -> dict:
    """Métricas técnicas e de negócio a partir da confusão."""
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n = tn + fp + fn + tp
    approved = tn + fn
    denied = tp + fp
    return {
        "accuracy": _safe_div(tp + tn, n),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "approval_rate": _safe_div(approved, n),
        # As duas métricas que o negócio realmente lê:
        "default_rate_approved": _safe_div(fn, approved),
        "default_rate_denied": _safe_div(tp, denied),
    }


def threshold_sweep(y_true, y_score, cost_fn: float, cost_fp: float,
                    n_points: int = 99, lo: float = 0.01, hi: float = 0.99) -> list[dict]:
    """Varredura de thresholds: confusão + custo + derivadas em cada ponto.

    Com n_points=99, lo=0.01, hi=0.99 o mínimo desta varredura é exatamente
    o que `business_threshold` devolve (travado em tests/test_metrics_lib.py).
    """
    out = []
    for t in np.linspace(lo, hi, n_points):
        cm = confusion_at(y_true, y_score, float(t))
        out.append({**cm, "cost": cost_at(cm, cost_fn, cost_fp), **derived_metrics(cm)})
    return out


# --------------------------------------------------------------------------
# Curvas (gravadas em artifacts/curves.json e servidas pela API)
# --------------------------------------------------------------------------


def _downsample(n: int, max_points: int) -> np.ndarray:
    """Índices preservando extremos — evita JSON de MBs com 30k pontos de ROC."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def roc_points(y_true, y_score, max_points: int = 300) -> dict:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = _downsample(len(fpr), max_points)
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "points": [
            {"fpr": float(fpr[i]), "tpr": float(tpr[i]),
             "threshold": float(thr[i]) if np.isfinite(thr[i]) else None}
            for i in idx
        ],
    }


def ks_curve(y_true, y_score, max_points: int = 300) -> dict:
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ks = tpr - fpr
    best = int(np.argmax(ks))
    idx = _downsample(len(fpr), max_points)
    return {
        "ks": float(ks[best]),
        "ks_threshold": float(thr[best]) if np.isfinite(thr[best]) else None,
        "points": [
            {"threshold": float(thr[i]) if np.isfinite(thr[i]) else None,
             "tpr": float(tpr[i]), "fpr": float(fpr[i]), "ks": float(ks[i])}
            for i in idx
        ],
    }


def decile_table(y_true, y_score, q: int = 10) -> list[dict]:
    """Tabela de decis de risco — a linguagem que um banco reconhece."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)               # decil 1 = maior risco
    y_sorted, s_sorted = y_true[order], y_score[order]
    n = y_true.size
    events, nonevents = y_true.sum(), n - y_true.sum()
    overall = _safe_div(events, n)

    bounds = np.linspace(0, n, q + 1).astype(int)
    rows, cum_e, cum_ne = [], 0, 0
    for d in range(q):
        lo, hi = bounds[d], bounds[d + 1]
        chunk, scores = y_sorted[lo:hi], s_sorted[lo:hi]
        e, ne = int(chunk.sum()), int(len(chunk) - chunk.sum())
        cum_e += e
        cum_ne += ne
        rows.append({
            "decile": d + 1,
            "n": int(hi - lo),
            "min_score": float(scores.min()) if len(scores) else None,
            "max_score": float(scores.max()) if len(scores) else None,
            "avg_score": float(scores.mean()) if len(scores) else None,
            "events": e,
            "event_rate": _safe_div(e, hi - lo),
            "lift": _safe_div(_safe_div(e, hi - lo), overall) if overall else 0.0,
            "cum_event_pct": _safe_div(cum_e, events),
            "cum_nonevent_pct": _safe_div(cum_ne, nonevents),
            "ks": abs(_safe_div(cum_e, events) - _safe_div(cum_ne, nonevents)),
        })
    return rows


# --------------------------------------------------------------------------
# Calibração
# --------------------------------------------------------------------------


def brier_score(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


def calibration_points(y_true, y_score, n_bins: int = 10, strategy: str = "quantile") -> list[dict]:
    """Curva de confiabilidade: probabilidade prevista vs. frequência observada.

    Um modelo calibrado fica na diagonal. Com `is_unbalance=true` e sem
    calibração, a curva fica bem acima dela — que é o diagnóstico da §1.1.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(y_score, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(y_score.min(), y_score.max(), n_bins + 1)
    idx = np.clip(np.digitize(y_score, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": b,
            "n": int(m.sum()),
            "mean_predicted": float(y_score[m].mean()),
            "observed_rate": float(y_true[m].mean()),
            "gap": float(y_score[m].mean() - y_true[m].mean()),
        })
    return rows


# --------------------------------------------------------------------------
# Incerteza — separa fraqueza real de ruído amostral (§1.3 do plano)
# --------------------------------------------------------------------------


def auc_bootstrap_ci(y_true, y_score, n_boot: int = 1000, alpha: float = 0.05,
                     random_state: int = 42) -> dict:
    """AUC com intervalo de confiança por bootstrap.

    Existe por um motivo específico: o segmento <25 anos tem só ~2.4k clientes
    no teste. Sem IC, não dá para saber se o AUC menor é fraqueza do modelo ou
    tamanho de amostra — e essa distinção muda a conclusão do trabalho.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = y_true.size
    if n == 0 or y_true.min() == y_true.max():
        return {"auc": None, "ci_low": None, "ci_high": None, "n": int(n),
                "n_events": int(y_true.sum()), "note": "sem ambas as classes"}

    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():          # reamostra degenerada: descarta
            continue
        boots.append(roc_auc_score(yb, y_score[idx]))

    if not boots:
        return {"auc": point, "ci_low": None, "ci_high": None, "n": int(n),
                "n_events": int(y_true.sum()), "note": "bootstrap degenerado"}

    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {
        "auc": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_width": float(hi - lo),
        "n": int(n),
        "n_events": int(y_true.sum()),
        "n_boot": len(boots),
    }


# --------------------------------------------------------------------------
# Comparação entre segmentos — o AUC de um grupo contra os DEMAIS
#
# Por que não comparar o AUC de um segmento com o AUC geral: o segmento é
# SUBCONJUNTO do geral (as estimativas são aninhadas) e o AUC agregado conta
# pares de grupos diferentes, que não existem em nenhum AUC intra-grupo.
# São duas quantidades que não medem a mesma coisa.
# --------------------------------------------------------------------------


def _auc_par(y_true, y_score, mask_ev, mask_nev) -> float | None:
    """AUC de um bloco de pares: eventos de um grupo x não-eventos de outro."""
    s_ev, s_nev = y_score[mask_ev], y_score[mask_nev]
    if s_ev.size == 0 or s_nev.size == 0:
        return None
    y = np.concatenate([np.ones(s_ev.size, dtype=int), np.zeros(s_nev.size, dtype=int)])
    return float(roc_auc_score(y, np.concatenate([s_ev, s_nev])))


def auc_within_between(y_true, y_score, groups) -> dict:
    """Decompõe o AUC agregado em pares DENTRO e ENTRE grupos.

    Identidade exata: o AUC é a proporção de pares (evento, não-evento) bem
    ordenados, então particionar os pares por (grupo do evento, grupo do
    não-evento) reconstrói o total. `w_within` diz que fração da evidência do
    AUC geral vem de comparações dentro do mesmo grupo — é o número que mostra
    sobre quantos pares um segmento pequeno está realmente sendo medido.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups, dtype=object)

    nomes = sorted({g for g in groups[~pd_isna(groups)]})
    ev = {g: (groups == g) & (y_true == 1) for g in nomes}
    nev = {g: (groups == g) & (y_true == 0) for g in nomes}

    matriz, pesos = {}, {}
    for i in nomes:
        for j in nomes:
            p = int(ev[i].sum()) * int(nev[j].sum())
            if not p:
                continue
            a = _auc_par(y_true, y_score, ev[i], nev[j])
            if a is None:
                continue
            matriz[f"{i}|{j}"] = a
            pesos[f"{i}|{j}"] = p

    total = sum(pesos.values())
    if not total:
        return {"auc_overall": None, "note": "sem pares utilizáveis"}

    p_in = sum(v for k, v in pesos.items() if k.split("|")[0] == k.split("|")[1])
    s_in = sum(matriz[k] * pesos[k] for k in pesos if k.split("|")[0] == k.split("|")[1])
    p_out, s_out = total - p_in, sum(matriz[k] * pesos[k] for k in pesos) - s_in

    return {
        "auc_overall": float(sum(matriz[k] * pesos[k] for k in pesos) / total),
        "auc_within": float(s_in / p_in) if p_in else None,
        "auc_between": float(s_out / p_out) if p_out else None,
        "w_within": float(p_in / total),
        "w_between": float(p_out / total),
        "pares_total": int(total),
        "pares_por_grupo": {g: int(ev[g].sum()) * int(nev[g].sum()) for g in nomes},
        "matriz": matriz,
    }


def pd_isna(arr):
    """isna elemento a elemento num array de objetos (evita depender do pandas)."""
    return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in arr])


def auc_diff_bootstrap(y_true, y_score, groups, target, n_boot: int = 1000,
                       alpha: float = 0.05, random_state: int = 42) -> dict:
    """Bootstrap estratificado da DIFERENÇA entre o AUC de um grupo e o dos demais.

    A referência é o AUC medido DENTRO de cada um dos outros grupos, ponderado
    pelo número de pares de cada um — a mesma quantidade que `auc_within_between`
    chama de `auc_within`. É comparável ao AUC do grupo porque as duas contam
    só pares intra-grupo.

    Reamostra cada grupo separadamente, preservando o n de cada um, e recalcula
    os dois lados na MESMA réplica: o IC sai da distribuição da diferença, e não
    da sobreposição de dois ICs independentes (que não é teste de hipótese).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups, dtype=object)

    nomes = sorted({g for g in groups[~pd_isna(groups)]})
    if target not in nomes:
        return {"target": target, "note": "grupo ausente"}

    idx = {g: np.flatnonzero(groups == g) for g in nomes}
    usaveis = [g for g in nomes
               if y_true[idx[g]].min() != y_true[idx[g]].max()]
    outros = [g for g in usaveis if g != target]
    if target not in usaveis or not outros:
        return {"target": target, "note": "sem ambas as classes no grupo ou na referência"}

    def auc_de(g, ii):
        y = y_true[ii]
        return None if y.min() == y.max() else float(roc_auc_score(y, y_score[ii]))

    # Pesos congelados nos valores originais: o estimando não pode se mover
    # entre réplicas, senão o IC mistura variação do alvo com a da referência.
    peso = {g: int((y_true[idx[g]] == 1).sum()) * int((y_true[idx[g]] == 0).sum())
            for g in outros}
    soma_peso = sum(peso.values())

    def referencia(aucs: dict) -> float | None:
        num = sum(peso[g] * aucs[g] for g in outros if aucs.get(g) is not None)
        den = sum(peso[g] for g in outros if aucs.get(g) is not None)
        return float(num / den) if den else None

    ponto = {g: auc_de(g, idx[g]) for g in usaveis}
    auc_alvo, auc_ref = ponto[target], referencia(ponto)
    if auc_alvo is None or auc_ref is None:
        return {"target": target, "note": "AUC indefinido"}

    rng = np.random.default_rng(random_state)
    difs = []
    for _ in range(n_boot):
        rep = {}
        for g in usaveis:
            ii = idx[g][rng.integers(0, idx[g].size, idx[g].size)]
            rep[g] = auc_de(g, ii)
        r = referencia(rep)
        if rep.get(target) is None or r is None:
            continue
        difs.append(rep[target] - r)

    if not difs:
        return {"target": target, "auc_target": auc_alvo, "auc_reference": auc_ref,
                "diff": auc_alvo - auc_ref, "note": "bootstrap degenerado"}

    difs = np.asarray(difs)
    lo, hi = np.quantile(difs, [alpha / 2, 1 - alpha / 2])
    # p-valor bicaudal do bootstrap, com a correção de continuidade usual.
    p_uni = (1 + min((difs >= 0).sum(), (difs <= 0).sum())) / (difs.size + 1)
    return {
        "target": target,
        "auc_target": auc_alvo,
        "auc_reference": auc_ref,
        "reference_kind": "auc_intra_grupo_dos_demais_ponderado_por_pares",
        "reference_groups": outros,
        "reference_weights": {g: peso[g] / soma_peso for g in outros},
        "diff": float(auc_alvo - auc_ref),
        "diff_ci_low": float(lo),
        "diff_ci_high": float(hi),
        "p_value": float(min(1.0, 2 * p_uni)),
        "alpha": alpha,
        "n_boot": int(difs.size),
        "significativo": bool(hi < 0 or lo > 0),
        "pior_que_referencia": bool(hi < 0),
    }



def auc_diff_all_groups(y_true, y_score, groups, n_boot: int = 1000,
                        alpha: float = 0.05, random_state: int = 42) -> dict:
    """`auc_diff_bootstrap` para todos os grupos de um eixo, numa passada só.

    As réplicas são COMPARTILHADAS entre os grupos: cada réplica reamostra o
    eixo inteiro uma vez e todos os grupos são medidos nela. Além de custar
    n_boot em vez de n_boot x n_grupos, isso mantém as comparações coerentes
    entre si — dois grupos do mesmo eixo são julgados na mesma reamostragem.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    groups = np.asarray(groups, dtype=object)

    nomes = sorted({g for g in groups[~pd_isna(groups)]})
    idx = {g: np.flatnonzero(groups == g) for g in nomes}
    usaveis = [g for g in nomes
               if idx[g].size and y_true[idx[g]].min() != y_true[idx[g]].max()]
    if len(usaveis) < 2:
        return {}

    peso = {g: int((y_true[idx[g]] == 1).sum()) * int((y_true[idx[g]] == 0).sum())
            for g in usaveis}

    def auc_de(ii):
        y = y_true[ii]
        return None if y.min() == y.max() else float(roc_auc_score(y, y_score[ii]))

    def referencia(aucs: dict, alvo: str):
        num = sum(peso[g] * aucs[g] for g in usaveis
                  if g != alvo and aucs.get(g) is not None)
        den = sum(peso[g] for g in usaveis if g != alvo and aucs.get(g) is not None)
        return float(num / den) if den else None

    ponto = {g: auc_de(idx[g]) for g in usaveis}

    rng = np.random.default_rng(random_state)
    difs: dict[str, list[float]] = {g: [] for g in usaveis}
    for _ in range(n_boot):
        rep = {g: auc_de(idx[g][rng.integers(0, idx[g].size, idx[g].size)])
               for g in usaveis}
        for g in usaveis:
            r = referencia(rep, g)
            if rep[g] is not None and r is not None:
                difs[g].append(rep[g] - r)

    saida = {}
    for g in usaveis:
        ref = referencia(ponto, g)
        d = np.asarray(difs[g])
        if ref is None or d.size == 0:
            saida[g] = {"target": g, "auc_target": ponto[g], "note": "sem referência"}
            continue
        lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        p_uni = (1 + min((d >= 0).sum(), (d <= 0).sum())) / (d.size + 1)
        soma = sum(peso[o] for o in usaveis if o != g) or 1
        saida[g] = {
            "target": g, "auc_target": ponto[g], "auc_reference": ref,
            "reference_kind": "auc_intra_grupo_dos_demais_ponderado_por_pares",
            "reference_groups": [o for o in usaveis if o != g],
            "reference_weights": {o: peso[o] / soma for o in usaveis if o != g},
            "diff": float(ponto[g] - ref),
            "diff_ci_low": float(lo), "diff_ci_high": float(hi),
            "p_value": float(min(1.0, 2 * p_uni)),
            "alpha": alpha, "n_boot": int(d.size),
            "significativo": bool(hi < 0 or lo > 0),
            "pior_que_referencia": bool(hi < 0),
        }
    return saida

# --------------------------------------------------------------------------
# Estabilidade populacional (PSI) — monitoramento de drift
# --------------------------------------------------------------------------


def psi(esperado, observado, n_bins: int = 10, eps: float = 1e-6) -> dict:
    """Population Stability Index entre duas distribuições.

    Mede o quanto a distribuição de uma variável (ou do próprio score) mudou
    em relação à referência de treino. É a métrica padrão de monitoramento de
    crédito, e a leitura de mercado é fixa:

        PSI < 0,10   estável
        0,10 – 0,25  atenção — investigar a fonte
        > 0,25       mudança relevante — o modelo pode ter deixado de valer

    Os cortes vêm dos quantis do ESPERADO (a referência), não do observado:
    o ponto é medir o quanto o novo se afasta do antigo, com a régua do antigo.
    """
    esperado = np.asarray(esperado, dtype=float)
    observado = np.asarray(observado, dtype=float)
    esperado = esperado[np.isfinite(esperado)]
    observado = observado[np.isfinite(observado)]
    if esperado.size == 0 or observado.size == 0:
        return {"psi": None, "note": "sem dados suficientes",
                "n_esperado": int(esperado.size), "n_observado": int(observado.size)}

    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, n_bins + 1)))
    if cortes.size < 3:
        return {"psi": None, "note": "variável quase constante na referência",
                "n_esperado": int(esperado.size), "n_observado": int(observado.size)}
    internos = cortes[1:-1]

    e_pct = np.bincount(np.digitize(esperado, internos), minlength=len(internos) + 1) / esperado.size
    o_pct = np.bincount(np.digitize(observado, internos), minlength=len(internos) + 1) / observado.size
    e_pct = np.clip(e_pct, eps, None)
    o_pct = np.clip(o_pct, eps, None)

    contrib = (o_pct - e_pct) * np.log(o_pct / e_pct)
    valor = float(contrib.sum())
    faixa = "estável" if valor < 0.10 else ("atenção" if valor < 0.25 else "mudança relevante")

    return {
        "psi": valor,
        "faixa": faixa,
        "n_esperado": int(esperado.size),
        "n_observado": int(observado.size),
        "bins": [
            {"limite_inf": float(cortes[i]) if i > 0 else None,
             "limite_sup": float(cortes[i + 1]) if i + 1 < len(cortes) else None,
             "pct_esperado": float(e_pct[i]), "pct_observado": float(o_pct[i]),
             "contribuicao": float(contrib[i])}
            for i in range(len(e_pct))
        ],
    }
