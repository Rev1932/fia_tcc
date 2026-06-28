# Plano de Desenvolvimento — TCC Home Credit Default Risk

## Context

Projeto Final do MBA Big Data e Analytics (FIA/LABDATA) que vale nota da disciplina
de IA **e** etapa avaliativa do TCC. Estruturado por CRISP-DM, exige um ciclo
completo de ML "como numa empresa": negócio → dados → modelo → narrativa → deploy.

A base escolhida é a **Home Credit Default Risk** (já disponível localmente em
`data/risco_fraude/home-credit-default-risk/`, 9 tabelas relacionais por `SK_ID_CURR`).
A análise exploratória inicial (ver `OKR.md`) revelou: target desbalanceado (8,07%
default), 14,3% de clientes thin-file, scores externos de bureau faltantes (EXT_SOURCE_1
56% nulo), e anomalias de dados (18% com `DAYS_EMPLOYED=365243`).

**Decisões fechadas:**
- **Dor central:** Dor 1 — credit scoring para aprovação/negação de crédito.
- **ABT:** todas as 9 tabelas, com feature engineering incremental.
- **Modelagem:** Regressão Logística (baseline) → LightGBM (campeão) → SHAP (explicabilidade).

Este plano documenta **Problema, Objetivo e Método** e detalha a execução end-to-end
(etapa em grupo + etapa individual de MLOps), produzindo o repositório no formato
exigido pelo PDF.

---

## 1. Problema (Business Understanding)

**Empresa fictícia:** financeira de crédito ao consumidor (perfil Home Credit), atuando
em público sub-bancarizado.

**A dor:** a decisão de conceder crédito gera perda nos dois extremos —
(a) **aprovar quem não paga** = perda direta de capital (8% da carteira entra em default);
(b) **negar quem pagaria** = receita perdida e exclusão financeira. Hoje a decisão
depende fortemente de scores externos de bureau, que **faltam em 20–56% dos casos**.

**Pergunta de negócio:** *Dado um pedido de crédito, qual a probabilidade de o cliente
ter dificuldade de pagamento, e qual ponto de corte maximiza o resultado financeiro
(perda evitada vs. volume aprovado)?*

**Formulação ML:** classificação binária supervisionada, `TARGET ∈ {0,1}`, saída
`P(default)` calibrada + régua de decisão (threshold) orientada a custo.

---

## 2. Objetivo

**Objetivo geral:** desenvolver e disponibilizar um modelo de credit scoring que estime
o risco de inadimplência e suporte a decisão de aprovação, com performance e
explicabilidade defensáveis perante uma banca.

**Objetivos específicos:**
1. Construir pipeline reprodutível: `raw → clean → ABT` agregando as 9 tabelas.
2. Treinar baseline interpretável + modelo campeão (LightGBM) com controle de overfitting.
3. Avaliar com métricas técnicas e **traduzir em métrica de negócio (R$ de perda evitada)**.
4. Explicar o modelo (SHAP) e discutir vieses/limitações (governança).
5. Empacotar como serviço de predição (API/app) com infra docker-compose e monitoramento.

**Métricas de sucesso:**
- Técnicas: **AUC-ROC** (primária), **KS**, recall na classe default, PR-AUC, Brier (calibração).
- Negócio: **perda esperada evitada (R$)** e **taxa de aprovação** numa matriz de custo
  (custo do falso negativo = inadimplência ≫ custo do falso positivo = receita perdida).

---

## 3. Método (CRISP-DM)

### 3.1 Data Understanding (EDA) — `exp_analysis.ipynb`
- Perfil de cada tabela, chaves e cardinalidade do relacionamento com `application`.
- Distribuição do target, nulos, outliers (tratar sentinela `DAYS_EMPLOYED=365243`),
  correlações, análise por segmento (tipo de contrato, renda, thin-file).
- Conclusões que justificam as decisões de limpeza e features.

### 3.2 Data Preparation (pipeline) — `data_sanitization.py` + `abt_transform.py`
- **`data_sanitization.py`**: limpeza/padronização → `clean_data.csv`
  (corrige sentinelas → NaN, normaliza categóricas, trata outliers de renda, tipa colunas).
- **`abt_transform.py`**: construção incremental da ABT (1 linha por `SK_ID_CURR`) →
  `abt.csv`:
  - Base: features de `application`.
  - Agregações de `bureau` (+`bureau_balance`): nº de créditos, ativos, em atraso, somas/médias.
  - Agregações de `previous_application`: taxa de aprovação prévia, valores.
  - Agregações dos balances (`POS_CASH`, `credit_card`, `installments_payments`):
    comportamento de pagamento (atrasos, utilização de limite, DPD).
  - Ratios derivados (credit/income, annuity/income, employed/age).
- Config YAML com lista de variáveis, parâmetros e metadados.

### 3.3 Modeling — `train.py` + `evaluation.ipynb`
- Split estratificado treino/validação/teste (atenção a vazamento na agregação).
- **Baseline:** Regressão Logística (com imputação + scaling), referência interpretável.
- **Campeão:** LightGBM — lida com nulos nativamente, `scale_pos_weight`/`is_unbalance`
  para o desbalanceamento; tuning de hiperparâmetros (early stopping, validação cruzada)
  para controlar overfitting.
- `train.py` serializa o modelo + metadados; config YAML com hiperparâmetros.

### 3.4 Evaluation — `evaluation.ipynb`
- Métricas técnicas + curva ROC/PR + KS + calibração.
- **Análise de threshold com matriz de custo** → ponto de corte ótimo de negócio.
- **SHAP**: importância global + casos individuais.
- Diagnóstico crítico: vieses (ex.: gênero/idade), desempenho no grupo thin-file,
  cenários de falha — atende critério "Results" e "Métricas e governança".

### 3.5 Deployment / MLOps (etapa individual)
- **`/Model/predict.py`**: carrega modelo e expõe função de predição.
- **`/app`**: serviço (FastAPI para API REST e/ou Streamlit para demo interativa).
- **`pipeline_orchestration.py`**: orquestra `raw → clean → abt → train` (Airflow).
- **`docker-compose`**: sobe pipeline + serviço de predição.
- **Monitoramento**: estratégia para data/feature drift, perda de performance, alertas.
- **Ações automatizadas (item iv do PDF)**: ganchos a partir do score
  (ex.: auto-aprovação abaixo de X, revisão humana na faixa cinza, gatilho de
  early-warning/cobrança) — conexão ML + automação + agentes de IA.

---

## 4. Estrutura do repositório (entregável exigido)

```
/Dados
    raw_data.csv          # amostra/symlink da application (a base bruta vive em data/)
    clean_data.csv        # saída de data_sanitization.py
    abt.csv               # saída de abt_transform.py
/DataPipeline
    data_sanitization.py
    abt_transform.py
    exp_analysis.ipynb
    config.yaml
/Model
    train.py
    predict.py            # (etapa individual)
    config.yaml
    evaluation.ipynb
/MLOps                    # (etapa individual)
    Readme.md             # desenho da arquitetura + próximos passos (monitoramento, automação)
    docker-compose.yml
    pipeline_orchestration.py
    /app                  # streamlit ou api
requirements.txt
Readme.md                 # projeto + objetivo de negócio + metodologia + como treinar/rodar
```

> Observação: a base bruta (>3 GB) permanece em `data/risco_fraude/...` e fica fora do
> versionamento (`.gitignore`); `/Dados` guarda amostra/saídas reprodutíveis.

---

## 5. Ordem de execução (mapeada ao cronograma)

1. **Setup**: estrutura de pastas, `.gitignore`, `requirements.txt`, ambiente.
2. **Dia 2 — Dados**: EDA (`exp_analysis.ipynb`) → `data_sanitization.py` → `abt_transform.py`.
3. **Dia 3 — Modelo**: `train.py` (baseline + LightGBM) → `evaluation.ipynb`.
4. **Dia 4 — Narrativa**: análise crítica + SHAP + PPT (5 slides) + pitch.
5. **Etapa individual**: `predict.py` → `/app` → `pipeline_orchestration.py` →
   `docker-compose` → MLOps Readme (arquitetura, monitoramento, automação).
6. **Readme** final com instruções de treino e de execução do serviço.

---

## 6. Stack técnica

- Python 3.11, pandas/polars (volume grande → considerar polars/chunks na ABT).
- scikit-learn (baseline, métricas), LightGBM (campeão), SHAP (explicabilidade).
- Jupyter (notebooks), PyYAML (config), joblib (serialização).
- FastAPI + Uvicorn e/ou Streamlit (serviço); Apache Airflow (orquestração); Docker Compose.

---

## 7. Verificação (como validar end-to-end)

- **Pipeline**: rodar `data_sanitization.py` e `abt_transform.py` gera `clean_data.csv`
  e `abt.csv` sem erro; conferir nº de linhas da ABT == nº de clientes únicos e ausência
  de vazamento (sem colunas pós-decisão).
- **Treino**: `python Model/train.py` produz artefato do modelo + métricas logadas;
  AUC de validação reportado e estável vs. treino (sem overfitting gritante).
- **Avaliação**: `evaluation.ipynb` roda fim a fim e gera ROC/KS/SHAP + tabela de
  threshold por custo.
- **Serviço**: `docker-compose up` sobe a stack; chamada à API/Streamlit com um payload
  de exemplo retorna `P(default)` e a decisão; Readme com o comando exato.
- **Reprodutibilidade**: `pip install -r requirements.txt` + instruções do Readme
  reproduzem o fluxo do zero.

---

## 8. Documento de TCC (Problema / Método / Objetivo)

Consolidar as seções 1–3 deste plano em um documento entregável (ex.: `OKR.md`
estendido ou `docs/TCC.md`): Problema de negócio, Objetivo (geral/específicos/métricas)
e Método (CRISP-DM), servindo de base textual para o PPT e a defesa na banca.
