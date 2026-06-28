# OKR — Projeto Final (MBA Big Data e Analytics · FIA / LABDATA)

> Análise dos requisitos do `ProjetoFinal_v2.pdf`. Documento de referência para
> orientar o planejamento e a execução do Projeto Final, que vale como nota da
> disciplina de **Inteligência Artificial** e como etapa avaliativa do **TCC**.

---

## 📌 O que é o projeto

Simulação do **ciclo real de um projeto de ML/IA dentro de uma empresa**,
estruturado pelo framework **CRISP-DM**. O foco não é "treinar um modelo", e sim
conectar **negócio + dados + tecnologia + tomada de decisão**.

O trabalho tem duas frentes:
- **Grupo** — Dias 1 a 4 (problema, dados, modelo, narrativa de negócio).
- **Individual** — Deploy e arquitetura da solução.

---

## 🎯 Objetivos (o que se espera demonstrar)

| Frente | Critério | O que precisa entregar |
|---|---|---|
| **Grupo** | **Business Knowledge** | Traduzir um desafio real de mercado em problema de modelagem; defender alinhamento estratégico, viabilidade operacional, impacto e retorno |
| **Grupo** | **Results** | Analisar criticamente as métricas, diagnosticar problemas de aprendizado e defender por que dá pra confiar que o modelo resolve a dor (métricas técnicas → insights de negócio + governança) |
| **Individual** | **Fundamentals** | Domínio conceitual: fundamentação teórica dos algoritmos + mecanismos técnicos |
| **Individual** | **Coding** | Pipelines robustos e código de qualidade: tratamento de dados + construção de pipeline + arquitetura de treino |

---

## 🗓️ Passo a passo / Cronograma

### Etapa em grupo
1. **Dia 1 — Kickoff (22/06):** apresentação do desafio, objetivos e aula-exemplo.
2. **Dia 2 — Dados (29/06):**
   - Definir **qual problema resolver**, o **impacto no negócio** e as **métricas de sucesso**.
   - **Análise exploratória (EDA):** padrões, qualidade, nulos, inconsistências, comportamento das variáveis.
   - Estruturar a **ABT (Analytical Base Table)** — visão analítica para a modelagem.
3. **Dia 3 — Modelo (06/07):** desenvolver a modelagem + avaliar em cenário de teste.
4. **Dia 4 — Narrativa de negócio:** análise crítica do modelo (limitações, vieses, cenários de falha) + **storytelling** + **Demoday** (pitch de **15 min** defendendo a solução como para uma banca).
   - 📅 **Entrega Final do grupo: 13/07**

### Etapa individual — Deploy e arquitetura
5. Propor **arquitetura funcional completa** (da origem dos dados até o deploy como serviço de predição).
6. Montar a infra com **docker-compose** (pipeline `bruta → clear → abt` → Airflow → Model → **FastAPI** / **Streamlit**).
7. Definir **monitoramento** de dados e modelo em produção (falhas, perda de performance, drift).
8. Propor **ações automatizadas** acionadas pelas previsões (ML + automação + **agentes de IA** aplicados ao negócio).
   - 📅 **Entrega Final individual: 15/07**

---

## 📦 Entregáveis concretos

### Grupo
- **A) Pitch (Demoday)** — 15 min.
- **B) PowerPoint (5 slides):** problema de negócio · EDA · ABT · modelo (técnica, overfitting, hiperparâmetros) · avaliação (performance + explicabilidade).
- **C) GitHub** com estrutura fixa:

```
/Dados
    raw_data.csv
    clean_data.csv
    abt.csv
/DataPipeline
    data_sanitization.py    # limpeza e padronização dos dados
    abt_transform.py        # transformação dos dados brutos em entrada do modelo
    exp_analysis.ipynb      # análise exploratória dos dados limpos
    config                  # variáveis, parâmetros e metadados
/Model
    train.py                # treinamento do modelo
    config                  # variáveis, parâmetros e metadados
    evaluation.ipynb        # avaliação e análise de interpretabilidade
requirements.txt
Readme.md                   # descrição + objetivo de negócio + metodologia + como treinar
```

### Individual (mesmo GitHub + adicionais)
```
/Model
    predict.py
/MLOps
    Readme.md               # desenho da arquitetura + próximos passos (itens iii e iv)
    Docker-compose
    pipeline_orchestration.py
    /app                    # resultado via Streamlit ou API
Readme.md                   # + instruções de execução do serviço de predição
```
- Demonstração ao vivo da solução (máquina própria ou laboratório).
- ⚠️ **"Estejam preparados para realizar modificações no projeto"** — na banca podem pedir alterações ao vivo.

---

## ⚖️ Peso da avaliação

| Etapa | Peso | Tipo |
|---|---|---|
| Pitch da solução | 20% | Nota do grupo |
| Entregável técnico (PPT/código/protótipo) | 20% | Nota individual |
| **Banca Final** | **60%** | Nota individual |

➡️ O peso está fortemente na **defesa individual** (60% banca). Domínio teórico e
capacidade de modificar o projeto ao vivo são decisivos.

---

## ✅ O que precisa ser definido para atender os objetivos

### Decisões de negócio/dados (Dia 2 — bloqueiam todo o resto)
1. **Dataset** entre os 3 oferecidos:
   - *Ventilator Pressure Prediction* — regressão / séries temporais
     <https://www.kaggle.com/competitions/ventilator-pressure-prediction/overview>
   - *IEEE Fraud Detection* — classificação, dados desbalanceados
     <https://www.kaggle.com/competitions/ieee-fraud-detection>
   - *Home Credit Default Risk* — risco de crédito, múltiplas tabelas
     <https://www.kaggle.com/competitions/home-credit-default-risk/overview>
2. **Contexto de negócio e a "dor"** — empresa fictícia, problema real, por que importa.
3. **Métricas de sucesso** — técnicas (AUC, F1, RMSE…) **e** de negócio (R$ evitados, redução de fraude/inadimplência).
4. **Composição do grupo**.

### Decisões técnicas
5. Desenho da **ABT** (granularidade, variáveis, janela temporal, fonte do target).
6. **Técnica de modelagem** + estratégia de overfitting/hiperparâmetros + **explicabilidade** (SHAP, etc.).
7. **Arquitetura de deploy** (individual): orquestrador (Airflow), serviço (FastAPI vs Streamlit), estratégia de **monitoramento/drift**, e quais **ações automatizadas/agentes de IA** o modelo dispara.

---

## 🧭 Próximo passo recomendado

A escolha de **dataset + dor de negócio + métricas** é a decisão-raiz: determina a
ABT, o modelo e a arquitetura. Recomenda-se fechar essa definição primeiro e, em
seguida, montar o esqueleto do repositório no formato exigido acima.
