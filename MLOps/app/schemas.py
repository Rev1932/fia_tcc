"""
schemas.py — Contratos de entrada e saída da API (Pydantic v2).

Num arquivo só, de propósito: sob pressão numa banca, Ctrl+F num arquivo
bate abrir quatro. Organizado em blocos com banner.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# Filtros multivalorados chegam como texto separado por vírgula
# (`?education=Higher education,Lower secondary`). São declarados como escalar
# de propósito: o FastAPI trata qualquer anotação de sequência como query param
# repetido, e isso não é preenchido quando o modelo entra via `Depends()`.
# A quebra em lista acontece em `db.split_csv`, no ponto de uso.
CsvList = str | None

from MLOps.app import settings

Decision = Literal["APROVAR", "REVISAR", "NEGAR"]


# ==========================================================================
# COMUNS — erro e paginação
# ==========================================================================

class ErrorBody(BaseModel):
    code: str
    message: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class PageMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class Page[T](BaseModel):
    meta: PageMeta
    items: list[T]


def make_page(items: list, total: int, page: int, page_size: int) -> dict:
    total_pages = (total + page_size - 1) // page_size if page_size else 0
    return {
        "meta": {"page": page, "page_size": page_size, "total": total,
                 "total_pages": total_pages,
                 "has_next": page < total_pages, "has_prev": page > 1},
        "items": items,
    }


# ==========================================================================
# SAÚDE E METADADOS
# ==========================================================================

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    data_loaded: bool
    artifacts_loaded: bool
    explainer_loaded: bool
    run_id: str | None = None
    trained_at: str | None = None
    threshold: float | None = None
    n_features: int | None = None
    n_clients: int | None = None
    uptime_seconds: float
    errors: dict[str, str] = Field(default_factory=dict)


class ColumnInfo(BaseModel):
    name: str
    type: str
    missing_rate: float | None = None
    n_unique: int | None = None


class ColumnsResponse(BaseModel):
    n_columns: int
    returned: int
    columns: list[ColumnInfo]


class DimensionInfo(BaseModel):
    key: str
    label: str
    expression: str


# ==========================================================================
# CLIENTES — filtros
# ==========================================================================

class ClientFilters(BaseModel):
    """Filtros aceitos por GET /clients E por todos os /stats/*.

    Reusar o mesmo modelo é o que permite perguntar
    "taxa de inadimplência por escolaridade, só entre thin-file com menos de
    25 anos" numa única chamada.
    """
    model_config = ConfigDict(extra="forbid")   # typo em query param vira 422

    age_min: float | None = Field(None, ge=0, le=120, description="Idade mínima (AGE_YEARS)")
    age_max: float | None = Field(None, ge=0, le=120, description="Idade máxima")
    income_min: float | None = Field(None, ge=0, description="Renda anual mínima")
    income_max: float | None = Field(None, ge=0)
    credit_min: float | None = Field(None, ge=0, description="Valor de crédito mínimo")
    credit_max: float | None = Field(None, ge=0)
    annuity_min: float | None = Field(None, ge=0)
    annuity_max: float | None = Field(None, ge=0)
    children_min: int | None = Field(None, ge=0)
    children_max: int | None = Field(None, ge=0)
    employed_years_min: float | None = None
    employed_years_max: float | None = None

    gender: Literal["M", "F"] | None = None
    contract_type: CsvList = Field(None, description="Ex.: Cash loans (separe múltiplos por vírgula)")
    education: CsvList = Field(None, description="Separe múltiplos por vírgula")
    income_type: CsvList = Field(None, description="Separe múltiplos por vírgula")
    family_status: CsvList = Field(None, description="Separe múltiplos por vírgula")
    housing_type: CsvList = Field(None, description="Separe múltiplos por vírgula")
    occupation: CsvList = Field(None, description="Separe múltiplos por vírgula")

    score_min: float | None = Field(None, ge=0, le=1, description="P(default) mínima")
    score_max: float | None = Field(None, ge=0, le=1)
    decision: Literal["APROVAR", "NEGAR"] | None = Field(
        None, description="Aplica o threshold vigente sobre o score")
    target: Literal[0, 1] | None = Field(None, description="Rótulo real de inadimplência")
    thin_file: bool | None = Field(
        None, description="true = sem nenhum registro no bureau (BUREAU_COUNT nulo)")
    split: Literal["train", "valid", "test"] | None = None


# ==========================================================================
# CLIENTES — respostas
# ==========================================================================

class ClientSummary(BaseModel):
    # extra="allow": o Swagger documenta o núcleo, e ?fields= continua
    # podendo pedir qualquer uma das 470+ colunas.
    model_config = ConfigDict(extra="allow")

    SK_ID_CURR: int
    TARGET: int | None = None
    AGE_YEARS: float | None = None
    CODE_GENDER: str | None = None
    NAME_CONTRACT_TYPE: str | None = None
    NAME_EDUCATION_TYPE: str | None = None
    AMT_INCOME_TOTAL: float | None = None
    AMT_CREDIT: float | None = None
    CREDIT_INCOME_RATIO: float | None = None
    EXT_SOURCE_2: float | None = None
    BUREAU_COUNT: float | None = None
    proba_champion: float | None = None
    decision: Decision | None = None
    split: str | None = None


class ScoreResponse(BaseModel):
    sk_id_curr: int
    probability_default: float
    threshold: float
    decision: Decision
    score_band: str
    percentile: float | None = Field(
        None, description="Posição do cliente na carteira (0 = menor risco)")
    source: Literal["batch", "live"] = Field(
        "batch", description="'batch' vem de scores.parquet; 'live' recalcula pelo modelo")
    target: int | None = None
    baseline_probability: float | None = None
    live_probability: float | None = Field(
        None, description="Preenchido com recompute=true: prova que artefato e modelo concordam")
    agreement_error: float | None = None


class ClientDetail(BaseModel):
    sk_id_curr: int
    thin_file: bool
    identificacao: dict[str, Any]
    financeiro: dict[str, Any]
    historico: dict[str, Any]
    scores_externos: dict[str, Any]
    score: ScoreResponse | None = None
    features: dict[str, Any] | None = None


# ==========================================================================
# ESTATÍSTICAS / EDA
# ==========================================================================

class OverviewResponse(BaseModel):
    n_clients: int
    n_defaults: int
    default_rate: float
    thin_file_rate: float
    avg_age: float | None = None
    median_income: float | None = None
    avg_credit: float | None = None
    avg_credit_income_ratio: float | None = None
    missing_ext_source_1_rate: float
    missing_ext_source_3_rate: float
    scored: int
    avg_score: float | None = None
    approval_rate: float | None = None
    threshold: float
    filters_applied: dict[str, Any]


class SegmentBucket(BaseModel):
    value: str | None
    n: int
    defaults: int | None = None
    default_rate: float | None = None
    lift: float | None = Field(None, description="default_rate do grupo / default_rate geral")
    avg_score: float | None = None
    approval_rate: float | None = None


class DefaultRateResponse(BaseModel):
    dimension: str
    label: str
    expression: str
    overall_default_rate: float | None
    n_total: int
    buckets: list[SegmentBucket]


class Bin(BaseModel):
    lower: float | None
    upper: float | None
    count: int
    pct: float
    default_rate: float | None = None


class FeatureStats(BaseModel):
    count: int
    missing: int
    missing_rate: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None


class DistributionResponse(BaseModel):
    feature: str
    type: Literal["numeric", "categorical"]
    n_total: int
    stats: FeatureStats | None = None
    bins: list[Bin] | None = None
    categories: list[SegmentBucket] | None = None


class MissingItem(BaseModel):
    column: str
    dtype: str | None = None
    missing_rate: float
    n_unique: int | None = None


class MissingResponse(BaseModel):
    n_rows: int
    n_columns: int
    columns_with_missing: int
    items: list[MissingItem]


class CrosstabResponse(BaseModel):
    rows_dimension: str
    cols_dimension: str
    metric: Literal["count", "default_rate"]
    n_total: int
    row_values: list[str]
    col_values: list[str]
    cells: list[dict[str, Any]]


# ==========================================================================
# MODELO
# ==========================================================================

class RunInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    tag: str | None = None
    trained_at: str | None = None
    git_sha: str | None = None
    n_rows: int | None = None
    n_features: int | None = None
    n_train: int | None = None
    n_valid: int | None = None
    n_test: int | None = None
    versions: dict[str, str] = Field(default_factory=dict)


class BaselineMetrics(BaseModel):
    auc: float
    ks: float
    brier: float | None = None


class ChampionMetrics(BaseModel):
    auc: float
    ks: float
    auc_train: float
    auc_valid: float
    best_iteration: int
    brier: float | None = None
    overfit_gap: float | None = None


class BusinessMetrics(BaseModel):
    threshold: float
    approval_rate: float
    cost_false_negative: float
    cost_false_positive: float


class ServedMetrics(BaseModel):
    """Métricas do modelo que a API de fato serve.

    É daqui que sai todo número de capa. Com calibração ativa, a isotônica
    introduz empates no score e move o KS na terceira casa em relação ao
    modelo cru — pequeno, mas suficiente para dois números circularem.
    """
    model: str
    auc: float
    ks: float
    brier: float
    threshold: float
    approval_rate: float


class CalibrationMetrics(BaseModel):
    method: str
    auc: float
    ks: float
    brier: float
    brier_before: float
    n_calib: int


class MetricsResponse(BaseModel):
    run: RunInfo
    served: ServedMetrics | None = Field(
        None, description="O modelo servido — a fonte dos números de capa")
    baseline: BaselineMetrics
    champion: ChampionMetrics
    calibrated: CalibrationMetrics | None = None
    business: BusinessMetrics
    lift_vs_baseline: float = Field(..., description="AUC do campeão - AUC do baseline")


class RocPoint(BaseModel):
    fpr: float
    tpr: float
    threshold: float | None = None


class RocResponse(BaseModel):
    model: str
    split: str
    auc: float
    gini: float
    points: list[RocPoint]


class DecileRow(BaseModel):
    model_config = ConfigDict(extra="allow")
    decile: int
    n: int
    events: int
    event_rate: float
    lift: float
    ks: float


class KsResponse(BaseModel):
    model: str
    split: str
    ks: float
    ks_threshold: float | None = None
    points: list[dict[str, Any]]
    deciles: list[DecileRow] | None = None


class ImportanceItem(BaseModel):
    rank: int
    feature: str
    importance: float
    importance_pct: float
    source_table: str


class FeatureImportanceResponse(BaseModel):
    kind: Literal["gain", "split"]
    n_features_total: int
    items: list[ImportanceItem]
    by_source: dict[str, float] = Field(
        ..., description="Fração da importância por tabela de origem — responde "
                         "com número se a ABT das 9 tabelas valeu a pena")


class SweepPoint(BaseModel):
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    cost: float
    approval_rate: float
    precision: float
    recall: float
    default_rate_approved: float


class ThresholdAnalysisResponse(BaseModel):
    cost_fn: float
    cost_fp: float
    cost_ratio: float
    split: str
    model: str
    n: int
    frozen_threshold: float
    best: SweepPoint
    current: SweepPoint
    delta_cost_vs_current: float
    points: list[SweepPoint]


class ConfusionMatrixResponse(BaseModel):
    threshold: float
    split: str
    n: int
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    approval_rate: float
    default_rate_approved: float
    default_rate_denied: float
    cost: float


class FairnessGroup(BaseModel):
    group: str
    n: int
    auc: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_width: float | None = None
    n_events: int | None = None
    pct_da_base: float | None = None
    approval_rate: float | None = None
    default_rate: float | None = None
    avg_score: float | None = None
    brier: float | None = None
    overlaps_overall: bool | None = Field(
        None, description="IC do grupo sobrepõe o IC geral? Se não, a diferença "
                          "de AUC é real e não ruído amostral")


class FairnessResponse(BaseModel):
    dimension: str
    threshold: float
    overall: dict[str, Any]
    groups: list[FairnessGroup]


class ImprovementRun(BaseModel):
    model_config = ConfigDict(extra="allow")
    run_id: str
    tag: str | None = None
    status: str | None = Field(
        None, description="'aceita' ou 'rejeitada' pela regra de aceite")
    motivo: str | None = Field(None, description="Por que foi rejeitada")
    n_features: int | None = None
    auc: float | None = None
    ks: float | None = None
    brier: float | None = None
    threshold: float | None = None
    approval_rate: float | None = None


class ImprovementsResponse(BaseModel):
    runs: list[ImprovementRun]
    baseline_tag: str | None = None
    latest_tag: str | None = Field(
        None, description="Última rodada ACEITA — é a que está servindo")
    deltas: dict[str, Any] = Field(default_factory=dict)
    rejeitadas: list[dict[str, Any]] = Field(
        default_factory=list,
        description="O que foi tentado e não passou na regra de aceite")


class PsiBin(BaseModel):
    limite_inf: float | None = None
    limite_sup: float | None = None
    pct_esperado: float
    pct_observado: float
    contribuicao: float


class PsiItem(BaseModel):
    feature: str
    psi: float | None = None
    faixa: str | None = None
    n_esperado: int
    n_observado: int
    note: str | None = None
    bins: list[PsiBin] | None = None


class PsiResponse(BaseModel):
    referencia: str
    comparado: str
    n_features: int
    limiares: dict[str, str]
    resumo: dict[str, int]
    items: list[PsiItem]


# ==========================================================================
# PREDIÇÃO, SIMULAÇÃO E EXPLICABILIDADE
# ==========================================================================

class Contribution(BaseModel):
    rank: int
    feature: str
    value: Any = None
    shap_value: float
    abs_shap: float
    effect: Literal["aumenta risco", "reduz risco"]
    source_table: str


class ConsistencyCheck(BaseModel):
    """`base_value + Σ shap` tem de reconstruir a probabilidade do modelo cru.
    É a prova de que a explicação é fiel, e não uma aproximação."""
    reconstructed_probability: float
    model_probability: float
    max_abs_error: float


class ExplainResponse(BaseModel):
    sk_id_curr: int | None = None
    probability_default: float
    raw_probability: float | None = Field(
        None, description="Score antes da calibração — é o que o SHAP explica")
    threshold: float
    decision: Decision
    base_value: float
    base_probability: float
    top_risk_drivers: list[Contribution]
    top_protective_factors: list[Contribution]
    sum_other_features: float
    consistency_check: ConsistencyCheck
    narrative: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "records": [{
            "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "F",
            "AMT_INCOME_TOTAL": 150000, "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000, "DAYS_BIRTH": -12775,
            "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.42,
        }],
        "threshold": None, "explain": False,
    }]})

    records: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=settings.MAX_PREDICT_RECORDS)
    threshold: float | None = Field(None, ge=0, le=1)
    explain: bool = False
    explain_top: int = Field(5, ge=1, le=settings.MAX_EXPLAIN_TOP)


class Prediction(BaseModel):
    index: int
    probability_default: float
    threshold: float
    decision: Decision
    score_band: str
    features_informed: int
    features_expected: int
    coverage: float = Field(..., description="Fração das features do modelo informadas")
    unknown_features: list[str] = Field(
        default_factory=list, description="Campos enviados que não são features do modelo")
    contributions: list[Contribution] | None = None


class PredictResponse(BaseModel):
    run_id: str | None = None
    threshold: float
    n_records: int
    predictions: list[Prediction]


class SimulateRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "sk_id_curr": 100002,
        "changes": {"AMT_CREDIT": 900000},
    }, {
        "sk_id_curr": 100002,
        "sweep": {"feature": "EXT_SOURCE_2", "start": 0.0, "stop": 1.0, "steps": 11},
    }]})

    sk_id_curr: int | None = Field(None, description="Parte de um cliente real da base")
    record: dict[str, Any] | None = Field(None, description="...ou de um payload livre")
    changes: dict[str, Any] = Field(default_factory=dict)
    sweep: dict[str, Any] | None = Field(
        None, description='{"feature": "AMT_CREDIT", "start": 1e5, "stop": 1e6, "steps": 10} '
                          'ou {"feature": "...", "values": [...]}')
    threshold: float | None = Field(None, ge=0, le=1)
    explain_top: int = Field(8, ge=1, le=settings.MAX_EXPLAIN_TOP)

    @model_validator(mode="after")
    def _exatamente_um(self):
        if (self.sk_id_curr is None) == (self.record is None):
            raise ValueError("informe exatamente um entre 'sk_id_curr' e 'record'")
        return self


class ScoreLite(BaseModel):
    probability_default: float
    decision: Decision
    score_band: str


class SweepRow(BaseModel):
    value: Any
    probability_default: float
    decision: Decision


class SimulateResponse(BaseModel):
    base: ScoreLite
    simulated: ScoreLite | None = None
    delta_probability: float | None = None
    decision_changed: bool = False
    threshold: float
    changes_applied: dict[str, Any] = Field(default_factory=dict)
    ignored_changes: list[str] = Field(
        default_factory=list, description="Campos que não são features do modelo")
    sweep: list[SweepRow] | None = None
    top_drivers: list[Contribution] | None = None
