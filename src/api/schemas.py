from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class DatabaseHealthResponse(BaseModel):
    status: str
    database_url: str
    schema: str
    driver: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    counts: dict[str, Optional[int]]
    error: Optional[str] = None


class PredictRequest(BaseModel):
    fixture_id: int = Field(..., description="Fixture id from match table")


class PredictResponse(BaseModel):
    market: str
    fixture_id: int
    prediction: int
    probability: float
    model_run_id: Optional[str] = None
    model_name: Optional[str] = None


class JobImportRequest(BaseModel):
    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    fixture_date: Optional[str] = None
    statuses: Optional[str] = None
    days_ahead: Optional[int] = None
    async_run: bool = True


class JobTodayUpdateRequest(BaseModel):
    target_date: Optional[str] = None
    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    async_run: bool = True


class JobFutureSyncRequest(BaseModel):
    days_ahead: int = 7
    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    async_run: bool = True


class JobSettlementRequest(BaseModel):
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    async_run: bool = True


class JobRetrainRequest(BaseModel):
    markets: Optional[list[str]] = None
    seasons: Optional[list[int]] = None
    selection_method: str = "kbest"
    async_run: bool = True


class JobResponse(BaseModel):
    queued: bool
    message: str
    details: Optional[dict[str, Any]] = None


class MetricsResponse(BaseModel):
    market: str
    latest: Optional[dict[str, Any]] = None
    production: Optional[dict[str, Any]] = None
    history: list[dict[str, Any]] = []


class JobsHistoryResponse(BaseModel):
    rows: list[dict[str, Any]]


class PredictionLogResponse(BaseModel):
    rows: list[dict[str, Any]]


class DashboardDayResponse(BaseModel):
    date: str
    total: int
    returned: int
    model_markets: list[str]
    rows: list[dict[str, Any]]


class DashboardLiveResponse(BaseModel):
    date: str
    total: int
    returned: int
    model_markets: list[str]
    rows: list[dict[str, Any]]


class DashboardOverviewResponse(BaseModel):
    date: str
    generated_at: str
    counts: dict[str, Any]
    model_markets: list[str]
    live_preview: list[dict[str, Any]]
    day_highlights: list[dict[str, Any]]


class DashboardMatchDetailResponse(BaseModel):
    fixture: Optional[dict[str, Any]] = None
    timeline: list[dict[str, Any]]
    odds_summary: dict[str, list[dict[str, Any]]]
    bookmaker_baseline: dict[str, Any] = {}
    decision_cards: list[dict[str, Any]]
    predictions: dict[str, Any]
    model_markets: list[str]
    odds_updated_at: Optional[str] = None


class ModelConsensusResponse(BaseModel):
    fixture_id: int
    market: str
    experts: list[dict[str, Any]]
    oracle_final: Optional[dict[str, Any]] = None
    consensus: dict[str, Any]
    warnings: list[str] = []


class OracleMatchDetailResponse(BaseModel):
    """MATCH-02: dettaglio Oracle completo per una fixture. Ogni sezione e'
    opzionale/vuota quando non disponibile (acceptance criteria "Dati
    mancanti gestiti") - `warnings` elenca esplicitamente quali."""

    fixture_id: int
    overview: dict[str, Any]
    probabilities: list[dict[str, Any]] = []
    value_bets: list[dict[str, Any]] = []
    team_strength: Optional[dict[str, Any]] = None
    expected_goals: Optional[dict[str, Any]] = None
    score_matrix: Optional[dict[str, Any]] = None
    odds_movement: list[dict[str, Any]] = []
    model_consensus: dict[str, Any] = {}
    model_markets: list[str] = []
    warnings: list[str] = []


class DataQualityResponse(BaseModel):
    generated_at: str
    source: dict[str, Any]
    coverage: dict[str, Any]
    anomalies: dict[str, Any]
    distribution: dict[str, Any]
    temporal_checks: dict[str, Any]
    settlement: dict[str, Any]


class PredictionLedgerLogRequest(BaseModel):
    """Input per BET-06: il chiamante fornisce fixture/market/outcome +
    probabilita'/quota gia' note (tipicamente da model consensus/ORACLE-04 e
    dalle quote correnti); l'endpoint calcola fair_odd/prob_edge/ev/decision
    (BET-01/BET-04, riusati) prima di salvare — mai un client che calcola la
    decisione da solo."""

    fixture_id: int
    market: str
    outcome: str
    p_model: Optional[float] = None
    p_market_fair: Optional[float] = None
    odd: Optional[float] = None
    samples: int = 0
    model_run_id: Optional[str] = None
    model_name: Optional[str] = None
    kickoff_at: Optional[str] = None
    stake: float = 1.0
    dedupe: bool = True


class PredictionLedgerResponse(BaseModel):
    rows: list[dict[str, Any]]
    total: int


class PredictionSettlementResponse(BaseModel):
    candidates: int
    settled: int
    still_pending: int
    void_no_result: int


class PaperPnlResponse(BaseModel):
    market: Optional[str] = None
    raw_summary: dict[str, Any]
    report: dict[str, Any]


class BetslipPoolResponse(BaseModel):
    """SLIP-01: pool di pick candidati per la Schedina Oracle. `picks`/
    `excluded` sono liste di dict gia' "appiattiti" da `PickPoolResult`
    (`dataclasses.asdict`) - vedi `src/oracle/betslip/pick_pool.py`."""

    pool_id: str
    generated_at: str
    policy_version: str
    picks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []


class BetslipGenerateResponse(BaseModel):
    """SLIP-03: schedine 2/3/4 eventi generate per i tre profili di rischio
    (Safe/Balanced/Aggressive). `profiles` mappa il nome del profilo alla
    lista di schedine generate (gia' "appiattite" da `dataclasses.asdict`) -
    vedi `src/oracle/betslip/betslip_builder.py`, ciascuna con quota
    combinata e probabilita' (naive/corretta per la correlazione)
    dichiarate con metodo esplicito. `pool_id`/`pool_policy_version`
    tracciano il Pick Pool (SLIP-01) da cui le schedine sono state
    generate."""

    generated_at: str
    correlation_ruleset_version: str
    pool_id: Optional[str] = None
    pool_policy_version: Optional[str] = None
    pool_considered: int
    profiles: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []








