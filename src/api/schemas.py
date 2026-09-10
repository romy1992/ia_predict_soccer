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


class JobLiveSyncRequest(BaseModel):
    """LIVE-01: sync manuale del dataset LIVE distinto (`live_fixture_snapshot`
    /`live_match_event`/`live_fixture_stat_snapshot`) - MAI le tabelle
    pre-match `match`/`statistics`/`odds`."""

    leagues: Optional[list[int]] = None
    include_events: bool = True
    include_statistics: bool = True
    async_run: bool = True


class JobFutureSyncRequest(BaseModel):
    days_ahead: int = 7
    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    async_run: bool = True


class JobDailyRefreshRequest(BaseModel):
    """Bottone "Aggiorna tutto" del Data Center: combina in un'unica azione
    l'import delle partite disputate IERI (tutti i campionati censiti se
    `leagues` non e' valorizzato) con la sync del calendario prossimo
    (`days_ahead` giorni, default 7)."""

    seasons: Optional[list[int]] = None
    leagues: Optional[list[int]] = None
    days_ahead: int = 7
    async_run: bool = True


class JobDataQualityReportRequest(BaseModel):
    """Bottone "Aggiorna report" della pagina Data Quality: ricalcola il
    report (coverage odds/anomalie/distribuzione) e lo registra come job -
    stessa funzione (`run_data_quality_report`) usata dal job schedulato
    omonimo (vedi `src/jobs/scheduler.py`). Non chiama alcun provider
    esterno (solo dati gia' a DB)."""

    top_n: int = 20
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


class JobSettingRow(BaseModel):
    job_id: str
    label: str
    description: str
    enabled: bool
    # True per i job che chiamano DAVVERO il provider esterno API-Sports
    # (quindi coinvolti dall'auto-pausa per quota esaurita - vedi
    # `src/jobs/job_settings.py::QUOTA_SENSITIVE_JOB_IDS`): `data_settlement`
    # e `ml_training` lavorano solo su dati gia' a DB e restano `False`.
    calls_api_sports: bool = False
    # Orario/intervallo editabile (2026-09-09): `schedule_kind` distingue la
    # FORMA dello `schedule` ("daily" -> {"hour","minute"}, "interval_minutes"
    # -> {"interval_minutes"}, "interval_seconds" -> {"interval_seconds"}) -
    # il frontend sceglie il controllo giusto (time picker vs number input)
    # in base a questo campo. `schedule` e' il valore EFFETTIVO corrente
    # (override salvato se presente, altrimenti il default da variabili
    # d'ambiente) - vedi `src/jobs/job_settings.py::resolve_job_schedule`.
    schedule_kind: str
    schedule: dict[str, int]
    schedule_is_default: bool = True


class JobSettingsResponse(BaseModel):
    jobs: list[JobSettingRow]
    # Auto-pausa per quota API-Sports esaurita (vedi
    # `src/jobs/job_settings.py::sync_job_settings_with_quota`): quando
    # `quota_paused=True` tutti i job risultano forzatamente disabilitati
    # fino al reset della quota (`quota_paused_since`, data UTC ISO), poi
    # ripristinati automaticamente allo stato precedente.
    quota_paused: bool = False
    quota_paused_since: Optional[str] = None


class JobSettingsUpdateRequest(BaseModel):
    updates: dict[str, bool] = Field(..., description="Mappa job_id -> enabled (aggiornamento parziale).")


class JobScheduleUpdateRequest(BaseModel):
    """Body di `PUT /settings/jobs/{job_id}/schedule`: le chiavi accettate
    dipendono dallo `schedule_kind` del job (vedi `JobSettingRow`) - un job
    "daily" richiede ESATTAMENTE `hour`+`minute`, "interval_minutes" solo
    `interval_minutes`, "interval_seconds" solo `interval_seconds`. Validato
    lato server (range sensati) da `job_settings.py::_validate_schedule` -
    un set di chiavi sbagliato o un valore fuori range risulta in 400."""

    hour: Optional[int] = None
    minute: Optional[int] = None
    interval_minutes: Optional[int] = None
    interval_seconds: Optional[int] = None


class JobScheduleResponse(BaseModel):
    job_id: str
    schedule_kind: str
    schedule: dict[str, int]
    schedule_is_default: bool


class ApiQuotaResponse(BaseModel):
    """Stato quota API-Sports (pagina Impostazioni).

    `source` distingue la fonte del dato:
    - `"live"`: risultato di una vera interrogazione dell'endpoint
      ufficiale `GET /status` di API-Sports (`POST /settings/quota/refresh`,
      bottone "Aggiorna") - stesso numero della dashboard account
      api-sports.io, autoritativo.
    - `"estimated"`: stima dedotta PASSIVAMENTE dagli ultimi header
      `x-ratelimit-*` osservati su una chiamata dati qualsiasi (api o
      scheduler, stato condiviso su disco) - puo' restare disallineata dal
      valore reale se la quota e' stata consumata da processi che non
      passano da questo provider (bug diagnosticato 2026-09-05: senza un
      controllo live, la barra poteva mostrare 0% con la quota reale gia'
      al 100%)."""

    available: bool
    source: Optional[str] = None
    plan: Optional[str] = None
    daily_limit: Optional[int] = None
    daily_remaining: Optional[int] = None
    daily_used: Optional[int] = None
    daily_used_percentage: Optional[float] = None
    minute_limit: Optional[int] = None
    minute_remaining: Optional[int] = None
    updated_at: Optional[str] = None
    message: Optional[str] = None


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


class LiveFixturesResponse(BaseModel):
    """LIVE-01: fixture attualmente "live" cosi' come risultano nel dataset
    distinto (ultimo `LiveFixtureSnapshot` per fixture non in stato finale) -
    NON il dettaglio arricchito con predizioni di `/dashboard/live` (quello
    resta invariato, questo endpoint espone il dato grezzo della pipeline)."""

    total: int
    rows: list[dict[str, Any]]


class LiveFixtureEventsResponse(BaseModel):
    fixture_id: int
    total: int
    rows: list[dict[str, Any]]


class LiveFixtureStatisticsResponse(BaseModel):
    fixture_id: int
    total: int
    rows: list[dict[str, Any]]


class DashboardOverviewResponse(BaseModel):
    date: str
    generated_at: str
    counts: dict[str, Any]
    model_markets: list[str]
    live_preview: list[dict[str, Any]]
    day_highlights: list[dict[str, Any]]


class DashboardAvailableDatesResponse(BaseModel):
    """Elenco date selezionabili nel filtro UI (TopFilters): dal giorno 1 di
    prediction salvata ad oggi, accumulato progressivamente (mai un
    calendario libero)."""

    dates: list[str]
    first_date: Optional[str] = None
    last_date: Optional[str] = None


class DashboardMatchDetailResponse(BaseModel):
    fixture: Optional[dict[str, Any]] = None
    timeline: list[dict[str, Any]]
    odds_summary: dict[str, list[dict[str, Any]]]
    bookmaker_baseline: dict[str, Any] = {}
    decision_cards: list[dict[str, Any]]
    predictions: dict[str, Any]
    model_markets: list[str]
    odds_updated_at: Optional[str] = None


class RecomputePredictionsRequest(BaseModel):
    markets: Optional[list[str]] = Field(
        default=None, description="Mercati da ricalcolare (default: tutti quelli registrati)."
    )


class RecomputePredictionsResponse(BaseModel):
    fixture_id: int
    found: bool
    predictions: dict[str, Any]
    model_markets: list[str]


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
    ledger_candidates: int
    ledger_settled_win: int
    ledger_settled_loss: int
    ledger_void: int
    ledger_still_pending: int
    errors: list[dict[str, Any]] = []


class PaperPnlResponse(BaseModel):
    market: Optional[str] = None
    raw_summary: dict[str, Any]
    report: dict[str, Any]


class OfficialPerformanceResponse(BaseModel):
    source: str
    cohort: str
    generated_at: str
    filters: dict[str, Any]
    sample_size: int
    settled_count: int
    void_count: int
    pending_count: int
    overall: dict[str, Any]
    breakdowns: dict[str, Any]


class OfficialClvResponse(BaseModel):
    source: str
    cohort: str
    generated_at: str
    filters: dict[str, Any]
    sample_size: int
    settled_count: int
    void_count: int
    pending_count: int
    overall: dict[str, Any]
    breakdowns: dict[str, Any]
    rows: list[dict[str, Any]]


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


class ModelRegistryOverviewResponse(BaseModel):
    """OPS-02: vista lifecycle per mercato — `latest` (ultimo run
    registrato, MAI implicitamente production) vs `production` (run
    attualmente in stage 'production', se presente) vs `history` (tutti i
    run registrati, decorati con `current_stage`/`promotion_history`)."""

    market: str
    latest: Optional[dict[str, Any]] = None
    production: Optional[dict[str, Any]] = None
    history: list[dict[str, Any]] = []


class PromotionEvaluationResponse(BaseModel):
    """OPS-02: verdetto del gate metriche + confronto candidate/production
    per un run, SENZA eseguire alcuna modifica (dry-run) — vedi
    `src/ml/registry/promotion_policy.py::evaluate_promotion`."""

    run_id: str
    market: str
    production_run_id: Optional[str] = None
    to_stage: str
    policy_version: str
    allowed: bool
    gate: dict[str, Any]
    comparison: Optional[dict[str, Any]] = None
    blocking_reasons: list[str] = []


class PromotionRequest(BaseModel):
    run_id: str
    to_stage: str = "production"
    reason: Optional[str] = None
    actor: str = "manual"
    force: bool = False


class RollbackRequest(BaseModel):
    to_run_id: Optional[str] = None
    reason: Optional[str] = None
    actor: str = "manual"


class PromotionResponse(BaseModel):
    """Esito di `POST /models/{market}/promote` o `/rollback`. Quando
    `promoted=False` (gate bloccante e `force=False`), `run` riflette lo
    stage INVARIATO del run — nessuna promozione e' avvenuta (acceptance
    criteria "Ultimo training non diventa automaticamente production")."""

    promoted: bool
    run_id: str
    market: str
    to_stage: str
    evaluation: Optional[dict[str, Any]] = None
    run: Optional[dict[str, Any]] = None


class PromotionHistoryResponse(BaseModel):
    """OPS-02 (acceptance criteria "Audit promotion"): tutti gli eventi di
    `promotion_history.jsonl` per il mercato, incluse le promozioni
    bloccate dal gate e i rollback - non solo le promozioni riuscite."""

    market: str
    events: list[dict[str, Any]] = []


class MonitoringOverviewResponse(BaseModel):
    """OPS-03: risposta aggregata per la dashboard di monitoring. Senza
    `market`, `calibration_drift`/`feature_coverage` restano `None`
    (richiedono un modello/mercato specifico)."""

    generated_at: str
    market: Optional[str] = None
    thresholds_version: str
    prediction_volume: dict[str, Any]
    calibration_drift: Optional[dict[str, Any]] = None
    roi_rolling: dict[str, Any]
    feature_coverage: Optional[dict[str, Any]] = None
    alerts: list[dict[str, Any]] = []


class MonitoringAlertsResponse(BaseModel):
    """OPS-03 (acceptance criteria "Alert base"): SOLO l'elenco alert,
    utile per un polling leggero e frequente senza ricalcolare l'intero
    `MonitoringOverviewResponse`."""

    generated_at: str
    market: Optional[str] = None
    thresholds_version: str
    alerts: list[dict[str, Any]] = []


# Con `from __future__ import annotations` attivo, Pydantic v2 puo' lasciare
# un modello in stato "deferred" (mock) finche' nessuno lo valida/serializza
# davvero - e se quel modello e' usato SOLO come body di un endpoint mai
# chiamato prima del primo `GET /openapi.json`, FastAPI incontra
# `PydanticUserError: ... is not fully defined` in fase di generazione dello
# schema OpenAPI (vedi https://errors.pydantic.dev/.../u/class-not-fully-defined).
# Forziamo qui il build EAGER di ogni modello del modulo, cosi' nessun nuovo
# schema aggiunto in futuro puo' ripresentare lo stesso problema.
for _name, _obj in list(globals().items()):
    if isinstance(_obj, type) and issubclass(_obj, BaseModel) and _obj is not BaseModel:
        _obj.model_rebuild(force=True)
del _name, _obj

