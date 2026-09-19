from __future__ import annotations

import dataclasses
import json
import logging
import os
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.dashboard_service import DashboardService
from src.api.model_diagnostics_service import ModelDiagnosticsService
from src.api.oracle_match_detail_service import OracleMatchDetailService
from src.api.schemas import (
    DataQualityResponse,
    DatabaseHealthResponse,
    ApiQuotaResponse,
    BettingStatisticsResponse,
    BetslipGenerateResponse,
    BetslipPoolResponse,
    BetslipProposalListResponse,
    DashboardAvailableDatesResponse,
    DashboardBundleResponse,
    DashboardDayResponse,
    DashboardLiveResponse,
    DashboardMatchDetailResponse,
    DashboardOverviewResponse,
    HealthResponse,
    JobDailyRefreshRequest,
    JobDataQualityReportRequest,
    JobFutureSyncRequest,
    JobImportRequest,
    JobLiveSyncRequest,
    JobOfficialCaptureRequest,
    JobPredictionSnapshotRefreshRequest,
    JobResponse,
    JobRetrainRequest,
    JobScheduleResponse,
    JobScheduleUpdateRequest,
    JobSettingsResponse,
    JobSettingsUpdateRequest,
    JobSettlementRequest,
    JobStatusResponse,
    JobTodayUpdateRequest,
    JobsHistoryResponse,
    LiveFixtureEventsResponse,
    LiveFixtureStatisticsResponse,
    LiveFixturesResponse,
    MetricsResponse,
    ModelConsensusResponse,
    ModelDiagnosticsResponse,
    ModelRegistryOverviewResponse,
    MonitoringAlertsResponse,
    MonitoringOverviewResponse,
    OracleMatchDetailResponse,
    OfficialClvResponse,
    OfficialBetslipListResponse,
    OfficialBetslipStatisticsResponse,
    OfficialPerformanceResponse,
    PaperPnlResponse,
    PredictRequest,
    PredictResponse,
    PredictionLedgerLogRequest,
    PredictionLedgerResponse,
    PredictionLogResponse,
    PredictionSettlementResponse,
    PromotionEvaluationResponse,
    PromotionHistoryResponse,
    PromotionRequest,
    PromotionResponse,
    RecomputePredictionsRequest,
    RecomputePredictionsResponse,
    RollbackRequest,
)
from src.data.quality_report_service import DataQualityService
from src.ml.ensemble.model_consensus import build_model_consensus_for_fixture
from src.ml.monitoring.monitoring_service import MonitoringService
from src.ml.registry.promotion_policy import DEFAULT_PROMOTION_POLICY
from src.oracle.betslip.pick_pool import PickPoolPolicy
from src.oracle.betslip.pick_pool_service import PickPoolService
from src.oracle.betslip.betslip_service import BetslipService
from src.oracle.betslip.official_betslip_service import OfficialBetslipService
from src.oracle.betslip.proposal_snapshot_service import BetslipProposalSnapshotService
from src.oracle.betting_statistics_service import BettingStatisticsService
from src.oracle.decision_engine.decision_policy import DEFAULT_DECISION_POLICY, evaluate_decision
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.oracle.ledger.official_clv_service import OfficialClvService
from src.oracle.ledger.official_performance_service import OFFICIAL_COHORT, OfficialPerformanceService
from src.repository.base.database_audit import get_database_audit
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.data.live.live_sync_job import run_manual_live_sync
from src.repository.live_data_repository import LiveDataRepository
from src.jobs.api_quota_state import get_quota_snapshot
from src.jobs.job_history import JobHistory
from src.jobs.job_settings import (
    JOB_DEFINITIONS,
    get_quota_pause_state,
    list_job_definitions,
    reset_job_schedule,
    resolve_job_schedule,
    sync_job_settings_with_quota,
    update_job_schedule,
    update_job_settings,
)
from src.jobs.scheduler import (
    run_daily_refresh,
    run_data_quality_report,
    run_manual_future_sync,
    run_manual_import,
    run_manual_retrain,
    run_manual_settlement,
    run_manual_today_update,
    run_official_prediction_capture,
    run_prediction_snapshot_refresh,
)
from src.service_ia.config.app_config import load_app_config
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider
from src.service_ia.pre_processing.settlement_service import SettlementService
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_paths import resolve_model_path
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.training.prediction_logger import PredictionLogger

app = FastAPI(title="Soccer ML Platform API", version="0.1.0")


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS")
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if values:
            return values
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _summary_path() -> str:
    return os.path.join(_project_root(), "best_models", "training_summary.json")


def _run_prediction_snapshot_refresh_job(target_date: Optional[str], job_id: str) -> None:
    try:
        run_prediction_snapshot_refresh(target_date=target_date, job_id=job_id)
    except Exception:
        logging.exception("prediction_snapshot_refresh job %s failed", job_id)


def _load_summary() -> list[dict[str, Any]]:
    path = _summary_path()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
            if isinstance(payload, list):
                return payload
            return []
        except json.JSONDecodeError:
            return []


def _extract_probability(model: Any, X) -> tuple[int, float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        first = np.asarray(probs[0], dtype=float)
        if first.size >= 2:
            p1 = float(first[-1])
            pred = int(p1 >= 0.5)
            return pred, p1
        if first.size == 1:
            p = float(first[0])
            pred = int(p >= 0.5)
            return pred, p

    pred_raw = model.predict(X)
    pred = int(np.asarray(pred_raw).ravel()[0])
    return pred, float(pred)


@app.get("/")
def index() -> dict[str, str]:
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "status": "ok",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/database", response_model=DatabaseHealthResponse)
def health_database() -> DatabaseHealthResponse:
    payload = get_database_audit()
    return DatabaseHealthResponse(**payload)


@app.get("/markets")
def markets() -> dict[str, list[str]]:
    # Alimenta SIA il filtro "Mercato" della Dashboard SIA il selettore
    # "Crea previsione manuale": esclude i mercati la cui produzione e'
    # finita in archivio/ (stessa distinzione di
    # ModelRegistry.list_active_markets(), gia' usata da Dashboard/
    # scheduler per non servire piu' un modello lasciato indietro dalla
    # riorganizzazione di best_models/ - es. under_over_4_5), cosi' non
    # restano selezionabili in nessuno dei due punti.
    active_markets = set(ModelRegistry().list_active_markets())
    values = sorted(m for m in FilterMarketService.SUPPORTED_MARKETS if m in active_markets)
    return {"markets": values}


def _parse_iso_date(value: Optional[str]) -> date:
    if not value:
        return date.today()
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Data non valida: {value}") from exc


def _parse_iso_datetime_optional(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Datetime non valido: {value}") from exc
    return parsed


def _parse_int_csv(value: Optional[str], field_name: str) -> Optional[list[int]]:
    if not value:
        return None

    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        return None

    parsed: list[int] = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Valore non valido per {field_name}: {item}") from exc
    return parsed


@app.get("/dashboard/overview", response_model=DashboardOverviewResponse)
def dashboard_overview(target_date: Optional[str] = None) -> DashboardOverviewResponse:
    service = DashboardService()
    payload = service.get_overview(target_date=_parse_iso_date(target_date))
    return DashboardOverviewResponse(**payload)


@app.get("/dashboard/bundle", response_model=DashboardBundleResponse)
def dashboard_bundle(
    target_date: Optional[str] = None,
    limit: int = 400,
    search: Optional[str] = None,
    force_refresh: bool = False,
) -> DashboardBundleResponse:
    payload = DashboardService().get_dashboard_bundle(
        target_date=_parse_iso_date(target_date),
        limit=limit,
        search_text=search,
        force_refresh=force_refresh,
    )
    return DashboardBundleResponse(**payload)


@app.get("/dashboard/available-dates", response_model=DashboardAvailableDatesResponse)
def dashboard_available_dates() -> DashboardAvailableDatesResponse:
    """Elenco date accumulato (dal giorno 1 di previsioni salvate ad oggi)
    usato da TopFilters al posto di un calendario libero."""
    service = DashboardService()
    payload = service.get_available_dates()
    return DashboardAvailableDatesResponse(**payload)


@app.get("/dashboard/live", response_model=DashboardLiveResponse)
def dashboard_live(
    target_date: Optional[str] = None,
    limit: int = 20,
    with_predictions: bool = True,
    markets: Optional[str] = None,
) -> DashboardLiveResponse:
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    service = DashboardService()
    payload = service.get_live_matches(
        target_date=_parse_iso_date(target_date),
        limit=limit,
        with_predictions=with_predictions,
        markets=selected_markets,
    )
    return DashboardLiveResponse(**payload)


@app.get("/dashboard/day", response_model=DashboardDayResponse)
def dashboard_day(
    target_date: Optional[str] = None,
    limit: int = 300,
    with_predictions: bool = True,
    markets: Optional[str] = None,
    phase: Optional[str] = None,
    search: Optional[str] = None,
    force_refresh: bool = False,
) -> DashboardDayResponse:
    """`force_refresh=true` (bottone "Forza aggiornamento" in Dashboard):
    bypassa lo skip DB-first per date storiche gia' sincronizzate (mai il
    guard quota-esaurita, vedi `DashboardService.get_day_matches`)."""
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    service = DashboardService()
    payload = service.get_day_matches(
        target_date=_parse_iso_date(target_date),
        limit=limit,
        with_predictions=with_predictions,
        markets=selected_markets,
        phase=phase,
        search_text=search,
        force_refresh=force_refresh,
    )
    return DashboardDayResponse(**payload.__dict__)


@app.get("/dashboard/match/{fixture_id}", response_model=DashboardMatchDetailResponse)
def dashboard_match_detail(
    fixture_id: int,
    with_predictions: bool = True,
    markets: Optional[str] = None,
) -> DashboardMatchDetailResponse:
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    service = DashboardService()
    payload = service.get_match_detail(
        fixture_id=fixture_id,
        with_predictions=with_predictions,
        markets=selected_markets,
    )
    return DashboardMatchDetailResponse(**payload)


@app.post("/dashboard/matches/{fixture_id}/recompute-predictions", response_model=RecomputePredictionsResponse)
def dashboard_match_recompute_predictions(
    fixture_id: int, payload: Optional[RecomputePredictionsRequest] = None
) -> RecomputePredictionsResponse:
    """Bottone "Ricalcola previsione" nel dettaglio partita (2026-09-10,
    punto 4/4 di `PROMPT_fast_historical_predictions.md`): forza il
    ricalcolo e il salvataggio di una riga NUOVA per questa fixture, anche
    se una gia' esiste (partita conclusa "congelata" inclusa) - uso
    ESPLICITO e manuale, mai chiamato da alcun job/vista lista."""
    markets = payload.markets if payload else None
    service = DashboardService()
    result = service.recompute_predictions(fixture_id=fixture_id, markets=markets)
    if not result.get("found"):
        raise HTTPException(status_code=404, detail=f"Fixture non trovata: {fixture_id}")
    return RecomputePredictionsResponse(**result)


@app.get("/dashboard/match/{fixture_id}/consensus", response_model=ModelConsensusResponse)
def dashboard_match_consensus(fixture_id: int, market: str) -> ModelConsensusResponse:
    """Model Consensus per spiegabilita' (ORACLE-04): output dei singoli
    Oracle Expert generici (Direct Expert EXP-05 + Market/Odds Expert EXP-04)
    per questa fixture/mercato, Oracle finale (meta-model ORACLE-02/03 se
    registrato, altrimenti media semplice) e dispersione del consensus.
    Nessuna logica scientifica nel frontend: tutto il calcolo avviene qui."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    report = build_model_consensus_for_fixture(market=market, fixture_id=fixture_id)
    return ModelConsensusResponse(
        fixture_id=report.fixture_id,
        market=report.market,
        experts=report.experts,
        oracle_final=report.oracle_final,
        consensus=report.consensus,
        warnings=report.warnings,
    )


@app.get("/dashboard/match/{fixture_id}/oracle-detail", response_model=OracleMatchDetailResponse)
def dashboard_match_oracle_detail(fixture_id: int, markets: Optional[str] = None) -> OracleMatchDetailResponse:
    """Oracle Match Detail (MATCH-02): overview, probabilities, value bets,
    team strength, expected goals, score matrix, odds movement e model
    consensus per QUESTA fixture, in un'unica risposta strutturata
    (acceptance criteria "Dettaglio navigabile per fixture"). Nessuna
    logica di betting/ML nel frontend: ogni sezione e' gia' calcolata qui
    da `OracleMatchDetailService` (riuso diretto di MATCH-01/EXP-01/EXP-02/
    ORACLE-04, nessuna duplicazione). Dati mancanti gestiti esplicitamente
    (`warnings`), mai un'eccezione che blocca l'intero dettaglio."""
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    service = OracleMatchDetailService()
    payload = service.build_oracle_match_detail(fixture_id=fixture_id, markets=selected_markets)
    return OracleMatchDetailResponse(**payload)


@app.post("/predict/{market}", response_model=PredictResponse)
def predict(market: str, payload: PredictRequest) -> PredictResponse:
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    active = registry.get_production(market=market) or registry.get_latest(market=market)
    if not active:
        raise HTTPException(status_code=404, detail=f"Nessun modello disponibile per mercato {market}")

    # Il percorso registrato viene risolto (`resolve_model_path`) e non usato
    # alla cieca: dopo la riorganizzazione di `best_models/` un modello puo'
    # stare in `under_over/<mercato>/` o in `archivio/`. Il valore registrato
    # vince sempre quando esiste; il messaggio d'errore mostra comunque quello,
    # non il percorso cercato, perche' e' li' che va corretta la riga.
    model_path = resolve_model_path(active.get("model_path"))
    if not model_path:
        raise HTTPException(status_code=404, detail=f"File modello non trovato: {active.get('model_path')}")

    model = joblib.load(model_path)

    builder = FilterMarketService()
    frame = builder.build_prediction_frame(market=market, fixture_id=payload.fixture_id)
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail=f"Fixture non trovata o feature insufficienti: {payload.fixture_id}")

    X = frame.drop(columns=["market", "id_fixture", "season", "league", "prediction_at"], errors="ignore")

    selected_features = active.get("feature_names") or []
    if selected_features:
        for feature in selected_features:
            if feature not in X.columns:
                X[feature] = 0.0
        X = X[selected_features]

    pred, prob = _extract_probability(model=model, X=X)

    logger = PredictionLogger()
    logger.log(
        fixture_id=payload.fixture_id,
        market=market,
        prediction=pred,
        probability=prob,
        model_run_id=active.get("run_id"),
        extra={"model_name": active.get("model_name")},
    )

    return PredictResponse(
        market=market,
        fixture_id=payload.fixture_id,
        prediction=pred,
        probability=prob,
        model_run_id=active.get("run_id"),
        model_name=active.get("model_name"),
    )


@app.post("/jobs/import", response_model=JobResponse)
def trigger_import(payload: JobImportRequest, background_tasks: BackgroundTasks) -> JobResponse:
    params = {
        "seasons": payload.seasons,
        "leagues": payload.leagues,
        "from_date": payload.from_date,
        "to_date": payload.to_date,
        "fixture_date": payload.fixture_date,
        "statuses": payload.statuses,
        "days_ahead": payload.days_ahead,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="import", params=params)
        background_tasks.add_task(
            run_manual_import,
            seasons=payload.seasons,
            leagues=payload.leagues,
            from_date=payload.from_date,
            to_date=payload.to_date,
            fixture_date=payload.fixture_date,
            statuses=payload.statuses,
            days_ahead=payload.days_ahead,
            is_next=False,
            job_id=row["job_id"],
            job_type="import",
        )
        return JobResponse(queued=True, message="Import job queued", details={"job_id": row["job_id"]})

    report = run_manual_import(
        seasons=payload.seasons,
        leagues=payload.leagues,
        from_date=payload.from_date,
        to_date=payload.to_date,
        fixture_date=payload.fixture_date,
        statuses=payload.statuses,
        days_ahead=payload.days_ahead,
        is_next=False,
    )
    return JobResponse(queued=False, message="Import job completed", details=report)


@app.post("/jobs/today-update", response_model=JobResponse)
def trigger_today_update(payload: JobTodayUpdateRequest, background_tasks: BackgroundTasks) -> JobResponse:
    params = {
        "target_date": payload.target_date,
        "seasons": payload.seasons,
        "leagues": payload.leagues,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="today_update", params=params)
        background_tasks.add_task(
            run_manual_today_update,
            target_date=payload.target_date,
            seasons=payload.seasons,
            leagues=payload.leagues,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Today update job queued", details={"job_id": row["job_id"]})

    report = run_manual_today_update(
        target_date=payload.target_date,
        seasons=payload.seasons,
        leagues=payload.leagues,
    )
    return JobResponse(queued=False, message="Today update job completed", details=report)


@app.post("/jobs/live-sync", response_model=JobResponse)
def trigger_live_sync(payload: JobLiveSyncRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """LIVE-01: sincronizza il dataset LIVE distinto (mai il dataset
    pre-match) - stesso pattern queued/sync degli altri job manuali."""
    params = {
        "leagues": payload.leagues,
        "include_events": payload.include_events,
        "include_statistics": payload.include_statistics,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="live_sync", params=params)
        background_tasks.add_task(
            run_manual_live_sync,
            leagues=payload.leagues,
            include_events=payload.include_events,
            include_statistics=payload.include_statistics,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Live sync job queued", details={"job_id": row["job_id"]})

    report = run_manual_live_sync(
        leagues=payload.leagues,
        include_events=payload.include_events,
        include_statistics=payload.include_statistics,
    )
    return JobResponse(queued=False, message="Live sync job completed", details=report)


@app.get("/live/fixtures", response_model=LiveFixturesResponse)
def live_fixtures() -> LiveFixturesResponse:
    """LIVE-01: fixture attualmente live nel dataset DISTINTO (ultimo
    snapshot per fixture non in stato finale) - dato grezzo della pipeline,
    NON il dettaglio arricchito con predizioni di `/dashboard/live`."""
    repository = LiveDataRepository()
    latest = repository.latest_fixture_snapshots()
    rows = [
        snapshot.to_dict()
        for fixture_id, snapshot in latest.items()
        if fixture_id in set(repository.list_active_fixture_ids())
    ]
    rows.sort(key=lambda row: row.get("fixture_id") or 0)
    return LiveFixturesResponse(total=len(rows), rows=rows)


@app.get("/live/fixtures/{fixture_id}/events", response_model=LiveFixtureEventsResponse)
def live_fixture_events(fixture_id: int) -> LiveFixtureEventsResponse:
    """LIVE-01: eventi live (con timestamp) per una fixture, dal dataset
    distinto - ordinati per minuto di gioco."""
    repository = LiveDataRepository()
    rows = [event.to_dict() for event in repository.list_events_for_fixture(fixture_id)]
    return LiveFixtureEventsResponse(fixture_id=fixture_id, total=len(rows), rows=rows)


@app.get("/live/fixtures/{fixture_id}/statistics", response_model=LiveFixtureStatisticsResponse)
def live_fixture_statistics(fixture_id: int) -> LiveFixtureStatisticsResponse:
    """LIVE-01: ultimo snapshot statistiche PER SQUADRA per una fixture."""
    repository = LiveDataRepository()
    rows = [row.to_dict() for row in repository.latest_stat_snapshots_for_fixture(fixture_id)]
    return LiveFixtureStatisticsResponse(fixture_id=fixture_id, total=len(rows), rows=rows)


@app.post("/jobs/future-sync", response_model=JobResponse)
def trigger_future_sync(payload: JobFutureSyncRequest, background_tasks: BackgroundTasks) -> JobResponse:
    if payload.days_ahead < 1:
        raise HTTPException(status_code=400, detail="days_ahead deve essere >= 1")

    params = {
        "days_ahead": payload.days_ahead,
        "seasons": payload.seasons,
        "leagues": payload.leagues,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="future_sync", params=params)
        background_tasks.add_task(
            run_manual_future_sync,
            days_ahead=payload.days_ahead,
            seasons=payload.seasons,
            leagues=payload.leagues,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Future sync job queued", details={"job_id": row["job_id"]})

    report = run_manual_future_sync(
        days_ahead=payload.days_ahead,
        seasons=payload.seasons,
        leagues=payload.leagues,
    )
    return JobResponse(queued=False, message="Future sync job completed", details=report)


@app.post("/jobs/daily-refresh", response_model=JobResponse)
def trigger_daily_refresh(payload: JobDailyRefreshRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Bottone "Aggiorna tutto" del Data Center: combina import partite
    disputate IERI (tutti i campionati censiti) + sync calendario prossimo
    (`days_ahead` giorni, con quote). Vedi `run_daily_refresh`."""
    if payload.days_ahead < 1:
        raise HTTPException(status_code=400, detail="days_ahead deve essere >= 1")

    params = {
        "seasons": payload.seasons,
        "leagues": payload.leagues,
        "days_ahead": payload.days_ahead,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="daily_refresh", params=params)
        background_tasks.add_task(
            run_daily_refresh,
            seasons=payload.seasons,
            leagues=payload.leagues,
            days_ahead=payload.days_ahead,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Daily refresh job queued", details={"job_id": row["job_id"]})

    report = run_daily_refresh(
        seasons=payload.seasons,
        leagues=payload.leagues,
        days_ahead=payload.days_ahead,
    )
    return JobResponse(queued=False, message="Daily refresh job completed", details=report)


@app.post("/jobs/prediction-snapshot-refresh", response_model=JobResponse)
def trigger_prediction_snapshot_refresh(payload: JobPredictionSnapshotRefreshRequest) -> JobResponse:
    """Bottone "Ricalcola previsioni del giorno" della Dashboard (2026-09-13):
    esegue ORA lo stesso giro del job schedulato `prediction_snapshot_refresh`
    ma scopato alla sola data richiesta. `async_run=true` (default del
    bottone) mette il lavoro in un thread e ritorna subito `job_id`: il
    frontend polla `GET /jobs/{job_id}` e mostra una barra, anche se la
    pagina viene ricaricata. `async_run=false` resta sincrono per i test.
    Lavora solo su dati gia' a DB: nessuna chiamata API-Sports."""
    if payload.target_date:
        try:
            date.fromisoformat(payload.target_date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail=f"target_date non valida (atteso YYYY-MM-DD): {payload.target_date}"
            )

    params = {"target_date": payload.target_date}
    if payload.async_run:
        row = JobHistory().queue_job(job_type="prediction_snapshot_refresh", params=params)
        # Thread dedicato (non FastAPI BackgroundTasks): il ricalcolo e'
        # sincrono e lungo, e BackgroundTasks bloccherebbe lo stesso worker
        # che deve servire il polling di avanzamento. Senza questo il bottone
        # restava disabilitato fino al refresh della pagina.
        thread = threading.Thread(
            target=_run_prediction_snapshot_refresh_job,
            kwargs={"target_date": payload.target_date, "job_id": row["job_id"]},
            daemon=True,
            name=f"prediction-snapshot-refresh-{str(row['job_id'])[:8]}",
        )
        thread.start()
        return JobResponse(
            queued=True, message="Prediction snapshot refresh job queued", details={"job_id": row["job_id"]}
        )

    report = run_prediction_snapshot_refresh(target_date=payload.target_date)
    return JobResponse(queued=False, message="Prediction snapshot refresh job completed", details=report)


@app.post("/jobs/settlement", response_model=JobResponse)
def trigger_settlement(payload: JobSettlementRequest, background_tasks: BackgroundTasks) -> JobResponse:
    params = {
        "from_date": payload.from_date,
        "to_date": payload.to_date,
        "seasons": payload.seasons,
        "leagues": payload.leagues,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="settlement", params=params)
        background_tasks.add_task(
            run_manual_settlement,
            from_date=payload.from_date,
            to_date=payload.to_date,
            seasons=payload.seasons,
            leagues=payload.leagues,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Settlement job queued", details={"job_id": row["job_id"]})

    report = run_manual_settlement(
        from_date=payload.from_date,
        to_date=payload.to_date,
        seasons=payload.seasons,
        leagues=payload.leagues,
    )
    return JobResponse(queued=False, message="Settlement job completed", details=report)


@app.get("/settlement/overview")
def settlement_overview(limit: int = 200, settlement_status: Optional[str] = None) -> dict[str, Any]:
    service = SettlementService()
    return service.settlement_overview(limit=limit, settlement_status=settlement_status)


@app.post("/jobs/retrain", response_model=JobResponse)
def trigger_retrain(payload: JobRetrainRequest, background_tasks: BackgroundTasks) -> JobResponse:
    params = {
        "markets": payload.markets,
        "seasons": payload.seasons,
        "selection_method": payload.selection_method,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="retrain", params=params)
        background_tasks.add_task(
            run_manual_retrain,
            markets=payload.markets,
            seasons=payload.seasons,
            selection_method=payload.selection_method,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Retrain job queued", details={"job_id": row["job_id"]})

    report = run_manual_retrain(
        markets=payload.markets,
        seasons=payload.seasons,
        selection_method=payload.selection_method,
    )
    return JobResponse(queued=False, message="Retrain job completed", details=report)


@app.get("/metrics/{market}", response_model=MetricsResponse)
def metrics(market: str, limit: int = 30) -> MetricsResponse:
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    latest = registry.get_latest(market=market)
    production = registry.get_production(market=market)
    history = registry.tail(limit=limit, market=market)

    return MetricsResponse(market=market, latest=latest, production=production, history=history)


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    return {"summary": _load_summary()}


@app.get("/models/diagnostics", response_model=ModelDiagnosticsResponse)
def model_diagnostics(markets: Optional[str] = None, force_refresh: bool = False) -> ModelDiagnosticsResponse:
    """Model Diagnostics (2026-09-12): confusion matrix/precision/recall/F1
    per classe/ROC+AUC via walk-forward OOF (`clone()` del modello
    registrato, rifittato fold per fold - MAI valutato sui dati di
    training) per ogni mercato BINARIO con un modello registrato (esclude
    "1x2" e i mercati a linea configurabile, vedi
    `model_diagnostics_service.py`). `markets` non specificato -> tutti i
    mercati diagnosticabili. Risposta cache 15 min (`force_refresh=true`
    per bypassarla) - il ricalcolo rifitta il modello per ogni fold/
    mercato, potenzialmente lento."""
    selected_markets = [item.strip() for item in markets.split(",") if item.strip()] if markets else None
    payload = ModelDiagnosticsService().get_diagnostics(markets=selected_markets, force_refresh=force_refresh)
    return ModelDiagnosticsResponse(**payload)


@app.get("/models/{market}/registry", response_model=ModelRegistryOverviewResponse)
def model_registry_overview(market: str, limit: int = 50) -> ModelRegistryOverviewResponse:
    """OPS-02: vista lifecycle completa per mercato (latest/production/
    history con `current_stage`/`promotion_history` gia' decorati da
    `ModelRegistry`, nessuna logica duplicata)."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    latest = registry.get_latest(market=market)
    production = registry.get_production(market=market)
    history = registry.tail(limit=limit, market=market)
    return ModelRegistryOverviewResponse(market=market, latest=latest, production=production, history=history)


@app.get("/models/{market}/promotion/evaluate", response_model=PromotionEvaluationResponse)
def evaluate_model_promotion(market: str, run_id: str, to_stage: str = "production") -> PromotionEvaluationResponse:
    """OPS-02 (dry-run, nessuna modifica): gate metriche + confronto con
    l'attuale production per lo stesso mercato — usato dall'operatore PER
    DECIDERE prima di chiamare `POST /models/{market}/promote`."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run non trovato: {run_id}")
    if run.get("market") != market:
        raise HTTPException(status_code=400, detail=f"Il run {run_id} non appartiene al mercato '{market}'")

    result = registry.evaluate_promotion(run_id=run_id, to_stage=to_stage, policy=DEFAULT_PROMOTION_POLICY)
    return PromotionEvaluationResponse(**result)


@app.post("/models/{market}/promote", response_model=PromotionResponse)
def promote_model(market: str, payload: PromotionRequest) -> PromotionResponse:
    """OPS-02: Candidate -> Champion -> Production promotion CONTROLLATA
    (mai automatica — acceptance criteria "Ultimo training non diventa
    automaticamente production"). Applica gate metriche + confronto con
    l'attuale production; `force=True` permette un override manuale
    esplicito SEMPRE tracciato nell'audit (`promotion_history.jsonl`)."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    run = registry.get_run(payload.run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run non trovato: {payload.run_id}")
    if run.get("market") != market:
        raise HTTPException(status_code=400, detail=f"Il run {payload.run_id} non appartiene al mercato '{market}'")

    try:
        result = registry.promote_with_policy(
            run_id=payload.run_id,
            to_stage=payload.to_stage,
            reason=payload.reason,
            actor=payload.actor or "manual",
            force=payload.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PromotionResponse(**result)


@app.post("/models/{market}/rollback", response_model=PromotionResponse)
def rollback_model(market: str, payload: RollbackRequest) -> PromotionResponse:
    """OPS-02 (Rollback): riporta lo stage 'production' del mercato a un
    run precedente (esplicito `to_run_id` oppure l'ultima production
    precedente dall'audit trail). Bypassa deliberatamente il gate
    metriche — azione di emergenza, sempre tracciata come
    `event_type='rollback'`."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    try:
        result = registry.rollback(
            market=market,
            to_run_id=payload.to_run_id,
            reason=payload.reason,
            actor=payload.actor or "manual",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PromotionResponse(
        promoted=True,
        run_id=result["rolled_back_to"],
        market=market,
        to_stage="production",
        evaluation=None,
        run=result.get("run"),
    )


@app.get("/models/{market}/promotion-history", response_model=PromotionHistoryResponse)
def model_promotion_history(market: str, limit: int = 100) -> PromotionHistoryResponse:
    """OPS-02 (acceptance criteria "Audit promotion"): storico completo
    degli eventi di lifecycle per il mercato, incluse promozioni bloccate
    dal gate e rollback."""
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    events = registry.list_promotion_events(market=market, limit=limit)
    return PromotionHistoryResponse(market=market, events=events)


@app.post("/jobs/data-quality-report", response_model=JobResponse)
def trigger_data_quality_report(payload: JobDataQualityReportRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Bottone "Aggiorna report" della pagina Data Quality: ricalcola il
    report (stessa funzione del job schedulato omonimo, vedi
    `run_data_quality_report`) e lo logga in job history. Non chiama alcun
    provider esterno: mai coinvolto dall'auto-pausa per quota API-Sports."""
    params = {
        "top_n": payload.top_n,
        "seasons": payload.seasons,
        "leagues": payload.leagues,
    }
    if payload.async_run:
        row = JobHistory().queue_job(job_type="data_quality_report", params=params)
        background_tasks.add_task(
            run_data_quality_report,
            top_n=payload.top_n,
            seasons=payload.seasons,
            leagues=payload.leagues,
            job_id=row["job_id"],
        )
        return JobResponse(queued=True, message="Data quality report job queued", details={"job_id": row["job_id"]})

    report = run_data_quality_report(
        top_n=payload.top_n,
        seasons=payload.seasons,
        leagues=payload.leagues,
    )
    return JobResponse(queued=False, message="Data quality report job completed", details=report)


@app.post("/jobs/official-capture", response_model=JobResponse)
def trigger_official_capture(payload: JobOfficialCaptureRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Bottone "Esegui ora" di Impostazioni per il job 'Cattura PLAY ufficiali':
    stessa funzione del job schedulato omonimo (vedi `run_official_prediction_capture`
    in `src/jobs/scheduler.py`). Non chiama alcun provider esterno."""
    if payload.async_run:
        row = JobHistory().queue_job(job_type="official_prediction_capture", params={})
        background_tasks.add_task(run_official_prediction_capture, job_id=row["job_id"])
        return JobResponse(queued=True, message="Official capture job queued", details={"job_id": row["job_id"]})

    report = run_official_prediction_capture()
    return JobResponse(queued=False, message="Official capture job completed", details=report)


@app.get("/data/quality", response_model=DataQualityResponse)
def data_quality(top_n: int = 20, seasons: Optional[str] = None, leagues: Optional[str] = None) -> DataQualityResponse:
    service = DataQualityService()
    payload = service.build_report(
        top_n=top_n,
        seasons=_parse_int_csv(seasons, "seasons"),
        leagues=_parse_int_csv(leagues, "leagues"),
    )
    return DataQualityResponse(**payload)


@app.get("/odds/snapshots/{fixture_id}")
def odds_snapshots(fixture_id: int) -> dict[str, Any]:
    rows = OddsSnapshotRepository().opening_latest_closing(fixture_id=fixture_id)
    return {
        "fixture_id": fixture_id,
        "rows": rows,
        "total": len(rows),
    }


@app.get("/jobs/history", response_model=JobsHistoryResponse)
def jobs_history(limit: int = 100, job_type: Optional[str] = None, status: Optional[str] = None) -> JobsHistoryResponse:
    rows = JobHistory().tail(limit=limit, job_type=job_type, status=status)
    return JobsHistoryResponse(rows=rows)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    """Stato + summary di avanzamento di un job. Usato dal bottone
    "Ricalcola previsioni": il frontend persiste `job_id` e, anche dopo un
    refresh, riprende la barra finche' il processo non termina."""
    row = JobHistory().get(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Job non trovato: {job_id}")
    return JobStatusResponse(
        job_id=str(row.get("job_id") or job_id),
        job_type=str(row.get("job_type") or ""),
        status=str(row.get("status") or ""),
        params=row.get("params") or {},
        summary=row.get("summary") or {},
        error=row.get("error"),
        queued_at=row.get("queued_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        duration_seconds=row.get("duration_seconds"),
    )


@app.get("/predictions/log", response_model=PredictionLogResponse)
def predictions_log(limit: int = 100, market: Optional[str] = None) -> PredictionLogResponse:
    rows = PredictionLogger().tail(limit=limit, market=market)
    return PredictionLogResponse(rows=rows)


@app.post("/predictions/ledger", response_model=PredictionLedgerResponse)
def log_prediction_ledger(payload: PredictionLedgerLogRequest) -> PredictionLedgerResponse:
    """Prediction Ledger / Paper Betting (BET-06): salva UNA prediction
    PRIMA del kickoff. Calcola fair_odd/prob_edge/ev/decision qui (BET-01/
    BET-04, riusati — mai un client che duplica la policy di decisione),
    poi persiste (idempotente per fixture/market/outcome/model_run_id)."""
    if payload.market not in (FilterMarketService.SUPPORTED_MARKETS | {"1x2"}):
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {payload.market}")

    decision = evaluate_decision(
        market=payload.market,
        outcome=payload.outcome,
        p_model=payload.p_model,
        p_market_fair=payload.p_market_fair,
        odd=payload.odd,
        samples=payload.samples,
        policy=DEFAULT_DECISION_POLICY,
    )

    try:
        row = PredictionLedgerService().log_prediction(
            fixture_id=payload.fixture_id,
            decision=decision,
            model_run_id=payload.model_run_id,
            model_name=payload.model_name,
            kickoff_at=_parse_iso_datetime_optional(payload.kickoff_at),
            stake=payload.stake,
            dedupe=payload.dedupe,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionLedgerResponse(rows=[row.to_dict()], total=1)


@app.get("/predictions/ledger", response_model=PredictionLedgerResponse)
def list_prediction_ledger(
    market: Optional[str] = None,
    is_settled: Optional[bool] = None,
    limit: int = 200,
) -> PredictionLedgerResponse:
    rows = PredictionLedgerService().list_ledger(market=market, is_settled=is_settled, limit=limit)
    return PredictionLedgerResponse(rows=rows, total=len(rows))


@app.get("/predictions/ledger/fixture/{fixture_id}", response_model=PredictionLedgerResponse)
def list_prediction_ledger_for_fixture(fixture_id: int, market: Optional[str] = None) -> PredictionLedgerResponse:
    rows = [row.to_dict() for row in PredictionLedgerService().repo.list_for_fixture(fixture_id=fixture_id, market=market)]
    return PredictionLedgerResponse(rows=rows, total=len(rows))


@app.post("/predictions/ledger/settle", response_model=PredictionSettlementResponse)
def settle_prediction_ledger() -> PredictionSettlementResponse:
    """Settlement batch (BET-06): settla SOLO le prediction il cui match e'
    effettivamente concluso (`FINAL_STATUSES`); le altre restano pending."""
    report = PredictionLedgerService().settle_pending()
    return PredictionSettlementResponse(**report)


@app.get("/predictions/ledger/pnl", response_model=PaperPnlResponse)
def prediction_ledger_pnl(market: Optional[str] = None, stake: float = 1.0) -> PaperPnlResponse:
    """PnL paper calcolabile (acceptance criteria BET-06): somma grezza dei
    pnl gia' salvati per riga (`raw_summary`, sempre affidabile) piu' il
    report ricco ROI/hit-rate/drawdown (`report`, riusa BET-03, assume
    stake flat uniforme)."""
    service = PredictionLedgerService()
    raw_summary = service.raw_pnl_summary(market=market)
    report = service.paper_pnl_report(market=market, stake=stake)
    return PaperPnlResponse(market=market, raw_summary=raw_summary, report=dataclasses.asdict(report))


@app.get("/predictions/official/performance", response_model=OfficialPerformanceResponse)
def official_prediction_performance(
    days: Optional[int] = None,
    from_at: Optional[str] = None,
    to_at: Optional[str] = None,
    market: Optional[str] = None,
    league: Optional[int] = None,
    model: Optional[str] = None,
    policy_version: Optional[str] = None,
) -> OfficialPerformanceResponse:
    now = datetime.now(timezone.utc)
    since = _parse_iso_datetime_optional(from_at)
    if days is not None:
        if days not in {7, 30, 90}:
            raise HTTPException(status_code=400, detail="days deve essere 7, 30 o 90")
        since = now - timedelta(days=days)
    payload = OfficialPerformanceService().report(
        since=since,
        until=_parse_iso_datetime_optional(to_at),
        market=market,
        league=league,
        model_name=model,
        policy_version=policy_version,
    )
    clv = OfficialClvService().report(
        since=since,
        until=_parse_iso_datetime_optional(to_at),
        market=market,
        league=league,
        model_name=model,
        policy_version=policy_version,
    )
    payload["overall"].update(
        {
            "avg_clv_odd_pct": clv["overall"].get("avg_clv_odd_pct"),
            "positive_clv_rate": clv["overall"].get("positive_clv_rate"),
            "clv_coverage": clv["overall"].get("coverage"),
        }
    )
    return OfficialPerformanceResponse(**payload)


@app.get("/predictions/official/plays", response_model=PredictionLedgerResponse)
def official_prediction_plays(
    market: Optional[str] = None,
    is_settled: Optional[bool] = None,
    limit: int = 200,
) -> PredictionLedgerResponse:
    rows = PredictionLedgerService().repo.list_all(
        market=market,
        is_settled=is_settled,
        cohort=OFFICIAL_COHORT,
        limit=min(max(limit, 0), 2_000),
    )
    return PredictionLedgerResponse(rows=[row.to_dict() for row in rows], total=len(rows))


@app.get("/predictions/official/clv", response_model=OfficialClvResponse)
def official_prediction_clv(
    days: Optional[int] = None,
    from_at: Optional[str] = None,
    to_at: Optional[str] = None,
    market: Optional[str] = None,
    league: Optional[int] = None,
    model: Optional[str] = None,
    policy_version: Optional[str] = None,
) -> OfficialClvResponse:
    now = datetime.now(timezone.utc)
    since = _parse_iso_datetime_optional(from_at)
    if days is not None:
        if days not in {7, 30, 90}:
            raise HTTPException(status_code=400, detail="days deve essere 7, 30 o 90")
        since = now - timedelta(days=days)
    payload = OfficialClvService().report(
        since=since,
        until=_parse_iso_datetime_optional(to_at),
        market=market,
        league=league,
        model_name=model,
        policy_version=policy_version,
    )
    return OfficialClvResponse(**payload)


@app.get("/predictions/official/clv/{id_prediction}")
def official_prediction_clv_detail(id_prediction: str) -> dict[str, Any]:
    payload = OfficialClvService().detail(id_prediction)
    if payload is None:
        raise HTTPException(status_code=404, detail="Prediction ufficiale non trovata")
    return payload


@app.get("/betslip/pool", response_model=BetslipPoolResponse)
def betslip_pool(
    target_date: Optional[str] = None,
    include_borderline: bool = False,
    min_odd: Optional[float] = None,
    max_odd: Optional[float] = None,
    min_ev: Optional[float] = None,
    markets: Optional[str] = None,
) -> BetslipPoolResponse:
    """Pick Pool per Schedina Oracle (SLIP-01): pool DETERMINISTICO e
    TRACCIABILE di pick candidati (solo PLAY di default, opzionalmente
    anche BORDERLINE) per la giornata richiesta, filtrato da vincoli
    quota/EV opzionali (soglie versionate, mai hardcoded). Nessuna nuova
    logica di betting: riusa le `decision_cards` gia' calcolate da
    `DashboardService` (MATCH-01/BET-01/02/04)."""
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    policy = PickPoolPolicy.with_overrides(
        include_borderline=include_borderline,
        min_odd=min_odd,
        max_odd=max_odd,
        min_ev=min_ev,
    )
    service = PickPoolService()
    result = service.build_pool_for_day(
        target_date=_parse_iso_date(target_date), policy=policy, markets=selected_markets
    )
    return BetslipPoolResponse(**dataclasses.asdict(result))


@app.get("/betslip/generate", response_model=BetslipGenerateResponse)
def betslip_generate(
    target_date: Optional[str] = None,
    include_borderline: bool = False,
    min_odd: Optional[float] = None,
    max_odd: Optional[float] = None,
    min_ev: Optional[float] = None,
    markets: Optional[str] = None,
) -> BetslipGenerateResponse:
    """Schedine 2/3/4 eventi con profili Safe/Balanced/Aggressive (SLIP-03):
    per ciascun profilo, combina le pick del Pick Pool (SLIP-01) validate
    dal Correlation Engine (SLIP-02, mai una coppia EXCLUDE nella stessa
    schedina), dichiarando quota combinata e probabilita' (naive vs
    corretta per la correlazione) con metodo esplicito. Nessuna nuova
    logica di betting: riusa il Pick Pool COSI' COM'E'."""
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    policy = PickPoolPolicy.with_overrides(
        include_borderline=include_borderline,
        min_odd=min_odd,
        max_odd=max_odd,
        min_ev=min_ev,
    )
    selected_date = _parse_iso_date(target_date)
    if selected_date < datetime.now(timezone.utc).date():
        raise HTTPException(
            status_code=409,
            detail="Le date passate sono disponibili tramite /betslip/proposals",
        )
    service = BetslipService()
    pool_result, generation = service.generate_exploration_for_day(
        target_date=selected_date,
        pool_policy=policy,
        markets=selected_markets,
    )
    payload = dataclasses.asdict(generation)
    payload["pool_id"] = pool_result.pool_id
    payload["pool_policy_version"] = pool_result.policy_version
    return BetslipGenerateResponse(**payload)


@app.post("/betslip/generate/snapshot", response_model=BetslipGenerateResponse)
def betslip_generate_snapshot(
    target_date: Optional[str] = None,
    include_borderline: bool = False,
    min_odd: Optional[float] = None,
    max_odd: Optional[float] = None,
    min_ev: Optional[float] = None,
    markets: Optional[str] = None,
) -> BetslipGenerateResponse:
    """Genera e salva una revisione solo se i dati della proposta cambiano."""
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    policy = PickPoolPolicy.with_overrides(
        include_borderline=include_borderline,
        min_odd=min_odd,
        max_odd=max_odd,
        min_ev=min_ev,
    )
    try:
        pool_result, generation, snapshot_report = BetslipService().generate_and_snapshot_for_day(
            target_date=_parse_iso_date(target_date),
            pool_policy=policy,
            markets=selected_markets,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    payload = dataclasses.asdict(generation)
    payload["pool_id"] = pool_result.pool_id
    payload["pool_policy_version"] = pool_result.policy_version
    payload["snapshot_report"] = snapshot_report
    return BetslipGenerateResponse(**payload)


@app.get("/betslip/proposals", response_model=BetslipProposalListResponse)
def betslip_saved_proposals(
    reference_date: str,
    latest_only: bool = True,
    limit: int = 200,
) -> BetslipProposalListResponse:
    """Consulta snapshot già salvati; non genera né modifica dati."""
    _parse_iso_date(reference_date)
    rows = BetslipProposalSnapshotService().list_saved(
        reference_date=reference_date,
        latest_only=latest_only,
        limit=min(max(limit, 0), 2_000),
    )
    return BetslipProposalListResponse(total=len(rows), rows=rows)


@app.get("/betslip/official", response_model=OfficialBetslipListResponse)
def betslip_official(
    reference_date: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 200,
) -> OfficialBetslipListResponse:
    """Sole schedine congelate dal job server-side; endpoint strettamente read-only."""
    rows = OfficialBetslipService().list_official(
        reference_date=reference_date,
        status=status.upper() if status else None,
        limit=limit,
    )
    return OfficialBetslipListResponse(total=len(rows), rows=rows)


@app.get("/betslip/official/statistics", response_model=OfficialBetslipStatisticsResponse)
def betslip_official_statistics() -> OfficialBetslipStatisticsResponse:
    return OfficialBetslipStatisticsResponse(statistics=OfficialBetslipService().statistics())


@app.get("/betting/statistics", response_model=BettingStatisticsResponse)
def betting_statistics(days: int = 30) -> BettingStatisticsResponse:
    try:
        return BettingStatisticsResponse(**BettingStatisticsService().report(days=days))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/monitoring/overview", response_model=MonitoringOverviewResponse)
def monitoring_overview(market: Optional[str] = None) -> MonitoringOverviewResponse:
    """OPS-03: prediction volume, calibration drift, ROI rolling (SOLO
    diagnostico) e feature coverage in un'unica risposta. Senza `market`,
    calibration drift/feature coverage restano `None` (richiedono un
    modello specifico); prediction volume/ROI rolling sono invece
    calcolati su TUTTI i mercati insieme."""
    if market is not None and market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    service = MonitoringService()
    payload = service.full_report(market=market)
    return MonitoringOverviewResponse(**payload)


@app.get("/monitoring/alerts", response_model=MonitoringAlertsResponse)
def monitoring_alerts(market: Optional[str] = None) -> MonitoringAlertsResponse:
    """OPS-03 (acceptance criteria "Alert base"): SOLO l'elenco alert, per
    un polling leggero e frequente (es. badge nella sidebar) senza
    ricalcolare l'intero `/monitoring/overview`."""
    if market is not None and market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    service = MonitoringService()
    payload = service.alerts_report(market=market)
    return MonitoringAlertsResponse(**payload)


@app.get("/settings/jobs", response_model=JobSettingsResponse)
def get_settings_jobs() -> JobSettingsResponse:
    """Pagina Impostazioni: stato enabled/disabled corrente di ciascun job
    schedulato (`src/jobs/scheduler.py`), letto da `job_settings.json`
    (file condiviso tra i container `api` e `scheduler`).

    Prima della lettura sincronizza l'auto-pausa per quota esaurita
    (`sync_job_settings_with_quota`, idempotente): se la quota API-Sports
    e' al 100% oggi, tutti i job risultano qui gia' disabilitati e
    `quota_paused=True` - il container `scheduler` fa la STESSA verifica
    autonomamente ad ogni tick, quindi il risultato e' coerente anche se
    nessuno apre mai questa pagina."""
    sync_job_settings_with_quota()
    pause_state = get_quota_pause_state()
    return JobSettingsResponse(
        jobs=list_job_definitions(),
        quota_paused=bool(pause_state.get("paused_date")),
        quota_paused_since=pause_state.get("paused_date"),
    )


@app.post("/settings/jobs", response_model=JobSettingsResponse)
def post_settings_jobs(payload: JobSettingsUpdateRequest) -> JobSettingsResponse:
    """Aggiorna (merge parziale) i flag enabled/disabled dei job. Il
    container `scheduler` verifica questo stato PRIMA di ogni esecuzione
    (non solo all'avvio), quindi il toggle ha effetto immediato senza
    restart di nessun container.

    Se la quota e' esaurita oggi, l'auto-pausa (`sync_job_settings_with_quota`,
    chiamata anche qui SUBITO DOPO l'update) corregge di nuovo a
    disabilitato un eventuale toggle manuale sui SOLI job che chiamano
    API-Sports (`data_daily_refresh`/`data_sync_today`/`data_future_sync`/
    `data_sync_live`): la risposta di questa chiamata riflette gia' il
    valore corretto (mai "acceso" per un solo istante). `data_settlement`/
    `ml_training` non sono mai toccati dall'auto-pausa (non chiamano il
    provider esterno) e un loro toggle ha sempre effetto reale."""
    try:
        update_job_settings(payload.updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sync_job_settings_with_quota()
    pause_state = get_quota_pause_state()
    return JobSettingsResponse(
        jobs=list_job_definitions(),
        quota_paused=bool(pause_state.get("paused_date")),
        quota_paused_since=pause_state.get("paused_date"),
    )


@app.put("/settings/jobs/{job_id}/schedule", response_model=JobScheduleResponse)
def put_settings_job_schedule(job_id: str, payload: JobScheduleUpdateRequest) -> JobScheduleResponse:
    """Salva un override di orario/intervallo per `job_id` (2026-09-09,
    "maggiore controllo" richiesto dall'operatore). Il container
    `scheduler` rilegge questo valore PRIMA di ogni esecuzione (heartbeat
    ogni 30s, vedi `src/jobs/scheduler.py::_is_job_due`), quindi ha effetto
    immediato senza restart di nessun container - stesso principio gia'
    applicato al toggle enabled/disabled.

    Le chiavi accettate dipendono dallo `schedule_kind` del job (vedi
    `GET /settings/jobs`): un `job_id` sconosciuto o un set di chiavi/
    valori non valido per quel job risulta in 400 (mai un override
    silenziosamente scartato o applicato a meta')."""
    if job_id not in JOB_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Job id non riconosciuto: {job_id}")

    schedule = {key: value for key, value in payload.model_dump().items() if value is not None}
    try:
        update_job_schedule(job_id, schedule)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    cfg = load_app_config()
    return JobScheduleResponse(
        job_id=job_id,
        schedule_kind=JOB_DEFINITIONS[job_id]["schedule_kind"],
        schedule=resolve_job_schedule(job_id, cfg=cfg),
        schedule_is_default=False,
    )


@app.delete("/settings/jobs/{job_id}/schedule", response_model=JobScheduleResponse)
def delete_settings_job_schedule(job_id: str) -> JobScheduleResponse:
    """Rimuove l'override di orario/intervallo per `job_id`, tornando al
    default calcolato da `AppConfig` (variabili d'ambiente) - stesso
    effetto immediato (senza restart) del salvataggio."""
    if job_id not in JOB_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Job id non riconosciuto: {job_id}")

    reset_job_schedule(job_id)
    cfg = load_app_config()
    return JobScheduleResponse(
        job_id=job_id,
        schedule_kind=JOB_DEFINITIONS[job_id]["schedule_kind"],
        schedule=resolve_job_schedule(job_id, cfg=cfg),
        schedule_is_default=True,
    )


def _is_today_utc(iso_ts: Optional[str]) -> bool:
    """API-Sports resetta la quota giornaliera a mezzanotte UTC: uno
    snapshot `status_*` di un giorno UTC precedente NON e' piu' affidabile
    (es. "100% esaurita ieri" mostrato ancora oggi che la quota si e'
    resettata sarebbe un falso allarme, opposto ma speculare al bug
    originale) - va quindi trattato come scaduto e si ricade sulla stima
    passiva finche' l'utente non fa un nuovo controllo reale."""
    if not iso_ts:
        return False
    try:
        dt = datetime.fromisoformat(iso_ts)
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()


def _build_quota_response(cfg, snapshot: Optional[dict[str, Any]]) -> ApiQuotaResponse:
    """Costruisce la risposta di `/settings/quota` preferendo SEMPRE, se
    disponibile e riferito ALLA GIORNATA UTC CORRENTE (vedi `_is_today_utc`),
    l'ultimo check live autoritativo (`status_*`, popolato da
    `ApiSportsProvider.get_status()` via `POST /settings/quota/refresh` o
    automaticamente non appena una chiamata dati incontra `errors` -
    vedi `_mark_quota_exhausted`) rispetto alla stima passiva dedotta dagli
    header `x-ratelimit-*` (`daily_remaining`, popolato da QUALSIASI
    chiamata dati - vedi `record_quota_snapshot`). La stima passiva puo'
    essere disallineata dalla realta' (bug diagnosticato 2026-09-05:
    mostrava 0% anche con quota reale al 100%, perche' l'header
    `x-ratelimit-requests-remaining` NON e' un segnale affidabile di
    esaurimento - solo il campo `errors` del body lo e')."""
    if not snapshot:
        return ApiQuotaResponse(
            available=False,
            daily_limit=cfg.api_sports_daily_limit,
            message="Nessuna chiamata API-Sports ancora registrata da questo deploy.",
        )

    status_current = snapshot.get("status_requests_current")
    status_limit = snapshot.get("status_requests_limit_day")
    status_updated_at = snapshot.get("status_updated_at")
    if (
        isinstance(status_current, int)
        and isinstance(status_limit, int)
        and status_limit > 0
        and _is_today_utc(status_updated_at)
    ):
        daily_used = status_current
        daily_limit = status_limit
        daily_remaining = max(daily_limit - daily_used, 0)
        daily_used_percentage = round(min(daily_used / daily_limit * 100, 100.0), 1)
        error_message = snapshot.get("status_error_message")
        return ApiQuotaResponse(
            available=True,
            source="live",
            plan=snapshot.get("status_plan"),
            daily_limit=daily_limit,
            daily_remaining=daily_remaining,
            daily_used=daily_used,
            daily_used_percentage=daily_used_percentage,
            minute_limit=snapshot.get("minute_limit"),
            minute_remaining=snapshot.get("minute_remaining"),
            updated_at=snapshot.get("status_updated_at") or snapshot.get("updated_at"),
            message=f"Provider API-Sports: {error_message}" if error_message else None,
        )

    daily_remaining = snapshot.get("daily_remaining")
    if daily_remaining is None:
        return ApiQuotaResponse(
            available=False,
            daily_limit=cfg.api_sports_daily_limit,
            message="Nessuna chiamata API-Sports ancora registrata da questo deploy.",
        )

    # Preferisce il limite REALE osservato nell'header `x-ratelimit-requests-
    # limit` (vedi `ApiSportsProvider._handle_quota_headers`) al fallback
    # hardcoded `API_SPORTS_DAILY_LIMIT`, che puo' disallinearsi dal piano
    # effettivo se questo non viene aggiornato manualmente.
    daily_limit = snapshot.get("daily_limit") or cfg.api_sports_daily_limit
    daily_used: Optional[int] = None
    daily_used_percentage: Optional[float] = None
    if isinstance(daily_remaining, int) and daily_limit > 0:
        daily_used = max(daily_limit - daily_remaining, 0)
        daily_used_percentage = round(min(daily_used / daily_limit * 100, 100.0), 1)

    return ApiQuotaResponse(
        available=True,
        source="estimated",
        daily_limit=daily_limit,
        daily_remaining=daily_remaining,
        daily_used=daily_used,
        daily_used_percentage=daily_used_percentage,
        minute_limit=snapshot.get("minute_limit"),
        minute_remaining=snapshot.get("minute_remaining"),
        updated_at=snapshot.get("updated_at"),
        message=(
            'Stima non verificata (l\'header di rate-limit NON e\' affidabile per capire se la quota giornaliera'
            ' e\' esaurita): clicca "Aggiorna" per un controllo reale su API-Sports.'
        ),
    )


@app.get("/settings/quota", response_model=ApiQuotaResponse)
def get_settings_quota() -> ApiQuotaResponse:
    """Pagina Impostazioni: stato quota giornaliera API-Sports. Lettura
    VELOCE e senza consumare chiamate reali - mostra l'ultimo valore noto
    (live se gia' stato eseguito un `POST /settings/quota/refresh`, stima
    altrimenti). Per un controllo certo vedi `post_settings_quota_refresh`."""
    cfg = load_app_config()
    snapshot = get_quota_snapshot()
    return _build_quota_response(cfg, snapshot)


@app.post("/settings/quota/refresh", response_model=ApiQuotaResponse)
def post_settings_quota_refresh() -> ApiQuotaResponse:
    """Bottone "Aggiorna" in Impostazioni: a differenza di `GET
    /settings/quota` (che rilegge solo la cache locale), questo endpoint
    interroga DAVVERO l'endpoint ufficiale `GET /status` di API-Sports
    (`ApiSportsProvider.get_status`, 1 chiamata reale consumata) e
    restituisce il consumo AUTORITATIVO - lo stesso numero della dashboard
    account api-sports.io. Non e' in polling automatico apposta (per non
    sprecare quota solo per controllarla): parte SOLO dal click esplicito
    dell'utente."""
    cfg = load_app_config()
    provider = ApiSportsProvider()
    status = provider.get_status()
    snapshot = get_quota_snapshot()
    response = _build_quota_response(cfg, snapshot)
    if status is None and response.source != "live":
        response.message = (
            "Controllo reale su API-Sports non riuscito (rete/config): mostrato l'ultimo valore stimato noto."
        )
    return response





















