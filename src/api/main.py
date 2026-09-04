from __future__ import annotations

import dataclasses
import json
import os
from datetime import date, datetime
from typing import Any, Optional

import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.dashboard_service import DashboardService
from src.api.oracle_match_detail_service import OracleMatchDetailService
from src.api.schemas import (
    DataQualityResponse,
    DatabaseHealthResponse,
    BetslipGenerateResponse,
    BetslipPoolResponse,
    DashboardDayResponse,
    DashboardLiveResponse,
    DashboardMatchDetailResponse,
    DashboardOverviewResponse,
    HealthResponse,
    JobFutureSyncRequest,
    JobImportRequest,
    JobResponse,
    JobRetrainRequest,
    JobSettlementRequest,
    JobTodayUpdateRequest,
    JobsHistoryResponse,
    MetricsResponse,
    ModelConsensusResponse,
    OracleMatchDetailResponse,
    PaperPnlResponse,
    PredictRequest,
    PredictResponse,
    PredictionLedgerLogRequest,
    PredictionLedgerResponse,
    PredictionLogResponse,
    PredictionSettlementResponse,
)
from src.data.quality_report_service import DataQualityService
from src.ml.ensemble.model_consensus import build_model_consensus_for_fixture
from src.oracle.betslip.pick_pool import PickPoolPolicy
from src.oracle.betslip.pick_pool_service import PickPoolService
from src.oracle.betslip.betslip_service import BetslipService
from src.oracle.decision_engine.decision_policy import DEFAULT_DECISION_POLICY, evaluate_decision
from src.oracle.ledger.ledger_service import PredictionLedgerService
from src.repository.base.database_audit import get_database_audit
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.jobs.job_history import JobHistory
from src.jobs.scheduler import (
    run_manual_future_sync,
    run_manual_import,
    run_manual_retrain,
    run_manual_settlement,
    run_manual_today_update,
)
from src.service_ia.pre_processing.settlement_service import SettlementService
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
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
    # sorted for stable UI rendering
    values = sorted(FilterMarketService.SUPPORTED_MARKETS)
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
) -> DashboardDayResponse:
    selected_markets = [item.strip() for item in markets.split(",")] if markets else None
    service = DashboardService()
    payload = service.get_day_matches(
        target_date=_parse_iso_date(target_date),
        limit=limit,
        with_predictions=with_predictions,
        markets=selected_markets,
        phase=phase,
        search_text=search,
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

    model_path = active.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"File modello non trovato: {model_path}")

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
    if payload.market not in FilterMarketService.SUPPORTED_MARKETS:
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

    row = PredictionLedgerService().log_prediction(
        fixture_id=payload.fixture_id,
        decision=decision,
        model_run_id=payload.model_run_id,
        model_name=payload.model_name,
        kickoff_at=_parse_iso_datetime_optional(payload.kickoff_at),
        stake=payload.stake,
        dedupe=payload.dedupe,
    )
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
    service = BetslipService()
    pool_result, generation = service.generate_for_day(
        target_date=_parse_iso_date(target_date), pool_policy=policy, markets=selected_markets
    )
    payload = dataclasses.asdict(generation)
    payload["pool_id"] = pool_result.pool_id
    payload["pool_policy_version"] = pool_result.policy_version
    return BetslipGenerateResponse(**payload)





















