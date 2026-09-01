from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Optional

import joblib
import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.dashboard_service import DashboardService
from src.api.schemas import (
    DashboardDayResponse,
    DashboardLiveResponse,
    DashboardOverviewResponse,
    HealthResponse,
    JobImportRequest,
    JobResponse,
    JobRetrainRequest,
    JobsHistoryResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    PredictionLogResponse,
)
from src.jobs.job_history import JobHistory
from src.jobs.scheduler import run_manual_import, run_manual_retrain
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


@app.post("/predict/{market}", response_model=PredictResponse)
def predict(market: str, payload: PredictRequest) -> PredictResponse:
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    latest = registry.get_latest(market=market)
    if not latest:
        raise HTTPException(status_code=404, detail=f"Nessun modello disponibile per mercato {market}")

    model_path = latest.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"File modello non trovato: {model_path}")

    model = joblib.load(model_path)

    builder = FilterMarketService()
    frame = builder.build_prediction_frame(market=market, fixture_id=payload.fixture_id)
    if frame is None or frame.empty:
        raise HTTPException(status_code=404, detail=f"Fixture non trovata o feature insufficienti: {payload.fixture_id}")

    X = frame.drop(columns=["market", "id_fixture", "season"], errors="ignore")

    selected_features = latest.get("feature_names") or []
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
        model_run_id=latest.get("run_id"),
        extra={"model_name": latest.get("model_name")},
    )

    return PredictResponse(
        market=market,
        fixture_id=payload.fixture_id,
        prediction=pred,
        probability=prob,
        model_run_id=latest.get("run_id"),
        model_name=latest.get("model_name"),
    )


@app.post("/jobs/import", response_model=JobResponse)
def trigger_import(payload: JobImportRequest, background_tasks: BackgroundTasks) -> JobResponse:
    if payload.async_run:
        background_tasks.add_task(run_manual_import, payload.seasons, payload.leagues)
        return JobResponse(queued=True, message="Import job queued")

    run_manual_import(seasons=payload.seasons, leagues=payload.leagues)
    return JobResponse(queued=False, message="Import job completed")


@app.post("/jobs/retrain", response_model=JobResponse)
def trigger_retrain(payload: JobRetrainRequest, background_tasks: BackgroundTasks) -> JobResponse:
    if payload.async_run:
        background_tasks.add_task(
            run_manual_retrain,
            payload.markets,
            payload.seasons,
            payload.selection_method,
        )
        return JobResponse(queued=True, message="Retrain job queued")

    run_manual_retrain(
        markets=payload.markets,
        seasons=payload.seasons,
        selection_method=payload.selection_method,
    )
    return JobResponse(queued=False, message="Retrain job completed")


@app.get("/metrics/{market}", response_model=MetricsResponse)
def metrics(market: str, limit: int = 30) -> MetricsResponse:
    if market not in FilterMarketService.SUPPORTED_MARKETS:
        raise HTTPException(status_code=400, detail=f"Mercato non supportato: {market}")

    registry = ModelRegistry()
    latest = registry.get_latest(market=market)
    history = registry.tail(limit=limit, market=market)

    return MetricsResponse(market=market, latest=latest, history=history)


@app.get("/metrics/summary")
def metrics_summary() -> dict[str, Any]:
    return {"summary": _load_summary()}


@app.get("/jobs/history", response_model=JobsHistoryResponse)
def jobs_history(limit: int = 100, job_type: Optional[str] = None) -> JobsHistoryResponse:
    rows = JobHistory().tail(limit=limit, job_type=job_type)
    return JobsHistoryResponse(rows=rows)


@app.get("/predictions/log", response_model=PredictionLogResponse)
def predictions_log(limit: int = 100, market: Optional[str] = None) -> PredictionLogResponse:
    rows = PredictionLogger().tail(limit=limit, market=market)
    return PredictionLogResponse(rows=rows)



