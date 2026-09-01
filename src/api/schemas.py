from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


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
    async_run: bool = True


class JobRetrainRequest(BaseModel):
    markets: Optional[list[str]] = None
    seasons: Optional[list[int]] = None
    selection_method: str = "kbest"
    async_run: bool = True


class JobResponse(BaseModel):
    queued: bool
    message: str


class MetricsResponse(BaseModel):
    market: str
    latest: Optional[dict[str, Any]] = None
    history: list[dict[str, Any]] = []


class JobsHistoryResponse(BaseModel):
    rows: list[dict[str, Any]]


class PredictionLogResponse(BaseModel):
    rows: list[dict[str, Any]]

