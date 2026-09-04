from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from src.service_ia.config.app_config import load_app_config


@dataclass
class ApiSportsProviderConfig:
    base_url: str
    api_key: str
    timeout_seconds: int = 20
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5


class ApiSportsProvider:
    """Provider API-Sports con retry e gestione centralizzata del rate limit."""

    def __init__(self, config: Optional[ApiSportsProviderConfig] = None):
        # Garantisce il caricamento di properties/config.env anche in esecuzione locale.
        load_app_config()

        if config is None:
            config = ApiSportsProviderConfig(
                base_url=(os.environ.get("API_SPORTS_BASE") or "").rstrip("/"),
                api_key=os.environ.get("API_SPORTS_KEY") or "",
            )
        self.config = config

    def request(self, path: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        if not self.config.base_url or not self.config.api_key:
            logging.warning("API-Sports config non completa: base_url o api_key mancante")
            return []

        url = f"{self.config.base_url}/{path.lstrip('/')}"
        request_params = dict(params or {})
        headers = {"x-apisports-key": self.config.api_key}

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = requests.get(
                    url=url,
                    headers=headers,
                    params=request_params,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                logging.warning("API-Sports request error attempt=%s path=%s err=%s", attempt, path, exc)
                if attempt >= self.config.max_retries:
                    return []
                time.sleep(self.config.retry_backoff_seconds * attempt)
                continue

            if response.status_code in {429, 500, 502, 503, 504}:
                logging.warning(
                    "API-Sports status=%s attempt=%s path=%s",
                    response.status_code,
                    attempt,
                    path,
                )
                if attempt >= self.config.max_retries:
                    return []
                time.sleep(self.config.retry_backoff_seconds * attempt)
                continue

            return self._parse_response(path=path, response=response)

        return []

    def _parse_response(self, path: str, response: requests.Response) -> list[dict[str, Any]]:
        self._handle_quota_headers(response)

        if response.status_code != 200:
            logging.warning("API-Sports non-200 path=%s status=%s", path, response.status_code)
            return []

        try:
            payload = response.json()
        except ValueError:
            logging.warning("API-Sports payload non JSON path=%s", path)
            return []

        rows = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return rows

    @staticmethod
    def _handle_quota_headers(response: requests.Response) -> None:
        remaining_daily = int(response.headers.get("x-ratelimit-requests-remaining", 0) or 0)
        minute_limit = int(response.headers.get("x-ratelimit-limit", 0) or 0)
        minute_remaining = int(response.headers.get("x-ratelimit-remaining", 0) or 0)

        logging.info(
            "API-Sports quota daily_remaining=%s minute_remaining=%s/%s",
            remaining_daily,
            minute_remaining,
            minute_limit,
        )

        if minute_limit > 0 and minute_remaining <= 0:
            logging.info("API-Sports minute quota esaurita: sleep 60s")
            time.sleep(60)

    def get_fixtures(self, **params: Any) -> list[dict[str, Any]]:
        return self.request(path="fixtures", params=params)

    def get_fixture_statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.request(path="fixtures/statistics", params={"fixture": int(fixture_id)})

    def get_fixture_odds(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.request(path="odds", params={"fixture": int(fixture_id)})

    def get_fixture_events(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.request(path="fixtures/events", params={"fixture": int(fixture_id)})

    def get_predictions(self, fixture_id: int) -> list[dict[str, Any]]:
        return self.request(path="predictions", params={"fixture": int(fixture_id)})
