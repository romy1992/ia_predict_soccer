"""Client HTTP FotMob via RapidAPI.

La chiave si legge solo da `RAPIDAPI_KEY`. Non viene scritta nei log, nei
messaggi di errore o nel payload restituito.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_HOST = "fotmob-api.p.rapidapi.com"
_RETRY_STATUS = {429, 500, 502, 503, 504}


class FotmobError(RuntimeError):
    pass


class FotmobAuthError(FotmobError):
    """401 o 403: la chiave manca o il piano non autorizza l'endpoint. Nessun retry."""


class FotmobUnavailableError(FotmobError):
    """429 o 5xx ancora in corso dopo i retry."""


class FotmobClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        timeout_seconds: int = 20,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        transport: Optional[Callable[..., tuple[int, Any]]] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = os.environ.get("RAPIDAPI_KEY", "") if api_key is None else api_key
        self.host = (host or os.environ.get("FOTMOB_RAPIDAPI_HOST") or DEFAULT_HOST).strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._transport = transport or self._http_transport
        self._sleep = sleep
        self.requests_made = 0
        self._cache: dict[tuple, Any] = {}

    def matches_by_date(self, day: str) -> Any:
        return self.get("/api/v1/matches/by-date", {"date": day})

    def lineups(self, match_id: int) -> Any:
        return self.get(f"/api/v1/matches/{int(match_id)}/lineups")

    def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        if not self.api_key:
            raise FotmobAuthError("RAPIDAPI_KEY non configurata")
        cache_key = (path, tuple(sorted((params or {}).items())))
        if cache_key in self._cache:
            return self._cache[cache_key]

        last_status = None
        for attempt in range(1, self.max_retries + 1):
            status, payload = self._transport(path, params or {})
            self.requests_made += 1
            last_status = status
            if status in {401, 403}:
                logger.warning("FotMob auth rifiutata status=%s path=%s", status, path)
                raise FotmobAuthError(f"FotMob ha risposto {status} su {path}")
            if status == 404:
                logger.info("FotMob 404 path=%s", path)
                self._cache[cache_key] = None
                return None
            if status in _RETRY_STATUS:
                logger.warning("FotMob status=%s attempt=%s path=%s", status, attempt, path)
                if attempt >= self.max_retries:
                    raise FotmobUnavailableError(f"FotMob ha risposto {status} su {path} dopo {attempt} tentativi")
                self._sleep(self.backoff_seconds * (2 ** (attempt - 1)))
                continue
            if status >= 400:
                logger.warning("FotMob status=%s path=%s", status, path)
                raise FotmobError(f"FotMob ha risposto {status} su {path}")
            if not isinstance(payload, (dict, list)) and payload is not None:
                raise FotmobError(f"FotMob ha risposto con un JSON non oggetto su {path}")
            self._cache[cache_key] = payload
            return payload
        raise FotmobUnavailableError(f"FotMob nessuna risposta valida su {path} (ultimo status {last_status})")

    def _http_transport(self, path: str, params: dict[str, Any]) -> tuple[int, Any]:
        url = f"https://{self.host}{path}"
        try:
            response = requests.get(
                url,
                params=params,
                headers={"x-rapidapi-host": self.host, "x-rapidapi-key": self.api_key},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            logger.warning("FotMob rete path=%s err=%s", path, type(exc).__name__)
            return 503, None
        try:
            body = response.json()
        except ValueError:
            body = None
        return response.status_code, body
