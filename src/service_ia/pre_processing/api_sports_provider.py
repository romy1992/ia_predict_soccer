from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from src.jobs.api_quota_state import record_quota_snapshot, record_status_snapshot
from src.service_ia.config.app_config import load_app_config


@dataclass
class ApiSportsProviderConfig:
    base_url: str
    api_key: str
    timeout_seconds: int = 20
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5


class ApiSportsQuotaExceededError(RuntimeError):
    """Sollevata quando API-Sports risponde HTTP 200 ma con un payload
    `errors` non vuoto (es. quota giornaliera/piano esaurito). API-Sports
    NON usa un vero status code 4xx per questo caso - senza un controllo
    esplicito del campo `errors`, il chiamante vede solo `response: []` e
    scambia "quota finita" per "nessuna partita trovata" (bug diagnosticato
    2026-09-05: ore di debug per capire perche' fixtures_seen restava 0 per
    QUALSIASI lega/stagione/data, quota daily gia' esaurita dai test)."""


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

            # Legge il body SUBITO (anche su status_code non-200): serve per
            # distinguere, PRIMA di decidere se aspettare/andare in retry,
            # la quota GIORNALIERA esaurita (permanente, campo `errors` con
            # chiave "requests") dal rate-limit AL MINUTO (transitorio,
            # chiave "rateLimit" - vedi `_is_daily_quota_error`).
            payload = self._safe_json(response)
            errors = payload.get("errors") if isinstance(payload, dict) else None
            is_daily_quota_error = errors is not None and self._is_daily_quota_error(errors)

            # BUGFIX 2026-09-06: gli header x-ratelimit-* sono catturati per
            # OGNI risposta (anche errori), MA lo sleep(60) per rate-limit al
            # MINUTO viene saltato quando la quota GIORNALIERA e' comunque
            # esaurita - aspettare un minuto non serve a nulla se il giorno
            # non si e' ancora resettato, e prima questo blocco il thread
            # inutilmente PRIMA di fallire comunque con la stessa eccezione.
            self._handle_quota_headers(response, skip_minute_sleep=is_daily_quota_error)

            if is_daily_quota_error:
                logging.error("API-Sports quota GIORNALIERA esaurita path=%s errors=%s", path, errors)
                self._mark_quota_exhausted(response, errors)
                raise ApiSportsQuotaExceededError(str(errors))

            if errors:
                # BUGFIX 2026-09-06: rate-limit AL MINUTO (chiave
                # "rateLimit", NON "requests") e' TRANSITORIO - la finestra
                # si resetta da sola, non e' la stessa cosa della quota
                # giornaliera esaurita. Diagnosticato da jobs_history.jsonl:
                # l'header/daily_remaining osservato PRIMA di questo errore
                # era ancora alto (es. 5161/7500) eppure veniva loggato
                # "quota giornaliera esaurita" e la fixture abbandonata per
                # sempre - invece di essere ritentata dopo una breve pausa,
                # come qualunque altro errore transitorio (429/5xx sotto).
                # Con 18 campionati e centinaia di fixture per "Aggiorna
                # tutto", questo blip sporadico colpiva ripetutamente,
                # abbandonando fixture che sarebbero andate a buon segno al
                # tentativo successivo.
                logging.warning(
                    "API-Sports rate-limit transitorio (minuto) path=%s attempt=%s errors=%s",
                    path,
                    attempt,
                    errors,
                )
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

            return self._parse_response(path=path, payload=payload, response=response)

        return []

    @staticmethod
    def _safe_json(response: requests.Response) -> Optional[dict[str, Any]]:
        try:
            return response.json()
        except ValueError:
            return None

    @staticmethod
    def _is_daily_quota_error(errors: Any) -> bool:
        """Distingue la quota GIORNALIERA/di piano esaurita (permanente fino
        al reset del giorno successivo) dal rate-limit AL MINUTO (transitorio
        - vedi `request()`). API-Sports usa chiavi diverse nel campo
        `errors` del payload per i due casi: `"requests"` per la quota
        giornaliera, `"rateLimit"` per il limite al minuto. Se `errors` non
        e' un dict riconoscibile (formato inatteso), trattiamo in modo
        CONSERVATIVO come quota giornaliera - mai un retry infinito su un
        errore sconosciuto non ancora osservato/documentato. Un `errors`
        vuoto/falsy (nessun errore) non e' MAI quota esaurita."""
        if not errors:
            return False
        if isinstance(errors, dict):
            keys = {str(k).lower() for k in errors.keys()}
            if keys.issubset({"ratelimit"}):
                return False
            return True
        return True

    def _parse_response(
        self,
        path: str,
        response: requests.Response,
        payload: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        # NOTA: gli header x-ratelimit-* e il controllo `errors` sono gia'
        # gestiti in `request()` per OGNI risposta (anche 429/5xx/quota) -
        # qui arriviamo SOLO con una risposta 200 "pulita" (nessun errors).
        if response.status_code != 200:
            logging.warning("API-Sports non-200 path=%s status=%s", path, response.status_code)
            return []

        if payload is None:
            payload = self._safe_json(response)
        if payload is None:
            logging.warning("API-Sports payload non JSON path=%s", path)
            return []

        rows = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return []
        return rows

    @staticmethod
    def _handle_quota_headers(response: requests.Response, skip_minute_sleep: bool = False) -> None:
        remaining_daily = int(response.headers.get("x-ratelimit-requests-remaining", 0) or 0)
        daily_limit = int(response.headers.get("x-ratelimit-requests-limit", 0) or 0)
        minute_limit = int(response.headers.get("x-ratelimit-limit", 0) or 0)
        minute_remaining = int(response.headers.get("x-ratelimit-remaining", 0) or 0)

        logging.info(
            "API-Sports quota daily_remaining=%s/%s minute_remaining=%s/%s",
            remaining_daily,
            daily_limit,
            minute_remaining,
            minute_limit,
        )
        record_quota_snapshot(
            daily_remaining=remaining_daily,
            daily_limit=daily_limit or None,
            minute_remaining=minute_remaining,
            minute_limit=minute_limit,
        )

        if skip_minute_sleep:
            return

        if minute_limit > 0 and minute_remaining <= 0:
            logging.info("API-Sports minute quota esaurita: sleep 60s")
            time.sleep(60)

    @staticmethod
    def _format_errors(errors: Any) -> str:
        """API-Sports restituisce `errors` come dict (es. `{"requests":
        "You have reached..."}`) o lista: qui estraiamo solo il testo
        leggibile, invece del repr Python grezzo (es. "{'requests': '...'}")
        che finirebbe altrimenti mostrato cosi' com'e' in UI."""
        if isinstance(errors, dict):
            return " | ".join(str(v) for v in errors.values())
        if isinstance(errors, list):
            return " | ".join(str(v) for v in errors)
        return str(errors)

    @staticmethod
    def _mark_quota_exhausted(response: requests.Response, errors: Any) -> None:
        """Marca lo stato condiviso come quota giornaliera esaurita AL 100%.

        Il campo `errors` nel body (qualsiasi endpoint, non solo dati) e'
        l'UNICA fonte davvero autoritativa: diagnosticato 2026-09-05 che
        l'header `x-ratelimit-requests-remaining` puo' riportare un valore
        "sano" (es. 7499 rimanenti su 7500) sulla STESSA risposta HTTP che
        nel body dice esplicitamente "You have reached the request limit
        for the day" - il provider evidentemente enforcea il blocco con un
        contatore diverso da quello esposto in quell'header, che quindi
        NON puo' essere usato da solo per decidere se la quota e' finita.
        Qui forziamo current=limit (100%) usando `x-ratelimit-requests-
        limit` se disponibile (il valore del PIANO, quello sì coerente
        con l'account reale)."""
        daily_limit = int(response.headers.get("x-ratelimit-requests-limit", 0) or 0) or None
        logging.error("API-Sports quota giornaliera esaurita (fonte: body errors): %s", errors)
        record_status_snapshot(
            requests_current=daily_limit,
            requests_limit_day=daily_limit,
            plan=None,
            active=None,
            error_message=ApiSportsProvider._format_errors(errors),
        )

    def get_status(self) -> Optional[dict[str, Any]]:
        """Interroga l'endpoint UFFICIALE `GET /status` di API-Sports:
        a differenza degli header `x-ratelimit-*` (osservati PASSIVAMENTE
        solo quando il codice fa una chiamata dati attraverso questo
        provider, quindi cieco a consumi avvenuti altrove, E dimostrato
        inaffidabile per il solo dato "remaining" - vedi
        `_mark_quota_exhausted`), questo endpoint restituisce
        `requests.current`/`requests.limit_day`: il consumo REALE e
        autoritativo della subscription, ESATTAMENTE lo stesso numero
        mostrato sulla dashboard account di api-sports.io.

        Usato SOLO su richiesta esplicita (bottone "Aggiorna" in
        Impostazioni -> `POST /settings/quota/refresh`), mai in polling
        automatico, per non consumare una chiamata reale ad ogni refresh
        periodico. Ritorna `None` solo per veri fallimenti (rete, config
        mancante, payload del tutto inatteso) - il chiamante ricade
        sull'ultimo snapshot noto."""
        if not self.config.base_url or not self.config.api_key:
            logging.warning("API-Sports config non completa: base_url o api_key mancante")
            return None

        url = f"{self.config.base_url}/status"
        headers = {"x-apisports-key": self.config.api_key}
        try:
            response = requests.get(url=url, headers=headers, timeout=self.config.timeout_seconds)
        except requests.RequestException as exc:
            logging.warning("API-Sports /status request error: %s", exc)
            return None

        if response.status_code != 200:
            logging.warning("API-Sports /status non-200 status=%s", response.status_code)
            return None
        try:
            payload = response.json()
        except ValueError:
            logging.warning("API-Sports /status payload non JSON")
            return None

        if isinstance(payload, dict) and payload.get("errors"):
            # `/status` risponde 200 anche a quota esaurita, con
            # `response: []` (lista vuota, non il dict con `requests.*`
            # atteso) e il messaggio reale dentro `errors` - vedi
            # `_mark_quota_exhausted`. Questo E' il segnale che l'utente
            # cerca quando la barra sembra "sana" ma non lo e' davvero.
            #
            # BUGFIX 2026-09-06: se `errors` e' SOLO un rate-limit al
            # MINUTO (transitorio, chiave "rateLimit" - vedi
            # `_is_daily_quota_error`), NON marchiamo la quota come esaurita
            # (sporcherebbe lo stato con un falso 100%/None) - ritorniamo
            # invece un errore esplicito ma SENZA toccare l'ultimo snapshot
            # quota noto, cosi' il chiamante (pagina Impostazioni) puo'
            # mostrare "riprova tra un minuto" invece di "quota esaurita".
            errors = payload.get("errors")
            if not self._is_daily_quota_error(errors):
                logging.warning("API-Sports /status rate-limit transitorio (minuto): %s", errors)
                return {
                    "requests_current": None,
                    "requests_limit_day": None,
                    "plan": None,
                    "active": None,
                    "error_message": self._format_errors(errors) + " (temporaneo, riprova tra un minuto)",
                }
            logging.error("API-Sports /status errors=%s", errors)
            self._mark_quota_exhausted(response, errors)
            daily_limit = int(response.headers.get("x-ratelimit-requests-limit", 0) or 0) or None
            return {
                "requests_current": daily_limit,
                "requests_limit_day": daily_limit,
                "plan": None,
                "active": None,
                "error_message": self._format_errors(errors),
            }

        self._handle_quota_headers(response)

        data = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return None

        requests_info = data.get("requests") if isinstance(data.get("requests"), dict) else {}
        subscription = data.get("subscription") if isinstance(data.get("subscription"), dict) else {}
        current = requests_info.get("current")
        limit_day = requests_info.get("limit_day")
        result = {
            "requests_current": current if isinstance(current, int) else None,
            "requests_limit_day": limit_day if isinstance(limit_day, int) else None,
            "plan": subscription.get("plan"),
            "active": subscription.get("active"),
        }
        record_status_snapshot(**result)
        return result

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
