from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Default runtime DB: punta SEMPRE al DB "dev" remoto ospitato su Railway
# (richiesto esplicitamente 2026-09-04, vedi README_PLATFORM.md sezione
# "Policy DATABASE_URL"). Usato SOLO se la env var DATABASE_URL non e'
# valorizzata affatto (nessun properties/config.env caricato). Vecchio
# valore locale (Postgres nativo Windows), mantenuto solo come riferimento:
# DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/match_db"
DEFAULT_DATABASE_URL = "postgresql://postgres:zwOQshmeSysBkmIefrYVqMQTgMKckJAz@sakura.proxy.rlwy.net:18862/railway"
DEFAULT_DATABASE_SCHEMA = "public"
_ENV_LOADED = False


def _load_project_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env_path = os.path.join(project_root, "properties", "config.env")
    load_dotenv(dotenv_path=env_path)
    _ENV_LOADED = True


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default

    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError:
            continue

    return values or default


def _int_env(*names: str, default: int) -> int:
    """Legge il primo env var valorizzato tra `names` (in ordine), utile
    per introdurre un nuovo nome di variabile PIU' esplicito (es.
    `TRAINING_HOUR`) mantenendo un fallback sul nome legacy (`SCHEDULER_HOUR`)
    gia' eventualmente configurato in ambienti esistenti (OPS-01: mai
    rompere una configurazione gia' in uso)."""
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return default


@dataclass
class AppConfig:
    leagues: list[int]
    seasons: list[int]
    # OPS-01: orari/intervalli dei job SEPARATI (mai un unico "scheduler_hour"
    # condiviso da data job e training job - vedi `src/jobs/scheduler.py`).
    # Data jobs: frequenti, in minuti (IntervalTrigger).
    data_sync_interval_minutes: int
    settlement_interval_minutes: int
    # Future sync: giornaliero (le fixture future non richiedono la stessa
    # frequenza delle fixture odierne/in corso).
    future_sync_hour: int
    future_sync_minute: int
    # Training job: giornaliero, orario INDIPENDENTE dai data job (mai un
    # retrain automatico legato al ciclo di import - acceptance criteria
    # "No retrain automatico ad ogni import"). Nome storico "scheduler_hour"/
    # "scheduler_minute" (env `SCHEDULER_HOUR`/`SCHEDULER_MINUTE`) mantenuto
    # come fallback per compatibilita' con ambienti gia' configurati.
    training_hour: int
    training_minute: int
    # LIVE-01: pipeline dati live SEPARATA dai data job pre-match sopra -
    # polling frequente (secondi, non minuti: i punteggi/eventi live cambiano
    # molto piu' spesso delle fixture odierne) e TTL di una cache dedicata
    # (mai la stessa `_api_cache` di `DashboardService`, che resta invariata)
    # per evitare chiamate HTTP ripetute entro la stessa finestra di poll.
    live_sync_interval_seconds: int
    live_cache_ttl_seconds: int
    database_url: str
    database_schema: str


def load_app_config() -> AppConfig:
    _load_project_env()

    default_leagues = [135, 136, 137, 138, 942, 943, 140, 144, 78, 61, 39, 94, 203, 88, 492, 2, 3, 848]
    default_seasons = [2026]

    leagues = _parse_int_list(os.environ.get("APP_LEAGUES"), default_leagues)
    seasons = _parse_int_list(os.environ.get("APP_SEASONS"), default_seasons)

    training_hour = _int_env("TRAINING_HOUR", "SCHEDULER_HOUR", default=23)
    training_minute = _int_env("TRAINING_MINUTE", "SCHEDULER_MINUTE", default=0)
    data_sync_interval_minutes = _int_env("DATA_SYNC_INTERVAL_MINUTES", default=30)
    settlement_interval_minutes = _int_env("SETTLEMENT_INTERVAL_MINUTES", default=60)
    future_sync_hour = _int_env("FUTURE_SYNC_HOUR", default=4)
    future_sync_minute = _int_env("FUTURE_SYNC_MINUTE", default=30)
    live_sync_interval_seconds = _int_env("LIVE_SYNC_INTERVAL_SECONDS", default=90)
    live_cache_ttl_seconds = _int_env("LIVE_CACHE_TTL_SECONDS", default=20)
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL).strip()
    database_schema = os.environ.get("DATABASE_SCHEMA", DEFAULT_DATABASE_SCHEMA).strip() or DEFAULT_DATABASE_SCHEMA

    return AppConfig(
        leagues=leagues,
        seasons=seasons,
        data_sync_interval_minutes=data_sync_interval_minutes,
        settlement_interval_minutes=settlement_interval_minutes,
        future_sync_hour=future_sync_hour,
        future_sync_minute=future_sync_minute,
        training_hour=training_hour,
        training_minute=training_minute,
        live_sync_interval_seconds=live_sync_interval_seconds,
        live_cache_ttl_seconds=live_cache_ttl_seconds,
        database_url=database_url,
        database_schema=database_schema,
    )

