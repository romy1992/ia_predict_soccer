from __future__ import annotations

import re
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from src.repository.base.repository_db import DATABASE_URL, engine
from src.service_ia.config.app_config import load_app_config


def _sanitize_identifier(value: str, fallback: str) -> str:
    if not value:
        return fallback
    clean = value.strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", clean):
        return clean
    return fallback


def _masked_database_url(url: str) -> str:
    parsed = make_url(url)
    if parsed.password:
        parsed = parsed.set(password="***")
    return str(parsed)


def _safe_count(table_name: str, schema_name: str) -> int | None:
    # Identificatori sanificati per evitare interpolation pericolosa.
    safe_schema = _sanitize_identifier(schema_name, "public")
    safe_table = _sanitize_identifier(table_name, table_name)
    statement = text(f'SELECT COUNT(*) FROM "{safe_schema}"."{safe_table}"')

    try:
        with engine.connect() as connection:
            value = connection.execute(statement).scalar()
        return int(value or 0)
    except Exception:
        return None


def get_database_audit() -> dict[str, Any]:
    cfg = load_app_config()
    schema_name = _sanitize_identifier(cfg.database_schema, "public")

    payload: dict[str, Any] = {
        "status": "ok",
        "database_url": _masked_database_url(DATABASE_URL),
        "schema": schema_name,
        "counts": {
            "match": None,
            "statistics": None,
            "odds": None,
            "odds_snapshot": None,
        },
    }

    try:
        parsed = make_url(DATABASE_URL)
        payload.update(
            {
                "driver": parsed.drivername,
                "host": parsed.host,
                "port": parsed.port,
                "database": parsed.database,
                "username": parsed.username,
            }
        )

        payload["counts"]["match"] = _safe_count("match", schema_name)
        payload["counts"]["statistics"] = _safe_count("statistics", schema_name)
        payload["counts"]["odds"] = _safe_count("odds", schema_name)
        payload["counts"]["odds_snapshot"] = _safe_count("odds_snapshot", schema_name)
    except SQLAlchemyError as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = str(exc)

    return payload
