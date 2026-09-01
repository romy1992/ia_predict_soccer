from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


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


@dataclass
class AppConfig:
    leagues: list[int]
    seasons: list[int]
    scheduler_hour: int
    scheduler_minute: int


def load_app_config() -> AppConfig:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env_path = os.path.join(project_root, "properties", "config.env")
    load_dotenv(dotenv_path=env_path)

    default_leagues = [135, 136, 137, 138, 942, 943, 140, 144, 78, 61, 39, 94, 203, 88, 492, 2, 3, 848]
    default_seasons = [2026]

    leagues = _parse_int_list(os.environ.get("APP_LEAGUES"), default_leagues)
    seasons = _parse_int_list(os.environ.get("APP_SEASONS"), default_seasons)

    hour = int(os.environ.get("SCHEDULER_HOUR", "23"))
    minute = int(os.environ.get("SCHEDULER_MINUTE", "0"))

    return AppConfig(
        leagues=leagues,
        seasons=seasons,
        scheduler_hour=hour,
        scheduler_minute=minute,
    )

