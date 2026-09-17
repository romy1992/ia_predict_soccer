"""Recupera statistiche finali mancanti per fixture gia' presenti nel DB.

Uso:
    python scripts/backfill_missing_fixture_statistics.py 1608651 1608652

Il comando e' intenzionalmente mirato: non reimporta fixture, quote o
predizioni e aggiorna solo match senza due righe statistiche finali.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.repository.base.repository_db import SessionLocal
from src.repository.match_repository import MatchRepository
from src.service_ia.model.match import Statistics
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider
from src.service_ia.pre_processing.download_match_service import map_statistic


def _fixture_payload(provider: ApiSportsProvider, fixture_id: int) -> dict | None:
    fixtures = provider.get_fixtures(id=fixture_id)
    return fixtures[0] if fixtures else None


def backfill(fixture_ids: list[int]) -> dict[str, int]:
    provider = ApiSportsProvider()
    repository = MatchRepository()
    report = {"requested": len(fixture_ids), "updated": 0, "already_complete": 0, "no_match": 0, "no_data": 0, "failed": 0}

    try:
        for fixture_id in fixture_ids:
            try:
                match = repository.filter_by(dict_search={"id_fixture": fixture_id}).first()
                if match is None:
                    report["no_match"] += 1
                    logging.warning("Fixture %s: match non presente nel DB", fixture_id)
                    continue

                if len(match.statistics or []) >= 2:
                    report["already_complete"] += 1
                    logging.info("Fixture %s: statistiche gia' presenti", fixture_id)
                    continue

                fixture = _fixture_payload(provider, fixture_id)
                statistics = provider.get_fixture_statistics(fixture_id)
                if not fixture or len(statistics) < 2:
                    report["no_data"] += 1
                    logging.warning(
                        "Fixture %s: API senza dati completi (fixture=%s, team_statistics=%s)",
                        fixture_id,
                        bool(fixture),
                        len(statistics),
                    )
                    continue

                teams = fixture.get("teams") or {}
                home_id = (teams.get("home") or {}).get("id")
                away_id = (teams.get("away") or {}).get("id")
                stat_team_ids = {(entry.get("team") or {}).get("id") for entry in statistics}
                if not home_id or not away_id or not {home_id, away_id}.issubset(stat_team_ids):
                    report["no_data"] += 1
                    logging.warning("Fixture %s: statistiche non disponibili per entrambe le squadre", fixture_id)
                    continue

                mapped = []
                for entry in statistics:
                    team_id = (entry.get("team") or {}).get("id")
                    side = "home" if team_id == home_id else "away"
                    mapped.append(
                        Statistics(
                            **map_statistic(match, entry, side, fixture_id, fixture)
                        )
                    )
                match.statistics = mapped
                repository.save(match)
                report["updated"] += 1
                logging.info("Fixture %s: statistiche salvate (%s righe)", fixture_id, len(mapped))
            except Exception:
                report["failed"] += 1
                logging.exception("Fixture %s: backfill fallito", fixture_id)
    finally:
        SessionLocal.remove()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill statistiche API-Sports mancanti")
    parser.add_argument("fixture_ids", nargs="+", type=int, help="ID fixture API-Sports da recuperare")
    args = parser.parse_args()
    report = backfill(args.fixture_ids)
    print(report)
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
