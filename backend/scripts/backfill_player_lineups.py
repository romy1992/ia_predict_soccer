"""Backfill storico delle formazioni (cantiere "giocatori che segnano") -
completa `backfill_player_match_events.py`: gli eventi dicono solo chi ha
segnato/preso cartellino, le formazioni dicono chi ha giocato (l'universo
completo, necessario per un dataset "ha segnato si/no" senza distorsioni -
vedi `src/data/player/player_lineup_models.py`).

One-off manuale, stesso pattern del backfill eventi: NON un job schedulato -
lo storico da qui in avanti si accumula da solo tramite l'aggancio in
`download_match_service.py::download_import_matches` (Step 4).

Stesse decisioni di scope del backfill eventi (operatore, 2026-10-06):
cutoff 2023-2026, solo APP_LEAGUES, budget richieste/giorno configurabile.

Uso:
    python scripts/backfill_player_lineups.py
    python scripts/backfill_player_lineups.py --seasons 2024 2025 --max-requests 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_player_match_events import DEFAULT_SEASONS, _candidate_fixtures
from src.data.player.player_lineup_service import extract_lineup_players
from src.repository.base.repository_db import SessionLocal
from src.repository.player_data_repository import PlayerDataRepository
from src.service_ia.config.app_config import load_app_config
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider, ApiSportsQuotaExceededError


def backfill(seasons: list[int], leagues: list[int], max_requests: int) -> dict[str, int | bool]:
    provider = ApiSportsProvider()
    repository = PlayerDataRepository()
    fixture_ids = _candidate_fixtures(seasons, leagues)

    already_captured_ids = repository.captured_lineup_fixture_ids(fixture_ids)
    logging.info(
        "Candidati: %s totali, %s gia' catturati, %s da processare",
        len(fixture_ids), len(already_captured_ids), len(fixture_ids) - len(already_captured_ids),
    )

    report = {
        "candidates": len(fixture_ids),
        "already_captured": len(already_captured_ids),
        "requests_used": 0,
        "fixtures_captured": 0,
        "players_saved": 0,
        "failed": 0,
        "quota_exceeded": False,
    }

    try:
        for fixture_id in fixture_ids:
            if report["requests_used"] >= max_requests:
                logging.info("Budget richieste esaurito (%s), stop backfill", max_requests)
                break

            if fixture_id in already_captured_ids:
                continue

            try:
                raw_lineups = provider.get_fixture_lineups(fixture_id)
                report["requests_used"] += 1
            except ApiSportsQuotaExceededError as quota_exc:
                logging.error("Quota API-Sports esaurita, stop backfill: %s", quota_exc)
                report["quota_exceeded"] = True
                break
            except Exception:
                report["failed"] += 1
                logging.exception("Fixture %s: fetch formazioni fallito", fixture_id)
                continue

            lineup_rows = [
                player_row
                for raw_team_lineup in raw_lineups
                for player_row in extract_lineup_players(fixture_id, raw_team_lineup)
            ]
            repository.save_lineups(lineup_rows)
            report["fixtures_captured"] += 1
            report["players_saved"] += len(lineup_rows)
    finally:
        SessionLocal.remove()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill storico formazioni (API-Sports)")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--leagues", nargs="+", type=int, default=None, help="Default: APP_LEAGUES da config")
    parser.add_argument("--max-requests", type=int, default=2000)
    args = parser.parse_args()

    cfg = load_app_config()
    leagues = args.leagues if args.leagues is not None else list(cfg.leagues)

    report = backfill(seasons=args.seasons, leagues=leagues, max_requests=args.max_requests)
    print(report)
    return 1 if report["failed"] and not report["fixtures_captured"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
