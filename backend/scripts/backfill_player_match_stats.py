"""Backfill delle statistiche individuali per partita (`fixtures/players`) -
cantiere "giocatori che segnano" (tiri, tiri in porta, minuti, rigori, rating).

Solo fixture FINITE delle prime 5 leghe (stessi default di
`build_player_scoring_dataset.py`) che non hanno ancora righe in
`player_match_stats`; riprendibile (salta le gia' presenti), commit a lotti
(ogni fixture e' salvata subito), 1 chiamata API per fixture, stop su quota.

Uso (da `backend/`):
    python scripts/backfill_player_match_stats.py --max-calls 500
    python scripts/backfill_player_match_stats.py --dry-run
    python scripts/backfill_player_match_stats.py --leagues 135 39 --seasons 2025 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_player_match_events import _candidate_fixtures
from scripts.build_player_scoring_dataset import DEFAULT_SEASONS
from src.data.player.coverage_report import TOP5_LEAGUES
from src.data.player.player_stats_service import extract_player_stats
from src.repository.base.repository_db import SessionLocal
from src.repository.player_data_repository import PlayerDataRepository
from src.service_ia.pre_processing.api_sports_provider import (
    ApiSportsProvider,
    ApiSportsQuotaExceededError,
)

PROGRESS_EVERY = 25


def backfill(seasons: list[int], leagues: list[int], max_calls: int, dry_run: bool = False) -> dict[str, int | bool]:
    repository = PlayerDataRepository()
    fixture_ids = _candidate_fixtures(seasons, leagues)
    captured = repository.captured_stats_fixture_ids(fixture_ids)
    todo = [fid for fid in fixture_ids if fid not in captured]
    print(f"Candidate: {len(fixture_ids)} | gia' con stats: {len(captured)} | da scaricare: {len(todo)}")

    report: dict[str, int | bool] = {
        "candidates": len(fixture_ids),
        "already_captured": len(captured),
        "to_download": len(todo),
        "calls": 0,
        "fixtures_saved": 0,
        "players_saved": 0,
        "failed": 0,
        "quota_exceeded": False,
    }
    if dry_run:
        print(f"[dry-run] chiamate che verrebbero fatte: {min(len(todo), max_calls)}")
        return report

    provider = ApiSportsProvider()
    try:
        for fixture_id in todo:
            if report["calls"] >= max_calls:
                print(f"Limite --max-calls raggiunto ({max_calls}), stop")
                break
            try:
                raw_players = provider.get_fixture_players(fixture_id)
                report["calls"] += 1
            except ApiSportsQuotaExceededError as quota_exc:
                logging.error("Quota API-Sports esaurita, stop backfill: %s", quota_exc)
                report["quota_exceeded"] = True
                break
            except Exception:
                report["calls"] += 1
                report["failed"] += 1
                logging.exception("Fixture %s: fetch statistiche fallito", fixture_id)
                continue

            rows = [row for block in raw_players for row in extract_player_stats(fixture_id, block)]
            repository.save_player_stats(rows)  # commit per fixture: riprendibile
            report["fixtures_saved"] += 1
            report["players_saved"] += len(rows)
            if report["calls"] % PROGRESS_EVERY == 0:
                print(f"[{report['calls']}/{min(len(todo), max_calls)}] fixture salvate={report['fixtures_saved']} "
                      f"giocatori={report['players_saved']} errori={report['failed']}")
    finally:
        SessionLocal.remove()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leagues", nargs="+", type=int, default=TOP5_LEAGUES, help="ID lega (default: prime 5)")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--max-calls", type=int, default=500, help="massimo chiamate API in questo giro")
    parser.add_argument("--dry-run", action="store_true", help="conta soltanto, nessuna chiamata API")
    args = parser.parse_args()

    report = backfill(args.seasons, args.leagues, args.max_calls, args.dry_run)
    print(report)
    return 1 if report["failed"] and not report["fixtures_saved"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
