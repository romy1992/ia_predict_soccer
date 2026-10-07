"""Backfill storico degli eventi-per-giocatore (cantiere "giocatori che
segnano", v1 = probabilita' dichiarata, NO motore EV/PLAY-BORDERLINE-NO BET).

One-off manuale (stesso pattern di `backfill_missing_fixture_statistics.py`):
NON un job schedulato - lo storico da qui in avanti si accumula da solo
tramite l'aggancio in `download_match_service.py::download_import_matches`
(Step 3b). Questo script copre solo le fixture GIA' concluse prima di quel
aggancio.

Decisioni di scope (operatore, 2026-10-06):
- cutoff alle ultime 4 stagioni (default 2023-2026), non tutte le 13 a DB -
  le stagioni vecchie pesano poco per "gol/90min ultime 5-10 partite" a
  fronte di un costo quota enorme; estendibile in un secondo giro;
- solo le leghe in `APP_LEAGUES` (`app_config.py`), non tutto cio' che e'
  a DB (alcune fixture a DB sono di leghe non piu' tracciate);
- budget di 2000 richieste/giorno riservate al backfill (il resto della
  quota giornaliera di 7500 serve al normale funzionamento - import
  regolare/settlement/odds usano la stessa chiave).

Uso:
    python scripts/backfill_player_match_events.py
    python scripts/backfill_player_match_events.py --seasons 2024 2025 --max-requests 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.player.player_event_service import extract_player_event
from src.repository.base.repository_db import SessionLocal
from src.repository.player_data_repository import PlayerDataRepository
from src.service_ia.config.app_config import load_app_config
from src.service_ia.model.match import Match
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider, ApiSportsQuotaExceededError
from src.service_ia.pre_processing.download_match_service import _FINISHED_STATUSES

DEFAULT_SEASONS = [2023, 2024, 2025, 2026]


def _candidate_fixtures(seasons: list[int], leagues: list[int]) -> list[int]:
    """Fixture concluse, con `id_fixture` valorizzato, nelle stagioni/leghe
    di scope - piu' recenti prima (budget limitato: se il giro si interrompe
    a meta', restano catturate per prime le partite piu' rilevanti)."""
    with SessionLocal() as session:
        rows = (
            session.query(Match.id_fixture)
            .filter(
                Match.season.in_(seasons),
                Match.league_match.in_(leagues),
                Match.id_fixture.isnot(None),
                Match.status.in_(_FINISHED_STATUSES),
            )
            .order_by(Match.date_match.desc())
            .all()
        )
    return [row[0] for row in rows]


def backfill(seasons: list[int], leagues: list[int], max_requests: int) -> dict[str, int | bool]:
    provider = ApiSportsProvider()
    repository = PlayerDataRepository()
    fixture_ids = _candidate_fixtures(seasons, leagues)

    report = {
        "candidates": len(fixture_ids),
        "already_captured": 0,
        "requests_used": 0,
        "fixtures_captured": 0,
        "goal_events_saved": 0,
        "failed": 0,
        "quota_exceeded": False,
    }

    try:
        for fixture_id in fixture_ids:
            if report["requests_used"] >= max_requests:
                logging.info("Budget richieste esaurito (%s), stop backfill", max_requests)
                break

            if repository.has_events_for_fixture(fixture_id):
                report["already_captured"] += 1
                continue

            try:
                raw_events = provider.get_fixture_events(fixture_id)
                report["requests_used"] += 1
            except ApiSportsQuotaExceededError as quota_exc:
                logging.error("Quota API-Sports esaurita, stop backfill: %s", quota_exc)
                report["quota_exceeded"] = True
                break
            except Exception:
                report["failed"] += 1
                logging.exception("Fixture %s: fetch eventi fallito", fixture_id)
                continue

            events = [extract_player_event(fixture_id, raw_event) for raw_event in raw_events]
            repository.save_events(events)
            report["fixtures_captured"] += 1
            report["goal_events_saved"] += sum(1 for event in events if event.event_type == "Goal")
    finally:
        SessionLocal.remove()

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill storico eventi-per-giocatore (API-Sports)")
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
