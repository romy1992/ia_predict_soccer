"""Controllo di copertura dei dati per il mercato "giocatori che segnano" - SOLA LETTURA.

Legge `match`, `player_match_event` e `player_lineup` dal DB di `DATABASE_URL` e stampa un
report (vedi `src/data/player/coverage_report.py`). Non scrive nulla e non chiama API-Sports.

Uso (da `backend/`, con `DATABASE_URL` impostato come per gli altri script):
    python scripts/check_player_data_coverage.py                       # prime 5 leghe, stagioni 2023-2026
    python scripts/check_player_data_coverage.py --all-leagues         # tutte le APP_LEAGUES
    python scripts/check_player_data_coverage.py --leagues 135 39 --seasons 2024 2025
    python scripts/check_player_data_coverage.py --output copertura.md --strict
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.player.coverage_report import TOP5_LEAGUES, build_report, render_report  # noqa: E402
from src.repository.base.repository_db import SessionLocal  # noqa: E402
from src.service_ia.config.app_config import load_app_config  # noqa: E402
from src.service_ia.pre_processing.download_match_service import _FINISHED_STATUSES  # noqa: E402

DEFAULT_SEASONS = [2023, 2024, 2025, 2026]

# Importare download_match_service imposta il log globale a DEBUG: qui serve solo il report.
logging.getLogger().setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leagues", nargs="+", type=int, help="ID lega (default: prime 5)")
    parser.add_argument("--all-leagues", action="store_true", help="tutte le leghe di APP_LEAGUES")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--output", help="scrive il report anche su questo file")
    parser.add_argument("--strict", action="store_true", help="exit code 1 se ci sono avvisi")
    args = parser.parse_args()

    leagues = args.leagues or (load_app_config().leagues if args.all_leagues else TOP5_LEAGUES)
    with SessionLocal() as session:
        report = build_report(session, leagues, args.seasons, _FINISHED_STATUSES)
    text = render_report(report)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    return 1 if (args.strict and report.warnings()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
