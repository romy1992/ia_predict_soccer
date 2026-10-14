"""Costruisce il dataset del mercato cartellini giocatori ("il giocatore prende un giallo?") dal DB - SOLA LETTURA.

Una riga per giocatore e partita, caratteristiche solo dal passato (vedi `src/data/player/cards_dataset.py`).
Scrive un file che il training leggera' senza toccare il DB.

Uso (da `backend/`, con `DATABASE_URL` impostato come per gli altri script):
    python scripts/build_player_cards_dataset.py                       # prime 5 leghe + coppe europee, 2023-2026
    python scripts/build_player_cards_dataset.py --output player_cards_dataset.parquet
    python scripts/build_player_cards_dataset.py --leagues 135 39 --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.player.cards_dataset import TARGET, build_from_session  # noqa: E402
from src.data.player.coverage_report import TOP5_LEAGUES  # noqa: E402
from src.repository.base.repository_db import SessionLocal  # noqa: E402
from src.service_ia.pre_processing.download_match_service import _FINISHED_STATUSES  # noqa: E402

DEFAULT_SEASONS = [2023, 2024, 2025, 2026]
EURO_CUPS = [2, 3, 848]
logging.getLogger().setLevel(logging.WARNING)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--leagues", nargs="+", type=int, help="ID lega (default: prime 5 + coppe europee)")
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--output", default="player_cards_dataset.parquet",
                        help="file di uscita (.parquet se pyarrow e' installato, altrimenti .csv)")
    args = parser.parse_args()

    leagues = args.leagues or (list(TOP5_LEAGUES) + EURO_CUPS)
    with SessionLocal() as session:
        df = build_from_session(session, leagues, args.seasons, _FINISHED_STATUSES)

    out = Path(args.output)
    try:
        df.to_parquet(out, index=False)
    except ImportError:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"Righe: {len(df)} | partite: {df['fixture_id'].nunique()} | giocatori: {df['player_id'].nunique()}")
    if len(df):
        print(f"Bersaglio: {int(df[TARGET].sum())} gialli ({100 * df[TARGET].mean():.1f}%)")
        print(df.groupby(["season"])[TARGET].agg(["count", "sum", "mean"]).round(3).to_string())
    print(f"Scritto: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
