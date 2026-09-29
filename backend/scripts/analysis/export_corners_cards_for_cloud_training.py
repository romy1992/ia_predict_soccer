"""Esporta le feature dataset GIA' ELABORATE (non i JSON grezzi di
odds/statistics/mean_statistics, troppo pesanti da trasferire) per Corners e
Cards a linea configurabile (MARKET-05/06, 2026-09-12) - stesso identico
pattern gia' usato per Under/Over 1.5-4.5
(`export_datasets_for_cloud_training.py`).

DA ESEGUIRE SOLO in un ambiente con accesso di rete reale al DB (es. sessione
bridge locale) - serve per trasferire un'istantanea del dataset verso un
ambiente SENZA accesso diretto al DB (es. sessione cloud sandboxed), dove
`train_corners_cards_from_export.py` esegue il training pesante senza dover
rifare le query (e senza dipendere dalla stabilita' della connessione di rete
locale, causa di ripetuti fallimenti di training diretto su bridge in
passato - vedi IMPLEMENTATION_LOG.md).

Output in `scripts/analysis/_export/` (NON gitignored di proposito: va
committato temporaneamente solo per trasferire i CSV via git, poi puo' essere
rimosso dal repo una volta che il training e' completato altrove).
"""

from __future__ import annotations

import os

from src.ml.markets.cards.cards_market import DEFAULT_LINES as CARDS_LINES
from src.ml.markets.cards.cards_market import build_cards_frame_from_records
from src.ml.markets.corners.corners_market import DEFAULT_LINES as CORNERS_LINES
from src.ml.markets.corners.corners_market import build_corners_frame_from_records
from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

OUTPUT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
_FILTERS = {"statistics": "not None", "mean_statistics": "not None", "odds": "not None", "status": ["FT"]}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    match_repo = MatchRepository()
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=_FILTERS))
    print(f"{len(matches)} fixture con statistics/mean_statistics/odds disponibili", flush=True)

    corners_frame = build_corners_frame_from_records(matches, lines=CORNERS_LINES)
    corners_path = os.path.join(OUTPUT_DIR, "corners.csv")
    corners_frame.to_csv(corners_path, index=False)
    print(f"corners: {len(corners_frame)} righe, {corners_frame.shape[1]} colonne -> {corners_path}")

    cards_frame = build_cards_frame_from_records(matches, lines=CARDS_LINES)
    cards_path = os.path.join(OUTPUT_DIR, "cards.csv")
    cards_frame.to_csv(cards_path, index=False)
    print(f"cards: {len(cards_frame)} righe, {cards_frame.shape[1]} colonne -> {cards_path}")

    total_size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, name)) for name in (corners_path, cards_path) if os.path.exists(name)
    ) / (1024 * 1024)
    print(f"\nDimensione totale export: {total_size_mb:.1f} MB in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
