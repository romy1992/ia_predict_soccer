"""Query diretta DB: pattern quote costanti per bookmaker su cards 3.5.

Stesso approccio usato per corners_line_8_5 (report_corners_8_5_passo1.md):
per ogni bookmaker che quota Over/Under 3.5 cards, conta fixture distinte e
quante di quelle hanno la quota Over ESATTAMENTE identica (2 decimali).
"""
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import Match
from src.service_ia.training.market_service.filter_market_service import FilterMarketService

session = SessionLocal()
try:
    rows = session.execute(select(Match).options(selectinload(Match.odds))).scalars().all()
    print("Match totali a DB:", len(rows))

    # bookmaker -> outcome ("over"/"under") -> fixture_id -> valore quota
    per_book_over = defaultdict(dict)
    per_book_under = defaultdict(dict)
    total_keys_35 = 0
    odds_records_with_cards = 0

    for m in rows:
        for o in m.odds:
            cards = o.cards
            if not cards:
                continue
            odds_records_with_cards += 1
            for key, raw_value in cards.items():
                line = FilterMarketService._extract_line_from_odds_key(str(key))
                if line is None or abs(line - 3.5) > 0.01:
                    continue
                value = FilterMarketService._safe_float(raw_value)
                if value <= 0:
                    continue
                outcome, bookmaker = FilterMarketService._split_outcome_and_bookmaker(str(key))
                slug = FilterMarketService._normalize_outcome_name(outcome)
                total_keys_35 += 1
                if slug == "over_3_5":
                    per_book_over[bookmaker][m.id_match_fk] = value
                elif slug == "under_3_5":
                    per_book_under[bookmaker][m.id_match_fk] = value

    print("Record Odds con bucket cards non vuoto:", odds_records_with_cards)
    print("Quote totali con linea == 3.5 (grezze):", total_keys_35)
    print()
    print(f"{'Bookmaker':<20}{'N fixture (over)':>18}{'N valori distinti':>20}{'Valore piu comune':>20}{'% su valore comune':>22}")

    suspicious = []
    for book, fixmap in sorted(per_book_over.items(), key=lambda kv: -len(kv[1])):
        n_fixture = len(fixmap)
        values = list(fixmap.values())
        counts = defaultdict(int)
        for v in values:
            counts[round(v, 2)] += 1
        most_common_value, most_common_count = max(counts.items(), key=lambda kv: kv[1])
        pct = most_common_count / n_fixture * 100 if n_fixture else 0
        print(f"{book:<20}{n_fixture:>18}{len(counts):>20}{most_common_value:>20.2f}{pct:>21.1f}%")
        if n_fixture >= 100 and pct >= 95.0:
            suspicious.append((book, n_fixture, most_common_value, pct))

    print()
    if suspicious:
        print("PATTERN SOSPETTO TROVATO (quota Over 3.5 costante):")
        for book, n_fixture, value, pct in suspicious:
            print(f"  - {book}: {n_fixture} fixture, quota Over 3.5 = {value:.2f} nel {pct:.1f}% dei casi")
    else:
        print("Nessuna quota costante trovata, quote cards sembrano pulite.")
finally:
    session.close()
