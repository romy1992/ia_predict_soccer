"""Esporta le feature dataset GIA' ELABORATE (non i JSON grezzi di
odds/statistics/mean_statistics, troppo pesanti da trasferire) per i mercati
Under/Over 1.5/2.5/3.5/4.5 (`FilterMarketService.build_dataset`) + il frame
'totals' usato da `phase3b_sequential_stacking_thresholds.py`
(`build_totals_evaluation_frame`), in CSV compatti.

DA ESEGUIRE SOLO in un ambiente con accesso di rete reale al DB (es. sessione
bridge locale) - serve per trasferire un'istantanea del dataset verso un
ambiente SENZA accesso diretto al DB (es. sessione cloud sandboxed), cosi'
il training pesante puo' girare li' senza bisogno di rifare le query.

Output in `scripts/analysis/_export/` (NON gitignored di proposito, a
differenza di `best_models/`: va committato temporaneamente solo per
trasferire i CSV via git, poi puo' essere rimosso dal repo una volta che il
training e' completato altrove).
"""
from __future__ import annotations

import json
import os

from src.ml.markets.totals.totals_market import build_totals_evaluation_frame
from src.repository.match_repository import MatchRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.utility.utils import convert_orm_match_to_dict

OUTPUT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    service = FilterMarketService()

    for market in MARKETS:
        df = service.build_dataset(market=market)
        path = os.path.join(OUTPUT_DIR, f"{market}.csv")
        df.to_csv(path, index=False)
        print(f"{market}: {len(df)} righe, {df.shape[1]} colonne -> {path}")

    match_repo = MatchRepository()
    filters = {"statistics": "not None", "mean_statistics": "not None", "odds": "not None", "status": ["FT"]}
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))
    frame, feature_columns = build_totals_evaluation_frame(matches=matches)

    frame_path = os.path.join(OUTPUT_DIR, "totals_evaluation_frame.csv")
    columns_path = os.path.join(OUTPUT_DIR, "totals_feature_columns.json")
    frame.to_csv(frame_path, index=False)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f)
    print(f"totals: {len(frame)} righe, {len(feature_columns)} feature -> {frame_path}")

    total_size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, name)) for name in os.listdir(OUTPUT_DIR)
    ) / (1024 * 1024)
    print(f"\nDimensione totale export: {total_size_mb:.1f} MB in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
