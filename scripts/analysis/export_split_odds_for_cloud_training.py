"""Variante SPERIMENTALE di `export_datasets_for_cloud_training.py`: esporta
i 4 mercati Under/Over 1.5/2.5/3.5/4.5 con le quote calcolate SEPARATAMENTE
per lato (Over vs Under), non messe nello stesso paniere come fa oggi
`FilterMarketService._extract_market_odds_features` in produzione.

Contesto (proposta dell'operatore, 2026-09-08): oggi `market_odds` (il
dizionario di quote per un mercato, es. "under_over_1_5") contiene le quote
di ENTRAMBI i lati ("over 1.5_bet365", "under 1.5_pinnacle", ...) e
`_extract_market_odds_features` le tratta come un'UNICA distribuzione
(count/mean/std/min/max + 10 quote ordinate) - quindi `odds_count` conta il
doppio dei bookmaker reali, e mean/std mescolano i due lati. Qui invece si
calcola una media/std/min/max separata per "over" e per "under" (le chiavi
del dizionario iniziano sempre con "over "/"under ", vedi `map_odds()` in
`download_match_service.py`: `alternate_value = str(value['value']).lower()`).

Nessuna modifica al codice di produzione (`FilterMarketService` NON viene
toccato): questo script usa un `SUBCLASS` locale che sovrascrive SOLO
`_extract_market_odds_features`, riusando INVARIATI target/mean_statistics
(`_label_by_market`/`_extract_mean_features`/`_resolve_team_stats` ereditati
cosi' come sono) - stesso principio "nessuna riscrittura non necessaria"
gia' seguito nel resto del progetto.

DA ESEGUIRE SOLO in un ambiente con accesso DB reale (sessione bridge).
Output in `scripts/analysis/_export/` (non gitignored di proposito, vedi
`export_datasets_for_cloud_training.py`).
"""
from __future__ import annotations

import os

import numpy as np

from src.service_ia.training.market_service.filter_market_service import FilterMarketService

OUTPUT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]


class SplitOddsFilterMarketService(FilterMarketService):
    @staticmethod
    def _extract_market_odds_features(market_odds: dict) -> dict:
        over_values: list[float] = []
        under_values: list[float] = []
        for key, value in market_odds.items():
            if value is None:
                continue
            v = FilterMarketService._safe_float(value)
            if v <= 0:
                continue
            key_lower = str(key).lower()
            if key_lower.startswith("over"):
                over_values.append(v)
            elif key_lower.startswith("under"):
                under_values.append(v)

        features: dict = {}
        for label, values in (("over", over_values), ("under", under_values)):
            if not values:
                continue
            features[f"{label}_odds_count"] = float(len(values))
            features[f"{label}_odds_mean"] = float(np.mean(values))
            features[f"{label}_odds_std"] = float(np.std(values))
            features[f"{label}_odds_min"] = float(np.min(values))
            features[f"{label}_odds_max"] = float(np.max(values))
        return features


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    service = SplitOddsFilterMarketService()

    for market in MARKETS:
        df = service.build_dataset(market=market)
        path = os.path.join(OUTPUT_DIR, f"{market}_split_odds.csv")
        df.to_csv(path, index=False)
        print(f"{market}: {len(df)} righe, {df.shape[1]} colonne -> {path}")
        over_cols = [c for c in df.columns if c.startswith("over_odds")]
        under_cols = [c for c in df.columns if c.startswith("under_odds")]
        print(f"  colonne quote: {over_cols + under_cols}")

    total_size_mb = sum(
        os.path.getsize(os.path.join(OUTPUT_DIR, name))
        for name in os.listdir(OUTPUT_DIR)
        if name.endswith("_split_odds.csv")
    ) / (1024 * 1024)
    print(f"\nDimensione totale export: {total_size_mb:.1f} MB in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
