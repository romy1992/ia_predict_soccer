"""Step B: training pipeline completa (save_model=True, no promote) per le 4 linee cards."""
from unittest import mock

import pandas as pd

from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.train_multi_market import train_market

for linea in ("3_5", "4_5", "5_5", "6_5"):
    market = f"cards_line_{linea}"
    QUOTE = [
        f"prob_norm_over_{linea}_line_{linea}",
        f"odds_mean_over_{linea}_line_{linea}",
        f"odds_mean_under_{linea}_line_{linea}",
        f"odds_count_line_{linea}",
        f"odds_std_over_{linea}_line_{linea}",
        f"overround_line_{linea}",
    ]

    df = FilterMarketService().build_dataset(market=market, fill_missing=False)
    df = df[df[f"odds_count_line_{linea}"].notna()].copy()
    df["prediction_at"] = pd.to_datetime(df["prediction_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["prediction_at"]).sort_values("prediction_at").reset_index(drop=True)

    with mock.patch.object(FilterMarketService, "build_dataset", return_value=df):
        result = train_market(market=market, feature_columns=QUOTE, save_model=True)

    print(f"\n=== {market} ===")
    print("status:", result.status)
    print("champion:", result.champion)
    print("rows:", result.rows)
    calib = result.details.get("calibration")
    print("calibration:", calib)
