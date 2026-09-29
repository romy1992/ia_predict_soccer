"""Esporta le feature dataset GIA' ELABORATE (`FilterMarketService.build_dataset`)
per il mercato "goal_no_goal" (GG/NG - Both Teams To Score), in CSV compatto.

Stesso identico pattern di `export_datasets_for_cloud_training.py` (gia'
usato per i 4 mercati Under/Over): DA ESEGUIRE SOLO in un ambiente con
accesso di rete reale al DB (es. sessione bridge locale) - serve per
trasferire un'istantanea del dataset verso un ambiente SENZA accesso diretto
al DB (es. sessione cloud sandboxed), cosi' il training pesante puo' girare
li' senza bisogno di rifare le query.

I dataset Under/Over gia' esportati in precedenza NON vengono ri-esportati
qui (sono recuperabili dalla history git, commit "export dataset per
training cloud", se servono di nuovo per l'esperimento di feature injection
GG/NG -> Under/Over): questo script e' scoped al solo nuovo mercato per non
appesantire la sessione bridge.

Output in `scripts/analysis/_export/goal_no_goal.csv` (dir NON gitignored di
proposito, a differenza di `best_models/`: va committata temporaneamente
solo per trasferire il CSV via git, poi va rimossa dal repo una volta che il
training e' completato altrove).
"""
from __future__ import annotations

import os

from src.service_ia.training.market_service.filter_market_service import FilterMarketService

OUTPUT_DIR = os.path.abspath(os.path.join("scripts", "analysis", "_export"))
MARKET = "goal_no_goal"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    service = FilterMarketService()

    df = service.build_dataset(market=MARKET)
    path = os.path.join(OUTPUT_DIR, f"{MARKET}.csv")
    df.to_csv(path, index=False)
    print(f"{MARKET}: {len(df)} righe, {df.shape[1]} colonne -> {path}")

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nDimensione export: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
