"""FASE 0 (task Under/Over 1.5/2.5/3.5/4.5): report di copertura dati REALE.

Riusa SOLO componenti gia' validati (nessuna nuova pipeline):
- `DataQualityService.build_report` (coverage generale gia' esistente)
- `DataQualityService.build_under_over_threshold_report` (estensione Fase 0,
  a sua volta basata su `FilterMarketService._build_row` +
  `expanding_window_splits`)
- `src.ml.markets.totals.totals_market.build_totals_evaluation_frame` (per
  misurare la size REALE del dataset condiviso hierarchical/goal_distribution)

Ottimizzazione IMPORTANTE (solo di performance, nessuna logica cambiata):
`Match.odds_snapshots` e' una relationship `lazy="selectin"` caricata EAGER
da QUALSIASI query su `Match" (anche quando il chiamante non la usa mai,
come qui) - con decine di migliaia di fixture puo' significare milioni di
righe trasferite inutilmente. Questo script carica le fixture UNA SOLA
VOLTA con quella relazione esplicitamente non caricata (`noload`, opzione
per-query di SQLAlchemy, nessuna modifica al modello condiviso) e riusa la
stessa lista in memoria per tutte le fasi sotto, invece di lasciare che
ciascun metodo rifaccia la propria query pesante.

Output: stampa un sommario leggibile e salva il report completo in
`best_models/under_over_phase0_report.json`.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logging.basicConfig(level=logging.WARNING)

from sqlalchemy.orm import noload

from src.data.quality_report_service import DataQualityService, UNDER_OVER_THRESHOLDS
from src.ml.markets.totals.totals_market import REFERENCE_ODDS_MARKET, THRESHOLDS, build_totals_evaluation_frame
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import Match
from src.service_ia.utility.utils import convert_orm_match_to_dict

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "under_over_phase0_report.json"))


def _load_all_matches_fast(seasons: Optional[list[int]] = None) -> list[Match]:
    """UNICA query pesante dello script: statistics/odds (necessari) restano
    `selectin` (invariato), `odds_snapshots` (mai acceduto da nessuna fase
    sotto) e' esplicitamente `noload` per evitare di trasferire in memoria
    milioni di righe di odds storici non pertinenti a questo report."""
    session = SessionLocal()
    try:
        query = session.query(Match).options(noload(Match.odds_snapshots))
        if seasons:
            query = query.filter(Match.season.in_(seasons))
        return query.all()
    finally:
        SessionLocal.remove()


class _StaticMatchRepo:
    """Adapter minimale per riusare `DataQualityService.build_report` (API
    invariata) sui `matches` GIA' caricati in memoria, senza fargli ripetere
    la stessa query pesante di `_load_all_matches_fast`."""

    def __init__(self, matches: list[Match]):
        self._matches = matches

    def search_all(self) -> list[Match]:
        return self._matches


def _hierarchical_shared_dataset_report(match_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Dimensione REALE del dataset condiviso da hierarchical/goal_distribution
    (build_totals_evaluation_frame, reference_market='under_over_2_5') + check
    minimo di CV temporale, STESSA logica di `run_totals_benchmark` - MA sui
    match GIA' caricati/filtrati FT dal chiamante (nessuna query aggiuntiva)."""
    frame, feature_columns = build_totals_evaluation_frame(
        matches=match_dicts, reference_market=REFERENCE_ODDS_MARKET, thresholds=THRESHOLDS
    )
    rows = int(len(frame))
    cv_folds_available = 0
    min_train_size = None
    min_valid_size = None
    if rows:
        min_train_size = max(30, int(rows * 0.45))
        min_valid_size = max(10, int(rows * 0.1))
        cv_folds_available = len(
            expanding_window_splits(
                frame=frame, time_col="prediction_at", n_splits=5,
                min_train_size=min_train_size, min_valid_size=min_valid_size,
            )
        )

    return {
        "description": (
            "Dataset condiviso da 'hierarchical'/'goal_distribution'/'binary_independent' "
            "COSI' COME confrontati DENTRO totals_market.py: feature odds+mean_stats dal "
            "reference_market 'under_over_2_5', target reale (goal_bin/y_over_*) da "
            "total_goals - richiede quindi solo copertura odds sul mercato 2.5, non sulle "
            "singole soglie 1.5/3.5/4.5."
        ),
        "reference_odds_market": REFERENCE_ODDS_MARKET,
        "thresholds": list(THRESHOLDS),
        "feature_columns_count": len(feature_columns),
        "usable_rows": rows,
        "cv_temporal": {
            "strategy": "expanding_window",
            "n_splits_requested": 5,
            "min_train_size": min_train_size,
            "min_valid_size": min_valid_size,
            "folds_available": cv_folds_available,
            "meets_minimum_for_training": cv_folds_available >= 2,
        },
    }


def main() -> None:
    t0 = time.time()

    print("=" * 100)
    print("FASE 0 - Report copertura dati Under/Over 1.5/2.5/3.5/4.5 (dati REALI da DB)")
    print("=" * 100)

    print("\n[0/3] Caricamento fixture (query unica, riusata per tutte le fasi sotto) ...")
    all_matches = _load_all_matches_fast()
    print(f"      fixture caricate: {len(all_matches)} ({time.time() - t0:.1f}s)")

    service = DataQualityService(match_repo=_StaticMatchRepo(all_matches))

    print("\n[1/3] DataQualityService.build_report() (coverage generale) ...")
    t1 = time.time()
    general_report = service.build_report(top_n=40)
    print(f"      fixtures_total={general_report['source']['fixtures_total']} "
          f"| coverage_odds={general_report['coverage']['coverage_odds_ratio']:.2%} "
          f"| coverage_statistics={general_report['coverage']['coverage_statistics_ratio']:.2%} "
          f"({time.time() - t1:.1f}s)")

    print("\n[2/3] DataQualityService.build_under_over_threshold_report() ...")
    t2 = time.time()
    uo_report = service.build_under_over_threshold_report(top_n=40, matches=all_matches)
    for market, payload in uo_report["per_threshold"].items():
        cv = payload["cv_temporal"]
        print(
            f"      {market:>16s}: odds={payload['fixtures_with_odds_for_market']:>6d} "
            f"({payload['odds_coverage_ratio']:.1%}) | usable_rows={payload['usable_rows_for_training']:>6d} "
            f"| over_ratio={payload['positive_class_ratio_over']} "
            f"| cv_folds={cv['folds_available']} | ok_min={cv['meets_minimum_for_training']}"
        )
    cross = uo_report["cross_threshold"]
    print(f"      fixture con TUTTE le 4 soglie (odds) disponibili: "
          f"{cross['fixtures_with_all_four_thresholds_odds_available']} "
          f"({cross['ratio_over_fixtures_ft_total']:.1%} delle FT) ({time.time() - t2:.1f}s)")

    print("\n[3/3] Dataset condiviso hierarchical/goal_distribution (build_totals_evaluation_frame) ...")
    t3 = time.time()
    matches_ft = [m for m in all_matches if (m.status or "").upper() == "FT"]
    match_dicts_ft = convert_orm_match_to_dict(matches_ft)
    hier_report = _hierarchical_shared_dataset_report(match_dicts_ft)
    print(f"      usable_rows={hier_report['usable_rows']} "
          f"| cv_folds={hier_report['cv_temporal']['folds_available']} "
          f"| ok_min={hier_report['cv_temporal']['meets_minimum_for_training']} "
          f"({time.time() - t3:.1f}s)")

    full_report = {
        "generated_at": uo_report["generated_at"],
        "elapsed_seconds": round(time.time() - t0, 2),
        "general_data_quality": general_report,
        "under_over_threshold_report": uo_report,
        "hierarchical_shared_dataset": hier_report,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nReport completo salvato in: {OUTPUT_PATH}")
    print(f"Tempo totale: {full_report['elapsed_seconds']}s")


if __name__ == "__main__":
    main()



