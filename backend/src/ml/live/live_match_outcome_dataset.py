"""Dataset per l'addestramento di modelli LIVE (LIVE-03, Fase LIVE ORACLE).

Costruisce, a partire dal feature store point-in-time di LIVE-02
(`LiveFeatureStore`), un DataFrame multi-riga-per-fixture per addestrare
modelli che producono PROBABILITA' AGGIORNATE durante il match (non solo
pre-match): una riga per ogni snapshot noto di ogni fixture GIA' CONCLUSA
(stato finale), con lo stesso target 1X2 (derivato dal risultato REALE
finale, mai dalle quote) ripetuto su tutte le righe di quella fixture.

Pipeline VOLUTAMENTE separata da quella pre-match
(`src.ml.markets.market_1x2`, acceptance criteria "Pipeline separata da
pre-match" di LIVE-03): questo modulo non tocca `build_1x2_dataset_from_db`
ne' `FilterMarketService` - riusa solo la funzione pura `label_1x2` (stessa
identica definizione dell'esito ovunque nel progetto, mai un duplicato
leggermente diverso che potrebbe divergere nel tempo).

Anti-leakage: ogni riga usa ESCLUSIVAMENTE
`LiveFeatureStore.build_feature_history` (LIVE-02, gia' point-in-time per
costruzione: ogni riga usa solo eventi/statistiche con timestamp <= al
proprio `as_of`). Il target e' noto qui solo perche' la fixture e' GIA'
conclusa al momento in cui si costruisce il dataset di TRAINING, non perche'
una singola riga "veda" il futuro della propria stessa partita.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.data.live.live_models import LiveFixtureSnapshot
from src.ml.live.live_feature_store import LiveFeatureStore
from src.ml.markets.market_1x2 import label_1x2
from src.repository.live_data_repository import LiveDataRepository

# Stessa semantica di `_FINAL_STATUSES` in `LiveDataRepository` / `FINAL_STATUSES`
# in `point_in_time_builder.py` (duplicazione deliberata di una costante, non
# di logica - stesso pattern gia' scelto in tutto il progetto per non
# accoppiare moduli via import di dettagli privati).
FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}

# Colonne di IDENTITA'/META: MAI feature di training, servono solo per
# split cronologico per match (`match_time`), report per finestra (`status`,
# `minute_total` - quest'ultima resta anche tra le feature, e' l'unica
# colonna presente in entrambi gli insiemi) e target (`y`).
META_COLUMNS = ["fixture_id", "as_of", "feature_available_at_max", "status", "match_time", "y"]


def completed_fixture_snapshots(live_repository: LiveDataRepository) -> dict[int, LiveFixtureSnapshot]:
    """Ultimo snapshot noto per ciascuna fixture il cui stato FINALE e' gia'
    disponibile (partita conclusa) - unica fonte affidabile del risultato
    reale nel dataset LIVE (mai le quote, mai un modello terzo)."""
    latest = live_repository.latest_fixture_snapshots()
    return {
        fixture_id: snapshot
        for fixture_id, snapshot in latest.items()
        if (snapshot.status or "").upper() in FINAL_STATUSES
        and snapshot.home_goals is not None
        and snapshot.away_goals is not None
    }


def build_match_outcome_dataset(
    feature_store: Optional[LiveFeatureStore] = None,
    live_repository: Optional[LiveDataRepository] = None,
    fixture_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Un frame con 1 riga per OGNI snapshot di OGNI fixture conclusa
    (`build_feature_history`, LIVE-02): il target `y` (HOME/DRAW/AWAY,
    `label_1x2`) e' costante per tutte le righe della stessa fixture, essendo
    il risultato REALE finale (non dipende da quando durante il match la
    riga e' stata calcolata).

    `match_time` = `as_of` della riga stessa (ISO8601): usato SOLO per
    l'ordinamento cronologico dello split per match
    (`match_level_temporal_splits` in `live_match_outcome_model.py`), mai
    come feature di training (e' in `META_COLUMNS`).
    """
    live_repository = live_repository or LiveDataRepository()
    feature_store = feature_store or LiveFeatureStore(live_repository=live_repository)

    completed = completed_fixture_snapshots(live_repository)
    target_fixture_ids = fixture_ids if fixture_ids is not None else sorted(completed.keys())

    rows: list[dict[str, Any]] = []
    for fixture_id in target_fixture_ids:
        final_snapshot = completed.get(int(fixture_id))
        if final_snapshot is None:
            continue
        outcome = label_1x2(final_snapshot.home_goals, final_snapshot.away_goals)
        if outcome is None:
            continue

        history = feature_store.build_feature_history(int(fixture_id))
        for feature_row in history:
            payload = feature_row.to_dict()
            payload["match_time"] = feature_row.as_of
            payload["y"] = outcome
            rows.append(payload)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["match_time"] = pd.to_datetime(frame["match_time"], utc=True, errors="coerce")
    frame = (
        frame.dropna(subset=["match_time"])
        .sort_values(by=["match_time", "fixture_id"])
        .reset_index(drop=True)
    )
    return frame

