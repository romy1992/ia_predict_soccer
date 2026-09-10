"""Backfill UNA TANTUM della banca dati predizioni (`match_prediction_snapshot`)
per TUTTO lo storico gia' concluso (2026-09-10, punto 3/4 di
`docs/soccer_oracle_v2_detailed/PROMPT_fast_historical_predictions.md`).

`match_prediction_snapshot` riceve una riga solo in tre casi: qualcuno apre
quella fixture in Dashboard, il job in background la intercetta mentre e'
ancora NS, oppure (dal 2026-09-10) il job la intercetta entro pochi giorni
dalla fine. Una fixture CONCLUSA mai vista in nessuno di questi tre modi
resta senza riga salvata per sempre, e la prima volta che qualcuno apre
quella data storica in Dashboard il calcolo scatta al volo (lento) -
esattamente il buco che questo script chiude una volta per tutte.

DA ESEGUIRE UNA VOLTA, dalla macchina/ambiente con accesso reale al DB
Postgres di produzione (mai da una sessione cloud senza quell'accesso).
Idempotente: rieseguirlo salta le fixture gia' coperte (stesso anti-join
del job schedulato, vedi `src.jobs.scheduler.run_prediction_snapshot_refresh`),
quindi puo' essere interrotto (Ctrl+C) e ripreso senza duplicare nulla -
una fixture conclusa gia' processata non viene MAI ricalcolata una seconda
volta (regime "congelato" di `PredictionSnapshotService`).

Uso:
    python -m scripts.backfill_prediction_snapshots
    python -m scripts.backfill_prediction_snapshots --batch-size 100
    python -m scripts.backfill_prediction_snapshots --limit 500  # prova su un sottoinsieme
"""
from __future__ import annotations

import argparse
import logging
import time

from sqlalchemy.orm import selectinload

from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService
from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import Match
from src.service_ia.training.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("backfill_prediction_snapshots")

# Stesso insieme di `PredictionSnapshotService._FINAL_STATUSES` - duplicato
# qui per lo stesso motivo li' documentato (niente dipendenza a ritroso tra
# package, scelta esplicita gia' presa in questa sessione di NON
# consolidare le copie sparse nel progetto).
_FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}

_DEFAULT_BATCH_SIZE = 200


def _fetch_final_fixture_ids_batch(after_fixture_id: int, batch_size: int) -> list[int]:
    """Keyset pagination su `id_fixture` (MAI `OFFSET` - su una tabella di
    decine/centinaia di migliaia di righe il costo di un OFFSET grande
    cresce linearmente ad ogni pagina, un `id_fixture > after_fixture_id`
    ordinato resta invece costante). Query leggera: carica SOLO gli id, mai
    le righe complete - evita di tenere in memoria migliaia di oggetti ORM
    con relazioni caricate in un colpo solo."""
    with SessionLocal() as session:
        rows = (
            session.query(Match.id_fixture)
            .filter(Match.id_fixture.is_not(None))
            .filter(Match.status.in_(_FINAL_STATUSES))
            .filter(Match.id_fixture > after_fixture_id)
            .order_by(Match.id_fixture.asc())
            .limit(batch_size)
            .all()
        )
    return [row[0] for row in rows]


def run_backfill(batch_size: int = _DEFAULT_BATCH_SIZE, limit: int | None = None) -> dict:
    markets = ModelRegistry().list_markets()
    service = PredictionSnapshotService()

    fixtures_scanned = 0
    fixtures_needing_backfill = 0
    fixtures_processed = 0
    predictions_resolved = 0
    errors: list[dict] = []

    start = time.perf_counter()
    after_fixture_id = -1

    while True:
        fixture_ids = _fetch_final_fixture_ids_batch(after_fixture_id=after_fixture_id, batch_size=batch_size)
        if not fixture_ids:
            break
        after_fixture_id = fixture_ids[-1]
        fixtures_scanned += len(fixture_ids)

        with SessionLocal() as session:
            matches = (
                session.query(Match)
                .options(selectinload(Match.statistics), selectinload(Match.odds))
                .filter(Match.id_fixture.in_(fixture_ids))
                .all()
            )

        # Anti-join (stessa logica del job schedulato, punto 2/4): scarta
        # le fixture GIA' completamente coperte - una riga congelata non va
        # mai ricalcolata.
        existing_snapshots = service.repo.get_latest_bulk(fixture_ids)
        matches_needing_backfill = [
            match
            for match in matches
            if any((match.id_fixture, market) not in existing_snapshots for market in markets)
        ]
        fixtures_needing_backfill += len(matches_needing_backfill)

        for match in matches_needing_backfill:
            if limit is not None and fixtures_processed >= limit:
                break
            fixtures_processed += 1
            try:
                payload = service.resolve_predictions(
                    fixture_id=match.id_fixture,
                    markets=markets,
                    db_match=match,
                    status=match.status,
                    allow_compute=True,
                )
                predictions_resolved += len(payload)
            except Exception as exc:
                errors.append({"fixture_id": match.id_fixture, "message": str(exc)})
                logger.warning("Fixture %s: errore durante il backfill - %s", match.id_fixture, exc)

        logger.info(
            "Batch fino a fixture_id=%s: scansionate=%d, da coprire=%d, coperte finora=%d, errori finora=%d",
            after_fixture_id,
            fixtures_scanned,
            fixtures_needing_backfill,
            fixtures_processed,
            len(errors),
        )

        if limit is not None and fixtures_processed >= limit:
            logger.info("Limite di %d fixture processate raggiunto: interruzione anticipata.", limit)
            break

    duration = time.perf_counter() - start
    summary = {
        "fixtures_scanned": fixtures_scanned,
        "fixtures_needing_backfill": fixtures_needing_backfill,
        "fixtures_processed": fixtures_processed,
        "predictions_resolved": predictions_resolved,
        "errors": errors,
        "duration_seconds": duration,
    }
    logger.info(
        "Backfill completato: %d fixture scansionate, %d da coprire, %d processate, "
        "%d predizioni salvate, %d errori, %.1fs",
        fixtures_scanned,
        fixtures_needing_backfill,
        fixtures_processed,
        predictions_resolved,
        len(errors),
        duration,
    )
    if errors:
        logger.warning("Fixture con errore (non hanno ricevuto una riga salvata):")
        for error in errors:
            logger.warning("  - fixture_id=%s: %s", error["fixture_id"], error["message"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Quante fixture caricare/processare per giro (default {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di fixture da PROCESSARE (utile per una prova su un sottoinsieme "
        "prima del giro completo). Default: nessun limite, copre tutto lo storico.",
    )
    args = parser.parse_args()

    run_backfill(batch_size=args.batch_size, limit=args.limit)


if __name__ == "__main__":
    main()
