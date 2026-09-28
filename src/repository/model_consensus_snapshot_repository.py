from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.repository.base.repository_db import SessionLocal
from src.service_ia.model.match import ModelConsensusSnapshot


class ModelConsensusSnapshotRepository:
    """Persistenza della cache di serving del Model Consensus (una riga per
    fixture_id+market, UPSERT dal job schedulato `run_model_consensus_refresh`).

    Stesso pattern "semplice" gia' in uso nelle altre repository del
    progetto: una sessione per operazione, nessuno stato condiviso tra
    chiamate."""

    def get_for_fixture(self, fixture_id: int, markets: list[str]) -> dict[str, ModelConsensusSnapshot]:
        if not markets:
            return {}
        with SessionLocal() as session:
            rows = (
                session.query(ModelConsensusSnapshot)
                .filter(ModelConsensusSnapshot.fixture_id == int(fixture_id))
                .filter(ModelConsensusSnapshot.market.in_(markets))
                .all()
            )
            return {row.market: row for row in rows}

    def get_latest_bulk(
        self,
        fixture_ids: list[int],
        markets: Optional[list[str]] = None,
    ) -> dict[tuple[int, str], ModelConsensusSnapshot]:
        """Tutte le righe (fixture_id, market) gia' presenti tra
        `fixture_ids`, in UNA sola query - usata dal job schedulato per
        l'anti-join sulle fixture concluse gia' completamente coperte
        (stesso principio di `MatchPredictionSnapshotRepository.get_latest_bulk`)."""
        if not fixture_ids:
            return {}
        with SessionLocal() as session:
            query = session.query(ModelConsensusSnapshot).filter(
                ModelConsensusSnapshot.fixture_id.in_([int(value) for value in fixture_ids])
            )
            if markets:
                query = query.filter(ModelConsensusSnapshot.market.in_(markets))
            rows = query.all()
        return {(row.fixture_id, row.market): row for row in rows}

    def upsert_many(self, reports: dict[tuple[int, str], dict[str, Any]]) -> int:
        """`reports` e' un dict {(fixture_id, market): {"experts":..,
        "oracle_final":.., "consensus":.., "warnings":..}} - stessa forma
        gia' prodotta da `OracleMatchDetailService._model_consensus_by_market`.
        `session.merge` fa UPSERT-per-PK (fixture_id, market): nessuna riga
        vecchia orfana, nessun DELETE preventivo necessario."""
        if not reports:
            return 0

        now = datetime.now(timezone.utc)
        with SessionLocal() as session:
            for (fixture_id, market), payload in reports.items():
                session.merge(
                    ModelConsensusSnapshot(
                        fixture_id=int(fixture_id),
                        market=market,
                        experts=payload.get("experts") or [],
                        oracle_final=payload.get("oracle_final"),
                        consensus=payload.get("consensus") or {},
                        warnings=payload.get("warnings") or [],
                        computed_at=now,
                    )
                )
            session.commit()
        return len(reports)
