from __future__ import annotations

from datetime import datetime, timezone

from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.model.match import OddsSnapshot

CAPTURED = datetime(2026, 11, 17, 18, 0, tzinfo=timezone.utc)


def _snapshot(id_snapshot: str, odd: float) -> OddsSnapshot:
    return OddsSnapshot(
        id_snapshot=id_snapshot, id_match="m1", fixture_id=1, bookmaker="b", market="h2h",
        period="full_time", line=None, outcome="Home", odd=odd, captured_at=CAPTURED, source="api_sports",
    )


def test_upsert_rows_ordinate_per_chiave_e_senza_duplicati():
    """Due scritture parallele (import e cattura delle quote di chiusura)
    devono bloccare le righe nello stesso ordine: righe ordinate per PK, una
    sola riga per PK con l'ultimo valore ricevuto."""
    rows = OddsSnapshotRepository._upsert_rows(
        [_snapshot("c", 1.5), _snapshot("a", 2.0), _snapshot("b", 3.0), _snapshot("a", 2.1)]
    )

    assert [row["id_snapshot"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["odd"] == 2.1
    assert rows[0]["captured_at"] == CAPTURED
