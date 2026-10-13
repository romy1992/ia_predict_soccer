"""Mapping grezzo->ORM per lo storico eventi-per-giocatore (vedi
`player_models.py` per il perche' di una tabella dedicata, separata da
`live_match_event`). Funzione pura, stesso stile di
`live_data_service.extract_event` - testabile senza DB/provider."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.data.player.player_models import PlayerMatchEvent, player_event_id


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_player_event(
    fixture_id: int, raw_event: dict[str, Any], recorded_at: Optional[datetime] = None
) -> PlayerMatchEvent:
    """Mappa UN evento grezzo (`fixtures/events`) in una riga permanente."""
    recorded_at = recorded_at or datetime.now(timezone.utc)
    time_block = raw_event.get("time") or {}
    team = raw_event.get("team") or {}
    player = raw_event.get("player") or {}
    assist = raw_event.get("assist") or {}

    elapsed = _safe_int(time_block.get("elapsed"))
    elapsed_extra = _safe_int(time_block.get("extra"))
    event_type = raw_event.get("type")
    event_detail = raw_event.get("detail")
    team_id = _safe_int(team.get("id"))
    player_id = _safe_int(player.get("id"))
    player_name = player.get("name")

    event_id = player_event_id(
        fixture_id=fixture_id,
        event_type=event_type,
        detail=event_detail,
        elapsed=elapsed,
        elapsed_extra=elapsed_extra,
        team_id=team_id,
        player_id=player_id,
        player_name=player_name,
    )
    return PlayerMatchEvent(
        id_event=event_id,
        fixture_id=fixture_id,
        recorded_at=recorded_at,
        elapsed_minute=elapsed,
        elapsed_extra=elapsed_extra,
        event_type=event_type,
        event_detail=event_detail,
        team_id=team_id,
        team_name=team.get("name"),
        player_id=player_id,
        player_name=player_name,
        assist_id=_safe_int(assist.get("id")),
        assist_name=assist.get("name"),
        comments=raw_event.get("comments"),
        source="api_sports",
    )
