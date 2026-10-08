"""Mapping grezzo->ORM per le formazioni (vedi `player_lineup_models.py`
per il perche'). Funzione pura, stesso stile di `extract_player_event` -
testabile senza DB/provider."""

from __future__ import annotations

from typing import Any, Optional

from src.data.player.player_lineup_models import PlayerLineup


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_lineup_players(fixture_id: int, raw_team_lineup: dict[str, Any]) -> list[PlayerLineup]:
    """Mappa UN blocco squadra grezzo (`fixtures/lineups`, un elemento
    dell'array per fixture - la risposta ne contiene due, una per squadra)
    in una riga per giocatore, titolari + panchina."""
    team = raw_team_lineup.get("team") or {}
    team_id = _safe_int(team.get("id"))
    team_name = team.get("name")
    formation = raw_team_lineup.get("formation")

    rows: list[PlayerLineup] = []
    for is_starter, entries in ((True, raw_team_lineup.get("startXI")), (False, raw_team_lineup.get("substitutes"))):
        for entry in entries or []:
            player = entry.get("player") or {}
            player_id = _safe_int(player.get("id"))
            if player_id is None:
                continue
            rows.append(
                PlayerLineup(
                    fixture_id=fixture_id,
                    player_id=player_id,
                    player_name=player.get("name"),
                    team_id=team_id,
                    team_name=team_name,
                    position=player.get("pos"),
                    is_starter=is_starter,
                    formation=formation,
                    source="api_sports",
                )
            )
    return rows
