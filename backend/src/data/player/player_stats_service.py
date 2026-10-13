"""Mapping grezzo->ORM per le statistiche per giocatore (vedi
`player_stats_models.py`). Funzione pura, testabile senza DB/provider."""

from __future__ import annotations

from typing import Any, Optional

from src.data.player.player_stats_models import PlayerMatchStats


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _block(stat: dict[str, Any], key: str) -> dict[str, Any]:
    value = stat.get(key)
    return value if isinstance(value, dict) else {}


def extract_player_stats(fixture_id: int, raw_team_block: dict[str, Any]) -> list[PlayerMatchStats]:
    """Mappa UN blocco squadra di `fixtures/players` (`team` + `players[]`,
    ognuno con `player` e `statistics[0]`) in una riga per giocatore.
    Valori mancanti/None -> None. Giocatori senza id o senza statistiche
    vengono saltati."""
    team_id = _safe_int((raw_team_block.get("team") or {}).get("id"))
    rows: list[PlayerMatchStats] = []
    for entry in raw_team_block.get("players") or []:
        player_id = _safe_int((entry.get("player") or {}).get("id"))
        statistics = entry.get("statistics") or []
        if player_id is None or not statistics or not isinstance(statistics[0], dict):
            continue
        stat = statistics[0]
        games, shots, goals = _block(stat, "games"), _block(stat, "shots"), _block(stat, "goals")
        passes, dribbles = _block(stat, "passes"), _block(stat, "dribbles")
        fouls, cards, penalty = _block(stat, "fouls"), _block(stat, "cards"), _block(stat, "penalty")
        rows.append(
            PlayerMatchStats(
                fixture_id=fixture_id,
                player_id=player_id,
                team_id=team_id,
                minutes=_safe_int(games.get("minutes")),
                rating=_safe_float(games.get("rating")),
                shots_total=_safe_int(shots.get("total")),
                shots_on=_safe_int(shots.get("on")),
                goals=_safe_int(goals.get("total")),
                assists=_safe_int(goals.get("assists")),
                saves=_safe_int(goals.get("saves")),
                passes_total=_safe_int(passes.get("total")),
                key_passes=_safe_int(passes.get("key")),
                dribbles_attempts=_safe_int(dribbles.get("attempts")),
                dribbles_success=_safe_int(dribbles.get("success")),
                fouls_drawn=_safe_int(fouls.get("drawn")),
                fouls_committed=_safe_int(fouls.get("committed")),
                yellow=_safe_int(cards.get("yellow")),
                red=_safe_int(cards.get("red")),
                pen_won=_safe_int(penalty.get("won")),
                pen_committed=_safe_int(penalty.get("commited", penalty.get("committed"))),
                pen_scored=_safe_int(penalty.get("scored")),
                pen_missed=_safe_int(penalty.get("missed")),
                offsides=_safe_int(stat.get("offsides")),
                captain=games.get("captain") if isinstance(games.get("captain"), bool) else None,
                substitute=games.get("substitute") if isinstance(games.get("substitute"), bool) else None,
                position=games.get("position"),
                source="api_sports",
            )
        )
    return rows
