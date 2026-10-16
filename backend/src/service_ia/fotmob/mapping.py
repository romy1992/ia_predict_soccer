"""Mappa partite, squadre e giocatori FotMob sugli ID API-Sports gia' a DB.

Il confronto e' per nome normalizzato e per data. Un nome che corrisponde a
piu' di un `player_id` resta non mappato: non si sceglie a caso.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.service_ia.fotmob.names import names_match, player_keys
from src.service_ia.fotmob.parse import ParsedMatch

# ID campionato FotMob -> ID lega API-Sports. Solo coppie note.
# Una lega FotMob assente da questa mappa non fa scartare la partita.
FOTMOB_LEAGUE_TO_API = {
    47: 39,    # Premier League
    55: 135,   # Serie A
    87: 140,  # La Liga
    54: 78,    # Bundesliga
    53: 61,    # Ligue 1
    42: 2,     # Champions League
    73: 3,     # Europa League
}

_ROLES = {"G", "D", "M", "F"}
_KICKOFF_CLOSE = timedelta(hours=3)
_KICKOFF_FAR = timedelta(hours=36)


@dataclass
class RosterAppearance:
    player_id: int
    position: Optional[str]
    last_seen: Optional[datetime]


@dataclass
class PlayerMap:
    api_player_id: Optional[int]
    method: Optional[str]
    position: Optional[str]
    warning: Optional[str]


@dataclass
class TeamRoster:
    by_full: dict[str, list[RosterAppearance]]
    by_initial: dict[str, list[RosterAppearance]]
    latest_team: dict[int, tuple[int, Optional[datetime]]]
    has_rows: bool


def empty_roster() -> TeamRoster:
    return TeamRoster(by_full={}, by_initial={}, latest_team={}, has_rows=False)


def _aware(value: Optional[datetime]) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def add_appearance(roster: TeamRoster, *, team_id: int, player_id: int, name: Optional[str],
                   position: Optional[str], last_seen: Optional[datetime]) -> None:
    roster.has_rows = True
    full, initial = player_keys(name)
    appearance = RosterAppearance(
        player_id=player_id,
        position=position if position in _ROLES else None,
        last_seen=last_seen,
    )
    if full:
        roster.by_full.setdefault(full, []).append(appearance)
    if initial:
        roster.by_initial.setdefault(initial, []).append(appearance)
    previous = roster.latest_team.get(player_id)
    if previous is None or _aware(last_seen) >= _aware(previous[1]):
        roster.latest_team[player_id] = (team_id, last_seen)


def _unique(candidates: list[RosterAppearance], method: str, team_id: int, roster: TeamRoster) -> PlayerMap:
    ids = {item.player_id for item in candidates}
    if len(ids) != 1:
        return PlayerMap(None, None, None, "ambiguous")
    player_id = next(iter(ids))
    latest = max(candidates, key=lambda item: _aware(item.last_seen))
    warning = None
    current = roster.latest_team.get(player_id)
    if current is not None and current[0] != team_id:
        warning = "roster_conflict"
    return PlayerMap(player_id, method, latest.position, warning)


def map_player(name: Optional[str], team_id: int, roster: TeamRoster) -> PlayerMap:
    full, initial = player_keys(name)
    if full and full in roster.by_full:
        return _unique(roster.by_full[full], "exact_name", team_id, roster)
    if initial and initial in roster.by_initial:
        return _unique(roster.by_initial[initial], "last_name_initial", team_id, roster)
    return PlayerMap(None, None, None, "unmapped")


def match_fixture(candidates: list[ParsedMatch], *, home: Optional[str], away: Optional[str],
                  kickoff: Optional[datetime], league_id: Optional[int]) -> tuple[Optional[ParsedMatch], str]:
    """Una sola corrispondenza, altrimenti niente. Non si prende la prima a caso."""
    hits: list[tuple[ParsedMatch, str]] = []
    our_kick = kickoff.astimezone(timezone.utc) if kickoff is not None else None
    for candidate in candidates:
        if not names_match(home, candidate.home_name) or not names_match(away, candidate.away_name):
            continue
        mapped_league = FOTMOB_LEAGUE_TO_API.get(candidate.league_id) if candidate.league_id is not None else None
        if mapped_league is not None and league_id is not None and mapped_league != league_id:
            continue
        if candidate.kickoff is not None and our_kick is not None:
            delta = abs(candidate.kickoff - our_kick)
            if delta > _KICKOFF_FAR:
                continue
            method = "teams_and_kickoff" if delta <= _KICKOFF_CLOSE else "teams_same_day"
        else:
            method = "teams_same_day"
        hits.append((candidate, method))
    if len(hits) != 1:
        return None, "ambiguous" if hits else "unmatched"
    return hits[0]
