"""Mappa partite, squadre e giocatori FotMob sugli ID API-Sports gia' a DB.

Il confronto e' per nome normalizzato e per data. Un nome che corrisponde a
piu' di un `player_id` resta non mappato: non si sceglie a caso.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.service_ia.fotmob.names import name_tokens, names_match, player_keys, token_key
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
# Tra due giocatori con la stessa chiave si sceglie il piu' recente solo se l'altro non gioca da almeno questo tempo.
_RECENCY_GAP = timedelta(days=60)


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
    by_tokens: dict[str, list[RosterAppearance]]
    tokens_by_player: dict[int, tuple[str, ...]]
    latest_team: dict[int, tuple[int, Optional[datetime]]]
    has_rows: bool


def empty_roster() -> TeamRoster:
    return TeamRoster(by_full={}, by_initial={}, by_tokens={}, tokens_by_player={}, latest_team={}, has_rows=False)


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
    ordered = token_key(name)
    if ordered:
        roster.by_tokens.setdefault(ordered, []).append(appearance)
    tokens = name_tokens(name)
    if tokens:
        roster.tokens_by_player[player_id] = tokens
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


def _recent_pick(candidates: list[RosterAppearance], method: str, team_id: int, roster: TeamRoster) -> Optional[PlayerMap]:
    """Due giocatori con lo stesso nome nella stessa rosa: vince quello che gioca ancora, se l'altro e' fermo da mesi."""
    latest: dict[int, RosterAppearance] = {}
    for item in candidates:
        kept = latest.get(item.player_id)
        if kept is None or _aware(item.last_seen) > _aware(kept.last_seen):
            latest[item.player_id] = item
    ranked = sorted(latest.values(), key=lambda item: _aware(item.last_seen), reverse=True)
    if len(ranked) < 2 or ranked[0].last_seen is None or ranked[1].last_seen is None:
        return None
    if _aware(ranked[0].last_seen) - _aware(ranked[1].last_seen) < _RECENCY_GAP:
        return None
    picked = _unique([ranked[0]], f"{method}_recent", team_id, roster)
    return PlayerMap(picked.api_player_id, picked.method, picked.position, "recency_pick")


def _surname_subset(name: Optional[str], team_id: int, roster: TeamRoster) -> Optional[PlayerMap]:
    """"Noel Aseko" contro "N. Aseko Nkili": stessa iniziale e cognomi uno contenuto nell'altro. Solo se unico."""
    tokens = name_tokens(name)
    if len(tokens) < 2:
        return None
    initial, surname = tokens[0][0], set(tokens[1:])
    hits = []
    for player_id, other in roster.tokens_by_player.items():
        if len(other) < 2 or other[0][0] != initial:
            continue
        other_surname = set(other[1:])
        if surname <= other_surname or other_surname <= surname:
            hits.append(player_id)
    if len(hits) != 1:
        return None
    appearances = [item for items in roster.by_full.values() for item in items if item.player_id == hits[0]]
    if not appearances:
        appearances = [item for items in roster.by_initial.values() for item in items if item.player_id == hits[0]]
    return _unique(appearances, "surname_subset", team_id, roster) if appearances else None


def position_for(roster: TeamRoster, player_id: int) -> Optional[str]:
    """Ruolo storico (il piu' recente) di un giocatore nella rosa, oppure None."""
    best: Optional[RosterAppearance] = None
    for items in roster.by_full.values():
        for item in items:
            if item.player_id == player_id and (best is None or _aware(item.last_seen) >= _aware(best.last_seen)):
                best = item
    return best.position if best is not None else None


def map_player(name: Optional[str], team_id: int, roster: TeamRoster) -> PlayerMap:
    full, initial = player_keys(name)
    ambiguous = False
    for key, table, method in (
        (full, roster.by_full, "exact_name"),
        (token_key(name), roster.by_tokens, "name_tokens"),
        (initial, roster.by_initial, "last_name_initial"),
    ):
        if not key or key not in table:
            continue
        result = _unique(table[key], method, team_id, roster)
        if result.api_player_id is not None:
            return result
        recent = _recent_pick(table[key], method, team_id, roster)
        if recent is not None:
            return recent
        ambiguous = True
    if not ambiguous:
        subset = _surname_subset(name, team_id, roster)
        if subset is not None:
            return subset
    return PlayerMap(None, None, None, "ambiguous" if ambiguous else "unmapped")


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
