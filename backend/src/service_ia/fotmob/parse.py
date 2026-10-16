"""Estrae partite e formazioni dal JSON FotMob senza assumere un solo involucro."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

_FORMATION = re.compile(r"^\d+(?:-\d+)+$")
_CONFIRMED = {"confirmed", "official"}
_PREDICTED = {"predicted"}


@dataclass
class ParsedPlayer:
    fotmob_id: Optional[int]
    name: Optional[str]
    position_id: Optional[int]
    shirt: Optional[str]
    market_value: Optional[int]
    unavailability_type: Optional[str] = None
    expected_return: Optional[str] = None
    expected_return_date: Optional[str] = None
    return_certainty: Optional[str] = None


@dataclass
class ParsedTeam:
    fotmob_id: Optional[int]
    name: Optional[str]
    formation: Optional[str]
    starters: list[ParsedPlayer] = field(default_factory=list)
    unavailable: list[ParsedPlayer] = field(default_factory=list)
    bench: list[ParsedPlayer] = field(default_factory=list)
    coach_name: Optional[str] = None


@dataclass
class ParsedLineup:
    lineup_type: str
    raw_lineup_type: Optional[str]
    source: Optional[str]
    home: ParsedTeam
    away: ParsedTeam


@dataclass
class ParsedMatch:
    match_id: int
    home_name: Optional[str]
    away_name: Optional[str]
    home_id: Optional[int]
    away_id: Optional[int]
    kickoff: Optional[datetime]
    league_id: Optional[int]
    league_name: Optional[str]


def _int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unwrap(payload: Any) -> Any:
    current = payload
    for _ in range(3):
        if not isinstance(current, dict):
            return current
        for key in ("data", "response", "result"):
            inner = current.get(key)
            if isinstance(inner, (dict, list)):
                current = inner
                break
        else:
            return current
    return current


def _kickoff(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _side(raw: Any) -> tuple[Optional[str], Optional[int]]:
    if isinstance(raw, str):
        return raw, None
    if not isinstance(raw, dict):
        return None, None
    name = raw.get("name") or raw.get("shortName") or raw.get("teamName")
    return (name if isinstance(name, str) else None), _int(raw.get("id"))


def iter_matches(payload: Any) -> list[ParsedMatch]:
    """Partite di `matches/by-date`. Liste vuote se lo schema non e' riconoscibile."""
    root = _unwrap(payload)
    if isinstance(root, list):
        blocks = [{"matches": root}]
    elif isinstance(root, dict):
        leagues = root.get("leagues") or root.get("competitions")
        if isinstance(leagues, list):
            blocks = leagues
        elif isinstance(root.get("matches"), list):
            blocks = [root]
        else:
            return []
    else:
        return []

    found: list[ParsedMatch] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        league_id = _int(block.get("id") or block.get("leagueId") or block.get("primaryId"))
        league_name = block.get("name") or block.get("leagueName")
        matches = block.get("matches") or block.get("events") or []
        if not isinstance(matches, list):
            continue
        for raw in matches:
            if not isinstance(raw, dict):
                continue
            match_id = _int(raw.get("id") or raw.get("matchId"))
            if match_id is None:
                continue
            home_name, home_id = _side(raw.get("home") or raw.get("homeTeam"))
            away_name, away_id = _side(raw.get("away") or raw.get("awayTeam"))
            status = raw.get("status") if isinstance(raw.get("status"), dict) else {}
            kickoff = _kickoff(
                status.get("utcTime") or raw.get("utcTime") or raw.get("matchTimeUTC") or raw.get("time")
            )
            found.append(ParsedMatch(
                match_id=match_id, home_name=home_name, away_name=away_name,
                home_id=home_id, away_id=away_id, kickoff=kickoff,
                league_id=league_id, league_name=league_name if isinstance(league_name, str) else None,
            ))
    return found


def formation_valid(value: Optional[str]) -> bool:
    if not value or not _FORMATION.match(value.strip()):
        return False
    return sum(int(part) for part in value.split("-")) == 10


def return_certainty(expected_return: Optional[str], expected_return_date: Optional[str]) -> str:
    """Una data ISO nel campo dedicato e' una data. Il testo libero non lo e'."""
    if expected_return_date:
        return "dated"
    text = (expected_return or "").strip().lower()
    if not text:
        return "unknown"
    return "uncertain"


def _player(raw: Any, *, absence: bool = False) -> Optional[ParsedPlayer]:
    if not isinstance(raw, dict):
        return None
    player = raw.get("player") if isinstance(raw.get("player"), dict) else raw
    unavail = raw.get("unavailability") if isinstance(raw.get("unavailability"), dict) else {}
    if absence and not unavail and isinstance(player.get("unavailability"), dict):
        unavail = player["unavailability"]
    expected_return = unavail.get("expectedReturn") if isinstance(unavail.get("expectedReturn"), str) else None
    expected_date = unavail.get("expectedReturnDate") if isinstance(unavail.get("expectedReturnDate"), str) else None
    market = player.get("marketValue")
    return ParsedPlayer(
        fotmob_id=_int(player.get("id")),
        name=player.get("name") if isinstance(player.get("name"), str) else None,
        position_id=_int(player.get("positionId") or player.get("usualPlayingPositionId")),
        shirt=str(player["shirtNumber"]) if player.get("shirtNumber") not in (None, "") else None,
        market_value=_int(market),
        unavailability_type=unavail.get("type") if isinstance(unavail.get("type"), str) else None,
        expected_return=expected_return,
        expected_return_date=expected_date,
        return_certainty=return_certainty(expected_return, expected_date) if absence else None,
    )


def _team(raw: Any) -> ParsedTeam:
    if not isinstance(raw, dict):
        return ParsedTeam(fotmob_id=None, name=None, formation=None)
    starters = [p for item in (raw.get("starters") or raw.get("startXI") or []) if (p := _player(item))]
    unavailable = [p for item in (raw.get("unavailable") or raw.get("injuries") or []) if (p := _player(item, absence=True))]
    bench = [p for item in (raw.get("subs") or raw.get("substitutes") or raw.get("bench") or []) if (p := _player(item))]
    coach = raw.get("coach") if isinstance(raw.get("coach"), dict) else {}
    formation = raw.get("formation") if isinstance(raw.get("formation"), str) else None
    nested = raw.get("team") if isinstance(raw.get("team"), dict) else {}
    return ParsedTeam(
        fotmob_id=_int(raw.get("id") or nested.get("id")),
        name=raw.get("name") if isinstance(raw.get("name"), str) else None,
        formation=formation,
        starters=starters,
        unavailable=unavailable,
        bench=bench,
        coach_name=coach.get("name") if isinstance(coach.get("name"), str) else None,
    )


def normalize_lineup_type(value: Optional[str]) -> str:
    key = (value or "").strip().lower()
    if key in _CONFIRMED:
        return "confirmed"
    if key in _PREDICTED:
        return "predicted"
    return "unknown"


def parse_lineup(payload: Any) -> Optional[ParsedLineup]:
    """None se la risposta non contiene un blocco formazione. Non inventa titolari."""
    if payload is None:
        return None
    root = _unwrap(payload)
    if not isinstance(root, dict):
        return None
    block = root.get("lineup") if isinstance(root.get("lineup"), dict) else root
    if not isinstance(block, dict):
        return None
    home = block.get("homeTeam") or block.get("home")
    away = block.get("awayTeam") or block.get("away")
    if not isinstance(home, dict) and not isinstance(away, dict):
        return None
    raw_type = block.get("lineupType") if isinstance(block.get("lineupType"), str) else None
    source = block.get("source") if isinstance(block.get("source"), str) else None
    return ParsedLineup(
        lineup_type=normalize_lineup_type(raw_type),
        raw_lineup_type=raw_type,
        source=source,
        home=_team(home),
        away=_team(away),
    )

