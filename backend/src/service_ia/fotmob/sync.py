"""Job giornaliero: partite future gia' a DB -> match FotMob -> snapshot -> previsione iniziale.

Se FotMob non ha la formazione, o i controlli non passano, la partita resta
senza previsione iniziale. Il modello e le formazioni ufficiali non cambiano.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.player.player_lineup_models import PlayerLineup
from src.service_ia.fotmob.client import FotmobAuthError, FotmobClient, FotmobError
from src.service_ia.fotmob.mapping import (
    PlayerMap, TeamRoster, add_appearance, empty_roster, map_player, match_fixture, position_for,
)
from src.service_ia.fotmob.models import FotmobLineupPlayer, FotmobLineupSnapshot, FotmobMatchMap
from src.service_ia.fotmob.parse import ParsedLineup, ParsedPlayer, iter_matches, parse_lineup
from src.service_ia.fotmob.predict import apply_initial_predictions
from src.service_ia.fotmob.validate import assess
from src.service_ia.fotmob.names import names_match
from src.service_ia.model.match import Match
from src.service_ia.scorer.prediction_service import SCORER_LEAGUES

logger = logging.getLogger(__name__)


def _utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _roster(session: Session, team_ids: list[int]) -> TeamRoster:
    roster = empty_roster()
    if not team_ids:
        return roster
    rows = session.execute(
        select(PlayerLineup.player_id, PlayerLineup.player_name, PlayerLineup.team_id,
               PlayerLineup.position, Match.date_match)
        .join(Match, Match.id_fixture == PlayerLineup.fixture_id)
        .where(PlayerLineup.team_id.in_(team_ids))
    ).all()
    for player_id, name, team_id, position, date_match in rows:
        add_appearance(
            roster, team_id=int(team_id), player_id=int(player_id), name=name,
            position=position, last_seen=_utc(date_match),
        )
    player_ids = list(roster.latest_team)
    if not player_ids:
        return roster
    global_rows = session.execute(
        select(PlayerLineup.player_id, PlayerLineup.team_id, Match.date_match)
        .join(Match, Match.id_fixture == PlayerLineup.fixture_id)
        .where(PlayerLineup.player_id.in_(player_ids))
    ).all()
    roster.latest_team = {}
    for player_id, team_id, date_match in global_rows:
        seen = _utc(date_match)
        previous = roster.latest_team.get(int(player_id))
        if previous is None or (seen or datetime.min.replace(tzinfo=timezone.utc)) >= (previous[1] or datetime.min.replace(tzinfo=timezone.utc)):
            roster.latest_team[int(player_id)] = (int(team_id), seen)
    return roster


def _hash(lineup: ParsedLineup) -> str:
    body = {
        "type": lineup.lineup_type,
        "home_xi": [player.fotmob_id for player in lineup.home.starters],
        "away_xi": [player.fotmob_id for player in lineup.away.starters],
        "home_formation": lineup.home.formation,
        "away_formation": lineup.away.formation,
        "home_out": [(player.fotmob_id, player.unavailability_type) for player in lineup.home.unavailable],
        "away_out": [(player.fotmob_id, player.unavailability_type) for player in lineup.away.unavailable],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _fresh(session: Session, fixture_id: int, now: datetime, refresh_hours: int) -> bool:
    latest = session.execute(
        select(FotmobLineupSnapshot)
        .where(FotmobLineupSnapshot.fixture_id == fixture_id)
        .order_by(FotmobLineupSnapshot.fetched_at.desc())
    ).scalars().first()
    fetched = _utc(latest.fetched_at) if latest is not None else None
    return fetched is not None and fetched >= now - timedelta(hours=refresh_hours)


def _map(session: Session, player: ParsedPlayer, team_id: int, roster: TeamRoster) -> PlayerMap:
    """Prima il collegamento gia' salvato nell'anagrafica (nessun confronto per nome), poi il nome; un abbinamento
    nuovo e univoco viene salvato per le volte dopo."""
    from src.service_ia.players import identity

    try:
        saved = identity.api_id_for_fotmob(session, player.fotmob_id)
    except Exception:  # noqa: BLE001 - l'anagrafica non deve mai bloccare il job
        logger.exception("FotMob: lettura anagrafica fallita per %s", player.fotmob_id)
        saved = None
    if saved is not None:
        return PlayerMap(saved, "saved_link", position_for(roster, saved), None)
    mapped = map_player(player.name, team_id, roster)
    if mapped.api_player_id is not None and player.fotmob_id is not None and mapped.warning != "roster_conflict":
        try:
            identity.link_fotmob_to_api(session, fotmob_id=player.fotmob_id, fotmob_name=player.name,
                                        api_player_id=int(mapped.api_player_id), method=mapped.method or "exact_name")
        except Exception:  # noqa: BLE001
            logger.exception("FotMob: collegamento non salvato per %s", player.fotmob_id)
    return mapped


def _player_row(snapshot_id: int, side: str, role: str, player: ParsedPlayer, api_team_id: int, mapped) -> FotmobLineupPlayer:
    return FotmobLineupPlayer(
        snapshot_id=snapshot_id, side=side, role=role, fotmob_player_id=player.fotmob_id,
        name=player.name, shirt_number=player.shirt, position_id=player.position_id,
        market_value=player.market_value, api_player_id=None if mapped is None else mapped.api_player_id,
        api_team_id=api_team_id, map_method=None if mapped is None else mapped.method,
        historical_position=None if mapped is None else mapped.position,
        unavailability_type=player.unavailability_type, expected_return=player.expected_return,
        expected_return_date=player.expected_return_date, return_certainty=player.return_certainty,
    )


def _store_players(session: Session, snapshot_id: int, lineup: ParsedLineup, home_maps, away_maps,
                   home_team_id: int, away_team_id: int, roster: TeamRoster) -> None:
    for side, team, maps, team_id in (
        ("home", lineup.home, home_maps, home_team_id),
        ("away", lineup.away, away_maps, away_team_id),
    ):
        for player, mapped in zip(team.starters, maps):
            session.add(_player_row(snapshot_id, side, "starter", player, team_id, mapped))
        for player in team.unavailable:
            session.add(_player_row(snapshot_id, side, "unavailable", player, team_id, _map(session, player, team_id, roster)))
        for player in team.bench:
            session.add(_player_row(snapshot_id, side, "bench", player, team_id, None))
        if team.coach_name:
            session.add(FotmobLineupPlayer(
                snapshot_id=snapshot_id, side=side, role="coach", name=team.coach_name, api_team_id=team_id,
            ))


def _bundles() -> tuple[Optional[dict], Optional[dict]]:
    from src.service_ia.player_cards import model_store as cards_store
    from src.service_ia.scorer import model_store as scorer_store

    return scorer_store.load_production_bundle(), cards_store.load_production_bundle()


def _starters(team, maps, team_id: int, team_name: Optional[str]) -> list[dict]:
    rows = []
    for player, mapped in zip(team.starters, maps):
        if mapped.api_player_id is None or mapped.warning:
            continue
        rows.append({
            "player_id": int(mapped.api_player_id),
            "team_id": int(team_id),
            "player_name": player.name,
            "team_name": team_name,
            "position": mapped.position,
        })
    return rows


def _refresh_identities(session: Session, now: datetime) -> dict[str, Any]:
    """Anagrafica: nuovi giocatori API-Sports e apprendimento dalle formazioni ufficiali arrivate dopo l'ultimo giro."""
    from src.service_ia.players import identity, lineups

    out: dict[str, Any] = {}
    try:
        out.update(identity.sync_api_sports_identities(session))
        out["learned"] = lineups.learn_recent(session, now=now)
    except Exception as exc:  # noqa: BLE001 - mai bloccare il job
        logger.exception("FotMob: aggiornamento anagrafica fallito")
        session.rollback()
        out["error"] = type(exc).__name__
    return out


def sync_predicted_lineups(
    session: Session,
    client: Optional[FotmobClient] = None,
    *,
    now: Optional[datetime] = None,
    horizon_hours: int = 72,
    max_requests: int = 120,
    refresh_hours: int = 20,
    leagues: Optional[list[int]] = None,
    scorer_bundle: Optional[dict] = None,
    cards_bundle: Optional[dict] = None,
    load_models: bool = True,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    client = client or FotmobClient()
    leagues = list(leagues or SCORER_LEAGUES)
    if load_models and scorer_bundle is None and cards_bundle is None:
        scorer_bundle, cards_bundle = _bundles()

    rows = session.execute(
        select(Match.id_fixture, Match.date_match, Match.league_match, Match.season,
               Match.id_team_home, Match.id_team_away, Match.name_home, Match.name_away)
        .where(Match.status == "NS", Match.league_match.in_(leagues), Match.id_fixture.isnot(None))
    ).all()
    window_end = now + timedelta(hours=horizon_hours)
    upcoming = []
    for row in rows:
        kickoff = _utc(row.date_match)
        if kickoff is not None and now <= kickoff <= window_end:
            upcoming.append((row, kickoff))

    report_identity = _refresh_identities(session, now)
    report: dict[str, Any] = {
        "identity": report_identity,
        "fixtures": len(upcoming), "mapped": 0, "unmatched": 0, "snapshots": 0,
        "skipped_fresh": 0, "predicted_scorers": 0, "predicted_cards": 0,
        "validated": 0, "partially_validated": 0, "unverified": 0, "rejected": 0,
        "no_lineup": 0, "truncated": False, "errors": [],
    }
    by_date: dict[str, list[Any]] = {}

    for row, kickoff in upcoming:
        fixture_id = int(row.id_fixture)
        if _fresh(session, fixture_id, now, refresh_hours):
            report["skipped_fresh"] += 1
            continue
        if client.requests_made >= max_requests:
            report["truncated"] = True
            break
        try:
            mapped = session.get(FotmobMatchMap, fixture_id)
            method = mapped.method if mapped is not None else None
            fotmob_id = mapped.fotmob_match_id if mapped is not None else None
            if fotmob_id is None:
                day = kickoff.strftime("%Y%m%d")
                if day not in by_date:
                    if client.requests_made >= max_requests:
                        report["truncated"] = True
                        break
                    by_date[day] = iter_matches(client.matches_by_date(day))
                found, method = match_fixture(
                    by_date[day], home=row.name_home, away=row.name_away,
                    kickoff=kickoff, league_id=int(row.league_match) if row.league_match is not None else None,
                )
                if found is None:
                    report["unmatched"] += 1
                    continue
                fotmob_id = found.match_id
                session.merge(FotmobMatchMap(
                    fixture_id=fixture_id, fotmob_match_id=fotmob_id, method=method, mapped_at=now,
                ))
                session.commit()
            report["mapped"] += 1
            if client.requests_made >= max_requests:
                report["truncated"] = True
                break
            raw_lineup = client.lineups(fotmob_id)
            parsed = parse_lineup(raw_lineup)
            if parsed is None:
                report["no_lineup"] += 1
                session.add(FotmobLineupSnapshot(
                    fixture_id=fixture_id, fotmob_match_id=fotmob_id, lineup_type="unknown",
                    fetched_at=now, reliability="rejected", warnings=["formazione_assente"],
                    content_hash="missing", starter_count=0, mapped_starters=0, payload={},
                    home_team_id=int(row.id_team_home), away_team_id=int(row.id_team_away),
                ))
                session.commit()
                report["snapshots"] += 1
                report["rejected"] += 1
                continue

            identity_ok = True
            if parsed.home.name and not names_match(parsed.home.name, row.name_home):
                identity_ok = False
            if parsed.away.name and not names_match(parsed.away.name, row.name_away):
                identity_ok = False
            roster = _roster(session, [int(row.id_team_home), int(row.id_team_away)])
            home_maps = [_map(session, player, int(row.id_team_home), roster) for player in parsed.home.starters]
            away_maps = [_map(session, player, int(row.id_team_away), roster) for player in parsed.away.starters]
            assessment = assess(
                parsed, home_maps, away_maps,
                match_method=method or "teams_same_day",
                roster_checked=roster.has_rows,
            )
            if not identity_ok:
                assessment.reliability = "rejected"
                assessment.warnings = ["identita_partita", *assessment.warnings]
                session.query(FotmobMatchMap).filter_by(fixture_id=fixture_id).delete()
            snapshot = FotmobLineupSnapshot(
                fixture_id=fixture_id, fotmob_match_id=fotmob_id, lineup_type=parsed.lineup_type,
                source=parsed.source, fetched_at=now, home_formation=parsed.home.formation,
                away_formation=parsed.away.formation, home_team_fotmob_id=parsed.home.fotmob_id,
                away_team_fotmob_id=parsed.away.fotmob_id, home_team_id=int(row.id_team_home),
                away_team_id=int(row.id_team_away), reliability=assessment.reliability,
                warnings=assessment.warnings, content_hash=_hash(parsed),
                starter_count=assessment.starter_count, mapped_starters=assessment.mapped_starters,
                payload=raw_lineup if isinstance(raw_lineup, dict) else {"lineupType": parsed.raw_lineup_type},
            )
            session.add(snapshot)
            session.flush()
            _store_players(session, int(snapshot.id), parsed, home_maps, away_maps,
                           int(row.id_team_home), int(row.id_team_away), roster)
            session.commit()
            report["snapshots"] += 1
            report[assessment.reliability] = report.get(assessment.reliability, 0) + 1

            if assessment.reliability == "validated" and parsed.lineup_type == "predicted":
                starters = _starters(parsed.home, home_maps, int(row.id_team_home), row.name_home)
                starters += _starters(parsed.away, away_maps, int(row.id_team_away), row.name_away)
                predicted = apply_initial_predictions(
                    session, fixture_id=fixture_id, kickoff=row.date_match, league=int(row.league_match),
                    season=int(row.season or now.year), home_id=int(row.id_team_home),
                    away_id=int(row.id_team_away), starters=starters, scorer_bundle=scorer_bundle,
                    cards_bundle=cards_bundle, now=now,
                )
                if predicted.get("scorers") == "predicted":
                    report["predicted_scorers"] += 1
                if predicted.get("cards") == "predicted":
                    report["predicted_cards"] += 1
        except FotmobAuthError:
            raise
        except FotmobError as exc:
            logger.warning("FotMob fixture=%s errore=%s", fixture_id, exc)
            report["errors"].append({"fixture_id": fixture_id, "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - una partita non blocca il giro
            logger.exception("FotMob fixture=%s", fixture_id)
            report["errors"].append({"fixture_id": fixture_id, "message": type(exc).__name__})
    report["requests"] = client.requests_made
    return report
