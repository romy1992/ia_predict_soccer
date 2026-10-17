"""Formazioni con gli ID di entrambi i provider e apprendimento dei collegamenti dalle formazioni ufficiali.

`lineup_for_fixture`: la formazione ufficiale API-Sports se e' completa (22 titolari), altrimenti l'ultima
prevista FotMob (non rifiutata). La prevista non si scrive mai in `player_lineup`: l'ufficiale la sostituisce
semplicemente perche' esiste."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.player.player_lineup_models import PlayerLineup
from src.service_ia.fotmob.models import FotmobLineupPlayer, FotmobLineupSnapshot
from src.service_ia.players import identity as ids
from src.service_ia.players.models import PlayerProviderLink


def _official_rows(session: Session, fixture_id: int) -> list[PlayerLineup]:
    return list(session.execute(select(PlayerLineup).where(PlayerLineup.fixture_id == fixture_id)).scalars())


def _latest_snapshot(session: Session, fixture_id: int) -> Optional[FotmobLineupSnapshot]:
    return session.execute(
        select(FotmobLineupSnapshot)
        .where(FotmobLineupSnapshot.fixture_id == fixture_id, FotmobLineupSnapshot.reliability != "rejected")
        .order_by(FotmobLineupSnapshot.fetched_at.desc())
    ).scalars().first()


def _ids_by_api(session: Session, api_ids: list[int]) -> dict[int, dict[str, Any]]:
    """{api_id: {identity_id, fotmob_id}} con due sole query."""
    if not api_ids:
        return {}
    api_links = session.execute(select(PlayerProviderLink).where(
        PlayerProviderLink.provider == ids.API, PlayerProviderLink.provider_player_id.in_([str(i) for i in api_ids])
    )).scalars().all()
    by_identity = {int(link.identity_id): link for link in api_links}
    fotmob = session.execute(select(PlayerProviderLink).where(
        PlayerProviderLink.provider == ids.FOTMOB, PlayerProviderLink.identity_id.in_(list(by_identity) or [0])
    )).scalars().all()
    fotmob_of = {int(link.identity_id): link.provider_player_id for link in fotmob}
    return {
        int(link.provider_player_id): {"identity_id": int(link.identity_id), "fotmob_id": fotmob_of.get(int(link.identity_id))}
        for link in api_links if str(link.provider_player_id).isdigit()
    }


def _ids_by_fotmob(session: Session, fotmob_ids: list[Any]) -> dict[str, dict[str, Any]]:
    if not fotmob_ids:
        return {}
    links = session.execute(select(PlayerProviderLink).where(
        PlayerProviderLink.provider == ids.FOTMOB, PlayerProviderLink.provider_player_id.in_([str(i) for i in fotmob_ids])
    )).scalars().all()
    identity_ids = [int(link.identity_id) for link in links]
    api = session.execute(select(PlayerProviderLink).where(
        PlayerProviderLink.provider == ids.API, PlayerProviderLink.identity_id.in_(identity_ids or [0])
    )).scalars().all()
    api_of = {int(link.identity_id): link.provider_player_id for link in api}
    return {link.provider_player_id: {"identity_id": int(link.identity_id), "api_id": api_of.get(int(link.identity_id))}
            for link in links}


def _official_payload(session: Session, fixture_id: int, rows: list[PlayerLineup]) -> dict[str, Any]:
    maps = _ids_by_api(session, [int(r.player_id) for r in rows])
    teams: dict[Any, dict[str, Any]] = {}
    for row in rows:
        team = teams.setdefault(row.team_id, {"team_api_id": row.team_id, "team": row.team_name,
                                              "formation": row.formation, "starters": [], "bench": []})
        known = maps.get(int(row.player_id), {})
        team["starters" if row.is_starter else "bench"].append({
            "name": row.player_name, "position": row.position, "api_sports_id": int(row.player_id),
            "fotmob_id": known.get("fotmob_id"), "identity_id": known.get("identity_id"),
        })
    return {"fixture_id": fixture_id, "lineup_type": "official", "source": "api_sports", "teams": list(teams.values())}


def _predicted_payload(session: Session, fixture_id: int, snapshot: FotmobLineupSnapshot) -> dict[str, Any]:
    players = list(session.execute(
        select(FotmobLineupPlayer).where(FotmobLineupPlayer.snapshot_id == snapshot.id)
    ).scalars())
    fotmob_maps = _ids_by_fotmob(session, [p.fotmob_player_id for p in players if p.fotmob_player_id is not None])
    teams = {
        "home": {"team_api_id": snapshot.home_team_id, "formation": snapshot.home_formation, "starters": [], "bench": [],
                 "unavailable": []},
        "away": {"team_api_id": snapshot.away_team_id, "formation": snapshot.away_formation, "starters": [], "bench": [],
                 "unavailable": []},
    }
    key = {"starter": "starters", "bench": "bench", "unavailable": "unavailable"}
    for p in players:
        side = teams.get(p.side)
        if side is None or p.role not in key:
            continue
        known = fotmob_maps.get(str(p.fotmob_player_id), {})
        api_id = p.api_player_id if p.api_player_id is not None else known.get("api_id")
        side[key[p.role]].append({
            "name": p.name, "position": p.historical_position, "api_sports_id": int(api_id) if api_id is not None else None,
            "fotmob_id": p.fotmob_player_id, "identity_id": known.get("identity_id"), "map_method": p.map_method,
        })
    return {"fixture_id": fixture_id, "lineup_type": snapshot.lineup_type or "predicted", "source": "fotmob",
            "reliability": snapshot.reliability, "fetched_at": snapshot.fetched_at.isoformat() if snapshot.fetched_at else None,
            "teams": [teams["home"], teams["away"]]}


def lineup_for_fixture(session: Session, fixture_id: int, prefer: str = "official") -> dict[str, Any]:
    official = _official_rows(session, fixture_id)
    complete = sum(1 for r in official if r.is_starter) == 22
    snapshot = _latest_snapshot(session, fixture_id)
    if prefer == "predicted":
        order = ["predicted", "official"]
    else:
        order = ["official", "predicted"]
    for kind in order:
        if kind == "official" and complete:
            return _official_payload(session, fixture_id, official)
        if kind == "predicted" and snapshot is not None:
            return _predicted_payload(session, fixture_id, snapshot)
    return {"fixture_id": fixture_id, "lineup_type": None, "source": None, "teams": []}


def learn_from_official(session: Session, fixture_id: int) -> dict[str, int]:
    """Con la formazione ufficiale in mano: conferma i collegamenti FotMob gia' trovati e abbina per eliminazione
    il solo titolare rimasto fuori da ciascuna squadra (un buco da una parte, un buco dall'altra)."""
    official = [r for r in _official_rows(session, fixture_id) if r.is_starter]
    snapshot = _latest_snapshot(session, fixture_id)
    report = {"confirmed": 0, "eliminated": 0, "unresolved": 0}
    if len(official) != 22 or snapshot is None:
        return report
    starters = list(session.execute(select(FotmobLineupPlayer).where(
        FotmobLineupPlayer.snapshot_id == snapshot.id, FotmobLineupPlayer.role == "starter")).scalars())
    for side, team_id in (("home", snapshot.home_team_id), ("away", snapshot.away_team_id)):
        official_team = {int(r.player_id): r for r in official if r.team_id == team_id}
        mine = [p for p in starters if p.side == side]
        matched = set()
        missing = []
        for p in mine:
            if p.api_player_id is not None and int(p.api_player_id) in official_team:
                matched.add(int(p.api_player_id))
                if p.fotmob_player_id is not None:
                    ids.link_fotmob_to_api(session, fotmob_id=p.fotmob_player_id, fotmob_name=p.name,
                                           api_player_id=int(p.api_player_id), method=p.map_method or "exact_name",
                                           source="official_confirm")
                    report["confirmed"] += 1
            elif p.api_player_id is None:
                missing.append(p)
        free = [r for pid, r in official_team.items() if pid not in matched]
        if len(missing) == 1 and len(free) == 1 and missing[0].fotmob_player_id is not None:
            ids.link_fotmob_to_api(session, fotmob_id=missing[0].fotmob_player_id, fotmob_name=missing[0].name,
                                   api_player_id=int(free[0].player_id), method="official_elimination",
                                   source="official_confirm")
            report["eliminated"] += 1
        elif missing:
            report["unresolved"] += len(missing)
    session.commit()
    return report


def learn_recent(session: Session, *, days: int = 4, now: Optional[datetime] = None) -> dict[str, int]:
    now = now or datetime.now(timezone.utc)
    fixtures = [int(f) for f in session.execute(
        select(FotmobLineupSnapshot.fixture_id).where(
            FotmobLineupSnapshot.fixture_id.isnot(None), FotmobLineupSnapshot.fetched_at >= now - timedelta(days=days)
        ).distinct()
    ).scalars()]
    total = {"fixtures": len(fixtures), "confirmed": 0, "eliminated": 0, "unresolved": 0}
    for fixture_id in fixtures:
        part = learn_from_official(session, fixture_id)
        for key in ("confirmed", "eliminated", "unresolved"):
            total[key] += part[key]
    return total
