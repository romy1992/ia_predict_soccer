"""Predizioni gialli giocatori pre-partita (job `player_cards_predictions`).

Stesso flusso dei marcatori (`service_ia/scorer/prediction_service.py`, di cui riusa competizioni, finestra
temporale e scaricamento delle formazioni ufficiali - idempotente: se le ha gia' scaricate il job marcatori
non si spende nessuna chiamata in piu'): per ogni partita con formazione completa (22 titolari) calcola le
feature point-in-time dei titolari (solo il passato), applica il modello e salva una revisione append-only.
Le predizioni pubblicate non vengono mai riscritte."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.data.player.card_prediction_models import PlayerCardPrediction
from src.data.player.cards_dataset import build_dataset, build_from_session, load_referees
from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.scoring_dataset import load_raw
from src.service_ia.player_cards import model_store
from src.service_ia.scorer.prediction_service import (
    HISTORY_SEASONS_BACK, PREDICT_HORIZON_HOURS, SCORER_HISTORY_LEAGUES, SCORER_LEAGUES, fetch_official_lineups,
    fixture_league, is_supported_fixture, upcoming_matches,
)
from src.service_ia.model.match import Match

logger = logging.getLogger(__name__)

__all__ = ["fetch_official_lineups", "fixture_league", "is_supported_fixture", "upcoming_matches",
           "SCORER_LEAGUES", "PREDICT_HORIZON_HOURS"]

FINISHED = ("FT", "AET", "PEN", "ABD")


def _latest_revisions(session: Session, fixture_ids: list[int]) -> dict[int, tuple[int, set[int]]]:
    if not fixture_ids:
        return {}
    mx = session.execute(
        select(PlayerCardPrediction.fixture_id, func.max(PlayerCardPrediction.revision))
        .where(PlayerCardPrediction.fixture_id.in_(fixture_ids)).group_by(PlayerCardPrediction.fixture_id)
    ).all()
    out: dict[int, tuple[int, set[int]]] = {}
    for fid, rev in mx:
        pids = session.execute(
            select(PlayerCardPrediction.player_id).where(
                PlayerCardPrediction.fixture_id == fid, PlayerCardPrediction.revision == rev)
        ).scalars().all()
        out[int(fid)] = (int(rev), set(int(p) for p in pids))
    return out


def predict_upcoming(session: Session, bundle: dict, *, leagues: Iterable[int] = SCORER_LEAGUES,
                     history_leagues: Optional[Iterable[int]] = None,
                     now: Optional[datetime] = None, hours: float = PREDICT_HORIZON_HOURS,
                     finished_statuses: Iterable[str] = FINISHED) -> dict:
    """Calcola e salva le predizioni per le partite in arrivo con formazione completa."""
    now = now or datetime.now(timezone.utc)
    cand = upcoming_matches(session, list(leagues), now, hours)
    ids = [int(m.id_fixture) for m in cand]
    if not ids:
        return {"fixtures": 0, "predicted": 0, "skipped_same_lineup": 0, "no_lineup": 0}
    lineups = pd.DataFrame(
        [(r.fixture_id, r.player_id, r.team_id, r.position, r.is_starter, r.player_name, r.team_name)
         for r in session.execute(select(PlayerLineup).where(PlayerLineup.fixture_id.in_(ids))).scalars()],
        columns=["fixture_id", "player_id", "team_id", "position", "is_starter", "player_name", "team_name"])
    if lineups.empty:
        return {"fixtures": len(ids), "predicted": 0, "skipped_same_lineup": 0, "no_lineup": len(ids)}
    ready = lineups[lineups["is_starter"]].groupby("fixture_id").size()
    ready_ids = [int(f) for f, n in ready.items() if n == 22]
    if not ready_ids:
        return {"fixtures": len(ids), "predicted": 0, "skipped_same_lineup": 0, "no_lineup": len(ids)}

    seasons = list(range(now.year - HISTORY_SEASONS_BACK, now.year + 1))
    hist_matches, hist_lineups, hist_events = load_raw(
        session, list(history_leagues) if history_leagues is not None else SCORER_HISTORY_LEAGUES,
        seasons, finished_statuses)
    up = pd.DataFrame([(int(m.id_fixture), m.date_match, m.league_match, m.season, m.id_team_home, m.id_team_away)
                       for m in cand if int(m.id_fixture) in ready_ids], columns=hist_matches.columns)
    up_lineups = lineups[lineups["fixture_id"].isin(ready_ids)][hist_lineups.columns]
    matches = pd.concat([hist_matches, up], ignore_index=True)
    all_lineups = pd.concat([hist_lineups, up_lineups], ignore_index=True)
    referees = load_referees(session, matches["fixture_id"].tolist())
    frame = build_dataset(matches, all_lineups, hist_events, referees, upcoming_fixture_ids=ready_ids)
    frame = frame[frame["fixture_id"].isin(ready_ids)].copy()
    if frame.empty:
        return {"fixtures": len(ids), "predicted": 0, "skipped_same_lineup": 0, "no_lineup": len(ids)}
    frame["probability"] = model_store.predict_proba(bundle, frame)

    names = lineups.set_index(["fixture_id", "player_id"])[["player_name", "team_name"]]
    kick = {int(m.id_fixture): m.date_match for m in cand}
    latest = _latest_revisions(session, ready_ids)
    predicted, skipped = 0, 0
    for fid, grp in frame.groupby("fixture_id"):
        fid = int(fid)
        pids = set(int(p) for p in grp["player_id"])
        prev = latest.get(fid)
        if prev and prev[1] == pids:
            skipped += 1
            continue
        revision = (prev[0] + 1) if prev else 1
        for r in grp.itertuples(index=False):
            nm = names.loc[(fid, int(r.player_id))]
            session.add(PlayerCardPrediction(
                fixture_id=fid, player_id=int(r.player_id), revision=revision,
                player_name=nm["player_name"], team_id=int(r.team_id), team_name=nm["team_name"],
                position=("F" if r.pos_F else "M" if r.pos_M else "D"), league=int(r.league),
                kickoff=kick.get(fid), probability=float(r.probability), lineup_type="official",
                model_version=bundle.get("run_id") or bundle["version"]))
        predicted += 1
    session.commit()
    return {"fixtures": len(ids), "predicted": predicted, "skipped_same_lineup": skipped,
            "no_lineup": len(ids) - len(ready_ids), "model_version": bundle.get("run_id") or bundle["version"]}


def dataset_for_training(session: Session, *, leagues: Iterable[int] = SCORER_HISTORY_LEAGUES,
                         finished_statuses: Iterable[str] = FINISHED) -> pd.DataFrame:
    year = datetime.now(timezone.utc).year
    return build_from_session(session, list(leagues), list(range(year - HISTORY_SEASONS_BACK, year + 1)),
                              finished_statuses)


def settle_predictions(session: Session, finished_statuses: Iterable[str] = FINISHED) -> dict:
    """Compila `yellow` sulle predizioni di partite concluse (ammonizione <= 90', secondo giallo incluso)."""
    from src.data.player.player_models import PlayerMatchEvent

    pending = session.execute(
        select(PlayerCardPrediction.fixture_id).where(PlayerCardPrediction.yellow.is_(None)).distinct()
    ).scalars().all()
    if not pending:
        return {"settled_fixtures": 0}
    done = set(session.execute(
        select(Match.id_fixture).where(Match.id_fixture.in_(pending), Match.status.in_(list(finished_statuses)))
    ).scalars().all())
    settled = 0
    for fid in done:
        events = session.execute(
            select(PlayerMatchEvent.event_type, PlayerMatchEvent.event_detail,
                   PlayerMatchEvent.elapsed_minute, PlayerMatchEvent.player_id)
            .where(PlayerMatchEvent.fixture_id == fid)
        ).all()
        if not events:  # eventi non ancora scaricati: riprova al prossimo giro
            continue
        booked = {
            int(pid) for typ, det, el, pid in events
            if pid is not None and str(typ or "").lower() == "card"
            and str(det or "").lower() in ("yellow card", "second yellow card")
            and el is not None and float(el) <= 90
        }
        for row in session.execute(select(PlayerCardPrediction).where(
                PlayerCardPrediction.fixture_id == fid, PlayerCardPrediction.yellow.is_(None))).scalars():
            row.yellow = int(row.player_id) in booked
        settled += 1
    session.commit()
    return {"settled_fixtures": settled}


def latest_predictions(session: Session, *, fixture_id: Optional[int] = None, day: Optional[str] = None,
                       limit: int = 2000) -> list[dict]:
    """Ultima revisione di ogni partita (filtro per fixture o per giorno ISO `YYYY-MM-DD` del kickoff)."""
    q = select(PlayerCardPrediction)
    if fixture_id is not None:
        q = q.where(PlayerCardPrediction.fixture_id == fixture_id)
    if day:
        q = q.where(PlayerCardPrediction.kickoff.like(f"{day}%"))
    rows = session.execute(q.limit(limit * 5)).scalars().all()
    latest: dict[int, int] = {}
    for r in rows:
        latest[r.fixture_id] = max(latest.get(r.fixture_id, 0), r.revision)
    out = [r.to_dict() for r in rows if r.revision == latest[r.fixture_id]]
    for d in out:
        d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    out.sort(key=lambda d: (d["kickoff"] or "", d["fixture_id"], -d["probability"]))
    return out[:limit]
