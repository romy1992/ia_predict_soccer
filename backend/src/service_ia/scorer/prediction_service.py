"""Predizioni marcatori pre-partita.

Flusso (job `scorer_predictions`):
1. Per le partite delle prime 5 leghe in arrivo (NS) il cui calcio d'inizio e' entro N minuti e che non
   hanno ancora la formazione, chiede a API-Sports la formazione UFFICIALE e la salva (poche chiamate).
2. Per ogni partita con formazione completa (22 titolari) calcola le feature point-in-time dei titolari
   (solo il passato, mai la partita stessa), applica il modello e salva una revisione append-only.
Tutto avviene PRIMA del calcio d'inizio: le predizioni pubblicate non vengono mai riscritte."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.data.player.coverage_report import TOP5_LEAGUES
from src.data.player.player_lineup_models import PlayerLineup
from src.data.player.player_lineup_service import extract_lineup_players
from src.data.player.scorer_prediction_models import PlayerScorerPrediction
from src.data.player.scoring_dataset import build_dataset, load_raw
from src.service_ia.model.match import Match
from src.service_ia.scorer import model_store
from src.service_ia.scorer.model_store import EURO_CUPS

logger = logging.getLogger(__name__)

# Competizioni coperte dal mercato marcatori: solo quelle con eventi+formazioni storici verificati
# (vedi `copertura.md`). Una partita di un'altra competizione non avra' mai predizioni.
# Cinque campionati + Champions/Europa/Conference League: validati sull'holdout temporale (calibrazione e
# guadagno sulla baseline anche sulle sole partite di coppa, vedi scripts/analysis/eval_scorer_v3.py).
SCORER_LEAGUES = list(TOP5_LEAGUES) + EURO_CUPS
# Campionati usati SOLO per lo storico dei giocatori e per l'addestramento (squadre delle coppe che
# giocano fuori dai 5 campionati: Portogallo, Olanda, Turchia, Belgio). Non hanno predizioni proprie.
SCORER_HISTORY_LEAGUES = SCORER_LEAGUES + [94, 88, 203, 144]

LINEUP_FETCH_MINUTES = 75     # la formazione ufficiale esce ~60' prima: da qui in poi la cerchiamo
PREDICT_HORIZON_HOURS = 6     # non predire partite lontane: la formazione non c'e' ancora
HISTORY_SEASONS_BACK = 3


def _parse(dt: Any) -> Optional[datetime]:
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


def upcoming_matches(session: Session, leagues: Iterable[int], now: datetime, hours: float) -> list[Any]:
    """Partite NS nella finestra (righe leggere, senza caricare le relazioni ORM di `Match`)."""
    rows = session.execute(
        select(Match.id_fixture, Match.date_match, Match.league_match, Match.season,
               Match.id_team_home, Match.id_team_away)
        .where(Match.status == "NS", Match.league_match.in_(list(leagues)), Match.id_fixture.isnot(None))
    ).all()
    end = now + timedelta(hours=hours)
    out = []
    for m in rows:
        ko = _parse(m.date_match)
        if ko is not None and now <= ko <= end:
            out.append(m)
    return out


def fetch_official_lineups(session: Session, provider, repo_player, *, leagues: Iterable[int] = SCORER_LEAGUES,
                           now: Optional[datetime] = None, minutes_before: int = LINEUP_FETCH_MINUTES) -> dict:
    """Scarica le formazioni ufficiali mancanti per le partite in partenza a breve. Idempotente."""
    now = now or datetime.now(timezone.utc)
    cand = upcoming_matches(session, leagues, now, minutes_before / 60.0)
    fetched, missing = 0, 0
    for m in cand:
        fid = int(m.id_fixture)
        if repo_player.has_lineup_for_fixture(fid):
            continue
        try:
            blocks = provider.get_fixture_lineups(fid)
        except Exception:  # noqa: BLE001 - una partita non deve bloccare le altre
            logger.exception("Scorer: errore scaricando la formazione di %s", fid)
            continue
        rows = [r for b in blocks for r in extract_lineup_players(fid, b)]
        starters = sum(1 for r in rows if r.is_starter)
        if starters != 22:  # non ancora pubblicata o incompleta: riprova al prossimo giro
            missing += 1
            continue
        repo_player.save_lineups(rows)
        fetched += 1
    return {"candidates": len(cand), "lineups_fetched": fetched, "lineups_not_ready": missing}


def _latest_revisions(session: Session, fixture_ids: list[int]) -> dict[int, tuple[int, set[int]]]:
    if not fixture_ids:
        return {}
    mx = session.execute(
        select(PlayerScorerPrediction.fixture_id, func.max(PlayerScorerPrediction.revision))
        .where(PlayerScorerPrediction.fixture_id.in_(fixture_ids)).group_by(PlayerScorerPrediction.fixture_id)
    ).all()
    out: dict[int, tuple[int, set[int]]] = {}
    for fid, rev in mx:
        pids = session.execute(
            select(PlayerScorerPrediction.player_id).where(
                PlayerScorerPrediction.fixture_id == fid, PlayerScorerPrediction.revision == rev)
        ).scalars().all()
        out[int(fid)] = (int(rev), set(int(p) for p in pids))
    return out


def predict_upcoming(session: Session, bundle: dict, *, leagues: Iterable[int] = SCORER_LEAGUES,
                     history_leagues: Optional[Iterable[int]] = None,
                     now: Optional[datetime] = None, hours: float = PREDICT_HORIZON_HOURS,
                     finished_statuses: Iterable[str] = ("FT", "AET", "PEN", "ABD")) -> dict:
    """Calcola e salva le predizioni per le partite in arrivo con formazione completa."""
    now = now or datetime.now(timezone.utc)
    leagues = list(leagues)
    cand = upcoming_matches(session, leagues, now, hours)
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

    year = now.year
    seasons = list(range(year - HISTORY_SEASONS_BACK, year + 1))
    hist_matches, hist_lineups, hist_events = load_raw(
        session, list(history_leagues) if history_leagues is not None else SCORER_HISTORY_LEAGUES,
        seasons, finished_statuses)
    up = pd.DataFrame([(int(m.id_fixture), m.date_match, m.league_match, m.season, m.id_team_home, m.id_team_away)
                       for m in cand if int(m.id_fixture) in ready_ids], columns=hist_matches.columns)
    up_lineups = lineups[lineups["fixture_id"].isin(ready_ids)][hist_lineups.columns]
    matches = pd.concat([hist_matches, up], ignore_index=True)
    all_lineups = pd.concat([hist_lineups, up_lineups], ignore_index=True)
    frame = build_dataset(matches, all_lineups, hist_events, upcoming_fixture_ids=ready_ids)
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
            session.add(PlayerScorerPrediction(
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
                         finished_statuses: Iterable[str] = ("FT", "AET", "PEN", "ABD")) -> pd.DataFrame:
    from src.data.player.scoring_dataset import build_from_session

    year = datetime.now(timezone.utc).year
    return build_from_session(session, list(leagues), list(range(year - HISTORY_SEASONS_BACK, year + 1)),
                              finished_statuses)


def settle_predictions(session: Session, finished_statuses: Iterable[str] = ("FT", "AET", "PEN", "ABD")) -> dict:
    """Compila `scored` sulle predizioni di partite concluse (regola del bersaglio: gol non autogol <= 90')."""
    from src.data.player.player_models import PlayerMatchEvent

    pending = session.execute(
        select(PlayerScorerPrediction.fixture_id).where(PlayerScorerPrediction.scored.is_(None)).distinct()
    ).scalars().all()
    if not pending:
        return {"settled_fixtures": 0}
    done = set(session.execute(
        select(Match.id_fixture).where(Match.id_fixture.in_(pending), Match.status.in_(list(finished_statuses)))
    ).scalars().all())
    settled = 0
    for fid in done:
        events = session.execute(
            select(PlayerMatchEvent.event_type, PlayerMatchEvent.event_detail, PlayerMatchEvent.comments,
                   PlayerMatchEvent.elapsed_minute, PlayerMatchEvent.player_id)
            .where(PlayerMatchEvent.fixture_id == fid)
        ).all()
        if not events:  # eventi non ancora scaricati: riprova al prossimo giro
            continue
        scorers = {
            int(pid) for typ, det, com, el, pid in events
            if pid is not None and str(typ or "").lower() == "goal" and det not in ("Own Goal", "Missed Penalty")
            and com != "Penalty Shootout" and el is not None and float(el) <= 90
        }
        for row in session.execute(select(PlayerScorerPrediction).where(
                PlayerScorerPrediction.fixture_id == fid, PlayerScorerPrediction.scored.is_(None))).scalars():
            row.scored = int(row.player_id) in scorers
        settled += 1
    session.commit()
    return {"settled_fixtures": settled}


def fixture_league(session: Session, fixture_id: int) -> Optional[int]:
    return session.execute(select(Match.league_match).where(Match.id_fixture == fixture_id)).scalar()


def is_supported_fixture(session: Session, fixture_id: int) -> bool:
    return fixture_league(session, fixture_id) in SCORER_LEAGUES


def latest_predictions(session: Session, *, fixture_id: Optional[int] = None, day: Optional[str] = None,
                       limit: int = 2000) -> list[dict]:
    """Ultima revisione di ogni partita (filtro per fixture o per giorno ISO `YYYY-MM-DD` del kickoff)."""
    q = select(PlayerScorerPrediction)
    if fixture_id is not None:
        q = q.where(PlayerScorerPrediction.fixture_id == fixture_id)
    if day:
        q = q.where(PlayerScorerPrediction.kickoff.like(f"{day}%"))
    rows = session.execute(q.limit(limit * 5)).scalars().all()
    latest: dict[int, int] = {}
    for r in rows:
        latest[r.fixture_id] = max(latest.get(r.fixture_id, 0), r.revision)
    out = [r.to_dict() for r in rows if r.revision == latest[r.fixture_id]]
    for d in out:
        d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    out.sort(key=lambda d: (d["kickoff"] or "", d["fixture_id"], -d["probability"]))
    return out[:limit]
