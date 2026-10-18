"""Previsione iniziale da una formazione FotMob gia' validata.

Usa il modello in production, senza nuovi pesi. Non scrive `player_lineup`:
lo storico ufficiale resta quello di API-Sports. Se esiste gia' una revisione
`official`, non se ne aggiunge una prevista dopo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.data.player.card_prediction_models import PlayerCardPrediction
from src.data.player.cards_dataset import build_dataset as build_cards
from src.data.player.cards_dataset import load_referees
from src.data.player.scoring_dataset import build_dataset as build_scorers
from src.data.player.scoring_dataset import load_raw
from src.data.player.scorer_prediction_models import PlayerScorerPrediction
from src.service_ia.scorer.prediction_service import HISTORY_SEASONS_BACK, SCORER_HISTORY_LEAGUES

logger = logging.getLogger(__name__)

_FINISHED = ("FT", "AET", "PEN", "ABD")


def _latest(session: Session, model, fixture_id: int):
    revision = session.execute(
        select(func.max(model.revision)).where(model.fixture_id == fixture_id)
    ).scalar()
    if not revision:
        return None
    rows = session.execute(
        select(model).where(model.fixture_id == fixture_id, model.revision == revision)
    ).scalars().all()
    if not rows:
        return None
    return int(revision), {int(row.player_id) for row in rows}, (rows[0].lineup_type or "official")


OUTFIELD_ROLES = {"F", "M", "D"}


def _same_saved_lineup(previous, starters: list[dict]) -> bool:
    """La revisione salvata contiene solo i giocatori di movimento (il dataset esclude i portieri), quindi va
    confrontata con i titolari F/M/D e non con tutti e 22. Se i ruoli mancano il confronto esatto avviene dopo
    il calcolo delle feature."""
    if not previous or previous[2] != "predicted":
        return False
    outfield = {int(item["player_id"]) for item in starters if item.get("position") in OUTFIELD_ROLES}
    return bool(outfield) and previous[1] == outfield


def _save(session: Session, model, frame: pd.DataFrame, names: pd.DataFrame, *, fixture_id: int,
          kickoff: str, revision: int, model_version: str) -> None:
    indexed = names.set_index("player_id")
    for row in frame.itertuples(index=False):
        info = indexed.loc[int(row.player_id)]
        session.add(model(
            fixture_id=fixture_id,
            player_id=int(row.player_id),
            revision=revision,
            player_name=info["player_name"],
            team_id=int(row.team_id),
            team_name=info["team_name"],
            position=("F" if row.pos_F else "M" if row.pos_M else "D"),
            league=int(row.league),
            kickoff=kickoff,
            probability=float(row.probability),
            lineup_type="predicted",
            model_version=model_version,
        ))


def _one_market(session: Session, model, bundle: dict, build, *, fixture_id: int, kickoff: str,
                league: int, season: int, home_id: int, away_id: int, starters: list[dict],
                history_leagues: list[int], seasons: list[int], extra_frame) -> str:
    previous = _latest(session, model, fixture_id)
    if previous and previous[2] == "official":
        return "gia_ufficiale"
    if _same_saved_lineup(previous, starters):
        return "stessa_formazione"
    matches, lineups, events = load_raw(session, history_leagues, seasons, _FINISHED)
    upcoming_match = pd.DataFrame(
        [(fixture_id, kickoff, league, season, home_id, away_id)],
        columns=["fixture_id", "date_match", "league", "season", "home_id", "away_id"],
    )
    upcoming_lineup = pd.DataFrame(
        [(fixture_id, int(item["player_id"]), int(item["team_id"]), item.get("position"), True) for item in starters],
        columns=["fixture_id", "player_id", "team_id", "position", "is_starter"],
    )
    all_matches = pd.concat([matches, upcoming_match], ignore_index=True)
    all_lineups = pd.concat([lineups, upcoming_lineup], ignore_index=True)
    frame = build(all_matches, all_lineups, events, *extra_frame(all_matches), upcoming_fixture_ids=[fixture_id])
    frame = frame[frame["fixture_id"] == fixture_id].copy()
    if frame.empty:
        return "senza_feature"
    if previous and previous[2] == "predicted" and previous[1] == {int(p) for p in frame["player_id"]}:
        return "stessa_formazione"
    from src.service_ia.player_cards import model_store as cards_store
    from src.service_ia.scorer import model_store as scorer_store

    predict = scorer_store.predict_proba if model is PlayerScorerPrediction else cards_store.predict_proba
    frame["probability"] = predict(bundle, frame)
    revision = (previous[0] + 1) if previous else 1
    names = pd.DataFrame(starters)[["player_id", "player_name", "team_name"]]
    _save(session, model, frame, names, fixture_id=fixture_id, kickoff=kickoff, revision=revision,
          model_version=bundle.get("run_id") or bundle.get("version") or "production")
    return "predicted"


def apply_initial_predictions(session: Session, *, fixture_id: int, kickoff: str, league: int, season: int,
                              home_id: int, away_id: int, starters: list[dict],
                              scorer_bundle: Optional[dict], cards_bundle: Optional[dict],
                              now: Optional[datetime] = None) -> dict[str, Any]:
    """`starters` usa gli ID API-Sports. Il valore di mercato non entra nel calcolo."""
    now = now or datetime.now(timezone.utc)
    if len(starters) != 22 or len({int(item["player_id"]) for item in starters}) != 22:
        return {"scorers": "formazione_incompleta", "cards": "formazione_incompleta"}
    seasons = list(range(now.year - HISTORY_SEASONS_BACK, now.year + 1))
    leagues = list(SCORER_HISTORY_LEAGUES)
    report: dict[str, Any] = {}
    if scorer_bundle is None:
        report["scorers"] = "nessun_modello"
    else:
        try:
            report["scorers"] = _one_market(
                session, PlayerScorerPrediction, scorer_bundle, build_scorers,
                fixture_id=fixture_id, kickoff=kickoff, league=league, season=season,
                home_id=home_id, away_id=away_id, starters=starters, history_leagues=leagues,
                seasons=seasons, extra_frame=lambda _matches: (),
            )
        except Exception:  # noqa: BLE001 - una partita non blocca le altre
            logger.exception("FotMob: previsione marcatori non calcolata per %s", fixture_id)
            report["scorers"] = "errore"
    if cards_bundle is None:
        report["cards"] = "nessun_modello"
    else:
        try:
            report["cards"] = _one_market(
                session, PlayerCardPrediction, cards_bundle, build_cards,
                fixture_id=fixture_id, kickoff=kickoff, league=league, season=season,
                home_id=home_id, away_id=away_id, starters=starters, history_leagues=leagues,
                seasons=seasons,
                extra_frame=lambda matches: (load_referees(session, matches["fixture_id"].tolist()),),
            )
        except Exception:  # noqa: BLE001
            logger.exception("FotMob: previsione ammoniti non calcolata per %s", fixture_id)
            report["cards"] = "errore"
    if report.get("scorers") == "predicted" or report.get("cards") == "predicted":
        session.commit()
    return report
