"""Modello ORM delle statistiche individuali per partita (cantiere
"giocatori che segnano"): endpoint API-Sports `fixtures/players`.

Completa eventi (`player_match_event`) e formazioni (`player_lineup`) con
minuti, tiri, tiri in porta, rigori, rating ecc. - utili come proxy di xG
per il modello "giocatore segna". Una riga = UN giocatore in UNA fixture;
PK composita (fixture_id, player_id) = unicita' garantita e upsert via
`session.merge`, come `PlayerLineup`. Chiave di join con gli altri dataset:
(fixture_id, player_id)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String

from src.service_ia.model.match import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class PlayerMatchStats(Base):
    __tablename__ = "player_match_stats"

    fixture_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, primary_key=True)
    team_id = Column(Integer, nullable=True, index=True)
    minutes = Column(Integer, nullable=True)
    rating = Column(Float, nullable=True)
    shots_total = Column(Integer, nullable=True)
    shots_on = Column(Integer, nullable=True)
    goals = Column(Integer, nullable=True)
    assists = Column(Integer, nullable=True)
    saves = Column(Integer, nullable=True)
    passes_total = Column(Integer, nullable=True)
    key_passes = Column(Integer, nullable=True)
    dribbles_attempts = Column(Integer, nullable=True)
    dribbles_success = Column(Integer, nullable=True)
    fouls_drawn = Column(Integer, nullable=True)
    fouls_committed = Column(Integer, nullable=True)
    yellow = Column(Integer, nullable=True)
    red = Column(Integer, nullable=True)
    pen_won = Column(Integer, nullable=True)
    pen_committed = Column(Integer, nullable=True)
    pen_scored = Column(Integer, nullable=True)
    pen_missed = Column(Integer, nullable=True)
    offsides = Column(Integer, nullable=True)
    captain = Column(Boolean, nullable=True)
    substitute = Column(Boolean, nullable=True)
    position = Column(String, nullable=True)
    source = Column(String, nullable=False, default="api_sports")
    created_at = Column(DateTime, nullable=False, default=_utcnow)

    def to_dict(self) -> dict:
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
