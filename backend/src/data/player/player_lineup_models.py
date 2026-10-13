"""Modello ORM delle formazioni (cantiere "giocatori che segnano").

Completa `player_models.py`/`PlayerMatchEvent`: gli eventi (gol, cartellini,
sostituzioni) dicono SOLO chi ha fatto qualcosa di notevole, non chi era in
campo. Senza le formazioni, un dataset di training "ha segnato si/no"
sarebbe fortemente distorto - vedremmo solo attaccanti/giocatori con
eventi, mai gli esempi negativi puliti (difensori/portieri che giocano 90
minuti senza mai comparire in `player_match_event`).

Una riga = UN giocatore in UNA fixture (titolare o panchina) - l'universo
completo su cui poi si incrociano gli eventi per sapere chi ha segnato.
Tabella PERMANENTE, stessa Base condivisa di `player_models.py`/
`src.service_ia.model.match` (stesso principio, nessuna configurazione
Alembic separata)."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, Integer, String

from src.service_ia.model.match import Base


class PlayerLineup(Base):
    """Un giocatore (titolare o panchina) in una fixture - PK composita
    naturale (fixture_id, player_id): un giocatore appare una sola volta
    per fixture nella formazione, mai duplicato."""

    __tablename__ = "player_lineup"

    fixture_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, primary_key=True)
    player_name = Column(String, nullable=True)
    team_id = Column(Integer, nullable=True, index=True)
    team_name = Column(String, nullable=True)
    position = Column(String, nullable=True)
    is_starter = Column(Boolean, nullable=False, default=False)
    formation = Column(String, nullable=True)
    source = Column(String, nullable=False, default="api_sports")

    def to_dict(self) -> dict:
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
