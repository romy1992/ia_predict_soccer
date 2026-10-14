"""Predizioni "il giocatore prende un giallo?" - append-only, una riga per giocatore e REVISIONE.

Stessa logica di `scorer_prediction_models.py`: una nuova revisione nasce solo se la formazione cambia; le
predizioni pubblicate non si riscrivono mai. `yellow` si compila dopo la partita (ammonizione nei tempi
regolamentari) per misurare la qualita' reale del modello."""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, func

from src.service_ia.model.match import Base


class PlayerCardPrediction(Base):
    __tablename__ = "player_card_prediction"

    fixture_id = Column(Integer, primary_key=True)
    player_id = Column(Integer, primary_key=True)
    revision = Column(Integer, primary_key=True, default=1)
    player_name = Column(String, nullable=True)
    team_id = Column(Integer, nullable=True)
    team_name = Column(String, nullable=True)
    position = Column(String, nullable=True)
    league = Column(Integer, nullable=True, index=True)
    kickoff = Column(String, nullable=True)
    probability = Column(Float, nullable=False)
    lineup_type = Column(String, nullable=False, default="official")
    model_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    yellow = Column(Boolean, nullable=True)

    def to_dict(self) -> dict:
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
