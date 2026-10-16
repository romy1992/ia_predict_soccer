"""Snapshot append-only delle formazioni FotMob.

Non sostituiscono `player_lineup`: quella tabella resta la formazione
ufficiale API-Sports usata per addestrare. Una riga nuova non cancella
le precedenti, nemmeno quando il tipo passa da `predicted` a `confirmed`.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, JSON, String, func

from src.service_ia.model.match import Base


class FotmobMatchMap(Base):
    __tablename__ = "fotmob_match_map"

    fixture_id = Column(Integer, primary_key=True)
    fotmob_match_id = Column(Integer, nullable=False, index=True)
    method = Column(String, nullable=False)
    mapped_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class FotmobLineupSnapshot(Base):
    __tablename__ = "fotmob_lineup_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fixture_id = Column(Integer, nullable=True, index=True)
    fotmob_match_id = Column(Integer, nullable=False, index=True)
    lineup_type = Column(String, nullable=False)
    source = Column(String, nullable=True)
    fetched_at = Column(DateTime(timezone=True), nullable=False, index=True)
    home_formation = Column(String, nullable=True)
    away_formation = Column(String, nullable=True)
    home_team_fotmob_id = Column(Integer, nullable=True)
    away_team_fotmob_id = Column(Integer, nullable=True)
    home_team_id = Column(Integer, nullable=True)
    away_team_id = Column(Integer, nullable=True)
    reliability = Column(String, nullable=False, index=True)
    warnings = Column(JSON, nullable=False)
    content_hash = Column(String, nullable=False)
    starter_count = Column(Integer, nullable=False, default=0)
    mapped_starters = Column(Integer, nullable=False, default=0)
    payload = Column(JSON, nullable=False)

    def to_dict(self) -> dict:
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class FotmobLineupPlayer(Base):
    __tablename__ = "fotmob_lineup_player"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_id = Column(Integer, nullable=False, index=True)
    side = Column(String, nullable=False)
    role = Column(String, nullable=False)
    fotmob_player_id = Column(Integer, nullable=True)
    name = Column(String, nullable=True)
    shirt_number = Column(String, nullable=True)
    position_id = Column(Integer, nullable=True)
    market_value = Column(Integer, nullable=True)
    api_player_id = Column(Integer, nullable=True)
    api_team_id = Column(Integer, nullable=True)
    map_method = Column(String, nullable=True)
    historical_position = Column(String, nullable=True)
    unavailability_type = Column(String, nullable=True)
    expected_return = Column(String, nullable=True)
    expected_return_date = Column(String, nullable=True)
    return_certainty = Column(String, nullable=True)
