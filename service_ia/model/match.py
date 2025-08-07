import uuid

from sqlalchemy import Column, Integer, String, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Match(Base):
    __tablename__ = 'match'
    id_match_fk = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_events = Column(String(32), unique=True)  # Id proveniente da odds.api (per quote)
    # id_alternate_events: Id proveniente da odds.api (per quote) - In caso di mach rinviato o spostato
    id_alternate_events = Column(String(32))
    id_fixture = Column(Integer, unique=True)  # Id proveniente da api.sports (per statistiche ed eventuali nuove quote)
    name_home = Column(String)  # Nome team casa
    id_team_home = Column(Integer)  # Id team casa
    name_away = Column(String)  # Nome tema ospite
    id_team_away = Column(Integer)  # Id team ospite
    date_match = Column(String)  # Data reale del match
    date_alternate_match = Column(String)  # In caso di mach rinviato o spostato
    sport_key = Column(String)  # Chiave della lega (Stringa per odds e numerica per statistics)
    title_league = Column(String)  # Nome della lega
    referee = Column(String)  # Arbitro
    round = Column(String)  # Giornata
    season = Column(Integer)  # Stagione
    statistics = relationship("Statistics",
                              back_populates="match",  # back_populates crea la relazione # 👈 One-to-Many
                              cascade="all, delete-orphan")
    odds = relationship("Odds",
                        back_populates="match",  # back_populates crea la relazione # 👈 One-to-Many
                        cascade="all, delete-orphan")

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class Statistics(Base):
    __tablename__ = 'statistics'
    id_statistics_fk = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_match = Column(String(36), ForeignKey("match.id_match_fk"))  # 👈 Foreign Key
    match = relationship("Match", back_populates="statistics")  # 👈 Many-to-One
    score_ht = Column(JSON, nullable=True)  # Risultato primo tempo
    score_ft = Column(JSON, nullable=True)  # Risultato secondo tempo
    shots = Column(JSON, nullable=True)  # Tiri
    fouls = Column(JSON, nullable=True)  # Falli
    corners = Column(JSON, nullable=True)  # Corner
    offside = Column(JSON, nullable=True)  # Fuorigioco
    bass_possession = Column(JSON, nullable=True)  # Possesso palla
    yellow_cards = Column(JSON, nullable=True)  # Cartellini gialli
    red_cards = Column(JSON, nullable=True)  # Cartellini rossi
    goal_keeper = Column(JSON, nullable=True)  # Palle salvate dal portiere
    passes = Column(JSON, nullable=True)  # Passaggi
    form = Column(JSON, nullable=True)  # Forma delle squadre
    for_ = Column(JSON, nullable=True)  # For comprende una serie di statistiche a FAVORE della squadra indicata
    against = Column(JSON, nullable=True)  # Against comprende una serie di statistiche a SFAVORE della squadra indicata
    # preview_matches: wins_home-wins_away-draws_home-draws_away-loses_home-loses_away -> Serie di statistiche che indicano il totale delle partite precedenti se hanno vinto,pareggiato o perso in casa o fuori
    preview_matches = Column(JSON, nullable=True)
    comparison = Column(JSON, nullable=True)  # Percentuali di comparazioni delle 2 squadre
    # Restanti statistiche :expected_goals-goals_prevented-Assists-Counter Attacks-Cross Attacks-Free Kicks-Goals-Goal Attempts-Substitutions-Throwins-Medical Treatment
    generic_statistics = Column(JSON, nullable=True)
    predict = Column(JSON, nullable=True)  # Predizioni del match provenienti da API

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class Odds(Base):
    __tablename__ = 'odds'
    id_odds_fk = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_match = Column(String(36), ForeignKey("match.id_match_fk"))  # 👈 Foreign Key
    match = relationship("Match", back_populates="odds")  # 👈 Many-to-One
    odds_from = Column(String)  # Da che API proviene la quota
    h2h = Column(JSON, nullable=True)  # Fisse(1X2)
    under_over_1_5 = Column(JSON, nullable=True)  # Per il match
    under_over_2_5 = Column(JSON, nullable=True)  # Per il match
    under_over_3_5 = Column(JSON, nullable=True)  # Per il match
    under_over_4_5 = Column(JSON, nullable=True)  # Per il match
    under_over_home_away = Column(JSON, nullable=True)  # Tutti gli under e over per singola squadra
    goal_no_goal = Column(JSON, nullable=True)  # Goal e No Goal
    corners = Column(JSON, nullable=True)  # Under e over dei corner
    cards = Column(JSON, nullable=True)  # Under e over dei cartellini
    dc = Column(JSON, nullable=True)  # Doppia chance

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}
