import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, Index, Integer, String, ForeignKey, JSON, Float, DateTime, Boolean
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Match(Base):
    __tablename__ = 'match'
    id_match_fk = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_events = Column(String(32))#, unique=True)  # Id proveniente da odds.api (per quote)
    # id_alternate_events: Id proveniente da odds.api (per quote) - In caso di mach rinviato o spostato
    id_alternate_events = Column(String(32))
    id_fixture = Column(Integer)#, unique=True)  # Id proveniente da api.sports (per statistiche ed eventuali nuove quote)
    name_home = Column(String)  # Nome team casa
    id_team_home = Column(Integer)  # Id team casa
    name_away = Column(String)  # Nome tema ospite
    id_team_away = Column(Integer)  # Id team ospite
    date_match = Column(String)  # Data reale del match
    date_alternate_match = Column(String)  # In caso di mach rinviato o spostato
    sport_key = Column(String)  # Chiave della lega (per odds)
    title_league = Column(String)  # Nome della lega (per odds)
    current_league = Column(Integer)  # Numero della lega corrente (per statistics)
    league_match = Column(Integer)  # Numero della lega partita (per statistics)
    # status della partita : # NS Non disputata - FT partita finita - AET è per partita finita ai supplementari (QUINDI PER COPPE) - PEN è per partita finita ai rigori (QUINDI PER COPPE)
    status = Column(String)
    referee = Column(String)  # Arbitro
    round = Column(String)  # Giornata
    season = Column(Integer)  # Stagione
    is_settled = Column(Boolean, nullable=True)
    settlement_status = Column(String, nullable=True)
    settled_at = Column(String, nullable=True)
    settlement_details = Column(JSON, nullable=True)
    # Punteggio finale (2026-09-10): salvato INDIPENDENTEMENTE da
    # `Statistics.score_ft`, dalla stessa risposta 'fixtures' che aggiorna
    # gia' `status` (sempre disponibile) - a differenza delle righe
    # `Statistics`, create SOLO se l'endpoint dedicato 'fixtures/statistics'
    # ha dati per quella fixture (spesso assente per campionati minori con
    # copertura limitata: una partita "Finita" restava senza alcun
    # punteggio mostrabile ne' un risultato reale per h2h/goal_no_goal/
    # under_over/dc). Vedi `download_match_service.map_base_match`.
    score_home = Column(Integer, nullable=True)
    score_away = Column(Integer, nullable=True)
    statistics = relationship("Statistics",
                              back_populates="match",  # back_populates crea la relazione # 👈 One-to-Many
                              cascade="all, delete-orphan", lazy="selectin")
    odds = relationship("Odds",
                        back_populates="match",  # back_populates crea la relazione # 👈 One-to-Many
                        cascade="all, delete-orphan", lazy="selectin")
    odds_snapshots = relationship(
        "OddsSnapshot",
        back_populates="match",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # Medie stagionali alla giornata corrente (cioè PRIMA CHE INIZIASSE LA PARTITA CORRENTE)
    # Bug fix: SQLAlchemy JSON di default (none_as_null=False) salva un
    # Python None come letterale JSON 'null' (NON SQL NULL) -> `IS NOT NULL`
    # risultava SEMPRE vero, rendendo il filtro "mean_statistics: not None"
    # un no-op silenzioso su tutta la pipeline multi-mercato. Con
    # none_as_null=True, assegnare None equivale a SQL NULL (comportamento
    # atteso da `CrudRepository.search_filter`).
    mean_statistics = Column(JSON(none_as_null=True), nullable=True)

    def to_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}


class Statistics(Base):
    __tablename__ = 'statistics'
    id_statistics_fk = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    id_match = Column(String(36), ForeignKey("match.id_match_fk"))  # 👈 Foreign Key
    match = relationship("Match", back_populates="statistics")  # 👈 Many-to-One
    statistics_team_id = Column(Integer)  # Discriminante per capire di che team si parla
    score_ht = Column(Integer)  # Risultato primo tempo
    score_ft = Column(Integer)  # Risultato secondo tempo
    shots = Column(JSON, nullable=True)  # Tiri
    fouls = Column(Integer)  # Falli
    corners = Column(Integer)  # Corner
    offside = Column(Integer)  # Fuorigioco
    bass_possession = Column(Integer)  # Possesso palla
    yellow_cards = Column(Integer)  # Cartellini gialli
    red_cards = Column(Integer)  # Cartellini rossi
    goal_keeper = Column(Integer)  # Palle salvate dal portiere
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


class OddsSnapshot(Base):
    __tablename__ = 'odds_snapshot'
    id_snapshot = Column(String(40), primary_key=True)
    id_match = Column(String(36), ForeignKey("match.id_match_fk"), nullable=True)
    fixture_id = Column(Integer, nullable=False)
    bookmaker = Column(String, nullable=False)
    market = Column(String, nullable=False)
    period = Column(String, nullable=False, default='full_time')
    line = Column(String, nullable=True)
    outcome = Column(String, nullable=False)
    odd = Column(Float, nullable=False)
    captured_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    source = Column(String, nullable=False, default='api_sports')

    match = relationship("Match", back_populates="odds_snapshots")

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("captured_at") is not None:
            payload["captured_at"] = payload["captured_at"].isoformat()
        return payload


class PredictionLedger(Base):
    """Prediction Ledger / Paper Betting (BET-06, Fase BETTING).

    Ogni riga rappresenta UNA prediction salvata PRIMA del kickoff
    ("Salvare prediction prima del kickoff", acceptance criteria): i campi
    "originali" (dal market/outcome fino a `created_at`) sono scritti UNA
    SOLA VOLTA da `PredictionLedgerService.log_prediction` e non vengono mai
    più modificati ("Immutabilita' logica della prediction originale") —
    il settlement (dopo il risultato reale) aggiorna ESCLUSIVAMENTE i campi
    dedicati sotto "Settlement" (mai i campi originali), stesso principio
    gia' applicato a `Match.is_settled/settlement_status/settlement_details`
    in `SettlementService`.
    """

    __tablename__ = 'prediction_ledger'

    id_prediction = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # --- Prediction originale (immutabile dopo la creazione) ---
    fixture_id = Column(Integer, nullable=False)
    market = Column(String, nullable=False)
    outcome = Column(String, nullable=False)
    model_run_id = Column(String, nullable=True)
    model_name = Column(String, nullable=True)
    policy_version = Column(String, nullable=True)
    p_model = Column(Float, nullable=True)
    p_market_raw = Column(Float, nullable=True)
    p_market_fair = Column(Float, nullable=True)
    odd = Column(Float, nullable=True)
    fair_odd = Column(Float, nullable=True)
    model_void_odd = Column(Float, nullable=True)
    market_fair_odd = Column(Float, nullable=True)
    odds_edge_absolute = Column(Float, nullable=True)
    odds_edge_percent = Column(Float, nullable=True)
    prob_edge = Column(Float, nullable=True)
    ev = Column(Float, nullable=True)
    expected_roi_percent = Column(Float, nullable=True)
    play_threshold_odd = Column(Float, nullable=True)
    min_edge_percent = Column(Float, nullable=True)
    value_label = Column(String, nullable=True)
    value_reason = Column(String, nullable=True)
    decision = Column(String, nullable=False)
    stake = Column(Float, nullable=False, default=1.0)
    period = Column(String, nullable=False, default="full_time")
    line = Column(String, nullable=True)
    source = Column(String, nullable=False, default="manual")
    cohort = Column(String, nullable=False, default="manual")
    captured_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    odds_captured_at = Column(DateTime(timezone=True), nullable=True)
    bookmaker_count = Column(Integer, nullable=False, default=0)
    league = Column(Integer, nullable=True)
    capture_key = Column(String(160), nullable=True, unique=True)
    kickoff_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    # --- Settlement (popolati SOLO dopo il risultato, mai in creazione) ---
    is_settled = Column(Boolean, nullable=False, default=False)
    settled_at = Column(DateTime(timezone=True), nullable=True)
    settlement_status = Column(String, nullable=True)
    actual_outcome = Column(String, nullable=True)
    won = Column(Boolean, nullable=True)
    pnl = Column(Float, nullable=True)

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        for key in ("captured_at", "odds_captured_at", "kickoff_at", "created_at", "settled_at"):
            if payload.get(key) is not None:
                payload[key] = payload[key].isoformat()
        return payload


class MatchPredictionSnapshot(Base):
    """Log APPEND-ONLY delle predizioni ML calcolate per fixture+mercato
    (2026-09-09, richiesto esplicitamente dall'operatore: "salvare le
    predizioni... anche perche' dobbiamo avere una banca dati da
    accumulare"). Distinta da `PredictionLedger` (BET-06, sopra): quella
    logga SOLO le decisioni di scommessa loggate esplicitamente (stake/EV/
    settlement); questa logga OGNI predizione GREZZA calcolata dal modello
    per OGNI mercato, usata sia come cache di serving (si legge l'ultima
    riga per fixture+market, `MatchPredictionSnapshotRepository.get_latest`)
    sia come storico di ricerca (come si e' mossa la stima del modello
    prima del calcio d'inizio, man mano che quote/feature cambiavano).

    Una riga NUOVA viene scritta SOLO quando `feature_fingerprint` (hash
    delle feature usate dal modello per QUEL mercato) o `model_run_id`
    cambiano rispetto all'ultima riga nota (`PredictionSnapshotService`) -
    mai un refresh "vuoto" senza che nulla sia davvero cambiato. Per le
    partite gia' concluse (status finale), la riga resta congelata per
    sempre: rappresenta "cosa prediceva il modello a quel tempo", un valore
    storico che non deve cambiare sotto i piedi nemmeno se in futuro viene
    promosso un modello nuovo (scelta esplicita dell'operatore).
    """

    __tablename__ = 'match_prediction_snapshot'
    __table_args__ = (
        Index(
            'ix_match_prediction_snapshot_fixture_market_computed',
            'fixture_id', 'market', 'computed_at',
        ),
    )

    id_snapshot = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    fixture_id = Column(Integer, nullable=False)
    market = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    model_name = Column(String, nullable=True)
    model_run_id = Column(String, nullable=True)
    feature_fingerprint = Column(String, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("computed_at") is not None:
            payload["computed_at"] = payload["computed_at"].isoformat()
        return payload


