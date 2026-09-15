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
    # Id proveniente da api.sports (per statistiche ed eventuali nuove quote).
    # UNICO (migration f1a2b3c4d5e6): il vincolo era commentato e senza di
    # esso due esecuzioni sovrapposte di `download_import_matches` (es. il
    # job `data_daily_refresh` e il bottone "Aggiorna tutto", processi
    # `scheduler`/`api` distinti) inserivano due righe per la stessa
    # partita - ne sono nate 116. Gli update successivi passano da
    # `filter_by(...).first()` e ne aggiornavano una sola: l'altra restava
    # a `NS` per sempre e la Dashboard mostrava la partita "In diretta" a
    # giorni di distanza (vedi `_classify_phase` e lo script
    # `scripts/maintenance/dedup_match_id_fixture.py`). L'indice unico e'
    # anche il presupposto dell'upsert `ON CONFLICT (id_fixture)` usato ora
    # da `download_import_matches` al posto di select-poi-insert.
    # Resta NULLABLE: in Postgres un unique index ammette piu' NULL, quindi
    # le righe legacy senza fixture (import storici da odds-api) restano valide.
    id_fixture = Column(Integer, unique=True)
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
    # `lazy="select"` e NON `selectin` come le due relazioni sopra: e' la
    # tabella piu' grande del DB (311.779 righe / 113 MB, media 693 snapshot
    # per match, massimo 3.060) e con l'eager load OGNI query su Match la
    # trascinava dietro. Misurato prima della modifica: caricare UNA fixture
    # con `filter_by({'id_fixture': ...}).first()` costava 1,54 s e 2.174
    # righe snapshot trasferite da Railway, che su 193 fixture spiega i 456 s
    # di un `future_sync`; `SettlementService` (44.983 match finali, nessun
    # filtro data) e `FilterMarketService._search_matches` ne soffrivano
    # ancora di piu' (1.331 s e fino a 8.708 s per esecuzione).
    # Nessun chiamante perde dati: l'UNICO punto che legge davvero questa
    # collezione e' `OfficialPredictionCaptureService`, che la chiede gia'
    # esplicitamente con `selectinload(Match.odds_snapshots)`; Dashboard e
    # `phase0_under_over_data_quality` usano `noload`; tutto il resto
    # (`Match.to_dict`, `convert_orm_match_to_dict`) non la tocca mai.
    # Non `lazy="raise"`: trasformerebbe un problema di prestazioni in un
    # errore a runtime su un percorso utente, e non serve - chi la vuole la
    # chiede con `selectinload`.
    odds_snapshots = relationship(
        "OddsSnapshot",
        back_populates="match",
        cascade="all, delete-orphan",
        lazy="select",
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
    id_match = Column(String(36), ForeignKey("match.id_match_fk"), index=True)  # 👈 Foreign Key
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
    id_match = Column(String(36), ForeignKey("match.id_match_fk"), index=True)  # 👈 Foreign Key
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
    __table_args__ = (
        Index("ix_prediction_ledger_cohort_fixture", "cohort", "fixture_id"),
        Index("ix_prediction_ledger_created_at", "created_at"),
    )

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


class BettingSlip(Base):
    """Schedina ufficiale: snapshot pre-partita immutabile e settlement separato."""

    __tablename__ = "betting_slips"
    __table_args__ = (
        Index("ix_betting_slips_reference_profile", "reference_date", "profile"),
        Index("ix_betting_slips_status", "status"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    capture_key = Column(String(160), nullable=False, unique=True)
    reference_date = Column(String(10), nullable=False)
    profile = Column(String(24), nullable=False)
    initial_situation = Column(String(16), nullable=False)
    initial_reason = Column(String, nullable=True)
    status = Column(String(16), nullable=False, default="PENDING")
    event_count = Column(Integer, nullable=False)
    combined_odd = Column(Float, nullable=False)
    naive_probability = Column(Float, nullable=True)
    adjusted_probability = Column(Float, nullable=True)
    combined_model_void_odd = Column(Float, nullable=True)
    combined_edge_absolute = Column(Float, nullable=True)
    combined_edge_percent = Column(Float, nullable=True)
    combined_expected_roi = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)
    combined_play_threshold = Column(Float, nullable=True)
    slip_min_edge_percent = Column(Float, nullable=True)
    stake = Column(Float, nullable=False, default=1.0)
    potential_return = Column(Float, nullable=True)
    effective_combined_odd = Column(Float, nullable=True)
    actual_return = Column(Float, nullable=True)
    realized_profit = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    policy_version = Column(String, nullable=False)
    correlation_version = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    settled_at = Column(DateTime(timezone=True), nullable=True)

    picks = relationship(
        "BettingSlipPick",
        back_populates="slip",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="BettingSlipPick.position",
    )

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        for key in ("created_at", "settled_at"):
            if payload.get(key) is not None:
                payload[key] = payload[key].isoformat()
        payload["picks"] = [pick.to_dict() for pick in self.picks]
        return payload


class BettingSlipPick(Base):
    """Snapshot della singola selezione appartenente a una schedina ufficiale."""

    __tablename__ = "betting_slip_picks"
    __table_args__ = (
        Index("ix_betting_slip_picks_slip_position", "slip_id", "position", unique=True),
        Index("ix_betting_slip_picks_fixture", "fixture_id"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    slip_id = Column(String(36), ForeignKey("betting_slips.id", ondelete="CASCADE"), nullable=False)
    prediction_id = Column(String(36), ForeignKey("prediction_ledger.id_prediction"), nullable=True)
    position = Column(Integer, nullable=False)
    fixture_id = Column(Integer, nullable=False)
    competition = Column(String, nullable=True)
    kickoff_at = Column(DateTime(timezone=True), nullable=False)
    home_team = Column(String, nullable=True)
    away_team = Column(String, nullable=True)
    market = Column(String, nullable=False)
    line = Column(String, nullable=True)
    outcome = Column(String, nullable=False)
    p_model = Column(Float, nullable=False)
    market_odd = Column(Float, nullable=False)
    model_void_odd = Column(Float, nullable=True)
    market_fair_odd = Column(Float, nullable=True)
    odds_edge_absolute = Column(Float, nullable=True)
    odds_edge_percent = Column(Float, nullable=True)
    expected_roi = Column(Float, nullable=True)
    situation = Column(String(16), nullable=False)
    bookmakers_count = Column(Integer, nullable=False, default=0)
    model_version = Column(String, nullable=True)
    policy_version = Column(String, nullable=True)
    status = Column(String(16), nullable=False, default="PENDING")
    final_score = Column(String(24), nullable=True)
    void_reason = Column(String, nullable=True)
    settled_at = Column(DateTime(timezone=True), nullable=True)

    slip = relationship("BettingSlip", back_populates="picks")

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        for key in ("kickoff_at", "settled_at"):
            if payload.get(key) is not None:
                payload[key] = payload[key].isoformat()
        payload["expected_roi_percent"] = (
            payload["expected_roi"] * 100.0 if payload.get("expected_roi") is not None else None
        )
        return payload


class BettingSlipProposalSnapshot(Base):
    """Snapshot append-only di una schedina proposta dal generatore.

    Le performance economiche ufficiali restano nelle tabelle
    ``betting_slips``/``betting_slip_picks``. Questa tabella conserva invece
    ogni revisione realmente diversa di una proposta, senza trasformarla in
    una giocata ufficiale e senza introdurre hindsight nel ROI.
    """

    __tablename__ = "betting_slip_proposal_snapshots"
    __table_args__ = (
        Index(
            "ix_betting_slip_proposal_reference_profile",
            "reference_date",
            "profile",
        ),
        Index(
            "ix_betting_slip_proposal_lineage_latest",
            "logical_slip_id",
            "is_latest",
        ),
        Index(
            "ix_betting_slip_proposal_shadow_status",
            "shadow_status",
            "is_latest",
        ),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    snapshot_key = Column(String(64), nullable=False, unique=True)
    logical_slip_id = Column(String(64), nullable=False)
    supersedes_id = Column(
        String(36),
        ForeignKey("betting_slip_proposal_snapshots.id"),
        nullable=True,
    )
    reference_date = Column(String(10), nullable=False)
    profile = Column(String(24), nullable=False)
    situation = Column(String(16), nullable=False)
    event_count = Column(Integer, nullable=False)
    combined_odd = Column(Float, nullable=False)
    adjusted_probability = Column(Float, nullable=True)
    combined_model_void_odd = Column(Float, nullable=True)
    combined_edge_absolute = Column(Float, nullable=True)
    combined_expected_roi = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    policy_version = Column(String, nullable=False)
    correlation_version = Column(String, nullable=False)
    diversification_version = Column(String, nullable=True)
    payload = Column(JSON, nullable=False)
    is_latest = Column(Boolean, nullable=False, default=True)
    shadow_status = Column(String(16), nullable=False, default="PENDING")
    shadow_stake = Column(Float, nullable=False, default=1.0)
    shadow_effective_odd = Column(Float, nullable=True)
    shadow_return = Column(Float, nullable=True)
    shadow_profit = Column(Float, nullable=True)
    shadow_settlement = Column(JSON, nullable=True)
    shadow_settled_at = Column(DateTime(timezone=True), nullable=True)
    staking_policy_version = Column(
        String,
        nullable=False,
        default="shadow_flat_unit_v1",
    )
    generated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self):
        payload = {column.name: getattr(self, column.name) for column in self.__table__.columns}
        if payload.get("generated_at") is not None:
            payload["generated_at"] = payload["generated_at"].isoformat()
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


