"""Prediction Ledger Service (BET-06, Fase BETTING) — orchestrazione DB.

Collega la parte "pura" (`prediction_ledger.py`) alla persistenza
(`PredictionLedgerRepository`) e al dato reale necessario per il settlement
(risultato finale del match, da `MatchRepository`/`FilterMarketService`,
STESSI moduli gia' usati dal training — nessuna nuova logica di
estrazione statistiche).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.oracle.backtest.betting_backtester import (
    DEFAULT_PLACED_DECISIONS,
    STAKE_DEFAULT,
    BacktestBet,
    BacktestReport,
    compute_backtest_report,
)
from src.oracle.decision_engine.decision_policy import Decision
from src.oracle.ledger.prediction_ledger import (
    DEFAULT_STAKE,
    PredictionRecord,
    build_prediction_record,
    resolve_actual_outcome,
    settle_prediction_record,
)
from src.oracle.ledger.settlement_rules import (
    PENDING_MATCH_STATUSES,
    RESULT_STATUSES,
    VOID_STATUS_BY_MATCH_STATUS,
    SettlementResolution,
    resolve_settlement,
)
from src.repository.match_repository import MatchRepository
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.model.match import PredictionLedger
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.utility.utils import convert_orm_match_to_dict


def _record_to_orm(record: PredictionRecord) -> PredictionLedger:
    row = PredictionLedger(
        fixture_id=record.fixture_id,
        market=record.market,
        outcome=record.outcome,
        model_run_id=record.model_run_id,
        model_name=record.model_name,
        policy_version=record.policy_version,
        p_model=record.p_model,
        p_market_raw=record.p_market_raw,
        p_market_fair=record.p_market_fair,
        odd=record.odd,
        fair_odd=record.fair_odd,
        prob_edge=record.prob_edge,
        ev=record.ev,
        decision=record.decision,
        stake=record.stake,
        period=record.period,
        line=record.line,
        source=record.source,
        cohort=record.cohort,
        captured_at=record.captured_at or record.created_at or datetime.now(timezone.utc),
        odds_captured_at=record.odds_captured_at,
        bookmaker_count=record.bookmaker_count,
        league=int(record.league) if record.league is not None else None,
        capture_key=record.capture_key,
        kickoff_at=record.kickoff_at,
        created_at=record.created_at or datetime.now(timezone.utc),
        is_settled=record.is_settled,
        settled_at=record.settled_at,
        settlement_status=record.settlement_status,
        actual_outcome=record.actual_outcome,
        won=record.won,
        pnl=record.pnl,
    )
    if record.id_prediction:
        row.id_prediction = record.id_prediction
    return row


def _orm_to_record(row: PredictionLedger) -> PredictionRecord:
    return PredictionRecord(
        id_prediction=row.id_prediction,
        fixture_id=row.fixture_id,
        market=row.market,
        outcome=row.outcome,
        decision=row.decision,
        stake=row.stake,
        model_run_id=row.model_run_id,
        model_name=row.model_name,
        policy_version=row.policy_version,
        p_model=row.p_model,
        p_market_raw=row.p_market_raw,
        p_market_fair=row.p_market_fair,
        odd=row.odd,
        fair_odd=row.fair_odd,
        prob_edge=row.prob_edge,
        ev=row.ev,
        period=row.period,
        line=row.line,
        source=row.source,
        cohort=row.cohort,
        captured_at=row.captured_at,
        odds_captured_at=row.odds_captured_at,
        bookmaker_count=row.bookmaker_count,
        league=str(row.league) if row.league is not None else None,
        capture_key=row.capture_key,
        kickoff_at=row.kickoff_at,
        created_at=row.created_at,
        is_settled=row.is_settled,
        settled_at=row.settled_at,
        settlement_status=row.settlement_status,
        actual_outcome=row.actual_outcome,
        won=row.won,
        pnl=row.pnl,
    )


class PredictionLedgerService:
    def __init__(
        self,
        repo: Optional[PredictionLedgerRepository] = None,
        match_repo: Optional[MatchRepository] = None,
    ):
        self.repo = repo or PredictionLedgerRepository()
        self.match_repo = match_repo or MatchRepository()

    def log_prediction(
        self,
        fixture_id: int,
        decision: Decision,
        model_run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        kickoff_at: Optional[datetime] = None,
        stake: float = DEFAULT_STAKE,
        dedupe: bool = True,
        p_market_raw: Optional[float] = None,
        period: str = "full_time",
        line: Optional[str] = None,
        source: str = "manual",
        cohort: str = "manual",
        captured_at: Optional[datetime] = None,
        odds_captured_at: Optional[datetime] = None,
        bookmaker_count: int = 0,
        league: Optional[int] = None,
        capture_key: Optional[str] = None,
    ) -> PredictionLedger:
        """Salva UNA prediction PRIMA del kickoff (acceptance criteria).

        Idempotente per costruzione (`dedupe=True`, default): se esiste gia'
        una riga per la STESSA chiave logica (fixture/market/outcome/
        model_run_id), la ritorna COSI' COM'E' invece di crearne un'altra —
        garanzia di immutabilita': una prediction gia' salvata non viene mai
        ri-scritta da una chiamata successiva con lo stesso `model_run_id`.
        """
        captured_at = captured_at or datetime.now(timezone.utc)
        is_official = cohort == "official_paper" or source == "scheduled_official_capture"
        if is_official and kickoff_at is None:
            raise ValueError("kickoff_at obbligatorio per registrare una giocata")
        captured_aware = captured_at if captured_at.tzinfo else captured_at.replace(tzinfo=timezone.utc)
        if kickoff_at is not None:
            kickoff_aware = kickoff_at if kickoff_at.tzinfo else kickoff_at.replace(tzinfo=timezone.utc)
            if captured_aware >= kickoff_aware:
                raise ValueError("Una giocata non può essere registrata al o dopo il kickoff")
        if decision.decision == "PLAY" and (decision.odd is None or float(decision.odd) <= 0):
            raise ValueError("Una PLAY richiede una quota valida")

        if capture_key:
            existing = self.repo.find_by_capture_key(capture_key)
            if existing is not None:
                return existing
        if dedupe:
            existing = self.repo.find_existing(
                fixture_id=fixture_id,
                market=decision.market,
                outcome=decision.outcome,
                model_run_id=model_run_id,
                cohort=cohort,
            )
            if existing is not None:
                return existing

        record = build_prediction_record(
            fixture_id=fixture_id,
            decision=decision,
            model_run_id=model_run_id,
            model_name=model_name,
            kickoff_at=kickoff_at,
            stake=stake,
            p_market_raw=p_market_raw,
            period=period,
            line=line,
            source=source,
            cohort=cohort,
            captured_at=captured_at,
            odds_captured_at=odds_captured_at,
            bookmaker_count=bookmaker_count,
            league=str(league) if league is not None else None,
            capture_key=capture_key,
        )
        return self.repo.save(_record_to_orm(record))

    def _resolve_outcome_for_fixture(
        self,
        fixture_id: int,
        market: str,
        period: str = "full_time",
        line: Optional[str] = None,
    ) -> SettlementResolution:
        match = self.match_repo.filter_by(dict_search={"id_fixture": int(fixture_id)}).first()
        if match is None:
            return SettlementResolution(pending=True)

        match_dict = convert_orm_match_to_dict([match])[0]
        stat_home, stat_away = FilterMarketService._resolve_team_stats(match=match_dict, with_full_stats=True)
        home_score = match.score_home
        away_score = match.score_away
        if home_score is None and stat_home:
            home_score = stat_home.get("score_ft")
        if away_score is None and stat_away:
            away_score = stat_away.get("score_ft")
        return resolve_settlement(
            market=market,
            match_status=str(match.status or ""),
            home_score=home_score,
            away_score=away_score,
            stat_home=stat_home,
            stat_away=stat_away,
            period=period,
            line=line,
        )

    def settle_prediction(self, id_prediction: str) -> Optional[PredictionLedger]:
        """Settlement di UNA prediction: nessun effetto se il match non e'
        ancora concluso (resta pending, riprovabile al prossimo run)."""
        row = self.repo.get_by_id(id_prediction)
        if row is None or row.is_settled:
            return row

        resolution = self._resolve_outcome_for_fixture(
            fixture_id=row.fixture_id,
            market=row.market,
            period=row.period,
            line=row.line,
        )
        if resolution.pending:
            return row  # match non ancora concluso: resta pending, nessuna modifica

        record = settle_prediction_record(
            record=_orm_to_record(row),
            actual_outcome=resolution.actual_outcome,
            void_status=resolution.void_status,
            is_push=resolution.is_push,
        )
        return self.repo.save(_record_to_orm(record))

    def settle_pending(self, before: Optional[datetime] = None, limit: int = 500) -> dict[str, Any]:
        """Settlement batch (job giornaliero/manuale): itera le prediction
        NON settled con kickoff gia' passato, settla SOLO quelle il cui
        match e' effettivamente concluso (`FINAL_STATUSES`) — le altre
        restano pending, mai forzate."""
        pending = self.repo.list_pending_settlement(before=before, limit=limit)
        report = {
            "ledger_candidates": len(pending),
            "ledger_settled_win": 0,
            "ledger_settled_loss": 0,
            "ledger_void": 0,
            "ledger_still_pending": 0,
            "errors": [],
        }

        for row in pending:
            try:
                resolution = self._resolve_outcome_for_fixture(
                    fixture_id=row.fixture_id,
                    market=row.market,
                    period=row.period,
                    line=row.line,
                )
                if resolution.pending:
                    report["ledger_still_pending"] += 1
                    continue

                record = settle_prediction_record(
                    record=_orm_to_record(row),
                    actual_outcome=resolution.actual_outcome,
                    void_status=resolution.void_status,
                    is_push=resolution.is_push,
                )
                saved = self.repo.save(_record_to_orm(record))
                if saved.settlement_status == "settled_win":
                    report["ledger_settled_win"] += 1
                elif saved.settlement_status == "settled_loss":
                    report["ledger_settled_loss"] += 1
                else:
                    report["ledger_void"] += 1
            except Exception as exc:
                report["errors"].append(
                    {"fixture_id": row.fixture_id, "id_prediction": row.id_prediction, "message": str(exc)}
                )

        # Alias retro-compatibili per consumer BET-06 precedenti.
        report["candidates"] = report["ledger_candidates"]
        report["settled"] = report["ledger_settled_win"] + report["ledger_settled_loss"] + report["ledger_void"]
        report["still_pending"] = report["ledger_still_pending"]
        report["void_no_result"] = report["ledger_void"]
        return report

    def list_ledger(
        self, market: Optional[str] = None, is_settled: Optional[bool] = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        rows = self.repo.list_all(market=market, is_settled=is_settled, limit=limit)
        return [row.to_dict() for row in rows]

    def raw_pnl_summary(self, market: Optional[str] = None) -> dict[str, Any]:
        """Somma DIRETTA dei `pnl` gia' calcolati per riga al momento del
        settlement (nessuna ricostruzione/assunzione di stake uniforme):
        fonte di verita' minimale per "PnL paper calcolabile" (acceptance
        criteria), sempre coerente per costruzione con quanto salvato."""
        rows = self.repo.list_all(market=market, is_settled=True, limit=1_000_000)
        wins = sum(1 for row in rows if row.settlement_status == "settled_win")
        losses = sum(1 for row in rows if row.settlement_status == "settled_loss")
        voids = [row for row in rows if str(row.settlement_status or "").startswith("void_")]
        realised_rows = [
            row
            for row in rows
            if str(row.settlement_status or "").startswith("void_")
            or row.settlement_status in {"settled_win", "settled_loss", "settled"}
        ]
        return {
            "market": market,
            "settled_total": len(rows),
            "settled_with_pnl": wins + losses,
            "void": len(voids),
            "wins": wins,
            "losses": losses,
            "total_pnl": float(sum(float(row.pnl or 0.0) for row in realised_rows)),
            "total_staked": float(
                sum(row.stake for row in realised_rows if not str(row.settlement_status or "").startswith("void_"))
            ),
        }

    def paper_pnl_report(
        self,
        market: Optional[str] = None,
        stake: float = STAKE_DEFAULT,
        include_decisions: frozenset[str] = DEFAULT_PLACED_DECISIONS,
    ) -> BacktestReport:
        """Report aggregato ricco (ROI/hit-rate/max drawdown/edge bucket),
        RIUSANDO `compute_backtest_report` (BET-03, non duplicato): assume
        stake flat uniforme su tutte le righe (stesso principio "Stake
        flat" di BET-03 — se il chiamante ha salvato prediction con stake
        eterogenei, si usi `raw_pnl_summary` per la somma esatta)."""
        rows = self.repo.list_all(market=market, is_settled=True, limit=1_000_000)
        bets = [
            BacktestBet(
                market=row.market,
                outcome=row.outcome,
                p_model=row.p_model,
                p_market_fair=row.p_market_fair,
                odd=row.odd,
                prob_edge=row.prob_edge,
                ev=row.ev,
                decision=row.decision,
                policy_version=row.policy_version or "",
                won=row.won,
                fixture_id=row.fixture_id,
                kickoff_at=row.kickoff_at.isoformat() if row.kickoff_at else None,
            )
            for row in rows
        ]
        return compute_backtest_report(bets=bets, stake=stake, include_decisions=include_decisions)
