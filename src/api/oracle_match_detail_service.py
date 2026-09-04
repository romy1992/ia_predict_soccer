"""Oracle Match Detail (MATCH-02, Fase MATCH CENTER).

Aggrega, per una singola fixture, le sezioni richieste dall'acceptance
criteria ("Dettaglio navigabile per fixture"):
1. Overview (fixture, timeline, quote, baseline bookmaker).
2. Probabilities (una riga per mercato con modello disponibile).
3. Value Bets (sottoinsieme delle probabilities con badge PLAY/BORDERLINE).
4. Team Strength (rating pre-match delle due squadre, EXP-01).
5. Expected Goals (lambda Poisson stimata dai rating, EXP-02).
6. Score Matrix (P(home=i, away=j), stessa fonte di Expected Goals).
7. Odds Movement (opening/latest/closing, sola visualizzazione).
8. Model Consensus per mercato (ORACLE-04).

Principio guida (stesso di MATCH-01/ORACLE-04): NESSUNA logica di
betting/ML duplicata qui. Ogni sezione riusa DIRETTAMENTE il modulo gia'
esistente e testato (`DashboardService.get_match_detail`, `TeamStrengthExpert`,
`GoalDistributionExpert`, `OddsSnapshotRepository`, `build_model_consensus_for_fixture`).

Acceptance criteria "Dati mancanti gestiti": ogni sezione e' calcolata in
modo indipendente e avvolta in try/except - se una fallisce o non ha dati
sufficienti (nessuno storico squadra, nessuno snapshot quote, nessun
modello registrato per un mercato), il campo resta `None`/vuoto con un
warning esplicito in `warnings`, MAI un'eccezione che blocca l'intero
dettaglio e MAI un valore inventato.
"""

from __future__ import annotations

from typing import Any, Optional

from src.api.dashboard_service import DashboardService
from src.ml.datasets.point_in_time_builder import FINAL_STATUSES
from src.ml.ensemble.model_consensus import build_model_consensus_for_fixture
from src.ml.experts.goal_distribution.goal_distribution_expert import GoalDistributionExpert
from src.ml.experts.team_strength.team_strength_expert import TeamStrengthExpert
from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

# Value Bets (acceptance criteria "Value Bets"): sottoinsieme delle
# probabilities con badge di interesse, stessa etichetta gia' prodotta da
# `evaluate_decision_from_fair_odds_outcome` (BET-04) - nessuna nuova soglia.
_VALUE_BET_LABELS = {"PLAY", "BORDERLINE"}


class OracleMatchDetailService:
    def __init__(
        self,
        dashboard_service: Optional[DashboardService] = None,
        match_repo: Optional[MatchRepository] = None,
        team_strength_expert: Optional[TeamStrengthExpert] = None,
        goal_distribution_expert: Optional[GoalDistributionExpert] = None,
        odds_snapshot_repo: Optional[OddsSnapshotRepository] = None,
    ):
        self.dashboard_service = dashboard_service or DashboardService()
        self.match_repo = match_repo or MatchRepository()
        self.team_strength_expert = team_strength_expert or TeamStrengthExpert()
        self.goal_distribution_expert = goal_distribution_expert or GoalDistributionExpert()
        self.odds_snapshot_repo = odds_snapshot_repo or OddsSnapshotRepository()

    # ------------------------------------------------------------------
    # Team Strength (EXP-01): rating PRE-match delle due squadre
    # ------------------------------------------------------------------
    def _team_strength_for_teams(
        self,
        home_team_id: Optional[int],
        away_team_id: Optional[int],
        before: Optional[str],
    ) -> Optional[dict[str, Any]]:
        if home_team_id is None or away_team_id is None:
            return None

        try:
            orm_matches = self.match_repo.search_filter(
                filters={
                    # Solo le partite delle DUE squadre coinvolte (non
                    # l'intero storico DB): query mirata, stesso principio
                    # di performance gia' applicato altrove nel progetto.
                    "OR": [
                        ("id_team_home", [int(home_team_id), int(away_team_id)]),
                        ("id_team_away", [int(home_team_id), int(away_team_id)]),
                    ],
                    "status": list(FINAL_STATUSES),
                }
            )
        except Exception:
            return None

        matches = convert_orm_match_to_dict(orm_matches)
        if before:
            # Point-in-time (no leakage): se la fixture corrente e' gia'
            # conclusa, esclude se stessa e ogni partita successiva -
            # confronto lessicografico valido su ISO 8601 con offset fisso
            # (stesso pattern gia' in uso in `DashboardService._fetch_matches`).
            before_str = str(before)
            matches = [m for m in matches if str(m.get("date_match") or "") < before_str]

        if not matches:
            return None

        try:
            ratings = self.team_strength_expert.current_ratings(matches)
        except Exception:
            return None

        home_rating = ratings.get(int(home_team_id))
        away_rating = ratings.get(int(away_team_id))
        if not home_rating or not away_rating:
            return None

        return {
            "rating_version": TeamStrengthExpert.VERSION,
            "sample_matches": len(matches),
            "home": home_rating,
            "away": away_rating,
        }

    # ------------------------------------------------------------------
    # Expected Goals + Score Matrix (EXP-02): stima deterministica dai
    # rating EXP-01, nessun training necessario.
    # ------------------------------------------------------------------
    def _goal_distribution_output(self, team_strength: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if not team_strength:
            return None

        home = team_strength.get("home") or {}
        away = team_strength.get("away") or {}
        try:
            home_lambda, away_lambda = self.goal_distribution_expert.estimate_lambdas_from_ratings(
                home_attack_rating=home["team_attack_rating"],
                away_defense_rating=away["team_defense_rating"],
                away_attack_rating=away["team_attack_rating"],
                home_defense_rating=home["team_defense_rating"],
            )
            return self.goal_distribution_expert.build_expert_output(home_lambda=home_lambda, away_lambda=away_lambda)
        except Exception:
            return None

    @staticmethod
    def _stringify_score_matrix(score_matrix_raw: dict[Any, dict[Any, float]]) -> dict[str, dict[str, float]]:
        """Le chiavi del DataFrame (`0..max_goals`) sono interi Python:
        forzate a stringa per un payload JSON/Pydantic senza ambiguita'."""
        return {
            str(row_key): {str(col_key): col_value for col_key, col_value in row.items()}
            for row_key, row in (score_matrix_raw or {}).items()
        }

    # ------------------------------------------------------------------
    # Model Consensus (ORACLE-04) per ciascun mercato con modello disponibile
    # ------------------------------------------------------------------
    def _model_consensus_by_market(self, fixture_id: int, model_markets: list[str]) -> dict[str, Any]:
        consensus_by_market: dict[str, Any] = {}
        for market in model_markets:
            try:
                report = build_model_consensus_for_fixture(market=market, fixture_id=fixture_id)
            except Exception:
                continue
            consensus_by_market[market] = {
                "experts": report.experts,
                "oracle_final": report.oracle_final,
                "consensus": report.consensus,
                "warnings": report.warnings,
            }
        return consensus_by_market

    # ------------------------------------------------------------------
    # Orchestratore principale
    # ------------------------------------------------------------------
    def build_oracle_match_detail(self, fixture_id: int, markets: Optional[list[str]] = None) -> dict[str, Any]:
        warnings: list[str] = []

        # 1/2/3: Overview + Probabilities + Value Bets - riusa DIRETTAMENTE
        # `get_match_detail` (MATCH-01), MAI una nuova query odds/predizioni.
        detail = self.dashboard_service.get_match_detail(fixture_id=fixture_id, with_predictions=True, markets=markets)
        fixture_row = detail.get("fixture")
        if not fixture_row:
            warnings.append("fixture_not_found")

        decision_cards = detail.get("decision_cards") or []
        value_bets = [card for card in decision_cards if card.get("value_label") in _VALUE_BET_LABELS]
        model_markets = detail.get("model_markets") or []

        # 4: Team Strength - id squadre/kickoff letti direttamente dal DB
        # locale (query leggera, nessun selectinload di statistics/odds).
        home_team_id = None
        away_team_id = None
        kickoff_at = None
        try:
            match_row = self.match_repo.filter_by(dict_search={"id_fixture": fixture_id}).first()
        except Exception:
            match_row = None
        if match_row is not None:
            home_team_id = match_row.id_team_home
            away_team_id = match_row.id_team_away
            kickoff_at = match_row.date_match

        team_strength = self._team_strength_for_teams(home_team_id, away_team_id, before=kickoff_at)
        if team_strength is None:
            warnings.append("team_strength_not_available")

        # 5/6: Expected Goals + Score Matrix
        goal_distribution = self._goal_distribution_output(team_strength)
        expected_goals: Optional[dict[str, Any]] = None
        score_matrix: Optional[dict[str, Any]] = None
        if goal_distribution:
            score_matrix = self._stringify_score_matrix(goal_distribution.get("score_matrix") or {})
            expected_goals = {k: v for k, v in goal_distribution.items() if k != "score_matrix"}
        else:
            warnings.append("expected_goals_not_available")

        # 7: Odds Movement (sola visualizzazione, mai una feature di training)
        try:
            odds_movement = self.odds_snapshot_repo.opening_latest_closing(fixture_id=fixture_id)
        except Exception:
            odds_movement = []
        if not odds_movement:
            warnings.append("odds_movement_not_available")

        # 8: Model Consensus per mercato
        model_consensus = self._model_consensus_by_market(fixture_id=fixture_id, model_markets=model_markets)
        if not model_consensus:
            warnings.append("model_consensus_not_available")

        return {
            "fixture_id": fixture_id,
            "overview": {
                "fixture": fixture_row,
                "timeline": detail.get("timeline") or [],
                "odds_summary": detail.get("odds_summary") or {},
                "bookmaker_baseline": detail.get("bookmaker_baseline") or {},
                "odds_updated_at": detail.get("odds_updated_at"),
            },
            "probabilities": decision_cards,
            "value_bets": value_bets,
            "team_strength": team_strength,
            "expected_goals": expected_goals,
            "score_matrix": score_matrix,
            "odds_movement": odds_movement,
            "model_consensus": model_consensus,
            "model_markets": model_markets,
            "warnings": warnings,
        }

