"""Team Strength Expert (EXP-01).

Calcola rating point-in-time (nessun leakage temporale) di:
- rating offensivo/difensivo per squadra (expanding average con shrinkage bayesiana),
- home advantage (boost storico della squadra quando gioca in casa),
- rolling form (ultime N partite, solo passato),
- output numerici versionati, riusabili da altri modelli/esperti.

Design:
- Le partite vengono ordinate cronologicamente (prediction_at, id_fixture).
- Per ogni partita viene prima emesso lo snapshot "PRE-match" delle due squadre
  (cioe' lo stato calcolato usando SOLO le partite precedenti) e SOLO DOPO lo
  stato viene aggiornato con il risultato della partita corrente. Questo
  garantisce l'assenza di leakage temporale per costruzione.
- Le squadre senza storico usano un prior di lega (shrinkage bayesiana verso
  `PRIOR_GOALS`), cosi' il cold-start e' deterministico e non introduce NaN.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.ml.datasets.point_in_time_builder import FINAL_STATUSES
from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

EXPERT_NAME = "team_strength"
EXPERT_CONFIG_VERSION = 1

PRIOR_GOALS = 1.35  # media storica goal/squadra/partita usata come prior di lega
SHRINKAGE_K = 5.0   # pseudo-partite di shrinkage verso il prior (piu' alto = piu' prudente)
FORM_WINDOW = 5      # numero di partite passate usate per la rolling form


@dataclass
class _TeamState:
    """Stato incrementale (point-in-time) di una singola squadra."""

    matches_played: int = 0
    sum_goals_for: float = 0.0
    sum_goals_against: float = 0.0

    home_matches_played: int = 0
    sum_home_goals_for: float = 0.0
    sum_home_goals_against: float = 0.0

    away_matches_played: int = 0
    sum_away_goals_for: float = 0.0
    sum_away_goals_against: float = 0.0

    recent_results: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    recent_goal_diff: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))

    @staticmethod
    def _shrunk(total: float, count: int, prior: float = PRIOR_GOALS, k: float = SHRINKAGE_K) -> float:
        return (total + k * prior) / (count + k)

    @property
    def attack_rating(self) -> float:
        return self._shrunk(self.sum_goals_for, self.matches_played)

    @property
    def defense_rating(self) -> float:
        return self._shrunk(self.sum_goals_against, self.matches_played)

    @property
    def home_attack_rating(self) -> float:
        return self._shrunk(self.sum_home_goals_for, self.home_matches_played)

    @property
    def home_defense_rating(self) -> float:
        return self._shrunk(self.sum_home_goals_against, self.home_matches_played)

    @property
    def away_attack_rating(self) -> float:
        return self._shrunk(self.sum_away_goals_for, self.away_matches_played)

    @property
    def away_defense_rating(self) -> float:
        return self._shrunk(self.sum_away_goals_against, self.away_matches_played)

    @property
    def home_advantage(self) -> float:
        """Boost storico (attacco + difesa) della squadra quando gioca in casa."""
        attack_boost = self.home_attack_rating - self.away_attack_rating
        defense_boost = self.away_defense_rating - self.home_defense_rating
        return attack_boost + defense_boost

    @property
    def rolling_form(self) -> float:
        """Media punti (W=3,D=1,L=0) normalizzata in [0,1] sulle ultime FORM_WINDOW partite."""
        if not self.recent_results:
            return 0.5
        return float(sum(self.recent_results) / (3.0 * len(self.recent_results)))

    @property
    def rolling_goal_diff(self) -> float:
        if not self.recent_goal_diff:
            return 0.0
        return float(sum(self.recent_goal_diff) / len(self.recent_goal_diff))

    def snapshot(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}_attack_rating": round(self.attack_rating, 4),
            f"{prefix}_defense_rating": round(self.defense_rating, 4),
            f"{prefix}_home_attack_rating": round(self.home_attack_rating, 4),
            f"{prefix}_home_defense_rating": round(self.home_defense_rating, 4),
            f"{prefix}_away_attack_rating": round(self.away_attack_rating, 4),
            f"{prefix}_away_defense_rating": round(self.away_defense_rating, 4),
            f"{prefix}_home_advantage": round(self.home_advantage, 4),
            f"{prefix}_rolling_form": round(self.rolling_form, 4),
            f"{prefix}_rolling_goal_diff": round(self.rolling_goal_diff, 4),
            f"{prefix}_matches_played": float(self.matches_played),
        }

    def update(self, goals_for: int, goals_against: int, is_home: bool) -> None:
        self.matches_played += 1
        self.sum_goals_for += goals_for
        self.sum_goals_against += goals_against

        if is_home:
            self.home_matches_played += 1
            self.sum_home_goals_for += goals_for
            self.sum_home_goals_against += goals_against
        else:
            self.away_matches_played += 1
            self.sum_away_goals_for += goals_for
            self.sum_away_goals_against += goals_against

        if goals_for > goals_against:
            points = 3
        elif goals_for < goals_against:
            points = 0
        else:
            points = 1
        self.recent_results.append(points)
        self.recent_goal_diff.append(goals_for - goals_against)


class TeamStrengthExpert:
    """Espone rating point-in-time riusabili come feature da altri modelli/esperti."""

    VERSION: str = ""  # valorizzato sotto la classe

    def __init__(self, match_repo: Optional[MatchRepository] = None):
        self.match_repo = match_repo or MatchRepository()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _extract_scores(match: dict[str, Any]) -> Optional[tuple[int, int]]:
        stats = match.get("statistics") or []
        id_home = match.get("id_team_home")
        id_away = match.get("id_team_away")
        stat_home = next((s for s in stats if s.get("statistics_team_id") == id_home), None)
        stat_away = next((s for s in stats if s.get("statistics_team_id") == id_away), None)
        if not stat_home or not stat_away:
            return None

        home_ft = stat_home.get("score_ft")
        away_ft = stat_away.get("score_ft")
        if home_ft is None or away_ft is None:
            return None
        return int(home_ft), int(away_ft)

    @classmethod
    def _sorted_matches(cls, matches: list[dict[str, Any]]) -> list[tuple[datetime, dict[str, Any]]]:
        parsed: list[tuple[datetime, dict[str, Any]]] = []
        for match in matches:
            status = str(match.get("status") or "").upper()
            if status and status not in FINAL_STATUSES:
                continue
            prediction_at = cls._parse_datetime(match.get("date_match"))
            if prediction_at is None:
                continue
            parsed.append((prediction_at, match))

        parsed.sort(key=lambda item: (item[0], item[1].get("id_fixture") or 0))
        return parsed

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------
    def build_ratings_dataset(self, matches: list[dict[str, Any]]) -> pd.DataFrame:
        """Ritorna un frame con 1 riga per fixture: rating PRE-match delle 2 squadre.

        Nessun leakage: lo stato usato per la riga N e' calcolato usando
        esclusivamente le partite con indice < N nell'ordinamento cronologico.
        Le colonne prefissate con `_actual_` servono solo per backtest/valutazione
        e NON vanno usate come feature di training (sono il target osservato).
        """
        states: dict[int, _TeamState] = {}
        rows: list[dict[str, Any]] = []

        for prediction_at, match in self._sorted_matches(matches):
            scores = self._extract_scores(match)
            if scores is None:
                continue
            home_goals, away_goals = scores

            home_id = match.get("id_team_home")
            away_id = match.get("id_team_away")
            if home_id is None or away_id is None:
                continue
            home_id = int(home_id)
            away_id = int(away_id)

            home_state = states.setdefault(home_id, _TeamState())
            away_state = states.setdefault(away_id, _TeamState())

            row: dict[str, Any] = {
                "id_fixture": match.get("id_fixture"),
                "season": match.get("season"),
                "prediction_at": prediction_at.isoformat(),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "rating_version": self.VERSION,
            }
            row.update(home_state.snapshot("home_team"))
            row.update(away_state.snapshot("away_team"))
            row["attack_rating_diff"] = round(home_state.attack_rating - away_state.attack_rating, 4)
            row["defense_rating_diff"] = round(away_state.defense_rating - home_state.defense_rating, 4)
            row["rolling_form_diff"] = round(home_state.rolling_form - away_state.rolling_form, 4)

            # Colonne di valutazione (non feature): risultato realmente osservato.
            row["_actual_home_win"] = int(home_goals > away_goals)
            row["_actual_goal_diff"] = int(home_goals - away_goals)

            rows.append(row)

            # Aggiorna lo stato SOLO dopo aver emesso la riga -> point-in-time garantito.
            home_state.update(goals_for=home_goals, goals_against=away_goals, is_home=True)
            away_state.update(goals_for=away_goals, goals_against=home_goals, is_home=False)

        return pd.DataFrame(rows)

    def current_ratings(self, matches: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
        """Stato aggiornato (post ultima partita disponibile) per fixture future/live."""
        states: dict[int, _TeamState] = {}
        for _, match in self._sorted_matches(matches):
            scores = self._extract_scores(match)
            if scores is None:
                continue
            home_goals, away_goals = scores

            home_id = match.get("id_team_home")
            away_id = match.get("id_team_away")
            if home_id is None or away_id is None:
                continue
            home_id = int(home_id)
            away_id = int(away_id)

            home_state = states.setdefault(home_id, _TeamState())
            away_state = states.setdefault(away_id, _TeamState())
            home_state.update(goals_for=home_goals, goals_against=away_goals, is_home=True)
            away_state.update(goals_for=away_goals, goals_against=home_goals, is_home=False)

        return {
            team_id: {**state.snapshot("team"), "rating_version": self.VERSION}
            for team_id, state in states.items()
        }

    def build_from_db(self, seasons: Optional[list[int]] = None) -> pd.DataFrame:
        filters: dict[str, Any] = {"statistics": "not None", "status": list(FINAL_STATUSES)}
        if seasons:
            filters["season"] = seasons

        orm_matches = self.match_repo.search_filter(filters=filters)
        matches = convert_orm_match_to_dict(orm_matches)
        return self.build_ratings_dataset(matches)

    # ------------------------------------------------------------------
    # Backtest base (acceptance criteria EXP-01)
    # ------------------------------------------------------------------
    @staticmethod
    def backtest_home_signal(frame: pd.DataFrame, min_matches_played: int = 3) -> dict[str, Any]:
        """Backtest minimale: il segnale rating batte la baseline 'home win rate'?

        Segnale = attack_rating_diff + defense_rating_diff + home_team_home_advantage.
        Regola naive: se segnale > 0 -> predici vittoria interna.
        """
        if frame.empty or "_actual_home_win" not in frame.columns:
            return {"n": 0, "accuracy": None, "baseline_home_win_rate": None, "edge": None}

        eligible = frame[
            (frame["home_team_matches_played"] >= min_matches_played)
            & (frame["away_team_matches_played"] >= min_matches_played)
        ]
        if eligible.empty:
            return {"n": 0, "accuracy": None, "baseline_home_win_rate": None, "edge": None}

        signal = (
            eligible["attack_rating_diff"]
            + eligible["defense_rating_diff"]
            + eligible["home_team_home_advantage"]
        )
        predicted_home_win = (signal > 0).astype(int)
        actual_home_win = eligible["_actual_home_win"].astype(int)

        accuracy = float((predicted_home_win == actual_home_win).mean())
        baseline = float(actual_home_win.mean())

        return {
            "n": int(len(eligible)),
            "accuracy": accuracy,
            "baseline_home_win_rate": baseline,
            "edge": float(accuracy - baseline),
        }

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------
    @staticmethod
    def _build_version() -> str:
        config = {
            "expert": EXPERT_NAME,
            "config_version": EXPERT_CONFIG_VERSION,
            "prior_goals": PRIOR_GOALS,
            "shrinkage_k": SHRINKAGE_K,
            "form_window": FORM_WINDOW,
        }
        serialized = json.dumps(config, sort_keys=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


TeamStrengthExpert.VERSION = TeamStrengthExpert._build_version()
