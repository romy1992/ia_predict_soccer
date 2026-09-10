from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import os
import re
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
from dateutil.parser import isoparse
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import selectinload

from src.ml.baselines.bookmaker_baseline import build_fixture_baseline, get_market_outcome_baseline
from src.ml.markets.totals.totals_market import enforce_monotonic_over_probabilities
from src.ml.serving.prediction_snapshot_service import PredictionSnapshotService
from src.jobs.api_quota_state import is_quota_exhausted_today
from src.oracle.decision_engine.decision_policy import evaluate_decision_from_fair_odds_outcome
from src.oracle.decision_engine.over_signal_policy import evaluate_over_signal
from src.oracle.fair_odds.fair_odds_engine import build_fair_odds_outcome
from src.repository.base.repository_db import SessionLocal
from src.repository.prediction_ledger_repository import PredictionLedgerRepository
from src.service_ia.config.app_config import load_app_config
from src.service_ia.model.match import Match
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.request_api import base_api_statistics

FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"}

# MATCH-01: priorita' per scegliere la decision card "migliore" da mostrare
# come badge sintetico in tabella (Match Center) tra quelle gia' calcolate
# da `_build_decision_cards` — nessuna nuova logica di decisione, solo una
# selezione tra righe gia' pronte.
_DECISION_LABEL_PRIORITY = {"PLAY": 0, "BORDERLINE": 1, "NO BET": 2}

# Coerenza monotona tra le 4 soglie Under/Over (2026-09-09, collegata al
# serving su richiesta esplicita dell'operatore dopo la verifica offline di
# questa sessione: violazioni pre-proiezione solo 0.3% sui champion reali,
# effetto sul selection_score sempre leggermente positivo o nullo, mai
# negativo - vedi `apply_monotonic_to_champions.py`). Vedi `_apply_monotonic_projection`.
UNDER_OVER_MARKETS = ["under_over_1_5", "under_over_2_5", "under_over_3_5", "under_over_4_5"]

# Finestra oraria per interrogare l'API-Sports da Dashboard per le partite
# ODIERNE (risparmio quota, 2026-09-09, richiesto esplicitamente
# dall'operatore): fuori da questa finestra (00:30-12:30, quando
# tipicamente non ci sono partite in corso ne' imminenti) le fetch passive
# per "oggi" (incluso l'auto-poll ogni 60s del frontend) restano bloccate,
# si mostra solo cio' che e' gia' a DB - stesso principio del guard
# `is_quota_exhausted_today`, gia' esistente. Orario LOCALE Europe/Rome
# (stesso fuso gia' usato da `src/jobs/scheduler.py` per i cron): la
# logica interna del progetto resta in UTC (confronti storico/oggi,
# timestamp DB) per non introdurre ambiguita' sul cambio ora legale -
# SOLO questa decisione "a che ora del giorno siamo" usa l'orario locale,
# perche' e' l'unico punto in cui conta davvero (partite di calcio si
# giocano in orario locale italiano, non UTC).
_DASHBOARD_TIMEZONE = ZoneInfo("Europe/Rome")
_DASHBOARD_API_WINDOW_START = time(12, 30)
_DASHBOARD_API_WINDOW_END = time(0, 30)


@dataclass
class DashboardDayData:
    date: str
    total: int
    returned: int
    model_markets: list[str]
    rows: list[dict[str, Any]]


class DashboardService:
    _api_cache: dict[str, tuple[datetime, Any]] = {}
    _api_cache_ttl_seconds = 60
    # Fix performance (2026-09-09): _fetch_api_day_fixtures interrogava i
    # campionati configurati (18 di default, APP_LEAGUES) UNO ALLA VOLTA -
    # con quote/timeout/retry di rete per ciascuno, il tempo totale scalava
    # linearmente col numero di campionati (osservato: risposta molto lenta
    # ogni volta che la cache 60s scade, tipicamente sulla data odierna).
    # Un pool limitato (non illimitato, per non bombardare l'API-Sports e
    # restare dentro ai rate limit del provider) taglia il tempo totale a
    # circa quello della chiamata piu' lenta invece che alla somma di tutte.
    _LEAGUE_FETCH_MAX_WORKERS = 6

    def __init__(self):
        self.registry = ModelRegistry()
        self.cfg = load_app_config()
        self.ledger_repo = PredictionLedgerRepository()
        self._snapshot_service = PredictionSnapshotService()
        self._prediction_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return isoparse(value)
        except Exception:
            return None

    @staticmethod
    def _format_time(dt_value: Optional[datetime]) -> str:
        if not dt_value:
            return "--:--"
        return dt_value.astimezone(timezone.utc).strftime("%H:%M UTC") if dt_value.tzinfo else dt_value.strftime("%H:%M")

    @staticmethod
    def _format_date(dt_value: Optional[datetime]) -> str:
        if not dt_value:
            return ""
        return dt_value.date().isoformat()

    @staticmethod
    def _classify_phase(status: Optional[str], dt_value: Optional[datetime]) -> str:
        status = (status or "").upper()
        if status in FINAL_STATUSES:
            return "finished"
        if status in LIVE_STATUSES:
            return "live"

        now_utc = datetime.now(timezone.utc)
        if dt_value and dt_value.tzinfo:
            return "to_play" if dt_value > now_utc else "live"
        if status == "NS":
            return "to_play"
        return "live"

    @staticmethod
    def _normalize_market_request(markets: Optional[list[str]]) -> Optional[list[str]]:
        if not markets:
            return None
        normalized = [m.strip() for m in markets if m and m.strip()]
        if not normalized:
            return None
        allowed = FilterMarketService.SUPPORTED_MARKETS
        return [m for m in normalized if m in allowed]

    @classmethod
    def _cache_get(cls, key: str) -> Optional[Any]:
        item = cls._api_cache.get(key)
        if not item:
            return None

        ts, payload = item
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > cls._api_cache_ttl_seconds:
            return None
        return payload

    @classmethod
    def _cache_set(cls, key: str, payload: Any) -> None:
        cls._api_cache[key] = (datetime.now(timezone.utc), payload)

    @staticmethod
    def _today_in_dashboard_timezone() -> date:
        return datetime.now(_DASHBOARD_TIMEZONE).date()

    @staticmethod
    def _is_within_dashboard_api_window(now_local: Optional[time] = None) -> bool:
        """Finestra 12:30->00:30 (Europe/Rome, attraversa la mezzanotte):
        dentro se l'ora locale e' >= 12:30 OPPURE < 00:30 - un range che
        attraversa la mezzanotte non e' confrontabile con un semplice
        start<=now<=end. `now_local` esplicito solo per i test - a runtime
        e' sempre l'ora corrente."""
        if now_local is None:
            now_local = datetime.now(_DASHBOARD_TIMEZONE).time()
        return now_local >= _DASHBOARD_API_WINDOW_START or now_local < _DASHBOARD_API_WINDOW_END

    @staticmethod
    def _fixture_id_from_api(fixture: dict[str, Any]) -> Optional[int]:
        fix = fixture.get("fixture") or {}
        fixture_id = fix.get("id")
        if fixture_id is None:
            return None
        try:
            return int(fixture_id)
        except Exception:
            return None

    @staticmethod
    def _dedupe_api_fixtures(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
        container: dict[int, dict[str, Any]] = {}
        for fixture in fixtures:
            fixture_id = DashboardService._fixture_id_from_api(fixture)
            if fixture_id is None:
                continue
            container[fixture_id] = fixture
        return list(container.values())

    def _season_for_date(self, target_date: date) -> int:
        seasons = list(self.cfg.seasons or [])
        if not seasons:
            return target_date.year
        if target_date.year in seasons:
            return target_date.year
        lower_or_equal = [s for s in seasons if s <= target_date.year]
        return max(lower_or_equal) if lower_or_equal else max(seasons)

    def _fetch_api_day_fixtures(self, target_date: date, force_refresh: bool = False) -> list[dict[str, Any]]:
        cache_key = f"day:{target_date.isoformat()}"
        if not force_refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                return cached
        # Fix (2026-09-07): con la quota GIA' segnalata esaurita OGGI (check
        # autoritativo, mai la sola stima passiva - vedi
        # `is_quota_exhausted_today`), non ha senso tentare comunque la
        # chiamata: fallirebbe e basta, sprecando tempo/rete e rischiando
        # ulteriori penalita' dal provider. Protegge OGNI punto di ingresso
        # (polling frontend ogni 60s, job schedulati, richieste manuali,
        # incluso il bottone "Forza aggiornamento" - `force_refresh` bypassa
        # SOLO la cache/lo skip DB-first, MAI questo guard) in un colpo
        # solo, indipendentemente da chi chiama questo metodo.
        if is_quota_exhausted_today():
            return []

        # Risparmio quota (2026-09-09): per la data ODIERNA (orario locale
        # Europe/Rome, non UTC - le partite si giocano in orario locale),
        # fuori dalla finestra 12:30-00:30 non c'e' motivo di interrogare
        # l'API esterna (nessuna partita in corso/imminente) - si mostra
        # solo cio' che e' gia' a DB. Bypassabile con `force_refresh`
        # (azione esplicita dell'operatore, non l'auto-poll passivo del
        # frontend), stesso principio dello skip DB-first per le date
        # storiche. Le altre date (storiche/future) non sono toccate da
        # questo guard: restano governate solo da `skip_api` in
        # `get_day_matches`.
        if not force_refresh and target_date == self._today_in_dashboard_timezone() and not self._is_within_dashboard_api_window():
            return []

        season = self._season_for_date(target_date)
        leagues = self.cfg.leagues or []

        def _fetch_one_league(league: int) -> list[dict[str, Any]]:
            try:
                return base_api_statistics(
                    path="fixtures",
                    params={"date": target_date.isoformat(), "league": league, "season": season},
                ) or []
            except Exception:
                return []

        fixtures: list[dict[str, Any]] = []
        if leagues:
            with ThreadPoolExecutor(max_workers=min(self._LEAGUE_FETCH_MAX_WORKERS, len(leagues))) as executor:
                for payload in executor.map(_fetch_one_league, leagues):
                    if payload:
                        fixtures.extend(payload)

        deduped = self._dedupe_api_fixtures(fixtures)
        self._cache_set(cache_key, deduped)
        return deduped

    def _fetch_api_live_fixtures(self) -> list[dict[str, Any]]:
        cache_key = "live"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if is_quota_exhausted_today():
            return []
        # Risparmio quota (2026-09-09): le partite live sono per definizione
        # "adesso" (nessun target_date da confrontare) - fuori dalla
        # finestra 12:30-00:30 (Europe/Rome) non ha senso interrogare
        # l'API esterna, nessun bottone la bypassa qui (nessun
        # force_refresh dedicato per il live).
        if not self._is_within_dashboard_api_window():
            return []

        fixtures: list[dict[str, Any]] = []
        for league in self.cfg.leagues or []:
            try:
                payload = base_api_statistics(path="fixtures", params={"live": "all", "league": league})
            except Exception:
                payload = []
            if payload:
                fixtures.extend(payload)

        deduped = self._dedupe_api_fixtures(fixtures)
        self._cache_set(cache_key, deduped)
        return deduped

    def _fetch_api_fixture_detail(self, fixture_id: int) -> Optional[dict[str, Any]]:
        cache_key = f"fixture:{fixture_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if is_quota_exhausted_today():
            return None

        try:
            payload = base_api_statistics(path="fixtures", params={"id": fixture_id})
        except Exception:
            payload = []

        item = payload[0] if payload else None
        self._cache_set(cache_key, item)
        return item

    def _fetch_api_events(self, fixture_id: int) -> list[dict[str, Any]]:
        cache_key = f"events:{fixture_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if is_quota_exhausted_today():
            return []

        try:
            payload = base_api_statistics(path="fixtures/events", params={"fixture": fixture_id})
        except Exception:
            payload = []

        events = payload if isinstance(payload, list) else []
        self._cache_set(cache_key, events)
        return events

    def _fetch_api_odds(self, fixture_id: int) -> Optional[dict[str, Any]]:
        cache_key = f"odds:{fixture_id}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        if is_quota_exhausted_today():
            return None

        try:
            payload = base_api_statistics(path="odds", params={"fixture": fixture_id})
        except Exception:
            payload = []

        item = payload[0] if payload else None
        self._cache_set(cache_key, item)
        return item

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            raw = value.strip().replace(",", ".")
            try:
                return float(raw)
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())

    def _match_market_from_bet_value(self, bet_name: str, value: str) -> Optional[str]:
        bet_norm = self._normalize_text(bet_name)

        if bet_norm == "matchwinner":
            return "h2h"
        if bet_norm == "bothteamsscore":
            return "goal_no_goal"
        if bet_norm == "doublechance":
            return "dc"
        if bet_norm == "cornersoverunder":
            return "corners"
        if bet_norm == "cardsoverunder":
            return "cards"
        if bet_norm == "goalsoverunder":
            point = self._extract_line_point(value)
            if point is None:
                return None
            if abs(point - 1.5) < 0.0001:
                return "under_over_1_5"
            if abs(point - 2.5) < 0.0001:
                return "under_over_2_5"
            if abs(point - 3.5) < 0.0001:
                return "under_over_3_5"
            if abs(point - 4.5) < 0.0001:
                return "under_over_4_5"
        return None

    @staticmethod
    def _normalize_outcome_value(value: str) -> str:
        val = (value or "").strip()
        mapping = {
            "home/draw": "Home/Draw",
            "draw/away": "Draw/Away",
            "home/away": "Home/Away",
            "yes": "Yes",
            "no": "No",
        }
        key = val.lower()
        if key in mapping:
            return mapping[key]
        if val.lower() == "home":
            return "Home"
        if val.lower() == "away":
            return "Away"
        if val.lower() == "draw":
            return "Draw"
        return val

    @staticmethod
    def _extract_line_point(value: str) -> Optional[float]:
        matches = re.findall(r"([0-9]+(?:\.[0-9]+)?)", value or "")
        if not matches:
            return None
        try:
            return float(matches[-1])
        except Exception:
            return None

    def _aggregate_odds_from_api(self, odds_payload: Optional[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not odds_payload:
            return {}

        aggregates: dict[str, dict[str, list[float]]] = {}
        bookmakers = odds_payload.get("bookmakers") or []
        for bookmaker in bookmakers:
            bets = bookmaker.get("bets") or []
            for bet in bets:
                bet_name = bet.get("name") or ""
                values = bet.get("values") or []
                for item in values:
                    raw_outcome = str(item.get("value") or "")
                    market = self._match_market_from_bet_value(bet_name, raw_outcome)
                    if not market:
                        continue

                    odd = self._to_float(item.get("odd"))
                    if odd is None or odd <= 0:
                        continue

                    outcome = self._normalize_outcome_value(raw_outcome)
                    market_dict = aggregates.setdefault(market, {})
                    market_dict.setdefault(outcome, []).append(odd)

        summary: dict[str, list[dict[str, Any]]] = {}
        for market, outcomes in aggregates.items():
            rows: list[dict[str, Any]] = []
            for outcome, odds in outcomes.items():
                if not odds:
                    continue
                rows.append(
                    {
                        "outcome": outcome,
                        "avg_odd": float(np.mean(odds)),
                        "min_odd": float(np.min(odds)),
                        "max_odd": float(np.max(odds)),
                        "bookmakers": len(odds),
                    }
                )
            rows.sort(key=lambda x: x["outcome"])
            summary[market] = rows

        return summary

    def _aggregate_odds_from_db(self, match: Optional[Match]) -> dict[str, list[dict[str, Any]]]:
        if not match or not match.odds:
            return {}

        odds_obj = match.odds[0].to_dict()
        summary: dict[str, list[dict[str, Any]]] = {}
        for market in FilterMarketService.SUPPORTED_MARKETS:
            market_values = odds_obj.get(market)
            if not isinstance(market_values, dict):
                continue

            outcome_buckets: dict[str, list[float]] = {}
            for key, raw_odd in market_values.items():
                odd = self._to_float(raw_odd)
                if odd is None or odd <= 0:
                    continue
                outcome, _, _book = key.rpartition("_")
                outcome = outcome or key
                outcome = self._normalize_outcome_value(outcome)
                outcome_buckets.setdefault(outcome, []).append(odd)

            rows: list[dict[str, Any]] = []
            for outcome, odds in outcome_buckets.items():
                rows.append(
                    {
                        "outcome": outcome,
                        "avg_odd": float(np.mean(odds)),
                        "min_odd": float(np.min(odds)),
                        "max_odd": float(np.max(odds)),
                        "bookmakers": len(odds),
                    }
                )

            rows.sort(key=lambda x: x["outcome"])
            if rows:
                summary[market] = rows

        return summary

    @staticmethod
    def _row_to_outcome_maps(rows: list[dict[str, Any]]) -> dict[str, float]:
        return {
            DashboardService._normalize_text(str(row.get("outcome") or "")): float(row.get("avg_odd") or 0)
            for row in rows
            if row.get("outcome") is not None
        }

    def _pick_and_odd_for_prediction(
        self,
        market: str,
        prediction: int,
        row_context: dict[str, Any],
        odds_summary: dict[str, list[dict[str, Any]]],
    ) -> tuple[str, Optional[float]]:
        rows = odds_summary.get(market) or []
        normalized = self._row_to_outcome_maps(rows)

        if market.startswith("under_over_"):
            threshold = market.replace("under_over_", "").replace("_", ".")
            pick = f"Over {threshold}" if prediction == 1 else f"Under {threshold}"
            odd = normalized.get(self._normalize_text(pick))
            return pick, odd

        if market == "goal_no_goal":
            pick = "Yes" if prediction == 1 else "No"
            odd = normalized.get(self._normalize_text(pick))
            return ("Goal" if pick == "Yes" else "No Goal"), odd

        if market == "dc":
            pick = "Home/Draw" if prediction == 1 else "Draw/Away"
            odd = normalized.get(self._normalize_text(pick))
            return pick, odd

        if market == "h2h":
            if prediction == 1:
                pick = row_context.get("home") or "Home"
                odd = normalized.get(self._normalize_text("Home"))
            else:
                # BET-02 ("Usare outcome corretto"): NON usare la quota
                # "Draw" come fallback per il pick "Away" — sono due
                # outcome diversi, mixarli produrrebbe un edge/EV calcolato
                # sulla quota sbagliata. Se manca la quota "Away", l'odd
                # resta None (gestito esplicitamente dal Value Engine).
                pick = row_context.get("away") or "Away"
                odd = normalized.get(self._normalize_text("Away"))
            return pick, odd

        if market in {"corners", "cards"}:
            target = 9.5 if market == "corners" else 4.5
            wanted_prefix = "over" if prediction == 1 else "under"
            candidates = []
            for r in rows:
                outcome = str(r.get("outcome") or "")
                norm = self._normalize_text(outcome)
                if not norm.startswith(wanted_prefix):
                    continue
                point = self._extract_line_point(outcome)
                if point is None:
                    continue
                candidates.append((abs(point - target), outcome, float(r.get("avg_odd") or 0)))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                _, outcome, odd = candidates[0]
                return outcome, odd

            fallback = "Over" if prediction == 1 else "Under"
            return fallback, None

        return str(prediction), None

    @staticmethod
    def _baseline_outcome_for_prediction(market: str, prediction: int, pick_label: str) -> str:
        if market == "h2h":
            return "Home" if prediction == 1 else "Away"
        if market == "goal_no_goal":
            return "Yes" if prediction == 1 else "No"
        if market == "dc":
            return "Home/Draw" if prediction == 1 else "Draw/Away"
        if market.startswith("under_over_"):
            threshold = market.replace("under_over_", "").replace("_", ".")
            return f"Over {threshold}" if prediction == 1 else f"Under {threshold}"
        return pick_label

    def _build_decision_cards(
        self,
        row_context: dict[str, Any],
        predictions: dict[str, Any],
        odds_summary: dict[str, list[dict[str, Any]]],
        bookmaker_baseline: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for market, payload in predictions.items():
            prediction = int(payload.get("prediction", 0))
            class1_probability = float(payload.get("probability", 0.5))
            predicted_probability = class1_probability if prediction == 1 else (1.0 - class1_probability)

            pick, odd = self._pick_and_odd_for_prediction(
                market=market,
                prediction=prediction,
                row_context=row_context,
                odds_summary=odds_summary,
            )

            baseline_outcome = self._baseline_outcome_for_prediction(
                market=market,
                prediction=prediction,
                pick_label=pick,
            )
            baseline_row = get_market_outcome_baseline(
                fixture_baseline=bookmaker_baseline or {},
                market=market,
                outcome=baseline_outcome,
            )
            market_baseline = ((bookmaker_baseline or {}).get("markets") or {}).get(market) or {}

            # BET-01 (Fair Odds Engine): p_market_raw/p_market_fair/fair_odd
            # standard per questo outcome, poi BET-04 (Decision Policy
            # versionata, che riusa compute_prob_edge/EV di BET-02) valuta
            # prob_edge/EV/decisione sullo STESSO outcome (mai un mix quota
            # di un outcome diverso, vedi fix in _pick_and_odd_for_prediction)
            # con soglie che possono variare per mercato/outcome, MAI
            # hardcoded qui (acceptance criteria BET-04).
            fair_odds_outcome = build_fair_odds_outcome(
                market=market,
                outcome=baseline_outcome,
                market_baseline_row=baseline_row,
                p_model=predicted_probability,
            )
            decision = evaluate_decision_from_fair_odds_outcome(fair_odds_outcome)

            # Segnale "Bet Over" (richiesto esplicitamente dall'operatore,
            # 2026-09-08): INDIPENDENTE dal pick sopra e da `decision` - non
            # ne altera il valore, aggiunge solo un campo. Vedi
            # `over_signal_policy.py` per il perche' non e' semplicemente
            # `prediction == 1`. `class1_probability` e' gia' P(Over) per
            # costruzione sui mercati under_over_* (classe 1 = over_X_5,
            # vedi `DIRECT_MARKET_SPECS`), quindi nessuna trasformazione
            # aggiuntiva necessaria qui.
            bet_over_signal = (
                evaluate_over_signal(market=market, p_over=class1_probability)
                if market.startswith("under_over_")
                else None
            )

            cards.append(
                {
                    "market": market,
                    "pick": pick,
                    "prediction": prediction,
                    "model_name": payload.get("model_name"),
                    "run_id": payload.get("run_id"),
                    "class_1_probability": class1_probability,
                    "predicted_probability": predicted_probability,
                    "bet_over_signal": bet_over_signal,
                    "odd": odd,
                    "bookmaker_implied_raw": fair_odds_outcome.p_market_raw,
                    "bookmaker_fair_probability": fair_odds_outcome.p_market_fair,
                    # MATCH-01 ("fair market"): quota equivalente alla fair
                    # probability del bookmaker, gia' calcolata da BET-01
                    # (`fair_odd = 1/p_market_fair`) e finora NON esposta qui.
                    "fair_odd": fair_odds_outcome.fair_odd,
                    "bookmaker_overround": market_baseline.get("overround"),
                    "bookmakers_count": fair_odds_outcome.bookmakers,
                    "model_minus_fair": decision.prob_edge,
                    "edge": decision.prob_edge,
                    "ev": decision.ev,
                    "value_label": decision.decision,
                    "value_reason": decision.reason,
                    "policy_version": decision.policy_version,
                }
            )

        cards.sort(key=lambda x: x.get("predicted_probability", 0), reverse=True)
        return cards

    @staticmethod
    def _select_best_decision_card(cards: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """MATCH-01: sintesi per la vista lista (Match Center) tra le
        decision_cards gia' calcolate da `_build_decision_cards` (mai un
        nuovo calcolo di edge/EV/decisione): sceglie quella con priorita'
        PLAY > BORDERLINE > NO BET e, a parita', la probabilita' predetta
        piu' alta. `None` se non ci sono card (nessun modello/quota)."""
        if not cards:
            return None
        return min(
            cards,
            key=lambda c: (
                _DECISION_LABEL_PRIORITY.get(c.get("value_label"), 99),
                -float(c.get("predicted_probability") or 0.0),
            ),
        )

    def _decisions_for_row(
        self,
        row: dict[str, Any],
        predictions: dict[str, Any],
        db_match: Optional[Match],
    ) -> tuple[list[dict[str, Any]], Optional[dict[str, Any]]]:
        """MATCH-01: badge decision/quote/edge/EV anche per le righe di
        LISTA (Match Center), non solo nel dettaglio match. Riusa
        ESATTAMENTE `_build_decision_cards` (mai una logica duplicata),
        calcolato SOLO quando le quote sono gia' disponibili SENZA fetch
        aggiuntive: da `db_match.odds` (relazione ORM gia' caricata da
        `_fetch_matches` con `selectinload`, la stessa query unica gia'
        eseguita per popolare la lista - fix performance esistente). MAI
        una chiamata odds API-Sports per riga: la quota giornaliera e'
        limitata (vedi job history) e centinaia di righe la esaurirebbero
        subito. Se la fixture non e' ancora nel DB locale, resta
        `([], None)`: aprendo il dettaglio (`get_match_detail`, che gia'
        fa una fetch odds dedicata per singola fixture) il badge completo
        resta comunque disponibile."""
        if not predictions or db_match is None:
            return [], None

        odds_summary = self._aggregate_odds_from_db(db_match)
        if not odds_summary:
            return [], None

        bookmaker_baseline = build_fixture_baseline(odds_summary)
        cards = self._build_decision_cards(
            row_context=row,
            predictions=predictions,
            odds_summary=odds_summary,
            bookmaker_baseline=bookmaker_baseline,
        )
        return cards, self._select_best_decision_card(cards)

    @staticmethod
    def _serialize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        for event in events:
            time_data = event.get("time") or {}
            elapsed = time_data.get("elapsed")
            extra = time_data.get("extra")

            minute = "-"
            if elapsed is not None:
                minute = f"{elapsed}'"
                if extra is not None:
                    minute = f"{elapsed}+{extra}'"

            rows.append(
                {
                    "minute": minute,
                    "elapsed": elapsed,
                    "team": (event.get("team") or {}).get("name"),
                    "type": event.get("type"),
                    "detail": event.get("detail"),
                    "player": (event.get("player") or {}).get("name"),
                    "assist": (event.get("assist") or {}).get("name"),
                    "comments": event.get("comments"),
                }
            )

        rows.sort(key=lambda x: (x.get("elapsed") is None, x.get("elapsed") or 0))
        return rows

    @staticmethod
    def _extract_scores(match: Match) -> dict[str, Optional[int]]:
        scores = {"home": None, "away": None}
        stats = match.statistics or []
        if not stats:
            return scores

        by_team = {s.statistics_team_id: s for s in stats}
        home_stat = by_team.get(match.id_team_home)
        away_stat = by_team.get(match.id_team_away)

        if home_stat:
            scores["home"] = home_stat.score_ft if home_stat.score_ft is not None else home_stat.score_ht
        if away_stat:
            scores["away"] = away_stat.score_ft if away_stat.score_ft is not None else away_stat.score_ht

        return scores

    @staticmethod
    def _score_from_api(fixture: dict[str, Any]) -> dict[str, Optional[int]]:
        goals = fixture.get("goals") or {}
        home = goals.get("home")
        away = goals.get("away")

        def _to_int(value):
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        return {"home": _to_int(home), "away": _to_int(away)}

    @staticmethod
    def _passes_search(row: dict[str, Any], search_text: Optional[str]) -> bool:
        if not search_text:
            return True
        search_l = search_text.lower()
        haystack = f"{row.get('home', '')} {row.get('away', '')} {row.get('league', '')}".lower()
        return search_l in haystack

    def _serialize_api_fixture(
        self,
        fixture: dict[str, Any],
        with_predictions: bool,
        markets: list[str],
        db_match: Optional[Match] = None,
        allow_compute: bool = True,
    ) -> Optional[dict[str, Any]]:
        fixture_meta = fixture.get("fixture") or {}
        teams = fixture.get("teams") or {}
        league = fixture.get("league") or {}

        fixture_id = self._fixture_id_from_api(fixture)
        if fixture_id is None:
            return None

        dt_value = self._parse_datetime(fixture_meta.get("date"))
        status = ((fixture_meta.get("status") or {}).get("short") or "NS").upper()

        row = {
            "fixture_id": fixture_id,
            "date": self._format_date(dt_value),
            "time": self._format_time(dt_value),
            "datetime": dt_value.isoformat() if dt_value else None,
            "league": league.get("name") or str(league.get("id") or ""),
            "round": league.get("round"),
            "home": ((teams.get("home") or {}).get("name") or "-").strip(),
            "away": ((teams.get("away") or {}).get("name") or "-").strip(),
            "status": status,
            "phase": self._classify_phase(status, dt_value),
            "score": self._score_from_api(fixture),
            "predictions": {},
            # MATCH-01: badge decision/edge/EV per la vista lista, vedi
            # `_decisions_for_row` - default vuoto finche' non calcolato.
            "decision_cards": [],
            "best_decision": None,
            "source": "api_sports",
        }

        if with_predictions and markets:
            predictions = self._predict_fixture(
                fixture_id=fixture_id, markets=markets, db_match=db_match, status=status, allow_compute=allow_compute
            )
            row["predictions"] = predictions
            row["decision_cards"], row["best_decision"] = self._decisions_for_row(
                row=row, predictions=predictions, db_match=db_match
            )

        return row

    def _predict_fixture(
        self,
        fixture_id: int,
        markets: list[str],
        db_match: Optional[Match] = None,
        status: Optional[str] = None,
        allow_compute: bool = True,
    ) -> dict[str, Any]:
        """Delega la risoluzione della predizione grezza per mercato a
        `PredictionSnapshotService` (2026-09-09: cache persistita su DB,
        `match_prediction_snapshot` - riusa una riga gia' calcolata quando
        possibile invece di ricaricare il modello e rifare l'inferenza ad
        ogni richiesta), poi applica la proiezione di coerenza monotona
        (sempre, sul payload GREZZO risolto - MAI persistita: e' una
        combinazione tra mercati diversi, non una proprieta' del singolo
        modello, va ricalcolata ad ogni lettura mantenendo la cache
        semplice "un mercato alla volta").

        `allow_compute=False` (2026-09-10): vietato qualunque calcolo
        nuovo, serve SOLO cio' che e' gia' salvato - usato dalla vista
        lista per le date storiche (`get_day_matches`), cosi' che scorrere
        lo storico resti SEMPRE veloce, mai "quasi sempre" (vedi
        `PredictionSnapshotService.resolve_predictions`)."""
        if not markets:
            return {}

        cache_key = f"{fixture_id}:{','.join(sorted(markets))}:{allow_compute}"
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        payload = self._snapshot_service.resolve_predictions(
            fixture_id=fixture_id, markets=markets, db_match=db_match, status=status, allow_compute=allow_compute
        )

        self._apply_monotonic_projection(payload)
        self._prediction_cache[cache_key] = payload
        return payload

    @staticmethod
    def _apply_monotonic_projection(payload: dict[str, Any]) -> None:
        """Impone P(Over1.5)>=P(Over2.5)>=P(Over3.5)>=P(Over4.5) sulle
        predizioni delle 4 soglie Under/Over per la STESSA fixture
        (`enforce_monotonic_over_probabilities`, gia' validata offline in
        questa sessione: nessun costo, piccolo beneficio). Applicata SOLO
        se tutte e 4 le soglie hanno gia' una predizione in `payload` (mai
        una proiezione parziale) - modifica `payload` in place, sia
        `probability` sia `prediction` (ricalcolato sulla probabilita'
        proiettata, non piu' quella grezza)."""
        if not all(market in payload for market in UNDER_OVER_MARKETS):
            return

        probabilities_by_threshold = {
            market.replace("under_over_", "over_"): np.array([payload[market]["probability"]])
            for market in UNDER_OVER_MARKETS
        }
        projected = enforce_monotonic_over_probabilities(probabilities_by_threshold)

        for market in UNDER_OVER_MARKETS:
            label = market.replace("under_over_", "over_")
            p = float(projected[label][0])
            payload[market]["probability"] = p
            payload[market]["prediction"] = int(p >= 0.5)

    def _serialize_match(
        self, match: Match, with_predictions: bool, markets: list[str], allow_compute: bool = True
    ) -> dict[str, Any]:
        dt_value = self._parse_datetime(match.date_match)
        phase = self._classify_phase(match.status, dt_value)

        row = {
            "fixture_id": match.id_fixture,
            "date": self._format_date(dt_value),
            "time": self._format_time(dt_value),
            "datetime": dt_value.isoformat() if dt_value else None,
            "league": match.title_league or str(match.current_league or ""),
            "round": match.round,
            "home": match.name_home,
            "away": match.name_away,
            "status": match.status,
            "phase": phase,
            "score": self._extract_scores(match),
            "predictions": {},
            # MATCH-01: badge decision/edge/EV per la vista lista, vedi
            # `_decisions_for_row` - default vuoto finche' non calcolato.
            "decision_cards": [],
            "best_decision": None,
            "source": "db",
        }

        if with_predictions and match.id_fixture:
            predictions = self._predict_fixture(
                fixture_id=match.id_fixture,
                markets=markets,
                db_match=match,
                status=match.status,
                allow_compute=allow_compute,
            )
            row["predictions"] = predictions
            # Riga gia' dal DB locale: `match.odds` e' gia' caricato via
            # `selectinload` dalla query unica di `_fetch_matches` (nessuna
            # nuova query/fetch per calcolare il badge decision).
            row["decision_cards"], row["best_decision"] = self._decisions_for_row(
                row=row, predictions=predictions, db_match=match
            )

        return row

    def _fetch_matches(self, target_date: date, day_margin: int = 1) -> list[Match]:
        """Match del DB locale rilevanti per `target_date` (+- day_margin giorni).

        Fix performance critico: la query precedente NON aveva alcun filtro
        SQL sulla data, quindi caricava l'INTERO storico (47k+ match, con
        selectin di statistics/odds -> centinaia di migliaia di righe) ad
        OGNI richiesta dashboard, impiegando 20-40+ secondi (misurato:
        22.6s in locale, timeout oltre 40s dal container via
        host.docker.internal). `date_match` e' una stringa ISO 8601 con
        offset fisso "+00:00" (es. "2026-09-03T18:00:00+00:00"): un
        confronto lessicografico su range di date (prefisso "YYYY-MM-DD")
        e' equivalente a un confronto temporale e riduce drasticamente le
        righe caricate (da tutto il DB a poche centinaia al massimo). Il
        margine di 1 giorno assorbe eventuali differenze di fuso orario; il
        filtro Python esistente su `dt_value.date() == target_date` scarta
        comunque le righe fuori target.
        """
        start = (target_date - timedelta(days=day_margin)).isoformat()
        end = (target_date + timedelta(days=day_margin + 1)).isoformat()
        try:
            with SessionLocal() as session:
                rows = (
                    session.query(Match)
                    .options(selectinload(Match.statistics), selectinload(Match.odds))
                    .filter(Match.id_fixture.is_not(None))
                    .filter(Match.date_match >= start)
                    .filter(Match.date_match < end)
                    .all()
                )
            return rows
        except (ProgrammingError, OperationalError):
            # Container avviato senza migrazioni: la dashboard resta disponibile mostrando stato vuoto.
            return []

    def _fetch_db_match_by_fixture(self, fixture_id: int) -> Optional[Match]:
        try:
            with SessionLocal() as session:
                row = (
                    session.query(Match)
                    .options(selectinload(Match.statistics), selectinload(Match.odds))
                    .filter(Match.id_fixture == fixture_id)
                    .first()
                )
            return row
        except (ProgrammingError, OperationalError):
            return None

    def _db_matches_lookup(self, target_date: date) -> dict[int, Match]:
        """MATCH-01: mappa fixture_id -> Match dal DB locale per la finestra
        di `target_date` (`_fetch_matches`, gia' ottimizzata con filtro SQL
        sulla data - fix performance esistente, query UNICA). Riusata per
        arricchire le righe del Match Center (lista giorno/live) con badge
        decision/edge/EV SENZA alcuna nuova fetch odds verso l'API esterna."""
        return {m.id_fixture: m for m in self._fetch_matches(target_date=target_date) if m.id_fixture is not None}

    def get_day_matches(
        self,
        target_date: date,
        limit: int = 300,
        with_predictions: bool = True,
        markets: Optional[list[str]] = None,
        phase: Optional[str] = None,
        search_text: Optional[str] = None,
        force_refresh: bool = False,
    ) -> DashboardDayData:
        model_markets = self._normalize_market_request(markets) or self.registry.list_markets()
        rows: list[dict[str, Any]] = []
        seen_fixtures: set[int] = set()

        # Query unica sul DB locale per la finestra di `target_date`,
        # riusata sia per arricchire le righe API (badge decision, MATCH-01)
        # sia per le righe DB-only piu' sotto (nessuna query duplicata).
        db_by_fixture = self._db_matches_lookup(target_date)

        # Fix quota API-Sports (2026-09-07): il DB locale viene sincronizzato
        # quotidianamente (job "Aggiorna tutto"/`data_daily_refresh` +
        # `data_sync_today`/`future_sync`, vedi
        # `AppConfig.daily_refresh_days_ahead`) per la finestra ieri ->
        # oggi+N giorni, su TUTTI i campionati censiti.
        #
        # Distinzione STORICHE vs OGGI/FUTURE (raffinata 2026-09-07): una
        # partita gia' CONCLUSA (data < oggi UTC) ha risultato/quote ormai
        # DEFINITIVI - se il DB ha gia' la fixture, il provider esterno non
        # verra' MAI piu' interrogato per quella data (spreco di quota
        # altrimenti evitabile). Una partita di OGGI o FUTURA invece puo'
        # ancora subire cambi (quote in movimento, rinvio/spostamento data,
        # nuove convocazioni) anche se il DB ha gia' un dato sincronizzato
        # nelle ultime 24h dal job schedulato - qui l'API resta la fonte
        # primaria, protetta comunque dalla cache TTL 60s
        # (`_fetch_api_day_fixtures`) e dal guard "quota esaurita" (nessuno
        # spreco quando i dati sono gia' freschi o la quota e' finita).
        today_utc = datetime.now(timezone.utc).date()
        is_historical_date = target_date < today_utc

        db_has_target_date = any(
            (dt := self._parse_datetime(match.date_match)) and dt.date() == target_date
            for match in db_by_fixture.values()
        )
        # `force_refresh` (bottone "Forza aggiornamento" in Dashboard, per i
        # rari casi in cui serve ri-sincronizzare a mano anche una data
        # storica gia' a DB - es. dato importato errato, correzione tardiva
        # del provider) bypassa lo skip DB-first e la cache TTL, MA MAI il
        # guard "quota esaurita" in `_fetch_api_day_fixtures` (nessun
        # bottone puo' forzare una chiamata quando la quota e' al 100%,
        # stessa policy di tutti gli altri bottoni che chiamano API-Sports).
        skip_api = not force_refresh and is_historical_date and db_has_target_date

        # Vista storica SEMPRE veloce (2026-09-10, richiesto esplicitamente
        # dall'operatore dopo aver segnalato lentezza anche al secondo giro
        # sulla stessa data): per una data passata, mai un calcolo nuovo di
        # predizione nella lista - un mercato senza riga gia' salvata
        # compare semplicemente come non disponibile invece di innescare un
        # ricalcolo lento (vedi `PredictionSnapshotService.resolve_predictions`).
        # Il dettaglio di una singola fixture (`get_match_detail`, un click
        # deliberato) NON e' toccato da questo flag - resta sempre veloce
        # tanto quanto completo.
        allow_compute = not is_historical_date

        if not skip_api:
            for fixture in self._fetch_api_day_fixtures(target_date, force_refresh=force_refresh):
                fixture_id = self._fixture_id_from_api(fixture)
                row = self._serialize_api_fixture(
                    fixture,
                    with_predictions=with_predictions,
                    markets=model_markets,
                    db_match=db_by_fixture.get(fixture_id) if fixture_id is not None else None,
                    allow_compute=allow_compute,
                )
                if not row:
                    continue
                if row.get("date") and row.get("date") != target_date.isoformat():
                    continue
                if phase and row["phase"] != phase:
                    continue
                if not self._passes_search(row, search_text):
                    continue

                rows.append(row)
                seen_fixtures.add(row["fixture_id"])

        for match in db_by_fixture.values():
            dt_value = self._parse_datetime(match.date_match)
            if not dt_value or dt_value.date() != target_date:
                continue
            if match.id_fixture in seen_fixtures:
                continue

            row = self._serialize_match(
                match, with_predictions=with_predictions, markets=model_markets, allow_compute=allow_compute
            )
            if phase and row["phase"] != phase:
                continue
            if not self._passes_search(row, search_text):
                continue

            rows.append(row)
            if row.get("fixture_id") is not None:
                seen_fixtures.add(row["fixture_id"])

        rows.sort(key=lambda x: (x.get("datetime") or "", x.get("league") or "", x.get("home") or ""))
        total_rows = len(rows)
        if limit > 0:
            rows = rows[:limit]

        return DashboardDayData(
            date=target_date.isoformat(),
            total=total_rows,
            returned=len(rows),
            model_markets=model_markets,
            rows=rows,
        )

    def get_live_matches(
        self,
        target_date: date,
        limit: int = 40,
        with_predictions: bool = True,
        markets: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        model_markets = self._normalize_market_request(markets) or self.registry.list_markets()
        rows: list[dict[str, Any]] = []
        seen_fixtures: set[int] = set()

        # Stessa lookup DB usata da `get_day_matches` (MATCH-01): arricchisce
        # le righe live-API con badge decision/edge/EV quando la fixture e'
        # gia' nel DB locale, senza nuove fetch odds verso l'API esterna.
        db_by_fixture = self._db_matches_lookup(target_date)

        for fixture in self._fetch_api_live_fixtures():
            fixture_id = self._fixture_id_from_api(fixture)
            row = self._serialize_api_fixture(
                fixture,
                with_predictions=with_predictions,
                markets=model_markets,
                db_match=db_by_fixture.get(fixture_id) if fixture_id is not None else None,
            )
            if not row or row.get("phase") != "live":
                continue
            # Evita sporadici live notturni fuori data target
            if row.get("date") and row.get("date") != target_date.isoformat():
                continue

            rows.append(row)
            seen_fixtures.add(row["fixture_id"])

        # fallback/integrazione da DB per eventuali match live non presenti nel feed API
        db_live = self.get_day_matches(
            target_date=target_date,
            limit=0,
            with_predictions=with_predictions,
            markets=model_markets,
            phase="live",
        )
        for row in db_live.rows:
            fixture_id = row.get("fixture_id")
            if fixture_id in seen_fixtures:
                continue
            rows.append(row)
            if fixture_id is not None:
                seen_fixtures.add(fixture_id)

        rows.sort(key=lambda x: (x.get("datetime") or "", x.get("league") or "", x.get("home") or ""))
        total_live = len(rows)
        rows = rows[:limit] if limit > 0 else rows

        return {
            "date": target_date.isoformat(),
            "total": total_live,
            "returned": len(rows),
            "model_markets": model_markets,
            "rows": rows,
        }

    def get_overview(self, target_date: date) -> dict[str, Any]:
        # Fix performance (cambio giorno lento): il frontend (DashboardPage)
        # usa SOLO `counts.total/live/to_play` e `model_markets.length` da
        # questa risposta - MAI i valori delle predizioni dentro
        # `day_highlights`/`live_preview`/`counts.with_prediction` (verificato
        # su tutto il frontend). Prima con `with_predictions=True` questo
        # endpoint ricalcolava le predizioni ML per OGNI fixture del giorno
        # (limit=0, nessun limite) in aggiunta a `/dashboard/day`, chiamato in
        # parallelo dallo stesso `loadDashboardData` - un secondo giro
        # IDENTICO e completamente inutilizzato di query DB + inferenza
        # modello per ogni cambio data/ricerca.
        day = self.get_day_matches(target_date=target_date, limit=0, with_predictions=False)

        live_count = 0
        to_play_count = 0
        finished_count = 0
        with_prediction_count = 0

        model_markets = self.registry.list_markets()
        for row in day.rows:
            if row["phase"] == "live":
                live_count += 1
            elif row["phase"] == "to_play":
                to_play_count += 1
            else:
                finished_count += 1

            if row.get("predictions"):
                with_prediction_count += 1

        highlights = day.rows[:8]
        live_preview = [row for row in day.rows if row["phase"] == "live"][:8]

        return {
            "date": target_date.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counts": {
                "total": len(day.rows),
                "live": live_count,
                "to_play": to_play_count,
                "finished": finished_count,
                "with_prediction": with_prediction_count,
            },
            "model_markets": model_markets,
            "live_preview": live_preview,
            "day_highlights": highlights,
        }

    _DATES_STATE_FILENAME = "dashboard_dates_state.json"

    def _dates_state_path(self) -> str:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "best_models"))
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, self._DATES_STATE_FILENAME)

    def _first_seen_date(self, today: date) -> date:
        """Data di "nascita" della select date accumulata (TopFilters):
        persistita su file (volume `best_models/`, sopravvive ai rebuild
        Docker) la PRIMA volta che questo endpoint viene chiamato, cosi' la
        lista cresce di un giorno alla volta da quel momento in poi (richiesta
        utente: "elenco di date dal giorno 1 ... che man mano viene
        accumulato"), SENZA dipendere da un trigger esterno che popoli il
        Prediction Ledger (usato comunque come fonte aggiuntiva in
        `get_available_dates`: se contiene una data ancora piu' vecchia,
        vince quella)."""
        path = self._dates_state_path()
        stored: Optional[date] = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                raw = payload.get("first_seen_date")
                if raw:
                    stored = date.fromisoformat(raw)
            except Exception:
                stored = None

        if stored is None:
            stored = today
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"first_seen_date": stored.isoformat()}, f)
            except Exception:
                pass

        return stored

    def get_available_dates(self) -> dict[str, Any]:
        """Elenco date selezionabili in UI (TopFilters): dal primo giorno in
        cui questa funzionalita' e' stata usata (persistito su file, vedi
        `_first_seen_date`) fino ad oggi, cosi' che la lista si accumuli
        automaticamente giorno dopo giorno invece di un calendario libero
        (richiesta esplicita utente: "elenco di date dal giorno 1 di
        previsioni ad oggi che man mano viene accumulato"). Se il Prediction
        Ledger contiene gia' una prediction salvata PRIMA di quella data,
        vince quella data ancora piu' vecchia."""
        today = datetime.now(timezone.utc).date()
        start = self._first_seen_date(today)

        earliest_dt = self.ledger_repo.get_earliest_created_date()
        if earliest_dt and earliest_dt.date() < start:
            start = earliest_dt.date()
        if start > today:
            start = today

        span_days = (today - start).days
        dates = [(start + timedelta(days=offset)).isoformat() for offset in range(span_days + 1)]
        dates.reverse()  # oggi per primo: piu' utile in una select

        return {
            "dates": dates,
            "first_date": dates[-1] if dates else today.isoformat(),
            "last_date": dates[0] if dates else today.isoformat(),
        }

    def get_match_detail(
        self,
        fixture_id: int,
        with_predictions: bool = True,
        markets: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        model_markets = self._normalize_market_request(markets) or self.registry.list_markets()

        api_fixture = self._fetch_api_fixture_detail(fixture_id)
        db_match = self._fetch_db_match_by_fixture(fixture_id)

        if api_fixture:
            fixture_row = self._serialize_api_fixture(api_fixture, with_predictions=False, markets=[])
        elif db_match:
            fixture_row = self._serialize_match(db_match, with_predictions=False, markets=[])
        else:
            return {
                "fixture": None,
                "timeline": [],
                "odds_summary": {},
                "bookmaker_baseline": {"markets": {}, "generated": False},
                "decision_cards": [],
                "predictions": {},
                "model_markets": model_markets,
            }

        predictions = (
            self._predict_fixture(
                fixture_id=fixture_id, markets=model_markets, db_match=db_match, status=fixture_row.get("status")
            )
            if with_predictions
            else {}
        )
        fixture_row["predictions"] = predictions

        events = self._fetch_api_events(fixture_id)
        timeline = self._serialize_events(events)

        odds_payload = self._fetch_api_odds(fixture_id)
        odds_summary = self._aggregate_odds_from_api(odds_payload)
        if not odds_summary:
            odds_summary = self._aggregate_odds_from_db(db_match)

        bookmaker_baseline = build_fixture_baseline(odds_summary)

        decision_cards = self._build_decision_cards(
            row_context=fixture_row,
            predictions=predictions,
            odds_summary=odds_summary,
            bookmaker_baseline=bookmaker_baseline,
        )

        return {
            "fixture": fixture_row,
            "timeline": timeline,
            "odds_summary": odds_summary,
            "bookmaker_baseline": bookmaker_baseline,
            "decision_cards": decision_cards,
            "predictions": predictions,
            "model_markets": model_markets,
            "odds_updated_at": (odds_payload or {}).get("update") if odds_payload else None,
        }






























