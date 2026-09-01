from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import os
from typing import Any, Optional

import joblib
import numpy as np
from dateutil.parser import isoparse
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import selectinload

from src.repository.base.repository_db import SessionLocal
from src.service_ia.config.app_config import load_app_config
from src.service_ia.model.match import Match
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.training.model_registry import ModelRegistry
from src.service_ia.utility.request_api import base_api_statistics

FINAL_STATUSES = {"FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"}
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"}


@dataclass
class DashboardDayData:
    date: str
    total: int
    returned: int
    model_markets: list[str]
    rows: list[dict[str, Any]]


class DashboardService:
    _api_cache: dict[str, tuple[datetime, list[dict[str, Any]]]] = {}
    _api_cache_ttl_seconds = 60

    def __init__(self):
        self.registry = ModelRegistry()
        self.filter_service = FilterMarketService()
        self.cfg = load_app_config()
        self._model_meta_cache: dict[str, dict[str, Any]] = {}
        self._model_cache: dict[str, Any] = {}
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
    def _cache_get(cls, key: str) -> Optional[list[dict[str, Any]]]:
        item = cls._api_cache.get(key)
        if not item:
            return None

        ts, payload = item
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        if age > cls._api_cache_ttl_seconds:
            return None
        return payload

    @classmethod
    def _cache_set(cls, key: str, payload: list[dict[str, Any]]) -> None:
        cls._api_cache[key] = (datetime.now(timezone.utc), payload)

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

    def _fetch_api_day_fixtures(self, target_date: date) -> list[dict[str, Any]]:
        cache_key = f"day:{target_date.isoformat()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        season = self._season_for_date(target_date)
        fixtures: list[dict[str, Any]] = []
        for league in self.cfg.leagues or []:
            try:
                payload = base_api_statistics(
                    path="fixtures",
                    params={"date": target_date.isoformat(), "league": league, "season": season},
                )
            except Exception:
                payload = []
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

    def _serialize_api_fixture(self, fixture: dict[str, Any], with_predictions: bool, markets: list[str]) -> Optional[dict[str, Any]]:
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
            "source": "api_sports",
        }

        if with_predictions and markets:
            row["predictions"] = self._predict_fixture(fixture_id=fixture_id, markets=markets)

        return row

    @staticmethod
    def _extract_probability(model: Any, X) -> tuple[int, float]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            first = np.asarray(probs[0], dtype=float)
            if first.size >= 2:
                p1 = float(first[-1])
                return int(p1 >= 0.5), p1
            if first.size == 1:
                p1 = float(first[0])
                return int(p1 >= 0.5), p1

        pred_raw = model.predict(X)
        pred = int(np.asarray(pred_raw).ravel()[0])
        return pred, float(pred)

    def _latest_model_for_market(self, market: str) -> Optional[dict[str, Any]]:
        if market not in self._model_meta_cache:
            self._model_meta_cache[market] = self.registry.get_latest(market=market) or {}
        model = self._model_meta_cache[market]
        return model or None

    def _load_model(self, model_path: str):
        if model_path not in self._model_cache:
            self._model_cache[model_path] = joblib.load(model_path)
        return self._model_cache[model_path]

    def _predict_fixture(self, fixture_id: int, markets: list[str]) -> dict[str, Any]:
        if not markets:
            return {}

        cache_key = f"{fixture_id}:{','.join(sorted(markets))}"
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached

        payload: dict[str, Any] = {}
        for market in markets:
            model_meta = self._latest_model_for_market(market)
            if not model_meta:
                continue

            model_path = model_meta.get("model_path")
            if not model_path or not os.path.exists(model_path):
                continue

            frame = self.filter_service.build_prediction_frame(market=market, fixture_id=fixture_id)
            if frame is None or frame.empty:
                continue

            X = frame.drop(columns=["market", "id_fixture", "season"], errors="ignore")
            selected_features = model_meta.get("feature_names") or []
            if selected_features:
                for feature_name in selected_features:
                    if feature_name not in X.columns:
                        X[feature_name] = 0.0
                X = X[selected_features]

            try:
                model = self._load_model(model_path)
                prediction, probability = self._extract_probability(model=model, X=X)
            except Exception:
                continue

            payload[market] = {
                "prediction": int(prediction),
                "probability": float(probability),
                "model_name": model_meta.get("model_name"),
                "run_id": model_meta.get("run_id"),
            }

        self._prediction_cache[cache_key] = payload
        return payload

    def _serialize_match(self, match: Match, with_predictions: bool, markets: list[str]) -> dict[str, Any]:
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
            "source": "db",
        }

        if with_predictions and match.id_fixture:
            row["predictions"] = self._predict_fixture(fixture_id=match.id_fixture, markets=markets)

        return row

    def _fetch_matches(self) -> list[Match]:
        try:
            with SessionLocal() as session:
                rows = (
                    session.query(Match)
                    .options(selectinload(Match.statistics), selectinload(Match.odds))
                    .filter(Match.id_fixture.is_not(None))
                    .all()
                )
            return rows
        except (ProgrammingError, OperationalError):
            # Container avviato senza migrazioni: la dashboard resta disponibile mostrando stato vuoto.
            return []

    def get_day_matches(
        self,
        target_date: date,
        limit: int = 300,
        with_predictions: bool = True,
        markets: Optional[list[str]] = None,
        phase: Optional[str] = None,
        search_text: Optional[str] = None,
    ) -> DashboardDayData:
        model_markets = self._normalize_market_request(markets) or self.registry.list_markets()
        rows: list[dict[str, Any]] = []
        seen_fixtures: set[int] = set()

        for fixture in self._fetch_api_day_fixtures(target_date):
            row = self._serialize_api_fixture(fixture, with_predictions=with_predictions, markets=model_markets)
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

        for match in self._fetch_matches():
            dt_value = self._parse_datetime(match.date_match)
            if not dt_value or dt_value.date() != target_date:
                continue
            if match.id_fixture in seen_fixtures:
                continue

            row = self._serialize_match(match, with_predictions=with_predictions, markets=model_markets)
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

        for fixture in self._fetch_api_live_fixtures():
            row = self._serialize_api_fixture(fixture, with_predictions=with_predictions, markets=model_markets)
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
        day = self.get_day_matches(target_date=target_date, limit=0, with_predictions=True)

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









