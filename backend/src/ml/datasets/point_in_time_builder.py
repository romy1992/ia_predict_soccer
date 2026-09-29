from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.training.market_service.filter_market_service import FilterMarketService
from src.service_ia.utility.utils import convert_orm_match_to_dict

FINAL_STATUSES = {"FT", "AET", "PEN", "ABD"}


@dataclass
class PointInTimeDataset:
    version: str
    market: str
    created_at: str
    frame: pd.DataFrame
    metadata: dict[str, Any]


class PointInTimeDatasetBuilder:
    def __init__(
        self,
        match_repo: Optional[MatchRepository] = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
    ):
        self.match_repo = match_repo or MatchRepository()
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()
        self.market_service = FilterMarketService()

    @staticmethod
    def _parse_datetime(value: str | None) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _line_from_market(market: str) -> Optional[str]:
        if not market.startswith("under_over_"):
            return None
        return market.replace("under_over_", "").replace("_", ".")

    @staticmethod
    def _safe_float(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(str(value).replace(",", "."))
        except Exception:
            return 0.0

    @staticmethod
    def _normalize_feature_name(key: str) -> str:
        import re

        return re.sub(r"[^a-z0-9]+", "_", (key or "").strip().lower()).strip("_")

    def _extract_mean_features(self, match: dict[str, Any]) -> dict[str, float]:
        mean_stats = match.get("mean_statistics")
        if not isinstance(mean_stats, list):
            return {}

        id_home = match.get("id_team_home")
        id_away = match.get("id_team_away")
        home_row = next((row for row in mean_stats if row.get("id_team") == id_home), None)
        away_row = next((row for row in mean_stats if row.get("id_team") == id_away), None)
        if not home_row or not away_row:
            return {}

        keys = set(home_row.keys()).union(set(away_row.keys()))
        features: dict[str, float] = {}
        for key in keys:
            if key == "id_team":
                continue
            normalized = self._normalize_feature_name(key)
            if not normalized:
                continue

            home_value = self._safe_float(home_row.get(key))
            away_value = self._safe_float(away_row.get(key))
            features[f"{normalized}_home_stat"] = home_value
            features[f"{normalized}_away_stat"] = away_value
            features[f"{normalized}_diff_stat"] = home_value - away_value

        return features

    def _extract_snapshot_features(self, snapshots: list[dict[str, Any]]) -> dict[str, float]:
        if not snapshots:
            return {}

        odds_values = [self._safe_float(row.get("odd")) for row in snapshots if self._safe_float(row.get("odd")) > 0]
        if not odds_values:
            return {}

        features = {
            "odds_snapshot_count": float(len(odds_values)),
            "odds_snapshot_mean": float(np.mean(odds_values)),
            "odds_snapshot_std": float(np.std(odds_values)),
            "odds_snapshot_min": float(np.min(odds_values)),
            "odds_snapshot_max": float(np.max(odds_values)),
        }

        outcomes: dict[str, list[float]] = {}
        for row in snapshots:
            odd = self._safe_float(row.get("odd"))
            if odd <= 0:
                continue
            outcome_key = self._normalize_feature_name(str(row.get("outcome") or "unknown"))
            outcomes.setdefault(outcome_key, []).append(odd)

        for outcome_key, values in outcomes.items():
            features[f"odd_mean_outcome_{outcome_key}"] = float(np.mean(values))
            features[f"odd_min_outcome_{outcome_key}"] = float(np.min(values))
            features[f"odd_max_outcome_{outcome_key}"] = float(np.max(values))

        return features

    def _filter_snapshots_until_prediction(
        self,
        fixture_snapshots: list[dict[str, Any]],
        market: str,
        period: str,
        line: Optional[str],
        prediction_at: datetime,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in fixture_snapshots:
            if row.get("market") != market:
                continue
            if (row.get("period") or "full_time") != period:
                continue

            snapshot_line = row.get("line")
            if line is not None and str(snapshot_line or "") != str(line):
                continue

            captured_at = self._parse_datetime(row.get("captured_at"))
            if captured_at is None or captured_at > prediction_at:
                continue

            rows.append(row)

        return rows

    def _build_row(
        self,
        match: dict[str, Any],
        market: str,
        period: str,
        line: Optional[str],
        outcome: Optional[str],
        snapshots_by_fixture: dict[int, list[dict[str, Any]]],
    ) -> Optional[dict[str, Any]]:
        fixture_id = match.get("id_fixture")
        if fixture_id is None:
            return None

        prediction_at = self._parse_datetime(match.get("date_match"))
        if prediction_at is None:
            return None

        stat_home, stat_away = self.market_service._resolve_team_stats(match=match, with_full_stats=True)
        if not stat_home or not stat_away:
            return None

        target = self.market_service._label_by_market(market=market, stat_home=stat_home, stat_away=stat_away)
        if target is None:
            return None

        fixture_snapshots = snapshots_by_fixture.get(int(fixture_id), [])
        selected_snapshots = self._filter_snapshots_until_prediction(
            fixture_snapshots=fixture_snapshots,
            market=market,
            period=period,
            line=line,
            prediction_at=prediction_at,
        )
        if not selected_snapshots:
            return None

        available_times = [self._parse_datetime(row.get("captured_at")) for row in selected_snapshots]
        available_times = [item for item in available_times if item is not None]
        if not available_times:
            return None

        max_available_at = max(available_times)
        if max_available_at > prediction_at:
            return None

        row = {
            "id_fixture": int(fixture_id),
            "season": match.get("season"),
            "market": market,
            "period": period,
            "line": line,
            "outcome": outcome,
            "prediction_at": prediction_at.isoformat(),
            "feature_available_at_max": max_available_at.isoformat(),
            "y": int(target),
        }

        row.update(self._extract_mean_features(match))
        row.update(self._extract_snapshot_features(selected_snapshots))

        # Le colonne di metadati da sole non sono sufficienti per il training.
        if len(row) <= 9:
            return None
        return row

    def build_from_records(
        self,
        matches: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        market: str,
        period: str = "full_time",
        line: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> PointInTimeDataset:
        normalized_line = line if line is not None else self._line_from_market(market)

        snapshots_by_fixture: dict[int, list[dict[str, Any]]] = {}
        for row in snapshots:
            fixture_id = row.get("fixture_id")
            if fixture_id is None:
                continue
            snapshots_by_fixture.setdefault(int(fixture_id), []).append(row)

        rows: list[dict[str, Any]] = []
        for match in matches:
            status = str(match.get("status") or "").upper()
            if status and status not in FINAL_STATUSES:
                continue

            built = self._build_row(
                match=match,
                market=market,
                period=period,
                line=normalized_line,
                outcome=outcome,
                snapshots_by_fixture=snapshots_by_fixture,
            )
            if built:
                rows.append(built)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
            numeric_columns = [
                col for col in frame.columns if col not in {"market", "period", "line", "outcome", "prediction_at", "feature_available_at_max"}
            ]
            frame[numeric_columns] = frame[numeric_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

        metadata = {
            "market": market,
            "period": period,
            "line": normalized_line,
            "outcome": outcome,
            "rows": int(len(frame)),
            "fixtures": sorted(frame["id_fixture"].astype(int).tolist()) if not frame.empty else [],
            "columns": frame.columns.tolist(),
        }
        version = self._build_version(metadata)
        return PointInTimeDataset(
            version=version,
            market=market,
            created_at=datetime.now(timezone.utc).isoformat(),
            frame=frame,
            metadata=metadata,
        )

    def build_from_db(
        self,
        market: str,
        seasons: Optional[list[int]] = None,
        period: str = "full_time",
        line: Optional[str] = None,
        outcome: Optional[str] = None,
        save_snapshot: bool = False,
        output_dir: str = os.path.join("best_models", "datasets"),
    ) -> PointInTimeDataset:
        filters = {
            "id_fixture": "not None",
            "statistics": "not None",
            "mean_statistics": "not None",
            "status": list(FINAL_STATUSES),
        }
        if seasons:
            filters["season"] = seasons

        orm_matches = self.match_repo.search_filter(filters=filters)
        matches = convert_orm_match_to_dict(orm_matches)

        snapshots: list[dict[str, Any]] = []
        for match in matches:
            fixture_id = match.get("id_fixture")
            if fixture_id is None:
                continue
            rows = self.snapshot_repo.list_for_fixture(fixture_id=int(fixture_id), market=market, period=period, line=line)
            snapshots.extend([row.to_dict() for row in rows])

        dataset = self.build_from_records(
            matches=matches,
            snapshots=snapshots,
            market=market,
            period=period,
            line=line,
            outcome=outcome,
        )

        if save_snapshot:
            self.save_dataset_snapshot(dataset=dataset, output_dir=output_dir)

        return dataset

    @staticmethod
    def _build_version(metadata: dict[str, Any]) -> str:
        serialized = json.dumps(metadata, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def assert_no_leakage(frame: pd.DataFrame) -> bool:
        if frame.empty:
            return True
        prediction_at = pd.to_datetime(frame["prediction_at"], utc=True, errors="coerce")
        feature_at = pd.to_datetime(frame["feature_available_at_max"], utc=True, errors="coerce")
        return bool((feature_at <= prediction_at).all())

    @staticmethod
    def save_dataset_snapshot(dataset: PointInTimeDataset, output_dir: str) -> dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)

        base_name = f"{dataset.market}_{dataset.version}"
        csv_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.csv"))
        metadata_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.metadata.json"))

        dataset.frame.to_csv(csv_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "version": dataset.version,
                    "market": dataset.market,
                    "created_at": dataset.created_at,
                    "metadata": dataset.metadata,
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )

        return {"csv": csv_path, "metadata": metadata_path}
