from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text

from src.repository.base.repository_db import SessionLocal
from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository

LEGACY_ODDS_MARKETS = [
    "h2h",
    "under_over_1_5",
    "under_over_2_5",
    "under_over_3_5",
    "under_over_4_5",
    "goal_no_goal",
    "corners",
    "cards",
    "dc",
]


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _safe_orphan_count(sql: str) -> Optional[int]:
    try:
        with SessionLocal() as session:
            return int(session.execute(text(sql)).scalar() or 0)
    except Exception:
        return None


class DataQualityService:
    def __init__(
        self,
        match_repo: Optional[MatchRepository] = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
    ):
        self.match_repo = match_repo or MatchRepository()
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()

    @staticmethod
    def _filter_matches(matches: list, seasons: Optional[list[int]], leagues: Optional[list[int]]) -> list:
        filtered = []
        for match in matches:
            if seasons and match.season not in seasons:
                continue
            if leagues and match.current_league not in leagues:
                continue
            filtered.append(match)
        return filtered

    def build_report(
        self,
        top_n: int = 20,
        seasons: Optional[list[int]] = None,
        leagues: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        matches = self._filter_matches(self.match_repo.search_all(), seasons=seasons, leagues=leagues)
        snapshots = self.snapshot_repo.list_all()
        filtered_fixture_ids = {int(match.id_fixture) for match in matches if match.id_fixture is not None}

        fixtures_total = len(matches)
        fixtures_with_statistics = 0
        fixtures_with_odds = 0
        invalid_match_datetime_count = 0

        null_counts = {
            "id_fixture": 0,
            "season": 0,
            "current_league": 0,
            "date_match": 0,
            "status": 0,
        }

        market_coverage_counter = {key: 0 for key in LEGACY_ODDS_MARKETS}
        by_season = Counter()
        by_league = Counter()
        fixture_counter = Counter()
        settlement_counter = Counter()

        match_date_by_fixture: dict[int, datetime] = {}
        for match in matches:
            fixture_id = match.id_fixture
            if fixture_id is not None:
                fixture_counter[int(fixture_id)] += 1

            if match.season is None:
                null_counts["season"] += 1
            else:
                by_season[int(match.season)] += 1

            if match.current_league is None:
                null_counts["current_league"] += 1
            else:
                by_league[int(match.current_league)] += 1

            if match.id_fixture is None:
                null_counts["id_fixture"] += 1
            if not match.status:
                null_counts["status"] += 1
            if not match.date_match:
                null_counts["date_match"] += 1

            match_dt = _parse_iso_datetime(match.date_match)
            if match_dt is None:
                invalid_match_datetime_count += 1
            elif fixture_id is not None:
                match_date_by_fixture[int(fixture_id)] = match_dt

            stats = match.statistics or []
            if len(stats) > 0:
                fixtures_with_statistics += 1

            odds_rows = match.odds or []
            if len(odds_rows) > 0:
                fixtures_with_odds += 1
                first_odds = odds_rows[0].to_dict()
                for market in LEGACY_ODDS_MARKETS:
                    payload = first_odds.get(market)
                    if isinstance(payload, dict) and payload:
                        market_coverage_counter[market] += 1

            settlement_status = (match.settlement_status or "pending").lower()
            if settlement_status not in {"complete", "incomplete", "pending"}:
                settlement_status = "pending"
            settlement_counter[settlement_status] += 1

        duplicate_fixture_count = sum(count - 1 for count in fixture_counter.values() if count > 1)
        fixtures_missing_statistics = max(0, fixtures_total - fixtures_with_statistics)
        fixtures_missing_odds = max(0, fixtures_total - fixtures_with_odds)
        fixtures_incomplete_count = int(settlement_counter.get("incomplete", 0) + settlement_counter.get("pending", 0))

        snapshot_fixture_counter = Counter()
        snapshot_market_counter = Counter()
        snapshot_after_kickoff_count = 0
        for row in snapshots:
            fixture_id = int(row.fixture_id)
            if filtered_fixture_ids and fixture_id not in filtered_fixture_ids:
                continue
            snapshot_fixture_counter[fixture_id] += 1
            snapshot_market_counter[str(row.market)] += 1

            kickoff = match_date_by_fixture.get(fixture_id)
            captured_at = row.captured_at
            if kickoff and captured_at and captured_at.tzinfo is None:
                captured_at = captured_at.replace(tzinfo=timezone.utc)
            if kickoff and captured_at and captured_at > kickoff:
                snapshot_after_kickoff_count += 1

        fixtures_with_snapshot = len(snapshot_fixture_counter)

        orphan_statistics_count = _safe_orphan_count(
            'SELECT COUNT(*) FROM statistics s LEFT JOIN "match" m ON s.id_match = m.id_match_fk '
            'WHERE s.id_match IS NOT NULL AND m.id_match_fk IS NULL'
        )
        orphan_odds_count = _safe_orphan_count(
            'SELECT COUNT(*) FROM odds o LEFT JOIN "match" m ON o.id_match = m.id_match_fk '
            'WHERE o.id_match IS NOT NULL AND m.id_match_fk IS NULL'
        )
        orphan_odds_snapshot_match_fk = _safe_orphan_count(
            'SELECT COUNT(*) FROM odds_snapshot os LEFT JOIN "match" m ON os.id_match = m.id_match_fk '
            'WHERE os.id_match IS NOT NULL AND m.id_match_fk IS NULL'
        )
        orphan_odds_snapshot_fixture_fk = _safe_orphan_count(
            'SELECT COUNT(*) FROM odds_snapshot os LEFT JOIN "match" m ON os.fixture_id = m.id_fixture '
            'WHERE m.id_fixture IS NULL'
        )

        fixtures_total_safe = max(fixtures_total, 1)
        market_coverage = {
            key: {
                "fixtures": int(value),
                "coverage_ratio": round(float(value) / fixtures_total_safe, 6),
            }
            for key, value in market_coverage_counter.items()
        }

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "fixtures_total": fixtures_total,
                "statistics_rows_total": int(sum(len(match.statistics or []) for match in matches)),
                "odds_rows_total": int(sum(len(match.odds or []) for match in matches)),
                "odds_snapshot_rows_total": int(len(snapshots)),
                "filters": {
                    "seasons": seasons or [],
                    "leagues": leagues or [],
                },
            },
            "coverage": {
                "fixtures_with_statistics": fixtures_with_statistics,
                "fixtures_with_odds": fixtures_with_odds,
                "fixtures_with_snapshot": fixtures_with_snapshot,
                "coverage_statistics_ratio": round(float(fixtures_with_statistics) / fixtures_total_safe, 6),
                "coverage_odds_ratio": round(float(fixtures_with_odds) / fixtures_total_safe, 6),
                "coverage_snapshot_ratio": round(float(fixtures_with_snapshot) / fixtures_total_safe, 6),
                "markets": market_coverage,
                "snapshot_markets": {
                    key: int(value)
                    for key, value in snapshot_market_counter.most_common(max(1, top_n))
                },
            },
            "anomalies": {
                "null_counts_match": null_counts,
                "duplicate_fixture_count": int(duplicate_fixture_count),
                "fixtures_missing_statistics": int(fixtures_missing_statistics),
                "fixtures_missing_odds": int(fixtures_missing_odds),
                "fixtures_incomplete_count": fixtures_incomplete_count,
                "orphan_statistics_count": orphan_statistics_count,
                "orphan_odds_count": orphan_odds_count,
                "orphan_odds_snapshot_match_fk": orphan_odds_snapshot_match_fk,
                "orphan_odds_snapshot_fixture_fk": orphan_odds_snapshot_fixture_fk,
            },
            "distribution": {
                "by_season": [
                    {"season": key, "count": int(value)} for key, value in by_season.most_common(max(1, top_n))
                ],
                "by_league": [
                    {"league": key, "count": int(value)} for key, value in by_league.most_common(max(1, top_n))
                ],
            },
            "temporal_checks": {
                "invalid_match_datetime_count": int(invalid_match_datetime_count),
                "snapshot_after_kickoff_count": int(snapshot_after_kickoff_count),
            },
            "settlement": {
                "complete": int(settlement_counter.get("complete", 0)),
                "incomplete": int(settlement_counter.get("incomplete", 0)),
                "pending": int(settlement_counter.get("pending", 0)),
            },
        }
        return report



