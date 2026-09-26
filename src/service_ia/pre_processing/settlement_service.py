from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional, Protocol

from src.repository.match_repository import MatchRepository
from src.repository.odds_snapshot_repository import OddsSnapshotRepository
from src.service_ia.model.match import Match
from src.service_ia.pre_processing.api_sports_provider import ApiSportsProvider
from src.service_ia.pre_processing.download_match_service import download_import_matches
from src.service_ia.utility.request_api import get_api_sports_provider

FINAL_STATUSES = {"FT", "AET", "PEN", "ABD"}


class SettlementHook(Protocol):
    def on_prediction_settled(self, match: Match, settlement_details: dict[str, Any]) -> None:
        ...


class NoOpSettlementHook:
    def on_prediction_settled(self, match: Match, settlement_details: dict[str, Any]) -> None:
        return None


def _parse_iso_date(value: Optional[str], fallback: date) -> date:
    if not value:
        return fallback
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return fallback


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _match_in_window(match: Match, from_day: date, to_day: date) -> bool:
    match_dt = _parse_iso_datetime(match.date_match)
    if match_dt is None:
        return False
    return from_day <= match_dt.date() <= to_day


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _scores_from_match(match: Match) -> tuple[Optional[int], Optional[int], bool]:
    stats = match.statistics or []
    if not stats:
        return None, None, False

    by_team = {row.statistics_team_id: row for row in stats}
    home_row = by_team.get(match.id_team_home)
    away_row = by_team.get(match.id_team_away)
    if not home_row or not away_row:
        return None, None, False

    home_score = _safe_int(home_row.score_ft)
    away_score = _safe_int(away_row.score_ft)
    if home_score is None or away_score is None:
        return None, None, False

    return home_score, away_score, True


def evaluate_match_completeness(match: Match, snapshot_count: int = 0) -> dict[str, Any]:
    final_status = str(match.status or "").upper() in FINAL_STATUSES
    home_score, away_score, has_scores = _scores_from_match(match)
    has_statistics = bool(match.statistics and len(match.statistics) >= 2)
    has_odds = bool(match.odds and len(match.odds) > 0) or snapshot_count > 0

    status = "complete" if final_status and has_scores and has_statistics and has_odds else "incomplete"

    return {
        "completeness_status": status,
        "final_status": final_status,
        "has_statistics": has_statistics,
        "has_final_scores": has_scores,
        "has_odds": has_odds,
        "snapshot_count": int(snapshot_count),
        "home_score": home_score,
        "away_score": away_score,
    }


def apply_settlement_metadata(match: Match, settlement_details: dict[str, Any], settled_at_iso: str) -> tuple[bool, bool]:
    previous_status = match.settlement_status
    previous_details = match.settlement_details or {}
    previously_settled = bool(match.is_settled)

    new_status = settlement_details.get("completeness_status")
    needs_update = (not previously_settled) or (previous_status != new_status) or (previous_details != settlement_details)
    transitioned_to_complete = previous_status != "complete" and new_status == "complete"

    if needs_update:
        match.is_settled = True
        match.settlement_status = str(new_status)
        if not match.settled_at:
            match.settled_at = settled_at_iso
        match.settlement_details = settlement_details

    return needs_update, transitioned_to_complete


class SettlementService:
    def __init__(
        self,
        match_repo: Optional[MatchRepository] = None,
        snapshot_repo: Optional[OddsSnapshotRepository] = None,
        import_runner: Optional[Callable[..., dict[str, Any]]] = None,
    ):
        self.match_repo = match_repo or MatchRepository()
        self.snapshot_repo = snapshot_repo or OddsSnapshotRepository()
        self.import_runner = import_runner or download_import_matches

    def run_settlement(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        seasons: Optional[list[int]] = None,
        leagues: Optional[list[int]] = None,
        provider: Optional[ApiSportsProvider] = None,
        hook: Optional[SettlementHook] = None,
    ) -> dict[str, Any]:
        hook = hook or NoOpSettlementHook()
        provider = provider or get_api_sports_provider()

        today = datetime.now(timezone.utc).date()
        from_day = _parse_iso_date(from_date, fallback=today - timedelta(days=3))
        to_day = _parse_iso_date(to_date, fallback=today)

        import_report = self.import_runner(
            seasons=seasons,
            leagues=leagues,
            from_date=from_day.isoformat(),
            to_date=to_day.isoformat(),
            statuses="FT-AET-PEN-ABD",
            provider=provider,
            is_next=False,
        )

        # Fix prestazioni 2026-09-15: PRIMA era un `search_filter` senza
        # alcun vincolo di data, che caricava TUTTI i match con stato finale
        # di TUTTE le stagioni (44.983 righe con relazioni) per poi
        # scartarne in Python oltre il 99% con `_match_in_window` - la
        # finestra utile e' di 4 giorni. Erano 1.331 s per esecuzione su un
        # job che gira ogni ora. Ora la finestra e' nel WHERE.
        matches = self.match_repo.search_by_date_window(
            from_day=from_day.isoformat(),
            to_day=to_day.isoformat(),
            statuses=sorted(FINAL_STATUSES),
            seasons=seasons,
            leagues=leagues,
            solo_con_id_fixture=True,
        )
        now_iso = datetime.now(timezone.utc).isoformat()

        report: dict[str, Any] = {
            "from_date": from_day.isoformat(),
            "to_date": to_day.isoformat(),
            "import_report": import_report,
            "final_matches_seen": 0,
            "updated": 0,
            "unchanged": 0,
            "complete": 0,
            "incomplete": 0,
            "hook_calls": 0,
            "rows": [],
        }

        for match in matches:
            if not _match_in_window(match, from_day=from_day, to_day=to_day):
                continue

            report["final_matches_seen"] += 1
            fixture_id = int(match.id_fixture)
            snapshots = self.snapshot_repo.list_for_fixture(fixture_id=fixture_id)
            details = evaluate_match_completeness(match=match, snapshot_count=len(snapshots))

            changed, transitioned_to_complete = apply_settlement_metadata(
                match=match,
                settlement_details=details,
                settled_at_iso=now_iso,
            )
            if changed:
                self.match_repo.save(match)
                report["updated"] += 1
            else:
                report["unchanged"] += 1

            status = details.get("completeness_status")
            if status == "complete":
                report["complete"] += 1
            else:
                report["incomplete"] += 1

            if transitioned_to_complete:
                hook.on_prediction_settled(match=match, settlement_details=details)
                report["hook_calls"] += 1

            if len(report["rows"]) < 300:
                report["rows"].append(
                    {
                        "fixture_id": fixture_id,
                        "season": match.season,
                        "league": match.current_league,
                        "status": str(match.status or ""),
                        "settlement_status": status,
                        "has_statistics": details.get("has_statistics"),
                        "has_odds": details.get("has_odds"),
                        "home_score": details.get("home_score"),
                        "away_score": details.get("away_score"),
                    }
                )

        return report

    def settlement_overview(
        self,
        limit: int = 200,
        settlement_status: Optional[str] = None,
    ) -> dict[str, Any]:
        filters = {
            "id_fixture": "not None",
            "status": sorted(FINAL_STATUSES),
        }
        matches = self.match_repo.search_filter(filters=filters)

        rows: list[dict[str, Any]] = []
        counts = {
            "complete": 0,
            "incomplete": 0,
            "pending": 0,
        }

        for match in matches:
            status = (match.settlement_status or "pending").lower()
            if status not in counts:
                status = "pending"
            counts[status] += 1

            if settlement_status and status != settlement_status.lower():
                continue

            if len(rows) < max(0, limit):
                details = match.settlement_details or {}
                rows.append(
                    {
                        "fixture_id": int(match.id_fixture),
                        "season": match.season,
                        "league": match.current_league,
                        "status": str(match.status or ""),
                        "settlement_status": status,
                        "settled_at": match.settled_at,
                        "details": details,
                    }
                )

        rows.sort(key=lambda row: (row.get("season") or 0, row.get("fixture_id") or 0), reverse=True)
        return {
            "total": len(matches),
            "counts": counts,
            "rows": rows,
        }
