from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
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

# FASE 0 (task Under/Over totals): stesse 4 soglie di `src/ml/markets/totals/totals_market.py`
# (`THRESHOLDS`), duplicata qui SOLO come costante di modulo (nessuna logica di
# filtro/training duplicata: `build_under_over_threshold_report` sotto riusa
# sempre `FilterMarketService.build_dataset` + `expanding_window_splits`).
UNDER_OVER_THRESHOLDS: tuple[float, ...] = (1.5, 2.5, 3.5, 4.5)


def _under_over_market_key(threshold: float) -> str:
    return f"under_over_{str(float(threshold)).replace('.', '_')}"


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

    # ------------------------------------------------------------------
    # FASE 0 (task Under/Over 1.5/2.5/3.5/4.5): copertura dati REALE per
    # soglia, propedeutica al training - NESSUNA logica di filtro/training
    # nuova, riusa sempre `FilterMarketService.build_dataset` (stesso
    # identico dataset che poi consuma `train_multi_market.train_market`)
    # ed `expanding_window_splits` (stessa CV walk-forward di
    # `train_multi_market.py`/`totals_market.py`).
    # ------------------------------------------------------------------
    @staticmethod
    def _match_team_statistics(match) -> tuple[Optional[Any], Optional[Any]]:
        stats = match.statistics or []
        if len(stats) < 2:
            return None, None
        stat_home = next((s for s in stats if s.statistics_team_id == match.id_team_home), None)
        stat_away = next((s for s in stats if s.statistics_team_id == match.id_team_away), None)
        return stat_home, stat_away

    @classmethod
    def _has_complete_statistics(cls, match) -> bool:
        stat_home, stat_away = cls._match_team_statistics(match)
        if not stat_home or not stat_away:
            return False
        return stat_home.score_ft is not None and stat_away.score_ft is not None

    @staticmethod
    def _has_mean_statistics(match) -> bool:
        mean_stats = match.mean_statistics
        if not isinstance(mean_stats, list) or len(mean_stats) < 2:
            return False
        mean_home = next((m for m in mean_stats if m.get("id_team") == match.id_team_home), None)
        mean_away = next((m for m in mean_stats if m.get("id_team") == match.id_team_away), None)
        return bool(mean_home and mean_away)

    @staticmethod
    def _has_odds_for_market(match, market: str) -> bool:
        odds_rows = match.odds or []
        if not odds_rows:
            return False
        payload = (odds_rows[0].to_dict() or {}).get(market)
        return isinstance(payload, dict) and bool(payload)

    def build_under_over_threshold_report(
        self,
        seasons: Optional[list[int]] = None,
        leagues: Optional[list[int]] = None,
        thresholds: tuple[float, ...] = UNDER_OVER_THRESHOLDS,
        top_n: int = 40,
        min_train_rows: int = 30,
        min_valid_rows: int = 10,
        n_splits: int = 5,
        matches: Optional[list] = None,
    ) -> dict[str, Any]:
        """Report Fase 0 (obbligatorio prima di addestrare, vedi task):
        per OGNI soglia under/over, quante fixture FT hanno odds valide PER
        QUELLA soglia, statistics/mean_statistics complete, distribuzione
        stagione/lega, sbilanciamento classi (over/under) e - soprattutto -
        quante RIGHE SONO REALMENTE UTILIZZABILI per il training (STESSA
        identica funzione `FilterMarketService._build_row` usata da
        `FilterMarketService.build_dataset`/`train_multi_market.train_market`,
        nessun filtro reinventato qui) e se il minimo per la CV temporale
        espandente e' raggiunto (STESSA soglia `len(cv_splits) < 2` gia'
        applicata da `train_multi_market._filter_valid_splits`/`train_market`).

        `matches` (opzionale): lista di ORM `Match` GIA' caricata dal
        chiamante (es. una sola query pesante condivisa con `build_report`
        in uno script di analisi) - evita di ripetere N volte la stessa
        query costosa (statistics/odds via `lazy=selectin` su tutte le
        fixture) SOLO per questo report aggiuntivo. Se `None` (default),
        comportamento invariato: `self.match_repo.search_all()`.
        """
        from src.ml.validation.temporal_split import expanding_window_splits
        from src.service_ia.training.market_service.filter_market_service import FilterMarketService
        from src.service_ia.utility.utils import convert_orm_match_to_dict

        source_matches = matches if matches is not None else self.match_repo.search_all()
        filtered_matches = self._filter_matches(source_matches, seasons=seasons, leagues=leagues)
        matches_ft = [m for m in filtered_matches if (m.status or "").upper() == "FT"]
        fixtures_ft_total = len(matches_ft)
        # Un'UNICA conversione a dict riusata per TUTTE le soglie sotto
        # (stesso formato consumato da `FilterMarketService._build_row`),
        # invece di richiamare `build_dataset` (che rifarebbe una query DB
        # completa) una volta per soglia.
        match_dicts_ft = convert_orm_match_to_dict(matches_ft)

        fixtures_with_complete_statistics = sum(1 for m in matches_ft if self._has_complete_statistics(m))
        fixtures_with_mean_statistics = sum(1 for m in matches_ft if self._has_mean_statistics(m))

        market_service = FilterMarketService()
        per_threshold: dict[str, Any] = {}
        fixture_ids_by_threshold: dict[str, set[int]] = {}

        for threshold in thresholds:
            market = _under_over_market_key(threshold)

            fixtures_with_odds_for_market = sum(1 for m in matches_ft if self._has_odds_for_market(m, market))

            # Stessa identica funzione di riga usata da `build_dataset` (nessuna
            # logica di filtro duplicata), applicata pero' sui dict GIA' in
            # memoria invece di rifare una query DB per ciascuna soglia.
            built_rows = []
            for match_dict in match_dicts_ft:
                if leagues and match_dict.get("current_league") not in leagues:
                    continue
                row = market_service._build_row(match=match_dict, market=market, with_target=True)
                if row:
                    built_rows.append(row)

            df = (
                pd.DataFrame(built_rows).replace([float("inf"), float("-inf")], pd.NA).fillna(0)
                if built_rows
                else pd.DataFrame()
            )

            rows = int(len(df))
            fixture_ids_by_threshold[market] = set(df["id_fixture"].astype(int).tolist()) if rows else set()

            class_balance: dict[str, int] = {"under_0": 0, "over_1": 0}
            positive_class_ratio_over: Optional[float] = None
            by_season: list[dict[str, Any]] = []
            by_league: list[dict[str, Any]] = []
            cv_folds_available = 0
            min_train_size: Optional[int] = None
            min_valid_size: Optional[int] = None

            if rows:
                y = df["y"].astype(int)
                counts = y.value_counts().to_dict()
                class_balance = {"under_0": int(counts.get(0, 0)), "over_1": int(counts.get(1, 0))}
                positive_class_ratio_over = round(float(y.mean()), 6)

                if "season" in df.columns:
                    season_counts = df["season"].value_counts(dropna=False).sort_index()
                    by_season = [
                        {"season": (int(k) if pd.notna(k) else None), "count": int(v)}
                        for k, v in season_counts.items()
                    ][:top_n]
                if "league" in df.columns:
                    league_counts = df["league"].value_counts(dropna=False).sort_values(ascending=False)
                    by_league = [
                        {"league": (int(k) if pd.notna(k) else None), "count": int(v)}
                        for k, v in league_counts.items()
                    ][:top_n]

                df_time = df.copy()
                if "prediction_at" in df_time.columns:
                    df_time["prediction_at"] = pd.to_datetime(df_time["prediction_at"], utc=True, errors="coerce")
                    df_time = df_time.dropna(subset=["prediction_at"]).sort_values(by=["prediction_at", "id_fixture"]).reset_index(drop=True)
                    min_train_size = max(min_train_rows, int(len(df_time) * 0.45))
                    min_valid_size = max(min_valid_rows, int(len(df_time) * 0.1))
                    cv_folds_available = len(
                        expanding_window_splits(
                            frame=df_time,
                            time_col="prediction_at",
                            n_splits=n_splits,
                            min_train_size=min_train_size,
                            min_valid_size=min_valid_size,
                        )
                    )

            per_threshold[market] = {
                "threshold": threshold,
                "fixtures_ft_total": fixtures_ft_total,
                "fixtures_with_odds_for_market": fixtures_with_odds_for_market,
                "odds_coverage_ratio": round(fixtures_with_odds_for_market / max(1, fixtures_ft_total), 6),
                "fixtures_with_complete_statistics": fixtures_with_complete_statistics,
                "fixtures_with_mean_statistics": fixtures_with_mean_statistics,
                "usable_rows_for_training": rows,
                "class_balance": class_balance,
                "positive_class_ratio_over": positive_class_ratio_over,
                "distribution_by_season": by_season,
                "distribution_by_league": by_league,
                "cv_temporal": {
                    "strategy": "expanding_window",
                    "n_splits_requested": n_splits,
                    "min_train_size": min_train_size,
                    "min_valid_size": min_valid_size,
                    "folds_available": cv_folds_available,
                    # Stessa condizione di `train_multi_market._filter_valid_splits`
                    # combinata con l'uso che ne fa `train_market` (< 2 fold -> skip).
                    "meets_minimum_for_training": cv_folds_available >= 2,
                },
            }

        non_empty_sets = [s for s in fixture_ids_by_threshold.values()]
        fixtures_with_all_thresholds = set.intersection(*non_empty_sets) if non_empty_sets and all(non_empty_sets) else set()

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "filters": {"seasons": seasons or [], "leagues": leagues or []},
            "fixtures_ft_total": fixtures_ft_total,
            "fixtures_with_complete_statistics": fixtures_with_complete_statistics,
            "fixtures_with_mean_statistics": fixtures_with_mean_statistics,
            "per_threshold": per_threshold,
            "cross_threshold": {
                "fixtures_with_all_four_thresholds_odds_available": len(fixtures_with_all_thresholds),
                "ratio_over_fixtures_ft_total": round(len(fixtures_with_all_thresholds) / max(1, fixtures_ft_total), 6),
                "note": (
                    "Intersezione delle fixture che hanno ODDS valide per TUTTE le soglie "
                    "richieste (oltre a statistics/mean_statistics complete). NON e' un "
                    "vincolo per l'approccio 'hierarchical'/'goal_distribution' di "
                    "totals_market.py (che usa le odds del solo reference_market "
                    "'under_over_2_5' per tutte le soglie, mentre il target reale deriva "
                    "sempre da total_goals): e' invece il vincolo rilevante per un "
                    "confronto 'binary_independent' che usasse feature odds specifiche "
                    "per soglia (vedi train_market('under_over_X') isolato)."
                ),
            },
        }


