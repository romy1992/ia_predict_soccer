import math
from datetime import datetime, timezone
from types import SimpleNamespace

from src.oracle.betting_statistics_service import BettingStatisticsService
from src.oracle.betslip.betslip_builder import (
    BetslipGenerationResult,
    GeneratedSlip,
)
from src.oracle.betslip.pick_pool import CandidatePick
from src.oracle.betslip.proposal_snapshot_service import BetslipProposalSnapshotService
from src.oracle.ledger.official_performance_service import OFFICIAL_COHORT, OFFICIAL_SOURCE
from src.oracle.ledger.settlement_rules import SettlementResolution


class FakeProposalRepo:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.keys = {}

    def save_revision(self, snapshot):
        if snapshot.snapshot_key in self.keys:
            return self.keys[snapshot.snapshot_key], False
        for row in self.rows:
            if getattr(row, "logical_slip_id", None) == snapshot.logical_slip_id:
                row.is_latest = False
        snapshot.is_latest = True
        self.keys[snapshot.snapshot_key] = snapshot
        self.rows.append(snapshot)
        return snapshot, True

    def list_all(self, **_kwargs):
        return self.rows

    def list_pending_settlement(self, **_kwargs):
        return [row for row in self.rows if row.shadow_status == "PENDING"]

    def save_settlement(self, snapshot):
        return snapshot


def _generation(odd=1.8):
    leg = CandidatePick(
        fixture_id=10,
        market="1x2",
        outcome="Home",
        decision="PLAY",
        p_model=0.6,
        p_market_fair=0.5,
        odd=odd,
        fair_odd=2.0,
        prob_edge=0.1,
        ev=0.08,
        samples=3,
        policy_version="policy-v2",
    )
    slip = GeneratedSlip(
        slip_id="logical-slip",
        profile_name="SAFE",
        profile_version="safe-v2",
        risk_label="LOW",
        n_legs=1,
        legs=[leg],
        combined_odd=odd,
        adjusted_probability=0.6,
        combined_model_void_odd=1 / 0.6,
        combined_edge_absolute=odd - (1 / 0.6),
        combined_expected_roi=0.6 * odd - 1,
        decision_policy_version="slip-policy-v1",
        correlation_ruleset_version="correlation-v1",
        situation="PLAY",
    )
    return BetslipGenerationResult(
        generated_at=datetime.now(timezone.utc).isoformat(),
        correlation_ruleset_version="correlation-v1",
        pool_considered=1,
        profiles={"SAFE": [slip]},
    )


def test_proposal_snapshots_are_idempotent_and_revisioned_on_odds_change():
    repo = FakeProposalRepo()
    service = BetslipProposalSnapshotService(repo=repo)

    first = service.save_generation(reference_date="2026-09-11", generation=_generation(1.8))
    same = service.save_generation(reference_date="2026-09-11", generation=_generation(1.8))
    changed = service.save_generation(reference_date="2026-09-11", generation=_generation(1.9))

    assert first["proposals_created"] == 1
    assert same["proposals_unchanged"] == 1
    assert changed["proposals_created"] == 1
    assert len(repo.rows) == 2
    assert repo.rows[0].logical_slip_id == repo.rows[1].logical_slip_id
    assert repo.rows[0].snapshot_key != repo.rows[1].snapshot_key


def test_saved_proposals_are_returned_without_regeneration():
    saved_at = datetime.now(timezone.utc)
    repo = FakeProposalRepo(
        [
            SimpleNamespace(
                id="snapshot-1",
                reference_date="2026-09-10",
                generated_at=saved_at,
                is_latest=True,
                payload={"slip_id": "slip-1", "profile_name": "SAFE"},
            )
        ]
    )

    rows = BetslipProposalSnapshotService(repo=repo).list_saved(
        reference_date="2026-09-10"
    )

    assert rows == [
        {
            "slip_id": "slip-1",
            "profile_name": "SAFE",
            "snapshot_id": "snapshot-1",
            "reference_date": "2026-09-10",
            "saved_at": saved_at.isoformat(),
            "is_latest": True,
        }
    ]


class FakeShadowLedger:
    def __init__(self, resolutions):
        self.resolutions = resolutions

    def _resolve_outcome_for_fixture(self, fixture_id, **_kwargs):
        return self.resolutions[fixture_id]


def test_shadow_settlement_tracks_result_without_becoming_official():
    now = datetime.now(timezone.utc)
    snapshot = SimpleNamespace(
        id="shadow-1",
        payload={
            "legs": [
                {
                    "fixture_id": 1,
                    "market": "1x2",
                    "outcome": "Home",
                    "odd": 1.8,
                    "kickoff_at": "2026-09-10T10:00:00+00:00",
                },
                {
                    "fixture_id": 2,
                    "market": "goal_no_goal",
                    "outcome": "Yes",
                    "odd": 1.5,
                    "kickoff_at": "2026-09-10T12:00:00+00:00",
                },
            ]
        },
        shadow_status="PENDING",
        shadow_stake=1.0,
        shadow_effective_odd=None,
        shadow_return=None,
        shadow_profit=None,
        shadow_settlement=None,
        shadow_settled_at=None,
    )
    repo = FakeProposalRepo([snapshot])
    service = BetslipProposalSnapshotService(
        repo=repo,
        ledger_service=FakeShadowLedger(
            {
                1: SettlementResolution(actual_outcome="Home"),
                2: SettlementResolution(actual_outcome="Yes"),
            }
        ),
    )

    report = service.settle_pending(before=now)

    assert report["shadow_settled"] == 1
    assert snapshot.shadow_status == "WON"
    assert math.isclose(snapshot.shadow_effective_odd, 2.7)
    assert math.isclose(snapshot.shadow_return, 2.7)
    assert math.isclose(snapshot.shadow_profit, 1.7)


class FakeLedgerRepo:
    def list_all(self, **_kwargs):
        now = datetime.now(timezone.utc)
        return [
            SimpleNamespace(
                source=OFFICIAL_SOURCE,
                cohort=OFFICIAL_COHORT,
                market="1x2",
                kickoff_at=now,
                created_at=now,
                decision="PLAY",
                odd=2.0,
                is_settled=True,
                settlement_status="settled_win",
                stake=1.0,
                pnl=1.0,
                prob_edge=0.1,
                ev=0.2,
                fixture_id=10,
            )
        ]


class FakeOfficialService:
    def statistics(self, **_kwargs):
        return {
            "total": 1,
            "won": 1,
            "lost": 0,
            "pending": 0,
            "void": 0,
            "net_profit": 1.0,
            "realized_roi": 1.0,
            "by_day": {"2026-09-11": {"total": 1}},
            "by_profile": {},
            "by_event_count": {},
            "by_market_combination": {},
            "bankroll_curve": [],
        }


def test_unified_statistics_keep_proposals_and_official_performance_separate():
    proposal_repo = FakeProposalRepo()
    BetslipProposalSnapshotService(repo=proposal_repo).save_generation(
        reference_date="2099-01-01",
        generation=_generation(),
    )
    service = BettingStatisticsService(
        ledger_repo=FakeLedgerRepo(),
        proposal_repo=proposal_repo,
        official_betslip_service=FakeOfficialService(),
    )
    report = service.report(days=30)

    assert report["overview"]["official_predictions"]["wins"] == 1
    assert report["overview"]["official_slips"]["total"] == 1
    assert report["overview"]["proposals"]["generated"] == 1
    assert report["slips"]["proposals_daily"][0]["date"] == "2099-01-01"
    assert report["markets"]["daily"][0]["market"] == "1x2"
    assert report["overview"]["simulated_portfolios"]["ALL"]["total"] == 1
    assert report["overview"]["simulated_portfolios"]["PLAY"]["pending"] == 1
