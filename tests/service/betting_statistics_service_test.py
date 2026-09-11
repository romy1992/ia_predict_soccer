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


class FakeProposalRepo:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.keys = {}

    def save_revision(self, snapshot):
        if snapshot.snapshot_key in self.keys:
            return self.keys[snapshot.snapshot_key], False
        self.keys[snapshot.snapshot_key] = snapshot
        self.rows.append(snapshot)
        return snapshot, True

    def list_all(self, **_kwargs):
        return self.rows


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
