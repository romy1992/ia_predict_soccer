import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.oracle.betslip.official_betslip_service import OfficialBetslipService
from src.oracle.ledger.settlement_rules import SettlementResolution
from src.repository.betting_slip_repository import BettingSlipRepository
from src.service_ia.model.match import Base, BettingSlip, BettingSlipPick


def _pick(position, fixture_id, outcome="Home", odd=2.0):
    return BettingSlipPick(
        position=position,
        fixture_id=fixture_id,
        kickoff_at=datetime.now(timezone.utc) - timedelta(hours=2),
        market="1x2",
        outcome=outcome,
        p_model=0.6,
        market_odd=odd,
        situation="PLAY",
        status="PENDING",
    )


def _slip(*picks):
    row = BettingSlip(
        id="slip-1",
        capture_key="capture",
        reference_date="2026-09-10",
        profile="SAFE",
        initial_situation="PLAY",
        status="PENDING",
        event_count=len(picks),
        combined_odd=4.0,
        stake=1.0,
        policy_version="slip_decision_policy_v1",
        correlation_version="correlation_ruleset_v1",
    )
    row.picks = list(picks)
    return row


class FakeRepo:
    def __init__(self, slip):
        self.slip = slip

    def list_pending(self, **_kwargs):
        return [self.slip]

    def save_settlement(self, slip):
        self.slip = slip
        return slip


class FakeLedger:
    def __init__(self, resolutions):
        self.resolutions = resolutions

    def _resolve_outcome_for_fixture(self, fixture_id, **_kwargs):
        return self.resolutions[fixture_id]


class FakeMatchRepo:
    def filter_by(self, dict_search):
        fixture_id = dict_search["id_fixture"]
        return SimpleNamespace(first=lambda: SimpleNamespace(score_home=2, score_away=1, id_fixture=fixture_id))


class OfficialBetslipSettlementTest(unittest.TestCase):
    def service(self, slip, resolutions):
        return OfficialBetslipService(
            repo=FakeRepo(slip),
            ledger_service=FakeLedger(resolutions),
            match_repo=FakeMatchRepo(),
        )

    def test_won_with_void_leg_recalculates_effective_odd(self):
        slip = _slip(_pick(1, 1, odd=1.8), _pick(2, 2, odd=1.5))
        service = self.service(
            slip,
            {
                1: SettlementResolution(actual_outcome="Home"),
                2: SettlementResolution(void_status="void_cancelled"),
            },
        )
        service.settle_pending()
        self.assertEqual(slip.status, "WON")
        self.assertEqual([pick.status for pick in slip.picks], ["WON", "VOID"])
        self.assertAlmostEqual(slip.effective_combined_odd, 1.8)
        self.assertAlmostEqual(slip.actual_return, 1.8)
        self.assertAlmostEqual(slip.realized_profit, 0.8)

    def test_lost_leg_makes_slip_lost(self):
        slip = _slip(_pick(1, 1), _pick(2, 2))
        service = self.service(
            slip,
            {
                1: SettlementResolution(actual_outcome="Away"),
                2: SettlementResolution(actual_outcome="Home"),
            },
        )
        service.settle_pending()
        self.assertEqual(slip.status, "LOST")
        self.assertEqual(slip.actual_return, 0.0)
        self.assertEqual(slip.realized_profit, -1.0)

    def test_all_void_refunds_stake_and_excludes_loss(self):
        slip = _slip(_pick(1, 1), _pick(2, 2))
        service = self.service(
            slip,
            {
                1: SettlementResolution(void_status="void_cancelled"),
                2: SettlementResolution(void_status="void_abandoned"),
            },
        )
        service.settle_pending()
        self.assertEqual(slip.status, "VOID")
        self.assertEqual(slip.effective_combined_odd, 1.0)
        self.assertEqual(slip.actual_return, 1.0)
        self.assertEqual(slip.realized_profit, 0.0)


class OfficialBetslipRepositoryTest(unittest.TestCase):
    def setUp(self):
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(engine)
        self.session_local = sessionmaker(bind=engine, expire_on_commit=False)
        self.patch = mock.patch("src.repository.betting_slip_repository.SessionLocal", self.session_local)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def test_capture_key_is_idempotent_and_snapshot_is_not_overwritten(self):
        repo = BettingSlipRepository()
        original = _slip(_pick(1, 1), _pick(2, 2))
        _, created = repo.save_with_picks(original, original.picks)
        self.assertTrue(created)

        replacement = _slip(_pick(1, 3), _pick(2, 4))
        replacement.combined_odd = 99.0
        existing, created_again = repo.save_with_picks(replacement, replacement.picks)
        self.assertFalse(created_again)
        self.assertEqual(existing.combined_odd, 4.0)
        self.assertEqual([pick.fixture_id for pick in existing.picks], [1, 2])


if __name__ == "__main__":
    unittest.main()
