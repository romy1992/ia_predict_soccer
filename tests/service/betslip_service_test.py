import unittest
from datetime import date, datetime, timedelta, timezone
from unittest import mock

from src.oracle.betslip.betslip_builder import DEFAULT_SLIP_PROFILES
from src.oracle.betslip.betslip_service import BetslipService
from src.oracle.betslip.pick_pool import CandidatePick, PickPoolResult, PoolPick


def _pool_pick(fixture_id, market, outcome, odd, p_model, ev=0.1, run_id="run-1", kickoff_at=None):
    candidate = CandidatePick(
        fixture_id=fixture_id,
        market=market,
        outcome=outcome,
        decision="PLAY",
        odd=odd,
        p_model=p_model,
        ev=ev,
        model_run_id=run_id,
        kickoff_at=kickoff_at,
    )
    return PoolPick(candidate=candidate, inclusion_reason="passed_policy:x", pool_policy_version="pick_pool_policy_v1")


class TestBetslipService(unittest.TestCase):
    def _service_with_pool(self, picks, snapshot_service=None):
        pick_pool_service = mock.Mock()
        pick_pool_service.build_pool_for_day.return_value = PickPoolResult(
            pool_id="deadbeef1234",
            generated_at="2026-09-05T10:00:00+00:00",
            policy_version="pick_pool_policy_v1",
            picks=picks,
            excluded=[],
        )
        return (
            BetslipService(
                pick_pool_service=pick_pool_service,
                proposal_snapshot_service=snapshot_service,
            ),
            pick_pool_service,
        )

    def test_generate_for_day_reuses_pick_pool_service(self):
        picks = [
            _pool_pick(1, "h2h", "Home", 1.8, 0.60),
            _pool_pick(2, "h2h", "Away", 1.9, 0.58),
        ]
        service, pick_pool_service = self._service_with_pool(picks)

        pool_result, generation = service.generate_for_day(target_date=date(2026, 9, 5))

        pick_pool_service.build_pool_for_day.assert_called_once()
        _, kwargs = pick_pool_service.build_pool_for_day.call_args
        self.assertEqual(kwargs["target_date"], date(2026, 9, 5))
        self.assertEqual(pool_result.pool_id, "deadbeef1234")
        self.assertEqual(generation.pool_considered, 2)
        self.assertEqual(set(generation.profiles.keys()), {p.name for p in DEFAULT_SLIP_PROFILES})
        self.assertEqual(len(generation.profiles["SAFE"]), 1)

    def test_generate_for_day_with_no_picks_returns_empty_profiles(self):
        service, _ = self._service_with_pool([])

        pool_result, generation = service.generate_for_day(target_date=date(2026, 9, 5))

        self.assertEqual(generation.pool_considered, 0)
        for slips in generation.profiles.values():
            self.assertEqual(slips, [])

    def test_generate_for_day_passes_markets_through(self):
        service, pick_pool_service = self._service_with_pool([])
        service.generate_for_day(target_date=date(2026, 9, 5), markets=["h2h", "goal_no_goal"])

        _, kwargs = pick_pool_service.build_pool_for_day.call_args
        self.assertEqual(kwargs["markets"], ["h2h", "goal_no_goal"])

    def test_snapshot_generation_rejects_past_dates(self):
        snapshot_service = mock.Mock()
        service, pick_pool_service = self._service_with_pool([], snapshot_service)
        now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)

        with self.assertRaisesRegex(ValueError, "solo in consultazione"):
            service.generate_and_snapshot_for_day(
                target_date=date(2026, 9, 10),
                now=now,
            )

        pick_pool_service.build_pool_for_day.assert_not_called()
        snapshot_service.save_generation.assert_not_called()

    def test_snapshot_generation_keeps_only_pre_kickoff_slips(self):
        snapshot_service = mock.Mock()
        snapshot_service.save_generation.return_value = {
            "proposals_seen": 0,
            "proposals_created": 0,
            "proposals_unchanged": 0,
        }
        now = datetime(2026, 9, 11, 10, 0, tzinfo=timezone.utc)
        picks = [
            _pool_pick(
                1,
                "h2h",
                "Home",
                1.8,
                0.60,
                kickoff_at=(now + timedelta(hours=2)).isoformat(),
            ),
            _pool_pick(
                2,
                "h2h",
                "Away",
                1.9,
                0.58,
                kickoff_at=(now - timedelta(minutes=1)).isoformat(),
            ),
        ]
        service, _ = self._service_with_pool(picks, snapshot_service)

        _, generation, report = service.generate_and_snapshot_for_day(
            target_date=now.date(),
            now=now,
        )

        self.assertEqual(generation.profiles["SAFE"], [])
        self.assertEqual(report["proposals_skipped_started"], 1)
        snapshot_service.save_generation.assert_called_once()


if __name__ == "__main__":
    unittest.main()

