import unittest

from src.ml.datasets.point_in_time_builder import PointInTimeDatasetBuilder


class TestPointInTimeDatasetBuilder(unittest.TestCase):
    def _sample_match(self):
        return {
            "id_fixture": 1001,
            "season": 2026,
            "status": "FT",
            "date_match": "2026-09-01T18:45:00+00:00",
            "id_team_home": 10,
            "id_team_away": 20,
            "mean_statistics": [
                {"id_team": 10, "mean_Shots on Goal": 5.0, "mean_Corner Kicks": 6.0},
                {"id_team": 20, "mean_Shots on Goal": 3.0, "mean_Corner Kicks": 4.0},
            ],
            "statistics": [
                {"statistics_team_id": 10, "score_ft": 2, "score_ht": 1},
                {"statistics_team_id": 20, "score_ft": 1, "score_ht": 1},
            ],
        }

    def test_build_from_records_filters_future_snapshots(self):
        builder = PointInTimeDatasetBuilder()
        matches = [self._sample_match()]
        snapshots = [
            {
                "fixture_id": 1001,
                "market": "under_over_2_5",
                "period": "full_time",
                "line": "2.5",
                "outcome": "Over 2.5",
                "odd": 1.95,
                "captured_at": "2026-09-01T17:00:00+00:00",
            },
            {
                "fixture_id": 1001,
                "market": "under_over_2_5",
                "period": "full_time",
                "line": "2.5",
                "outcome": "Over 2.5",
                "odd": 1.80,
                "captured_at": "2026-09-01T19:00:00+00:00",
            },
        ]

        dataset = builder.build_from_records(matches=matches, snapshots=snapshots, market="under_over_2_5")

        self.assertEqual(len(dataset.frame), 1)
        self.assertTrue(builder.assert_no_leakage(dataset.frame))
        self.assertEqual(dataset.frame.iloc[0]["line"], "2.5")

    def test_dataset_version_is_reproducible(self):
        builder = PointInTimeDatasetBuilder()
        matches = [self._sample_match()]
        snapshots = [
            {
                "fixture_id": 1001,
                "market": "under_over_2_5",
                "period": "full_time",
                "line": "2.5",
                "outcome": "Over 2.5",
                "odd": 1.95,
                "captured_at": "2026-09-01T17:00:00+00:00",
            }
        ]

        dataset_a = builder.build_from_records(matches=matches, snapshots=snapshots, market="under_over_2_5")
        dataset_b = builder.build_from_records(matches=matches, snapshots=snapshots, market="under_over_2_5")

        self.assertEqual(dataset_a.version, dataset_b.version)
        self.assertIn("prediction_at", dataset_a.frame.columns)


if __name__ == "__main__":
    unittest.main()

