"""Riepilogo di classificazione compatto per i mercati giocatore: soglie, confusion matrix, lift, JSON."""

from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.evaluation.classification_report import compute_compact_classification_summary
from src.service_ia.player_cards import model_store as cards_store
from src.service_ia.scorer import model_store as scorer_store
from tests.service.player_cards_lifecycle_test import _synthetic as cards_data
from tests.service.scorer_model_lifecycle_test import _synthetic as scorer_data


def _scores(n=20_000, rate=0.15, seed=1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    p = np.clip(0.15 + 0.08 * (y - rate) + rng.normal(0, 0.05, n), 0.01, 0.99)
    return y, p


class CompactSummaryTest(unittest.TestCase):
    def test_standard_threshold_predicts_no_positive_when_the_event_is_rare(self):
        y, p = _scores()
        s = compute_compact_classification_summary(y, p)
        std = s["thresholds"]["standard"]
        self.assertEqual(std["confusion_matrix"]["tp"] + std["confusion_matrix"]["fp"], 0)
        self.assertAlmostEqual(std["accuracy"], s["majority_accuracy"], places=6)
        self.assertEqual(std["class1"]["recall"], 0.0)

    def test_every_threshold_has_a_consistent_confusion_matrix(self):
        y, p = _scores()
        s = compute_compact_classification_summary(y, p)
        self.assertEqual(set(s["thresholds"]), {"standard", "base_rate", "best_f1", "youden"})
        for name, t in s["thresholds"].items():
            cm = t["confusion_matrix"]
            self.assertEqual(sum(cm.values()), len(y), name)
            self.assertEqual(cm["fn"] + cm["tp"], int(y.sum()), name)
            self.assertAlmostEqual(t["accuracy"], (cm["tn"] + cm["tp"]) / len(y), places=9)
        # la soglia "base_rate" prevede (circa) tanti positivi quanti sono davvero
        cm = s["thresholds"]["base_rate"]["confusion_matrix"]
        self.assertAlmostEqual((cm["tp"] + cm["fp"]) / len(y), y.mean(), delta=0.01)

    def test_lift_is_above_one_for_an_informative_model(self):
        y, p = _scores()
        s = compute_compact_classification_summary(y, p)
        self.assertGreater(s["lift"]["top_10pct"]["lift"], 1.0)
        self.assertEqual(set(s["lift"]), {"top_1pct", "top_5pct", "top_10pct", "top_20pct"})

    def test_single_class_and_json_safe(self):
        self.assertEqual(compute_compact_classification_summary([0, 0, 0], [0.1, 0.2, 0.3])["status"], "single_class")
        y, p = _scores(2000)
        json.dumps(compute_compact_classification_summary(y, p))


def _small(store):
    return mock.patch.object(store, "_estimator", lambda family="voting": make_pipeline(StandardScaler(), LogisticRegression()))


class RegistryStoresClassificationTest(unittest.TestCase):
    def test_cards_candidate_carries_the_classification_summary(self):
        with _small(cards_store):
            run = cards_store.train_candidate(cards_data())
        c = run["metrics"]["classification"]
        self.assertEqual(c["status"], "ok")
        json.dumps(run["metrics"])

    def test_scorer_candidate_carries_the_classification_summary(self):
        with _small(scorer_store):
            run = scorer_store.train_candidate(scorer_data())
        self.assertEqual(run["metrics"]["classification"]["status"], "ok")
        json.dumps(run["metrics"])


if __name__ == "__main__":
    unittest.main()
