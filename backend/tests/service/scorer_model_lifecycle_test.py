"""Ciclo di vita del modello marcatori: holdout -> candidate -> promozione con policy -> caricamento.
Dataset sintetico e stimatore minuscolo: verificano il CODICE (registro separato, gate, no auto-promozione)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.player.scoring_dataset import FEATURE_COLUMNS, TARGET
from src.service_ia.scorer import model_store
from src.service_ia.training.model_registry import ModelRegistry


def _synthetic(n: int = 40_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    role = rng.integers(0, 3, size=n)
    for i, c in enumerate(("pos_F", "pos_M", "pos_D")):
        df[c] = (role == i).astype(int)
    logit = -2.6 + 0.9 * df["pos_F"] + 0.6 * df["rate_last20_shrunk"]
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["fixture_id"] = np.arange(n) // 22
    df["match_date"] = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(df["fixture_id"] // 8, unit="D")
    return df


def _small():
    return mock.patch.object(model_store, "_estimator", lambda family="voting": make_pipeline(StandardScaler(), LogisticRegression()))


class ScorerLifecycleTest(unittest.TestCase):
    def test_candidate_is_registered_in_separate_registry_and_never_auto_promoted(self):
        with _small():
            run = model_store.train_candidate(_synthetic())
        self.assertEqual(run["stage"], "candidate")
        self.assertEqual(run["market"], model_store.MARKET)
        for key in ("log_loss", "brier", "ece", "auc", "baseline_log_loss", "by_role"):
            self.assertIn(key, run["metrics"])
        self.assertLess(run["metrics"]["log_loss"], run["metrics"]["baseline_log_loss"])
        # registro principale dei mercati di partita: nessun effetto collaterale
        self.assertEqual(ModelRegistry().list_markets(), [])
        self.assertIsNone(model_store.load_production_bundle())

    def test_promotion_goes_through_policy_then_bundle_loads(self):
        with _small():
            run = model_store.train_candidate(_synthetic())
        reg = model_store.registry()
        verdict = reg.evaluate_promotion(run["run_id"], to_stage="production")
        self.assertTrue(verdict["allowed"])
        self.assertTrue(reg.promote_with_policy(run["run_id"], to_stage="production")["promoted"])
        bundle = model_store.load_production_bundle()
        self.assertEqual(bundle["run_id"], run["run_id"])
        frame = _synthetic(100)
        p = model_store.predict_proba(bundle, frame)
        self.assertTrue(((p > 0) & (p < 1)).all())

    def test_incompatible_sklearn_version_is_not_loaded(self):
        with _small():
            run = model_store.train_candidate(_synthetic())
        reg = model_store.registry()
        reg.promote_with_policy(run["run_id"], to_stage="production")
        with mock.patch.object(model_store.sklearn, "__version__", "0.0.0"):
            self.assertIsNone(model_store.load_production_bundle())

    def test_auto_promote_requires_checks_and_policy(self):
        with _small():
            run = model_store.train_candidate(_synthetic())
        out = model_store.auto_promote(run)
        self.assertTrue(out["promoted"], out)
        self.assertEqual(model_store.load_production_bundle()["run_id"], run["run_id"])
        # candidate con calibrazione pessima: non promosso, production invariata
        bad = dict(run, run_id="x", metrics=dict(run["metrics"], ece=0.2))
        self.assertFalse(model_store.auto_promote(bad)["promoted"])
        # nessun guadagno sulla baseline
        flat = dict(run, metrics=dict(run["metrics"], log_loss=run["metrics"]["baseline_log_loss"]))
        self.assertIn("baseline", " ".join(model_store.auto_promote(flat)["reasons"]))

    def test_holdout_compares_all_families_and_keeps_the_best(self):
        with _small():
            m = model_store.holdout_metrics(_synthetic())
        self.assertEqual(set(m["families"]), set(model_store.FAMILIES))
        self.assertEqual(m["log_loss"], min(f["log_loss"] for f in m["families"].values()))

    def test_holdout_is_temporal(self):
        with _small():
            m = model_store.holdout_metrics(_synthetic())
        self.assertGreater(m["sample_size"], 3000)
        self.assertIn("holdout_from", m)


if __name__ == "__main__":
    unittest.main()
