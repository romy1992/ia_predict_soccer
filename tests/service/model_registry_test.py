import os
import tempfile
import unittest

from src.ml.registry.promotion_policy import PromotionGateThresholds, PromotionPolicy
from src.service_ia.training.model_registry import ModelRegistry


GOOD_METRICS = {"log_loss": 0.5, "brier": 0.15, "ece": 0.05, "auc": 0.7, "sample_size": 200}
BAD_METRICS = {"log_loss": 50.0, "brier": 0.9, "ece": 0.9, "auc": 0.1, "sample_size": 5}
PERMISSIVE_POLICY = PromotionPolicy(
    gate=PromotionGateThresholds(
        max_log_loss=None, max_brier=None, max_ece=None, min_auc=None, min_sample_size=0,
        require_at_least_one_metric=False,
    )
)


class TestModelRegistry(unittest.TestCase):
    def test_register_persists_lifecycle_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="under_over_2_5",
                model_name="logistic",
                metrics={"f1": 0.71},
                feature_names=["f1", "f2"],
                dataset_version="dataset:v1",
                feature_version="features:v1",
                windows={
                    "train": {"from": "2024-01-01", "to": "2025-06-01"},
                    "validation": {"from": "2025-06-02", "to": "2025-08-01"},
                    "test": {"from": "2025-08-02", "to": "2025-09-01"},
                },
                git_sha="abc1234",
                stage="candidate",
            )

            latest = registry.get_latest(market="under_over_2_5")
            self.assertIsNotNone(latest)
            self.assertEqual(latest["run_id"], payload["run_id"])
            self.assertEqual(latest["dataset_version"], "dataset:v1")
            self.assertEqual(latest["feature_version"], "features:v1")
            self.assertEqual(latest["git_sha"], "abc1234")
            self.assertEqual(latest["current_stage"], "candidate")
            self.assertEqual(len(latest["promotion_history"]), 0)

    def test_latest_is_not_implicitly_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload_1 = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="under_over_2_5",
                model_name="logistic",
                metrics={"f1": 0.71},
                feature_names=["f1", "f2"],
            )
            payload_2 = registry.register(
                model_path=os.path.join(tmp, "model_b.pkl"),
                market="under_over_2_5",
                model_name="stacking",
                metrics={"f1": 0.75},
                feature_names=["f1", "f2", "f3"],
            )

            self.assertIsNone(registry.get_production(market="under_over_2_5"))

            promoted = registry.promote(
                run_id=payload_1["run_id"],
                to_stage="production",
                reason="manual promote",
                actor="test",
            )
            self.assertIsNotNone(promoted)
            self.assertEqual(promoted["current_stage"], "production")

            latest = registry.get_latest(market="under_over_2_5")
            production = registry.get_production(market="under_over_2_5")
            self.assertIsNotNone(latest)
            self.assertIsNotNone(production)
            self.assertEqual(latest["market"], "under_over_2_5")
            self.assertEqual(latest["run_id"], payload_2["run_id"])
            self.assertEqual(production["run_id"], payload_1["run_id"])
            self.assertNotEqual(latest["run_id"], production["run_id"])

            tail = registry.tail(limit=10, market="under_over_2_5")
            self.assertEqual(len(tail), 2)

            markets = registry.list_markets()
            self.assertEqual(markets, ["under_over_2_5"])

    def test_promoting_new_production_demotes_previous(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload_1 = registry.register(
                model_path=os.path.join(tmp, "model_a.pkl"),
                market="h2h",
                model_name="logistic",
            )
            payload_2 = registry.register(
                model_path=os.path.join(tmp, "model_b.pkl"),
                market="h2h",
                model_name="stacking",
            )

            registry.promote(run_id=payload_1["run_id"], to_stage="production", reason="first production", actor="test")
            registry.promote(run_id=payload_2["run_id"], to_stage="production", reason="better model", actor="test")

            production = registry.get_production(market="h2h")
            run_1 = registry.get_run(payload_1["run_id"])
            run_2 = registry.get_run(payload_2["run_id"])

            self.assertIsNotNone(production)
            self.assertIsNotNone(run_1)
            self.assertIsNotNone(run_2)
            self.assertEqual(production["run_id"], payload_2["run_id"])
            self.assertEqual(run_1["current_stage"], "champion")
            self.assertEqual(run_2["current_stage"], "production")
            self.assertGreaterEqual(len(run_1["promotion_history"]), 2)
            self.assertGreaterEqual(len(run_2["promotion_history"]), 1)

    def test_lookup_production_per_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            under_prod = registry.register(
                model_path=os.path.join(tmp, "under_prod.pkl"),
                market="under_over_2_5",
                model_name="logistic",
            )
            under_latest = registry.register(
                model_path=os.path.join(tmp, "under_latest.pkl"),
                market="under_over_2_5",
                model_name="stacking",
            )
            h2h_prod = registry.register(
                model_path=os.path.join(tmp, "h2h_prod.pkl"),
                market="h2h",
                model_name="xgboost",
            )

            registry.promote(run_id=under_prod["run_id"], to_stage="production", reason="uo prod", actor="test")
            registry.promote(run_id=h2h_prod["run_id"], to_stage="production", reason="h2h prod", actor="test")

            production_under = registry.get_production(market="under_over_2_5")
            production_h2h = registry.get_production(market="h2h")
            latest_under = registry.get_latest(market="under_over_2_5")

            self.assertIsNotNone(production_under)
            self.assertIsNotNone(production_h2h)
            self.assertIsNotNone(latest_under)
            self.assertEqual(production_under["run_id"], under_prod["run_id"])
            self.assertEqual(production_h2h["run_id"], h2h_prod["run_id"])
            self.assertEqual(latest_under["run_id"], under_latest["run_id"])
            self.assertNotEqual(latest_under["run_id"], production_under["run_id"])
            self.assertIsNone(registry.get_production(market="1x2"))

    def test_lifecycle_reaches_retired_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)

            payload = registry.register(
                model_path=os.path.join(tmp, "model_retire.pkl"),
                market="under_over_2_5",
                model_name="logistic",
            )

            registry.promote(run_id=payload["run_id"], to_stage="champion", reason="validated", actor="test")
            registry.promote(run_id=payload["run_id"], to_stage="production", reason="go live", actor="test")
            registry.promote(run_id=payload["run_id"], to_stage="retired", reason="sunset", actor="test")

            run_state = registry.get_run(payload["run_id"])
            production = registry.get_production(market="under_over_2_5")

            self.assertIsNotNone(run_state)
            self.assertEqual(run_state["current_stage"], "retired")
            self.assertGreaterEqual(len(run_state["promotion_history"]), 3)
            self.assertIsNone(production)

    # ------------------------------------------------------------------
    # OPS-02: evaluate_promotion (dry-run)
    # ------------------------------------------------------------------
    def test_evaluate_promotion_returns_none_for_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            self.assertIsNone(registry.evaluate_promotion(run_id="does-not-exist", to_stage="production"))

    def test_evaluate_promotion_is_a_dry_run_and_never_mutates_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"),
                market="h2h",
                model_name="logistic",
                metrics=GOOD_METRICS,
            )

            result = registry.evaluate_promotion(run_id=payload["run_id"], to_stage="production")

            self.assertIsNotNone(result)
            self.assertTrue(result["allowed"])
            # Nessuna mutazione: ancora candidate, nessun evento in audit trail.
            run_state = registry.get_run(payload["run_id"])
            self.assertEqual(run_state["current_stage"], "candidate")
            self.assertEqual(registry.list_promotion_events(market="h2h"), [])

    def test_evaluate_promotion_reports_production_run_id_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            prod_payload = registry.register(
                model_path=os.path.join(tmp, "prod.pkl"), market="h2h", model_name="logistic", metrics=GOOD_METRICS
            )
            registry.promote(run_id=prod_payload["run_id"], to_stage="production", reason="seed", actor="test")

            candidate_payload = registry.register(
                model_path=os.path.join(tmp, "candidate.pkl"), market="h2h", model_name="stacking", metrics=GOOD_METRICS
            )
            result = registry.evaluate_promotion(run_id=candidate_payload["run_id"], to_stage="production")
            self.assertEqual(result["production_run_id"], prod_payload["run_id"])

    # ------------------------------------------------------------------
    # OPS-02: promote_with_policy (Gate metriche + Promozione controllata)
    # ------------------------------------------------------------------
    def test_promote_with_policy_blocks_bad_metrics_and_leaves_stage_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=BAD_METRICS
            )

            result = registry.promote_with_policy(run_id=payload["run_id"], to_stage="production")

            self.assertFalse(result["promoted"])
            self.assertFalse(result["evaluation"]["allowed"])
            run_state = registry.get_run(payload["run_id"])
            self.assertEqual(run_state["current_stage"], "candidate")
            self.assertIsNone(registry.get_production(market="h2h"))

    def test_promote_with_policy_blocked_attempt_is_still_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=BAD_METRICS
            )
            registry.promote_with_policy(run_id=payload["run_id"], to_stage="production", reason="try", actor="qa")

            events = registry.list_promotion_events(market="h2h")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["metadata"]["event_type"], "promotion_blocked")
            self.assertEqual(events[0]["metadata"]["attempted_stage"], "production")
            # from_stage == to_stage: nessuna transizione reale registrata.
            self.assertEqual(events[0]["from_stage"], events[0]["to_stage"])

    def test_promote_with_policy_allows_first_promotion_with_good_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=GOOD_METRICS
            )

            result = registry.promote_with_policy(run_id=payload["run_id"], to_stage="production", actor="qa")

            self.assertTrue(result["promoted"])
            self.assertEqual(registry.get_production(market="h2h")["run_id"], payload["run_id"])
            events = registry.list_promotion_events(market="h2h")
            self.assertEqual(events[-1]["metadata"]["event_type"], "promotion_approved")

    def test_promote_with_policy_blocks_candidate_worse_than_production(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            strong = registry.register(
                model_path=os.path.join(tmp, "strong.pkl"),
                market="h2h",
                model_name="logistic",
                metrics={"selection_score": 0.9},
            )
            registry.promote(run_id=strong["run_id"], to_stage="production", reason="seed", actor="test")

            weaker = registry.register(
                model_path=os.path.join(tmp, "weaker.pkl"),
                market="h2h",
                model_name="stacking",
                metrics={"selection_score": 0.3},
            )
            result = registry.promote_with_policy(run_id=weaker["run_id"], to_stage="production")

            self.assertFalse(result["promoted"])
            self.assertEqual(registry.get_production(market="h2h")["run_id"], strong["run_id"])

    def test_promote_with_policy_force_bypasses_gate_and_is_audited_as_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=BAD_METRICS
            )

            result = registry.promote_with_policy(run_id=payload["run_id"], to_stage="production", force=True, actor="admin")

            self.assertTrue(result["promoted"])
            self.assertEqual(registry.get_production(market="h2h")["run_id"], payload["run_id"])
            events = registry.list_promotion_events(market="h2h")
            self.assertEqual(events[-1]["metadata"]["event_type"], "promotion_forced")

    def test_promote_with_policy_raises_for_missing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            with self.assertRaises(ValueError):
                registry.promote_with_policy(run_id="does-not-exist", to_stage="production")

    def test_promote_with_policy_champion_stage_ignores_production_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            payload = registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=GOOD_METRICS
            )
            result = registry.promote_with_policy(run_id=payload["run_id"], to_stage="champion")
            self.assertTrue(result["promoted"])
            self.assertIsNone(result["evaluation"]["comparison"])

    def test_last_training_never_becomes_production_via_register_alone(self):
        """Acceptance criteria OPS-02: l'ultimo training registrato non
        diventa MAI automaticamente production, nemmeno passando per
        `promote_with_policy` se non viene esplicitamente invocato."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            registry.register(
                model_path=os.path.join(tmp, "model.pkl"), market="h2h", model_name="logistic", metrics=GOOD_METRICS
            )
            self.assertIsNone(registry.get_production(market="h2h"))

    # ------------------------------------------------------------------
    # OPS-02: rollback
    # ------------------------------------------------------------------
    def test_rollback_to_previous_production_without_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            first = registry.register(model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic")
            second = registry.register(model_path=os.path.join(tmp, "b.pkl"), market="h2h", model_name="stacking")

            registry.promote(run_id=first["run_id"], to_stage="production", reason="v1", actor="test")
            registry.promote(run_id=second["run_id"], to_stage="production", reason="v2", actor="test")
            self.assertEqual(registry.get_production(market="h2h")["run_id"], second["run_id"])

            result = registry.rollback(market="h2h", reason="v2 e' instabile", actor="oncall")

            self.assertEqual(result["rolled_back_to"], first["run_id"])
            self.assertEqual(result["rolled_back_from"], second["run_id"])
            self.assertEqual(registry.get_production(market="h2h")["run_id"], first["run_id"])

    def test_rollback_to_explicit_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            first = registry.register(model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic")
            second = registry.register(model_path=os.path.join(tmp, "b.pkl"), market="h2h", model_name="stacking")
            registry.promote(run_id=second["run_id"], to_stage="production", reason="v2", actor="test")

            result = registry.rollback(market="h2h", to_run_id=first["run_id"], actor="oncall")

            self.assertEqual(registry.get_production(market="h2h")["run_id"], first["run_id"])
            events = registry.list_promotion_events(market="h2h")
            self.assertEqual(events[-1]["metadata"]["event_type"], "rollback")

    def test_rollback_raises_when_no_previous_production_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            only = registry.register(model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic")
            registry.promote(run_id=only["run_id"], to_stage="production", reason="v1", actor="test")

            with self.assertRaises(ValueError):
                registry.rollback(market="h2h")

    def test_rollback_raises_when_run_id_belongs_to_different_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            h2h_run = registry.register(model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic")
            other_market_run = registry.register(
                model_path=os.path.join(tmp, "b.pkl"), market="under_over_2_5", model_name="logistic"
            )
            registry.promote(run_id=h2h_run["run_id"], to_stage="production", reason="v1", actor="test")

            with self.assertRaises(ValueError):
                registry.rollback(market="h2h", to_run_id=other_market_run["run_id"])

    # ------------------------------------------------------------------
    # OPS-02: audit trail (list_promotion_events)
    # ------------------------------------------------------------------
    def test_list_promotion_events_includes_blocked_promotions_and_rollbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            first = registry.register(
                model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic", metrics=GOOD_METRICS
            )
            second = registry.register(
                model_path=os.path.join(tmp, "b.pkl"), market="h2h", model_name="stacking", metrics=BAD_METRICS
            )
            third = registry.register(
                model_path=os.path.join(tmp, "c.pkl"), market="h2h", model_name="voting", metrics=GOOD_METRICS
            )

            registry.promote_with_policy(run_id=first["run_id"], to_stage="production", actor="qa")
            registry.promote_with_policy(run_id=second["run_id"], to_stage="production", actor="qa")  # bloccato
            registry.promote_with_policy(run_id=third["run_id"], to_stage="production", actor="qa")  # approvato
            registry.rollback(market="h2h", reason="third e' instabile", actor="oncall")  # torna a first

            events = registry.list_promotion_events(market="h2h")
            event_types = [event["metadata"].get("event_type") for event in events]
            self.assertIn("promotion_approved", event_types)
            self.assertIn("promotion_blocked", event_types)
            self.assertIn("rollback", event_types)
            self.assertEqual(registry.get_production(market="h2h")["run_id"], first["run_id"])

    def test_list_promotion_events_filters_by_market(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            h2h_run = registry.register(model_path=os.path.join(tmp, "a.pkl"), market="h2h", model_name="logistic")
            other_run = registry.register(
                model_path=os.path.join(tmp, "b.pkl"), market="under_over_2_5", model_name="logistic"
            )
            registry.promote(run_id=h2h_run["run_id"], to_stage="production", reason="v1", actor="test")
            registry.promote(run_id=other_run["run_id"], to_stage="production", reason="v1", actor="test")

            h2h_events = registry.list_promotion_events(market="h2h")
            self.assertTrue(all(event["market"] == "h2h" for event in h2h_events))

            all_events = registry.list_promotion_events()
            self.assertGreaterEqual(len(all_events), len(h2h_events))


class TestListActiveMarkets(unittest.TestCase):
    def test_esclude_il_mercato_con_produzione_in_archivio(self):
        # Scenario reale della riorganizzazione 2026-09-15: under_over_4_5
        # non e' mai stato rifatto con la procedura nuova, il suo .pkl e'
        # finito in archivio/ - deve sparire da Dashboard ma restare in
        # list_markets() per lo storico/diagnostics.
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            archiviato = registry.register(
                model_path=os.path.join(tmp, "best_models", "archivio", "under_over_4_5_champion.pkl"),
                market="under_over_4_5",
                model_name="logistic",
            )
            attivo = registry.register(
                model_path=os.path.join(tmp, "best_models", "under_over", "under_over_1_5", "champion.pkl"),
                market="under_over_1_5",
                model_name="logistic",
            )
            registry.promote(run_id=archiviato["run_id"], to_stage="production", actor="test")
            registry.promote(run_id=attivo["run_id"], to_stage="production", actor="test")

            self.assertIn("under_over_4_5", registry.list_markets())
            self.assertNotIn("under_over_4_5", registry.list_active_markets())
            self.assertIn("under_over_1_5", registry.list_active_markets())

    def test_mercato_senza_produzione_resta_attivo(self):
        # Un candidate ancora in valutazione (nessuna produzione) NON e'
        # "archiviato": deve restare visibile (badge "In coda" in Dashboard),
        # distinzione gia' esistente e voluta - vedi PredictionBadges.jsx.
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            registry.register(
                model_path=os.path.join(tmp, "candidate.pkl"), market="goal_no_goal", model_name="logistic"
            )

            self.assertIn("goal_no_goal", registry.list_active_markets())

    def test_produzione_fuori_da_archivio_resta_attiva(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = ModelRegistry(registry_dir=tmp)
            run = registry.register(
                model_path=os.path.join(tmp, "best_models", "h2h", "champion.pkl"),
                market="h2h",
                model_name="logistic",
            )
            registry.promote(run_id=run["run_id"], to_stage="production", actor="test")

            self.assertIn("h2h", registry.list_active_markets())


if __name__ == "__main__":
    unittest.main()


