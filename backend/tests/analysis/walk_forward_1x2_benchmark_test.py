"""Test del benchmark walk-forward 1X2 / Over-Under 2.5 / Goal-No Goal.

Verificano le proprieta' che rendono credibili i numeri del report: metriche
corrette su casi noti, matrice dei punteggi coerente, nessuna informazione dal
futuro in Elo, forma e forza squadre, parsing no-vig delle quote."""
import json
import unittest

import numpy as np
import pandas as pd

from scripts.analysis import walk_forward_1x2_benchmark as bench
from scripts.analysis.export_top5_matches_for_benchmark import no_vig_from_books


def _matches(n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Campionato sintetico: 8 squadre, una giornata ogni 3 giorni."""
    rng = np.random.default_rng(seed)
    teams = list(range(1, 9))
    rows = []
    fixture = 1
    start = pd.Timestamp("2020-08-01T18:00:00Z")
    for day in range(n_days):
        order = rng.permutation(teams)
        for i in range(0, 8, 2):
            home, away = int(order[i]), int(order[i + 1])
            rows.append(
                {
                    "id_fixture": fixture,
                    "date_utc": (start + pd.Timedelta(days=3 * day)).isoformat(),
                    "league": 135,
                    "season": 2020 + day // 60,
                    "home_team_id": home,
                    "away_team_id": away,
                    "home_goals": int(rng.poisson(1.2 + 0.1 * (8 - home))),
                    "away_goals": int(rng.poisson(0.8 + 0.1 * (8 - away))),
                    "home_xg": np.nan,
                    "away_xg": np.nan,
                    "odds_source": "sports-api",
                    "h2h_books": 3,
                    "p_mkt_home": 0.45,
                    "p_mkt_draw": 0.27,
                    "p_mkt_away": 0.28,
                    "h2h_overround": 0.05,
                    "ou25_books": 3,
                    "p_mkt_over25": 0.5,
                    "ou25_dispersion": 1.02,
                    "btts_books": 0,
                    "p_mkt_btts_yes": np.nan,
                }
            )
            fixture += 1
    return pd.DataFrame(rows)


def _load(frame: pd.DataFrame, tmp_path) -> pd.DataFrame:
    path = tmp_path / "matches.csv.gz"
    frame.to_csv(path, index=False)
    return bench.load_dataset(str(path))


class TestMetrics(unittest.TestCase):
    def test_previsione_perfetta_e_uniforme(self):
        y = np.array([0, 1, 2, 0])
        perfect = np.eye(3)[y]
        metrics = bench.multiclass_metrics(perfect, y)
        self.assertAlmostEqual(metrics["log_loss"], 0.0, places=4)
        self.assertAlmostEqual(metrics["brier"], 0.0, places=4)
        self.assertAlmostEqual(metrics["rps"], 0.0, places=4)
        self.assertEqual(metrics["accuracy"], 1.0)

        uniform = np.full((4, 3), 1 / 3)
        metrics = bench.multiclass_metrics(uniform, y)
        self.assertAlmostEqual(metrics["log_loss"], round(float(np.log(3)), 5), places=4)
        self.assertAlmostEqual(metrics["brier"], round(2 / 3, 5), places=4)

    def test_rps_premia_l_errore_vicino(self):
        y = np.array([0])
        near = bench.multiclass_metrics(np.array([[0.0, 1.0, 0.0]]), y)["rps"]
        far = bench.multiclass_metrics(np.array([[0.0, 0.0, 1.0]]), y)["rps"]
        self.assertLess(near, far)

    def test_binary_metrics_su_caso_noto(self):
        metrics = bench.binary_metrics(np.array([0.8, 0.2]), np.array([1, 0]))
        self.assertAlmostEqual(metrics["log_loss"], round(float(-np.log(0.8)), 5), places=4)
        self.assertAlmostEqual(metrics["brier"], 0.04, places=4)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_bootstrap_appaiato_riproducibile_e_centrato(self):
        delta = np.random.default_rng(1).normal(0.01, 0.05, 2000)
        a = bench.paired_bootstrap(delta, 200, np.random.default_rng(42))
        b = bench.paired_bootstrap(delta, 200, np.random.default_rng(42))
        self.assertEqual(a, b)
        self.assertLess(a["ci95_low"], a["mean_delta"])
        self.assertGreater(a["ci95_high"], a["mean_delta"])
        self.assertGreater(a["ci95_low"], 0.0)


class TestScoreMatrix(unittest.TestCase):
    def test_probabilita_coerenti(self):
        lam = np.array([1.6, 0.9, 2.4])
        mu = np.array([1.1, 1.3, 0.6])
        for rho in (0.0, -0.08):
            markets = bench.score_matrix_markets(lam, mu, rho)
            np.testing.assert_allclose(markets["1x2"].sum(axis=1), 1.0, atol=1e-9)
            self.assertTrue(np.all((markets["over25"] > 0) & (markets["over25"] < 1)))
            self.assertTrue(np.all((markets["btts"] > 0) & (markets["btts"] < 1)))

    def test_poisson_indipendente_ha_la_formula_chiusa_attesa(self):
        lam, mu = np.array([1.5]), np.array([1.0])
        markets = bench.score_matrix_markets(lam, mu, 0.0)
        expected_btts = (1 - np.exp(-1.5)) * (1 - np.exp(-1.0))
        total = 2.5
        expected_over = 1 - np.exp(-total) * (1 + total + total**2 / 2)
        self.assertAlmostEqual(float(markets["btts"][0]), expected_btts, places=5)
        self.assertAlmostEqual(float(markets["over25"][0]), expected_over, places=5)

    def test_dixon_coles_sposta_i_pareggi_bassi_ma_non_il_totale_gol(self):
        """La correzione redistribuisce massa fra 0-0, 1-0, 0-1 e 1-1 senza
        cambiarne la somma: l'Over 2.5 resta identico al Poisson, cambiano 1X2
        e Goal/No Goal."""
        lam, mu = np.array([1.4]), np.array([1.1])
        plain = bench.score_matrix_markets(lam, mu, 0.0)
        corrected = bench.score_matrix_markets(lam, mu, -0.1)
        self.assertAlmostEqual(float(plain["over25"][0]), float(corrected["over25"][0]), places=9)
        self.assertGreater(float(corrected["1x2"][0, 1]), float(plain["1x2"][0, 1]))
        self.assertGreater(float(corrected["btts"][0]), float(plain["btts"][0]))


class TestNoFutureInformation(unittest.TestCase):
    """Cambiare il risultato di una partita non deve cambiare nessuna
    feature pre-partita di quella partita ne' delle precedenti."""

    def setUp(self):
        import pathlib
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = pathlib.Path(self._tmp.name)
        self.raw = _matches()
        self.frame = _load(self.raw, self.tmp_path)
        self.pivot = len(self.frame) // 2
        altered = self.frame.copy()
        altered.loc[self.pivot, ["home_goals", "away_goals"]] = [9, 0]
        self.altered = altered

    def test_elo(self):
        config = bench.EloConfig(20, 70, True)
        base = bench.compute_elo(self.frame, config)
        changed = bench.compute_elo(self.altered, config)
        same_day = (self.frame.day <= self.frame.day[self.pivot]).to_numpy()
        np.testing.assert_allclose(base[same_day], changed[same_day])
        self.assertFalse(np.allclose(base[~same_day], changed[~same_day]))

    def test_forma(self):
        base = bench.compute_form_features(self.frame)
        changed = bench.compute_form_features(self.altered)
        same_day = (self.frame.day <= self.frame.day[self.pivot]).to_numpy()
        pd.testing.assert_frame_equal(base[same_day], changed[same_day])
        self.assertFalse(base[~same_day].equals(changed[~same_day]))

    def test_forza_squadre_usa_solo_i_mesi_precedenti(self):
        for frame in (self.frame, self.altered):
            home_eff, away_eff = bench._effective_teams(frame)
            frame["home_eff"] = home_eff
            frame["away_eff"] = away_eff
        start = self.frame.date.min() + pd.Timedelta(days=200)
        base = bench.compute_team_strength_predictions(self.frame, 0.002, start)
        changed = bench.compute_team_strength_predictions(self.altered, 0.002, start)
        pivot_month = self.frame.month_start[self.pivot]
        up_to_pivot_month = (self.frame.month_start <= pivot_month).to_numpy()
        np.testing.assert_allclose(base["lam"][up_to_pivot_month], changed["lam"][up_to_pivot_month], equal_nan=True)
        later = ~up_to_pivot_month & ~np.isnan(base["lam"])
        self.assertTrue(later.any())
        self.assertFalse(np.allclose(base["lam"][later], changed["lam"][later]))

    def test_prima_di_avere_storia_non_si_prevede(self):
        home_eff, away_eff = bench._effective_teams(self.frame)
        self.frame["home_eff"] = home_eff
        self.frame["away_eff"] = away_eff
        pred = bench.compute_team_strength_predictions(self.frame, 0.002, self.frame.date.min())
        first_month = (self.frame.month_start == self.frame.month_start.min()).to_numpy()
        self.assertTrue(np.isnan(pred["lam"][first_month]).all())


class TestTeamStrength(unittest.TestCase):
    def test_recupera_la_squadra_piu_forte_e_rho_resta_nei_limiti(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            frame = _load(_matches(n_days=200, seed=3), pathlib.Path(tmp))
        home_eff, away_eff = bench._effective_teams(frame)
        frame["home_eff"] = home_eff
        frame["away_eff"] = away_eff
        fit = bench.fit_team_strength(frame, frame.date.max() + pd.Timedelta(days=1), 0.0)
        self.assertIsNotNone(fit)
        self.assertTrue(-0.3 <= fit["rho"] <= 0.3)
        probe = pd.DataFrame({"home_eff": [1, 8], "away_eff": [8, 1]})
        lam, mu = bench.predict_team_strength(fit, probe)
        # La squadra 1 e' costruita come la piu' prolifica, la 8 come la meno.
        self.assertGreater(lam[0], lam[1])
        self.assertGreater(mu[1], mu[0])

    def test_squadra_sconosciuta_non_fa_fallire_la_previsione(self):
        import pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            frame = _load(_matches(n_days=100), pathlib.Path(tmp))
        home_eff, away_eff = bench._effective_teams(frame)
        frame["home_eff"] = home_eff
        frame["away_eff"] = away_eff
        fit = bench.fit_team_strength(frame, frame.date.max(), 0.001)
        lam, mu = bench.predict_team_strength(fit, pd.DataFrame({"home_eff": [999], "away_eff": [1]}))
        self.assertTrue(np.isfinite(lam).all() and np.isfinite(mu).all())

    def test_neopromossa_usa_la_pseudo_squadra_nelle_prime_partite(self):
        import pathlib
        import tempfile

        raw = _matches(n_days=80)
        late = raw[raw.season == 2021].copy()
        late.loc[late.home_team_id == 1, "home_team_id"] = 99
        late.loc[late.away_team_id == 1, "away_team_id"] = 99
        raw = pd.concat([raw[raw.season == 2020], late])
        with tempfile.TemporaryDirectory() as tmp:
            frame = _load(raw, pathlib.Path(tmp))
        home_eff, away_eff = bench._effective_teams(frame)
        played = ((frame.home_team_id == 99) | (frame.away_team_id == 99)).to_numpy()
        eff = np.where(frame.home_team_id.to_numpy() == 99, home_eff, away_eff)[played]
        self.assertTrue((eff[: bench.NEWCOMER_MATCHES] == bench.NEWCOMER).all())
        self.assertTrue((eff[bench.NEWCOMER_MATCHES:] == 99).all())
        # Le squadre presenti dalla prima stagione non sono mai "neopromosse".
        self.assertFalse(((home_eff == bench.NEWCOMER) & (frame.home_team_id != 99)).any())


class TestFolds(unittest.TestCase):
    def test_i_fold_partono_dal_periodo_di_test_e_non_si_sovrappongono(self):
        import pathlib
        import tempfile

        raw = _matches(n_days=400)
        with tempfile.TemporaryDirectory() as tmp:
            frame = _load(raw, pathlib.Path(tmp))
        folds = bench.build_folds(frame)
        self.assertTrue(folds)
        seen: set[int] = set()
        for fold in folds:
            rows = frame.loc[fold["test_index"]]
            self.assertTrue((rows.date >= bench.TEST_START).all())
            self.assertGreaterEqual(len(rows), bench.MIN_TEST_ROWS)
            self.assertTrue((rows.date >= fold["start"]).all())
            self.assertFalse(seen & set(fold["test_index"]))
            seen |= set(fold["test_index"])


class TestNoVigFromBooks(unittest.TestCase):
    def test_media_dei_bookmaker_completi(self):
        raw = json.dumps(
            {"home_A": "2.00", "draw_A": "4.00", "away_A": "4.00", "home_B": 2.0, "draw_B": 4.0, "away_B": 4.0, "home_C": "1.9"}
        )
        out = no_vig_from_books(raw, {"home": ("home_",), "draw": ("draw_",), "away": ("away_",)})
        self.assertEqual(out["books"], 2)  # C non quota tutti gli esiti
        self.assertAlmostEqual(out["home"], 0.5)
        self.assertAlmostEqual(out["draw"], 0.25)
        self.assertAlmostEqual(out["home"] + out["draw"] + out["away"], 1.0)
        self.assertAlmostEqual(out["overround"], 0.0)

    def test_le_due_grafie_storiche_dell_over_under(self):
        prefixes = {"over": ("over 2.5_", "over_2.5_"), "under": ("under 2.5_", "under_2.5_")}
        for raw in (
            {"over 2.5_Bet365": "1.80", "under 2.5_Bet365": "2.00"},
            {"over_2.5_Unibet": 1.8, "under_2.5_Unibet": 2.0},
        ):
            out = no_vig_from_books(json.dumps(raw), prefixes)
            self.assertEqual(out["books"], 1)
            self.assertAlmostEqual(out["over"], (1 / 1.8) / (1 / 1.8 + 1 / 2.0))

    def test_goal_e_no_goal_non_si_confondono(self):
        prefixes = {"yes": ("goal_",), "no": ("no_goal_",)}
        both = no_vig_from_books(json.dumps({"goal_X": 1.9, "no_goal_X": 1.9}), prefixes)
        self.assertEqual(both["books"], 1)
        self.assertAlmostEqual(both["yes"], 0.5)
        only_no = no_vig_from_books(json.dumps({"no_goal__Bet365": "1.70", "no_goal__Unibet": "1.77"}), prefixes)
        self.assertEqual(only_no["books"], 0)  # un solo lato: nessuna probabilita' inventata
        self.assertIsNone(only_no["yes"])

    def test_dispersione_segnala_quote_non_confrontabili(self):
        prefixes = {"over": ("over_2.5_",), "under": ("under_2.5_",)}
        raw = {"over_2.5_A": 2.42, "under_2.5_A": 1.66, "over_2.5_B": 1.85, "under_2.5_B": 1.97}
        out = no_vig_from_books(json.dumps(raw), prefixes)
        self.assertAlmostEqual(out["dispersion"], 2.42 / 1.85)

    def test_input_vuoti_o_non_validi(self):
        prefixes = {"home": ("home_",), "draw": ("draw_",), "away": ("away_",)}
        for raw in (None, float("nan"), "", "null", "{}", "non-json", "[1, 2]", json.dumps({"home_A": "1.0"})):
            out = no_vig_from_books(raw, prefixes)
            self.assertEqual(out["books"], 0, raw)
            self.assertIsNone(out["home"])


if __name__ == "__main__":
    unittest.main()
