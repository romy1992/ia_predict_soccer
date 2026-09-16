import os
import tempfile
import unittest

from src.service_ia.training.model_paths import (
    destination_subdir,
    is_archived_model_path,
    relative_to_best_models,
    resolve_model_path,
    to_container_path,
)


class TestRelativeToBestModels(unittest.TestCase):
    def test_riconosce_le_tre_forme_di_percorso(self):
        # Container, Windows (la macchina dell'operatore) e relativo dalla
        # radice del repo: tutte e tre devono dare lo stesso relativo.
        self.assertEqual(
            relative_to_best_models("/app/best_models/under_over/under_over_1_5/m.pkl"),
            "under_over/under_over_1_5/m.pkl",
        )
        self.assertEqual(
            relative_to_best_models(r"C:\Users\trott\git\ia_predict_soccer\best_models\m.pkl"),
            "m.pkl",
        )
        self.assertEqual(relative_to_best_models("best_models/archivio/m.pkl"), "archivio/m.pkl")

    def test_percorso_fuori_da_best_models(self):
        self.assertIsNone(relative_to_best_models("/tmp/altrove/m.pkl"))
        self.assertIsNone(relative_to_best_models(""))


class TestIsArchivedModelPath(unittest.TestCase):
    def test_riconosce_larchivio_in_ogni_forma_di_percorso(self):
        self.assertTrue(is_archived_model_path("/app/best_models/archivio/m.pkl"))
        self.assertTrue(is_archived_model_path(r"C:\git\repo\best_models\archivio\dc\m.pkl"))
        self.assertTrue(is_archived_model_path("best_models/archivio/m.pkl"))

    def test_percorso_attivo_o_estraneo_non_e_archiviato(self):
        self.assertFalse(is_archived_model_path("/app/best_models/under_over/under_over_1_5/m.pkl"))
        self.assertFalse(is_archived_model_path("/app/best_models/h2h/m.pkl"))
        self.assertFalse(is_archived_model_path("/tmp/altrove/m.pkl"))
        self.assertFalse(is_archived_model_path(None))
        self.assertFalse(is_archived_model_path(""))


class TestToContainerPath(unittest.TestCase):
    def test_preserva_la_sottocartella(self):
        # La regressione che questo test blocca: ricostruire il percorso dal
        # solo nome file appiattirebbe il modello nella radice.
        self.assertEqual(
            to_container_path(r"C:\git\repo\best_models\under_over\under_over_2_5\m.pkl"),
            "/app/best_models/under_over/under_over_2_5/m.pkl",
        )

    def test_percorso_gia_in_forma_container_invariato(self):
        percorso = "/app/best_models/archivio/m.pkl"
        self.assertEqual(to_container_path(percorso), percorso)

    def test_percorso_estraneo_restituito_invariato(self):
        # Meglio lasciare un percorso che non si sa tradurre che inventarne
        # uno dentro best_models.
        self.assertEqual(to_container_path("/tmp/altrove/m.pkl"), "/tmp/altrove/m.pkl")


class TestDestinationSubdir(unittest.TestCase):
    def test_mercati_della_procedura_nuova(self):
        self.assertEqual(destination_subdir("under_over_1_5"), os.path.join("under_over", "under_over_1_5"))
        self.assertEqual(destination_subdir("under_over_3_5"), os.path.join("under_over", "under_over_3_5"))

    def test_gli_altri_mercati_restano_nella_radice(self):
        self.assertEqual(destination_subdir("under_over_4_5"), "")
        self.assertEqual(destination_subdir("goal_no_goal"), "")

    def test_h2h_e_dc_stanno_nella_cartella_del_mercato(self):
        self.assertEqual(destination_subdir("h2h"), "h2h")
        self.assertEqual(destination_subdir("dc"), "dc")


class TestResolveModelPath(unittest.TestCase):
    def test_il_percorso_registrato_vince_quando_esiste(self):
        with tempfile.TemporaryDirectory() as tmp:
            registrato = os.path.join(tmp, "under_over", "under_over_1_5", "m.pkl")
            omonimo = os.path.join(tmp, "archivio", "m.pkl")
            for percorso in (registrato, omonimo):
                os.makedirs(os.path.dirname(percorso), exist_ok=True)
                open(percorso, "w").close()

            self.assertEqual(resolve_model_path(registrato, root=tmp), registrato)

    def test_traduce_il_percorso_container_in_locale(self):
        with tempfile.TemporaryDirectory() as tmp:
            locale = os.path.join(tmp, "under_over", "under_over_2_5", "m.pkl")
            os.makedirs(os.path.dirname(locale), exist_ok=True)
            open(locale, "w").close()

            risolto = resolve_model_path("/app/best_models/under_over/under_over_2_5/m.pkl", root=tmp)
            self.assertEqual(risolto, locale)

    def test_trova_il_file_spostato_da_una_riga_rimasta_indietro(self):
        # Caso reale della riorganizzazione: la riga punta ancora alla radice,
        # il file e' gia' in archivio. Meglio servire il modello che far
        # sparire la predizione.
        with tempfile.TemporaryDirectory() as tmp:
            spostato = os.path.join(tmp, "archivio", "goal_no_goal_champion.pkl")
            os.makedirs(os.path.dirname(spostato), exist_ok=True)
            open(spostato, "w").close()

            risolto = resolve_model_path("/app/best_models/goal_no_goal_champion.pkl", root=tmp)
            self.assertEqual(risolto, spostato)

    def test_trova_h2h_nella_cartella_del_mercato(self):
        with tempfile.TemporaryDirectory() as tmp:
            locale = os.path.join(tmp, "h2h", "h2h_champion_20260915.pkl")
            os.makedirs(os.path.dirname(locale), exist_ok=True)
            open(locale, "w").close()
            risolto = resolve_model_path("/app/best_models/h2h/h2h_champion_20260915.pkl", root=tmp)
            self.assertEqual(risolto, locale)

    def test_trova_dc_spostato_in_archivio_per_mercato(self):
        with tempfile.TemporaryDirectory() as tmp:
            spostato = os.path.join(tmp, "archivio", "dc", "dc_champion.pkl")
            os.makedirs(os.path.dirname(spostato), exist_ok=True)
            open(spostato, "w").close()
            risolto = resolve_model_path("/app/best_models/dc_champion.pkl", root=tmp)
            self.assertEqual(risolto, spostato)

    def test_nessun_file_da_nessuna_parte(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_model_path("/app/best_models/manca.pkl", root=tmp))
            self.assertIsNone(resolve_model_path(None, root=tmp))


if __name__ == "__main__":
    unittest.main()
