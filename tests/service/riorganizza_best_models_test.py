import json
import os
import tempfile
import unittest

from scripts.maintenance.riorganizza_best_models import (
    Spostamento,
    collisioni,
    normalizza_in_forma_container,
    pianifica,
    riscrivi_righe,
    sposta,
)

# Come si presentava la radice di `best_models` prima della riorganizzazione:
# i modelli di tutti i mercati mescolati, distinti solo dal nome.
RADICE = [
    "goal_no_goal_champion.pkl",
    "under_over_1_5_champion_20260914.pkl",
    "under_over_1_5_champion_calibrator_20260914.pkl",
    "under_over_4_5_champion.pkl",
    "corners_line_9_5_champion.pkl",
    "note.txt",
]
NUOVI = {
    "under_over_1_5": [
        "under_over_1_5_champion_20260914.pkl",
        "under_over_1_5_champion_calibrator_20260914.pkl",
    ]
}


class TestPianifica(unittest.TestCase):
    def test_i_modelli_nuovi_vanno_nella_cartella_del_mercato(self):
        piano = {s.sorgente: s.destinazione for s in pianifica(RADICE, NUOVI)}

        self.assertEqual(
            piano["under_over_1_5_champion_20260914.pkl"],
            "under_over/under_over_1_5/under_over_1_5_champion_20260914.pkl",
        )
        self.assertEqual(
            piano["under_over_1_5_champion_calibrator_20260914.pkl"],
            "under_over/under_over_1_5/under_over_1_5_champion_calibrator_20260914.pkl",
        )

    def test_tutto_il_resto_va_in_archivio(self):
        piano = {s.sorgente: s.destinazione for s in pianifica(RADICE, NUOVI)}

        for nome in ("goal_no_goal_champion.pkl", "under_over_4_5_champion.pkl", "corners_line_9_5_champion.pkl"):
            self.assertEqual(piano[nome], f"archivio/{nome}", nome)

    def test_i_file_che_non_sono_modelli_restano_dove_sono(self):
        piano = {s.sorgente for s in pianifica(RADICE, NUOVI)}
        self.assertNotIn("note.txt", piano)

    def test_senza_modelli_nuovi_va_tutto_in_archivio(self):
        # Un mercato non ancora rifatto (production a 69 feature) non compare
        # in `nuovi`: il suo modello e' vecchia procedura come gli altri.
        piano = pianifica(RADICE, {})
        self.assertTrue(all(s.destinazione.startswith("archivio/") for s in piano))


class TestRiscriviRighe(unittest.TestCase):
    def _righe(self):
        return [
            {
                "run_id": "under_over_1_5_A",
                "model_path": "/app/best_models/under_over_1_5_champion_20260914.pkl",
                "metadata_path": "/app/best_models/registry/under_over_1_5_A.json",
                "extra": {
                    "calibration": {
                        "calibrator_path": "/app/best_models/under_over_1_5_champion_calibrator_20260914.pkl"
                    }
                },
            },
            {"run_id": "goal_no_goal_A", "model_path": "/app/best_models/goal_no_goal_champion.pkl"},
            {"run_id": "goal_no_goal_B", "model_path": "/app/best_models/goal_no_goal_champion.pkl"},
            {"run_id": "vecchio", "model_path": "/app/best_models/archivio/under_over_2_5_champion.pkl"},
        ]

    def test_modello_e_calibratore_seguono_il_file(self):
        righe = self._righe()
        riscrivi_righe(righe, pianifica(RADICE, NUOVI))

        self.assertEqual(
            righe[0]["model_path"],
            "/app/best_models/under_over/under_over_1_5/under_over_1_5_champion_20260914.pkl",
        )
        self.assertEqual(
            righe[0]["extra"]["calibration"]["calibrator_path"],
            "/app/best_models/under_over/under_over_1_5/under_over_1_5_champion_calibrator_20260914.pkl",
        )

    def test_tutte_le_righe_che_condividono_il_file_vengono_aggiornate(self):
        # Il difetto trovato il 2026-09-14: piu' run puntavano allo stesso
        # nome file, e aggiornarne una sola lasciava le altre rotte.
        righe = self._righe()
        riscrivi_righe(righe, pianifica(RADICE, NUOVI))

        self.assertEqual(righe[1]["model_path"], "/app/best_models/archivio/goal_no_goal_champion.pkl")
        self.assertEqual(righe[2]["model_path"], "/app/best_models/archivio/goal_no_goal_champion.pkl")

    def test_le_righe_gia_in_archivio_non_vengono_toccate(self):
        righe = self._righe()
        riscrivi_righe(righe, pianifica(RADICE, NUOVI))

        self.assertEqual(righe[3]["model_path"], "/app/best_models/archivio/under_over_2_5_champion.pkl")

    def test_il_metadata_del_registry_resta_dov_e(self):
        righe = self._righe()
        riscrivi_righe(righe, pianifica(RADICE, NUOVI))

        self.assertEqual(righe[0]["metadata_path"], "/app/best_models/registry/under_over_1_5_A.json")


class TestNormalizzaInFormaContainer(unittest.TestCase):
    def test_traduce_i_percorsi_rimasti_di_un_altra_macchina(self):
        righe = [
            {
                "model_path": r"C:\Users\trott\git\ia_predict_soccer_export\best_models\m.pkl",
                "metadata_path": r"C:\Users\trott\git\ia_predict_soccer_export\best_models\registry\m.json",
                "extra": {"calibration": {"calibrator_path": r"C:\altro\best_models\c.pkl"}},
            }
        ]
        toccati = normalizza_in_forma_container(righe)

        self.assertEqual(toccati, 3)
        self.assertEqual(righe[0]["model_path"], "/app/best_models/m.pkl")
        self.assertEqual(righe[0]["metadata_path"], "/app/best_models/registry/m.json")
        self.assertEqual(righe[0]["extra"]["calibration"]["calibrator_path"], "/app/best_models/c.pkl")

    def test_le_righe_gia_in_forma_container_non_cambiano(self):
        righe = [{"model_path": "/app/best_models/archivio/m.pkl"}]
        self.assertEqual(normalizza_in_forma_container(righe), 0)


class TestSpostamentoSuDisco(unittest.TestCase):
    def _radice_finta(self, tmp):
        for nome in RADICE:
            open(os.path.join(tmp, nome), "w").close()

    def test_la_simulazione_non_tocca_il_disco(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._radice_finta(tmp)
            piano = pianifica(RADICE, NUOVI)

            sposta(tmp, piano, applica=False)

            self.assertTrue(os.path.exists(os.path.join(tmp, "goal_no_goal_champion.pkl")))
            self.assertFalse(os.path.isdir(os.path.join(tmp, "archivio")))

    def test_esecuzione_sposta_senza_cancellare(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._radice_finta(tmp)
            piano = pianifica(RADICE, NUOVI)

            spostati = sposta(tmp, piano, applica=True)

            self.assertEqual(spostati, len(piano))
            self.assertTrue(os.path.exists(os.path.join(tmp, "archivio", "goal_no_goal_champion.pkl")))
            self.assertTrue(
                os.path.exists(
                    os.path.join(tmp, "under_over", "under_over_1_5", "under_over_1_5_champion_20260914.pkl")
                )
            )
            # `note.txt` non era nel piano: resta dov'e'.
            self.assertTrue(os.path.exists(os.path.join(tmp, "note.txt")))

    def test_una_destinazione_occupata_viene_segnalata_prima_di_spostare(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._radice_finta(tmp)
            os.makedirs(os.path.join(tmp, "archivio"))
            open(os.path.join(tmp, "archivio", "goal_no_goal_champion.pkl"), "w").close()

            occupate = collisioni(tmp, pianifica(RADICE, NUOVI))

            self.assertEqual(occupate, ["archivio/goal_no_goal_champion.pkl"])


class TestRegistryScrittoDavvero(unittest.TestCase):
    def test_il_giro_completo_lascia_zero_righe_rotte(self):
        """Il criterio con cui l'intervento si dichiara riuscito: dopo lo
        spostamento, ogni `model_path` del registry punta a un file che
        esiste davvero."""
        with tempfile.TemporaryDirectory() as tmp:
            for nome in RADICE:
                open(os.path.join(tmp, nome), "w").close()
            righe = [
                {"model_path": f"/app/best_models/{nome}"}
                for nome in RADICE
                if nome.endswith(".pkl")
            ]

            piano = pianifica(RADICE, NUOVI)
            sposta(tmp, piano, applica=True)
            riscrivi_righe(righe, piano)

            rotte = [
                r["model_path"]
                for r in righe
                if not os.path.exists(r["model_path"].replace("/app/best_models", tmp))
            ]
            self.assertEqual(rotte, [])

            # E il file rimane leggibile come JSONL, cosi' com'e' scritto nel
            # registry vero.
            index = os.path.join(tmp, "index.jsonl")
            with open(index, "w", encoding="utf-8") as f:
                for r in righe:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(index, encoding="utf-8") as f:
                self.assertEqual(len([json.loads(l) for l in f if l.strip()]), len(righe))


class TestSpostamentoDataclass(unittest.TestCase):
    def test_il_piano_e_immutabile(self):
        s = Spostamento(sorgente="a.pkl", destinazione="archivio/a.pkl", motivo="vecchia procedura")
        with self.assertRaises(Exception):
            s.sorgente = "altro.pkl"


if __name__ == "__main__":
    unittest.main()
