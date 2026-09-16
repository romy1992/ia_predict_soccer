import os
import tempfile
import unittest

from scripts.maintenance.promuovi_h2h_dc import destinazione_archivio, sposta_senza_cancellare


class TestArchivioH2hDc(unittest.TestCase):
    def test_destinazione_per_mercato(self):
        self.assertEqual(destinazione_archivio("h2h/h2h_champion.pkl", "h2h"), "archivio/h2h/h2h_champion.pkl")
        self.assertEqual(destinazione_archivio("dc_champion.pkl", "dc"), "archivio/dc/dc_champion.pkl")

    def test_sposta_senza_sovrascrivere_ne_cancellare(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_rel = "h2h/h2h_champion.pkl"
            dst_rel = "archivio/h2h/h2h_champion.pkl"
            src = os.path.join(tmp, "h2h", "h2h_champion.pkl")
            dst = os.path.join(tmp, "archivio", "h2h", "h2h_champion.pkl")
            os.makedirs(os.path.dirname(src), exist_ok=True)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(src, "w") as f:
                f.write("nuovo")
            with open(dst, "w") as f:
                f.write("vecchio-archivio")

            finale, esito = sposta_senza_cancellare(tmp, src_rel, dst_rel, applica=True)
            self.assertEqual(esito, "kept_source_collision")
            self.assertEqual(finale, src_rel)
            self.assertTrue(os.path.exists(src))
            with open(dst) as f:
                self.assertEqual(f.read(), "vecchio-archivio")

    def test_sposta_quando_archivio_e_libero(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_rel = "dc_champion.pkl"
            dst_rel = "archivio/dc/dc_champion.pkl"
            src = os.path.join(tmp, src_rel)
            with open(src, "w") as f:
                f.write("prod-vecchia")

            finale, esito = sposta_senza_cancellare(tmp, src_rel, dst_rel, applica=True)
            self.assertEqual(esito, "moved")
            self.assertEqual(finale, dst_rel)
            self.assertFalse(os.path.exists(src))
            self.assertTrue(os.path.exists(os.path.join(tmp, *dst_rel.split("/"))))
            with open(os.path.join(tmp, *dst_rel.split("/"))) as f:
                self.assertEqual(f.read(), "prod-vecchia")

    def test_simulazione_non_muove(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_rel = "h2h_champion.pkl"
            with open(os.path.join(tmp, src_rel), "w") as f:
                f.write("x")
            finale, esito = sposta_senza_cancellare(
                tmp, src_rel, "archivio/h2h/h2h_champion.pkl", applica=False
            )
            self.assertEqual(esito, "moved")
            self.assertEqual(finale, "archivio/h2h/h2h_champion.pkl")
            self.assertTrue(os.path.exists(os.path.join(tmp, src_rel)))
            self.assertFalse(os.path.exists(os.path.join(tmp, "archivio", "h2h", "h2h_champion.pkl")))


if __name__ == "__main__":
    unittest.main()
