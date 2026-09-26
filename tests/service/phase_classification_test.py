"""`DashboardService._classify_phase`: una partita vecchia non e' "in diretta".

Bug fix 2026-09-15, segnalato dall'operatore con uno screenshot: Como-Parma
del 2026-09-14 mostrata "In diretta" (e senza punteggio) il giorno dopo.

La causa immediata era una riga `match` duplicata rimasta a `NS` - vedi
`scripts/maintenance/dedup_match_id_fixture.py` e
`MatchRepository.upsert_base_by_fixture` - ma il motivo per cui quella riga
appariva "In diretta" invece che "dato mancante" era qui: `_classify_phase`,
per uno stato non finale e non live, DEDUCEVA la fase dall'orario e
ricadeva su `return "live"` a qualunque distanza di tempo dal calcio
d'inizio. Bastava un import non arrivato perche' una partita restasse
"in corso" per sempre.
"""

import unittest
from datetime import datetime, timedelta, timezone

from src.api.dashboard_service import DashboardService


def _fra(**kwargs) -> datetime:
    return datetime.now(timezone.utc) + timedelta(**kwargs)


class TestClassifyPhase(unittest.TestCase):
    def test_stato_finale_vince_sempre(self):
        for stato in ("FT", "AET", "PEN", "ABD", "CANC", "PST", "WO"):
            self.assertEqual(
                DashboardService._classify_phase(stato, _fra(days=-3)), "finished", msg=stato
            )

    def test_stato_live_del_provider_vince_sull_orario(self):
        # Anche con un orario incoerente: se il provider dice "primo tempo",
        # quella e' la fonte autorevole.
        for stato in ("1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT"):
            self.assertEqual(
                DashboardService._classify_phase(stato, _fra(days=-3)), "live", msg=stato
            )

    def test_ns_nel_futuro_e_da_giocare(self):
        self.assertEqual(DashboardService._classify_phase("NS", _fra(hours=3)), "to_play")

    def test_ns_appena_iniziata_resta_live(self):
        """Caso REALE da preservare: il job di sync gira ogni 30 minuti, quindi
        una partita iniziata da poco puo' essere legittimamente ancora `NS` a
        DB. Qui dedurre "live" e' corretto."""
        self.assertEqual(DashboardService._classify_phase("NS", _fra(minutes=-30)), "live")
        self.assertEqual(DashboardService._classify_phase("NS", _fra(hours=-2)), "live")

    def test_ns_di_ieri_non_e_piu_live(self):
        """Il fix: oltre la finestra di durata massima di una partita, una riga
        non aggiornata e' un dato vecchio. Prima tornava "live" e la partita
        restava "In diretta" per giorni."""
        self.assertEqual(DashboardService._classify_phase("NS", _fra(days=-1)), "unknown")
        self.assertEqual(DashboardService._classify_phase("NS", _fra(days=-7)), "unknown")

    def test_confine_della_finestra_live(self):
        # Poco prima delle 4 ore: ancora plausibilmente in corso.
        self.assertEqual(DashboardService._classify_phase("NS", _fra(hours=-3, minutes=-50)), "live")
        # Poco dopo: non puo' piu' essere in corso.
        self.assertEqual(DashboardService._classify_phase("NS", _fra(hours=-4, minutes=-10)), "unknown")

    def test_stato_sconosciuto_nel_passato_non_e_live(self):
        """Stati che non sono ne' in FINAL_STATUSES ne' in LIVE_STATUSES (es.
        "TBD", "SUSP", "AWD", o una stringa vuota da un payload malformato)."""
        for stato in ("TBD", "SUSP", "AWD", "", None):
            self.assertEqual(
                DashboardService._classify_phase(stato, _fra(days=-2)), "unknown", msg=repr(stato)
            )

    def test_senza_orario_confrontabile(self):
        # `date_match` e' sempre una ISO con offset, quindi qui si arriva solo
        # con dati anomali: `NS` resta "da giocare", il resto e' incerto.
        self.assertEqual(DashboardService._classify_phase("NS", None), "to_play")
        self.assertEqual(DashboardService._classify_phase("TBD", None), "unknown")
        # Datetime naive (senza tzinfo): non confrontabile con `now(utc)`.
        self.assertEqual(DashboardService._classify_phase("NS", datetime(2026, 1, 1)), "to_play")


if __name__ == "__main__":
    unittest.main()
