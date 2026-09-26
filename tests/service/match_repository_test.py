"""Test per i due metodi aggiunti a `MatchRepository` il 2026-09-15.

Come `crud_repository_test.py`, usa un vero engine SQLite in-memory invece
di mock: entrambi i metodi generano SQL non banale (upsert con inferenza sul
conflitto, confronto lessicografico su una colonna VARCHAR di date) e un
mock non avrebbe esercitato la parte che conta.

NOTA sul percorso SQLite: `upsert_base_by_fixture` ha due rami, uno con
`INSERT ... ON CONFLICT (id_fixture) DO UPDATE ... RETURNING` per PostgreSQL
e uno portabile per gli altri dialetti. Qui si esercita il secondo. Il primo
non e' riproducibile su SQLite, ma il CONTRATTO verificato qui e' lo stesso
che il chiamante si aspetta in produzione: la riga viene persistita subito e
il metodo restituisce (id_definitivo, e_stata_inserita_ora).
"""

import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.repository.match_repository import MatchRepository, _giorno_successivo
from src.service_ia.model.match import Base, Match


def _repo_in_memoria() -> MatchRepository:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    repo = MatchRepository()
    repo.session.close()  # chiude la sessione Postgres di default (mai usata)
    repo.session = session_factory()
    return repo


def _colonne(id_fixture: int, *, status: str = "NS", score_home=None, date_match="2026-09-14T16:30:00+00:00") -> dict:
    return {
        "id_match_fk": str(uuid.uuid4()),
        "id_fixture": id_fixture,
        "name_home": "Como",
        "name_away": "Parma",
        "id_team_home": 1,
        "id_team_away": 2,
        "date_match": date_match,
        "status": status,
        "score_home": score_home,
        "season": 2026,
        "current_league": 135,
    }


class TestUpsertBaseByFixture(unittest.TestCase):
    def test_prima_scrittura_inserisce_e_segnala_inserted(self):
        repo = _repo_in_memoria()
        colonne = _colonne(1550118)

        id_definitivo, inserita = repo.upsert_base_by_fixture(colonne)

        self.assertTrue(inserita)
        self.assertEqual(id_definitivo, colonne["id_match_fk"])
        # Persistita SUBITO: e' il punto del fix (prima restava in una lista
        # Python fino alla fine del job).
        righe = repo.session.query(Match).all()
        self.assertEqual(len(righe), 1)
        self.assertEqual(righe[0].status, "NS")

    def test_seconda_scrittura_aggiorna_e_non_duplica(self):
        """Il cuore del fix: una seconda passata sulla stessa fixture - anche
        con un `id_match_fk` nuovo, come faceva ogni import concorrente
        generando il proprio `uuid4()` - deve AGGIORNARE la riga esistente e
        restituirne l'id, non creare una seconda riga."""
        repo = _repo_in_memoria()
        primo_id, _ = repo.upsert_base_by_fixture(_colonne(1550118, status="NS"))

        colonne_nuove = _colonne(1550118, status="FT", score_home=2)
        self.assertNotEqual(colonne_nuove["id_match_fk"], primo_id)

        id_definitivo, inserita = repo.upsert_base_by_fixture(colonne_nuove)

        self.assertFalse(inserita)
        self.assertEqual(id_definitivo, primo_id, "deve vincere la riga gia' a DB")
        righe = repo.session.query(Match).all()
        self.assertEqual(len(righe), 1, "nessuna riga duplicata")
        self.assertEqual(righe[0].id_match_fk, primo_id, "la PK non viene riscritta")
        self.assertEqual(righe[0].status, "FT", "le colonne base sono aggiornate")
        self.assertEqual(righe[0].score_home, 2)

    def test_non_sovrascrive_le_colonne_gestite_da_altri_servizi(self):
        """`mean_statistics` (calculate_mean) e `settlement_status`
        (SettlementService) non sono prodotte da `map_base_match`, quindi non
        sono tra le colonne dell'upsert e non devono essere azzerate."""
        repo = _repo_in_memoria()
        primo_id, _ = repo.upsert_base_by_fixture(_colonne(1550118))
        riga = repo.session.query(Match).one()
        riga.mean_statistics = {"mean_Corner Kicks": 4.2}
        riga.settlement_status = "complete"
        riga.is_settled = True
        repo.session.commit()

        repo.upsert_base_by_fixture(_colonne(1550118, status="FT", score_home=2))

        riga = repo.session.query(Match).one()
        self.assertEqual(riga.mean_statistics, {"mean_Corner Kicks": 4.2})
        self.assertEqual(riga.settlement_status, "complete")
        self.assertTrue(riga.is_settled)
        self.assertEqual(riga.status, "FT")

    def test_fixture_distinte_restano_righe_distinte(self):
        repo = _repo_in_memoria()
        repo.upsert_base_by_fixture(_colonne(1550118))
        repo.upsert_base_by_fixture(_colonne(1550120))
        self.assertEqual(repo.session.query(Match).count(), 2)


class TestSearchByDateWindow(unittest.TestCase):
    def _repo_con_partite(self) -> MatchRepository:
        repo = _repo_in_memoria()
        for id_fixture, giorno in (
            (1, "2026-09-11T20:45:00+00:00"),
            (2, "2026-09-12T13:00:00+00:00"),
            (3, "2026-09-14T16:30:00+00:00"),
            (4, "2026-09-14T23:59:00+00:00"),
            (5, "2026-09-15T18:00:00+00:00"),
        ):
            repo.upsert_base_by_fixture(_colonne(id_fixture, status="FT", date_match=giorno))
        return repo

    def test_finestra_inclusiva_su_entrambi_gli_estremi(self):
        repo = self._repo_con_partite()

        righe = repo.search_by_date_window(from_day="2026-09-12", to_day="2026-09-14")

        self.assertEqual(sorted(r.id_fixture for r in righe), [2, 3, 4])

    def test_l_ultimo_istante_del_giorno_finale_e_incluso(self):
        """Regressione sul confronto lessicografico: la fixture alle 23:59 del
        giorno `to_day` non deve cadere fuori (era il rischio di un confronto
        `<= to_day` sulla stringa "2026-09-14")."""
        repo = self._repo_con_partite()

        righe = repo.search_by_date_window(from_day="2026-09-14", to_day="2026-09-14")

        self.assertEqual(sorted(r.id_fixture for r in righe), [3, 4])

    def test_filtri_aggiuntivi_si_combinano(self):
        repo = self._repo_con_partite()
        riga = repo.session.query(Match).filter(Match.id_fixture == 3).one()
        riga.status = "NS"
        repo.session.commit()

        righe = repo.search_by_date_window(
            from_day="2026-09-11", to_day="2026-09-15", statuses=["FT"], seasons=[2026], leagues=[135]
        )

        self.assertNotIn(3, [r.id_fixture for r in righe])
        self.assertEqual(sorted(r.id_fixture for r in righe), [1, 2, 4, 5])

    def test_solo_con_id_fixture_scarta_le_righe_legacy(self):
        repo = self._repo_con_partite()
        repo.session.add(
            Match(id_match_fk=str(uuid.uuid4()), id_fixture=None, date_match="2026-09-12T15:00:00+00:00", status="FT")
        )
        repo.session.commit()

        senza_filtro = repo.search_by_date_window(from_day="2026-09-12", to_day="2026-09-12")
        con_filtro = repo.search_by_date_window(
            from_day="2026-09-12", to_day="2026-09-12", solo_con_id_fixture=True
        )

        self.assertEqual(len(senza_filtro), 2)
        self.assertEqual(len(con_filtro), 1)


class TestGiornoSuccessivo(unittest.TestCase):
    def test_incrementa_di_un_giorno(self):
        self.assertEqual(_giorno_successivo("2026-09-14"), "2026-09-15")

    def test_gestisce_il_cambio_di_mese(self):
        self.assertEqual(_giorno_successivo("2026-09-30"), "2026-10-01")

    def test_accetta_un_timestamp_completo(self):
        self.assertEqual(_giorno_successivo("2026-12-31T23:30:00+00:00"), "2027-01-01")


if __name__ == "__main__":
    unittest.main()
