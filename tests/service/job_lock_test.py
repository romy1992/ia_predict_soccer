"""Test per `src/jobs/job_lock.py` (2026-09-15).

`api` e `scheduler` sono due container distinti e `max_instances=1` di
APScheduler vale solo dentro il proprio processo: il 2026-09-15 il job
schedulato `data_daily_refresh` (06:21:20 -> 06:30:02) e una seconda catena
(06:23:01 -> 06:31:27) hanno girato in parallelo sulla STESSA finestra di 7
giorni, producendo 116 partite in doppia copia. L'advisory lock condiviso
sul DB e' la protezione cross-processo.

Il DB non viene mai toccato: l'`engine` del modulo e' sostituito da un
doppio finto che registra le query eseguite.
"""

import hashlib
import threading
import unittest
from unittest import mock

from src.jobs import job_lock as modulo
from src.jobs.job_lock import LOCK_IMPORT_MATCH, JobLockNotAcquired, _lock_key, job_lock


class _Dialetto:
    """Classe vera e non `mock.Mock(name=...)`: su un Mock il kwarg `name` e'
    riservato (imposta il nome del mock, non l'attributo), quindi
    `dialect.name` non sarebbe mai stata la stringa attesa e ogni test
    avrebbe silenziosamente preso il ramo no-op."""

    def __init__(self, nome: str):
        self.name = nome


class _ConnessioneFinta:
    def __init__(self, engine):
        self._engine = engine
        self.chiusa = False

    def execute(self, statement, params=None):
        testo = str(statement)
        self._engine.query_eseguite.append(testo)
        if self._engine.esplode_su_query:
            raise RuntimeError("connessione caduta")
        if self._engine.esplode_su_unlock and "advisory_unlock" in testo:
            raise RuntimeError("unlock non passato")
        risultato = mock.Mock()
        risultato.scalar.return_value = (
            self._engine.lock_ottenuto if "try_advisory_lock" in testo else True
        )
        return risultato

    def invalidate(self):
        self._engine.connessioni_invalidate += 1

    def close(self):
        self.chiusa = True
        self._engine.connessioni_chiuse += 1


class _EngineFinto:
    def __init__(self, dialetto="postgresql", lock_ottenuto=True,
                 esplode_su_query=False, esplode_su_connect=False, esplode_su_unlock=False):
        self.dialect = _Dialetto(dialetto)
        self.lock_ottenuto = lock_ottenuto
        self.esplode_su_query = esplode_su_query
        self.esplode_su_connect = esplode_su_connect
        self.esplode_su_unlock = esplode_su_unlock
        self.query_eseguite: list[str] = []
        self.connessioni_aperte = 0
        self.connessioni_chiuse = 0
        self.connessioni_invalidate = 0

    def connect(self):
        if self.esplode_su_connect:
            raise RuntimeError("pool esaurito")
        self.connessioni_aperte += 1
        return _ConnessioneFinta(self)


def _con_engine(engine_finto):
    return mock.patch.object(modulo, "engine", engine_finto)


def _azzera_stato_thread():
    """Il conteggio di rientranza vive in un `threading.local` di modulo:
    va azzerato tra i test, altrimenti un test che esce male lascerebbe il
    lock "tenuto" e falserebbe i successivi."""
    if hasattr(modulo._locali, "tenuti"):
        modulo._locali.tenuti.clear()


class TestLockKey(unittest.TestCase):
    def test_chiave_stabile_tra_processi(self):
        """Deve essere un hash STABILE e non `hash()`, che in Python e'
        randomizzato per processo (PYTHONHASHSEED): con chiavi diverse in
        `api` e `scheduler` i due lock non si vedrebbero, che e' esattamente
        il bug da evitare."""
        self.assertEqual(_lock_key("import_match"), _lock_key("import_match"))
        # Ricalcolo indipendente dell'algoritmo: se cambia, `api` e
        # `scheduler` con versioni diverse del codice userebbero chiavi
        # diverse durante un rollout, cioe' due lock che non si vedono.
        atteso = int.from_bytes(
            hashlib.sha1(LOCK_IMPORT_MATCH.encode("utf-8")).digest()[:8], byteorder="big", signed=True
        )
        self.assertEqual(_lock_key(LOCK_IMPORT_MATCH), atteso)

    def test_nomi_diversi_chiavi_diverse(self):
        self.assertNotEqual(_lock_key("import_match"), _lock_key("altro_job"))

    def test_rientra_in_un_bigint_con_segno(self):
        for nome in ("import_match", "a", "x" * 500):
            self.assertGreaterEqual(_lock_key(nome), -(2 ** 63))
            self.assertLess(_lock_key(nome), 2 ** 63)


class TestJobLock(unittest.TestCase):
    def setUp(self):
        _azzera_stato_thread()

    tearDown = setUp

    def test_lock_libero_esegue_e_rilascia(self):
        engine = _EngineFinto(lock_ottenuto=True)
        with _con_engine(engine):
            with job_lock("import_match") as preso:
                self.assertTrue(preso)

        self.assertTrue(any("pg_try_advisory_lock" in q for q in engine.query_eseguite))
        self.assertTrue(any("pg_advisory_unlock" in q for q in engine.query_eseguite))
        # La connessione DEDICATA va restituita al pool: senza `close()`
        # resterebbe occupata a ogni giro. Il rilascio del lock e' l'unlock
        # esplicito qui sopra, non la `close()` (vedi
        # `test_unlock_fallito_invalida_la_connessione`).
        self.assertEqual(engine.connessioni_chiuse, 1)

    def test_connessione_dedicata_non_dalla_session_condivisa(self):
        """Regressione sul difetto trovato durante l'implementazione: preso
        sulla Session "scoped" condivisa, il lock sarebbe stato restituito al
        pool da `SessionLocal.remove()` (che `download_import_matches` chiama
        a fine job) e l'unlock successivo sarebbe potuto finire su un'ALTRA
        connessione, lasciando il lock appeso per sempre."""
        engine = _EngineFinto()
        with _con_engine(engine):
            with job_lock("import_match"):
                pass
        self.assertEqual(engine.connessioni_aperte, 1)

    def test_lock_occupato_solleva_se_obbligatorio(self):
        engine = _EngineFinto(lock_ottenuto=False)
        with _con_engine(engine):
            with self.assertRaises(JobLockNotAcquired):
                with job_lock("import_match"):
                    self.fail("il corpo non deve essere eseguito")
        self.assertEqual(engine.connessioni_chiuse, 1)

    def test_lock_occupato_restituisce_false_se_non_obbligatorio(self):
        """E' la modalita' usata dai job: si salta il giro senza sollevare, cosi'
        lo storico registra un `success` con `skipped_locked` invece di un
        errore finto (vedi `scheduler._esito_job_saltato`)."""
        engine = _EngineFinto(lock_ottenuto=False)
        eseguito = []
        with _con_engine(engine):
            with job_lock("import_match", obbligatorio=False) as preso:
                eseguito.append(preso)

        self.assertEqual(eseguito, [False])
        # Nessun unlock: non lo avevamo preso.
        self.assertFalse(any("pg_advisory_unlock" in q for q in engine.query_eseguite))

    def test_lock_rilasciato_anche_se_il_corpo_solleva(self):
        engine = _EngineFinto(lock_ottenuto=True)
        with _con_engine(engine):
            with self.assertRaises(ValueError):
                with job_lock("import_match"):
                    raise ValueError("il job e' fallito")

        self.assertTrue(any("pg_advisory_unlock" in q for q in engine.query_eseguite))
        self.assertEqual(engine.connessioni_chiuse, 1)

    def test_rientrante_nello_stesso_thread(self):
        """`run_daily_refresh` prende il lock e poi chiama `run_manual_import`,
        che lo richiede: la chiamata interna deve passare SENZA interrogare di
        nuovo Postgres (una connessione nuova sarebbe un'altra sessione e
        `pg_try_advisory_lock` fallirebbe, facendo saltare al job le proprie
        sotto-fasi). Il rilascio avviene solo all'uscita del blocco ESTERNO."""
        engine = _EngineFinto(lock_ottenuto=True)
        with _con_engine(engine):
            with job_lock("import_match") as esterno:
                self.assertTrue(esterno)
                with job_lock("import_match") as interno:
                    self.assertTrue(interno)
                # Uscito il blocco interno, il lock NON deve essere gia' stato
                # rilasciato: siamo ancora dentro quello esterno.
                self.assertFalse(any("pg_advisory_unlock" in q for q in engine.query_eseguite))

        self.assertEqual(engine.connessioni_aperte, 1, "una sola connessione per tutto l'annidamento")
        self.assertEqual(len([q for q in engine.query_eseguite if "pg_try_advisory_lock" in q]), 1)
        self.assertEqual(len([q for q in engine.query_eseguite if "pg_advisory_unlock" in q]), 1)

    def test_nomi_diversi_non_si_bloccano(self):
        engine = _EngineFinto(lock_ottenuto=True)
        with _con_engine(engine):
            with job_lock("import_match") as a:
                with job_lock("altro_job") as b:
                    self.assertTrue(a)
                    self.assertTrue(b)
        self.assertEqual(engine.connessioni_aperte, 2)

    def test_un_altro_thread_viene_comunque_bloccato(self):
        """La rientranza e' per-THREAD a posta: due job in thread APScheduler
        distinti stanno duplicando lo stesso lavoro, quindi il secondo deve
        arrivare a Postgres e ricevere un rifiuto."""
        engine = _EngineFinto(lock_ottenuto=False)
        modulo._locali.tenuti = {"import_match": 1}  # come se lo tenesse QUESTO thread
        esito = {}

        def altro_thread():
            with _con_engine(engine):
                with job_lock("import_match", obbligatorio=False) as preso:
                    esito["preso"] = preso

        t = threading.Thread(target=altro_thread)
        t.start()
        t.join()

        self.assertFalse(esito["preso"], "il conteggio di rientranza non deve attraversare i thread")

    def test_db_irraggiungibile_non_blocca_il_job(self):
        """Il lock e' una protezione dal lavoro duplicato, non un prerequisito
        di correttezza: se non riusciamo nemmeno a chiederlo si procede. La
        doppia riga a DB e' comunque impedita dall'indice unico su
        `id_fixture`."""
        for engine in (_EngineFinto(esplode_su_connect=True), _EngineFinto(esplode_su_query=True)):
            _azzera_stato_thread()
            with _con_engine(engine):
                with job_lock("import_match") as preso:
                    self.assertTrue(preso)

    def test_unlock_fallito_invalida_la_connessione(self):
        """Se `pg_advisory_unlock` non passa, la connessione non va restituita
        al pool: e' ancora titolare del lock, e `close()` non lo rilascia
        (il pool tiene il backend Postgres vivo, e un lock di sessione
        sopravvive al rollback eseguito al rientro). `invalidate()` chiude
        davvero il socket ed e' l'unica garanzia che il lock cada, invece di
        restare appeso e bloccare ogni import successivo."""
        engine = _EngineFinto(lock_ottenuto=True, esplode_su_unlock=True)
        with _con_engine(engine):
            with job_lock("import_match") as preso:
                self.assertTrue(preso)

        self.assertEqual(engine.connessioni_invalidate, 1)
        # Il conteggio di rientranza deve essere pulito comunque, altrimenti
        # il thread crederebbe di tenere ancora il lock.
        self.assertFalse(modulo._locali.tenuti.get("import_match"))

    def test_unlock_riuscito_non_invalida_la_connessione(self):
        engine = _EngineFinto(lock_ottenuto=True)
        with _con_engine(engine):
            with job_lock("import_match"):
                pass
        self.assertEqual(engine.connessioni_invalidate, 0)
        self.assertEqual(engine.connessioni_chiuse, 1)

    def test_su_sqlite_e_un_no_op(self):
        engine = _EngineFinto(dialetto="sqlite")
        with _con_engine(engine):
            with job_lock("import_match") as preso:
                self.assertTrue(preso)

        self.assertEqual(engine.query_eseguite, [], "nessuna query su un dialetto senza advisory lock")
        self.assertEqual(engine.connessioni_aperte, 0)


if __name__ == "__main__":
    unittest.main()
