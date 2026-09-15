"""Lock cross-PROCESSO per i job che scrivono sulle stesse tabelle.

IL PROBLEMA
`api` e `scheduler` sono due container distinti (`docker-compose.yml`) e
APScheduler ha `max_instances=1` solo DENTRO il proprio processo: non sa
niente di un job identico lanciato a mano dal frontend. Il 2026-09-15 le due
cose si sono sovrapposte davvero:

    06:21:20 -> 06:30:02  daily_refresh        (schedulato, container scheduler)
    06:22:25 -> 06:30:02    future_sync          finestra 09-15 -> 09-22
    06:23:01 -> 06:23:53  daily_refresh_played   (seconda catena)
    06:23:54 -> 06:31:27    future_sync          STESSA finestra, sovrapposta

Due `download_import_matches` sulla stessa finestra di date hanno prodotto
116 partite in doppia copia a DB. L'upsert su `id_fixture`
(`MatchRepository.upsert_base_by_fixture`) impedisce ormai la doppia riga,
ma due giri sovrapposti restano comunque lavoro inutile: stesse centinaia
di chiamate API-Sports pagate due volte, stessa quota consumata, e i due
import che si sovrascrivono a vicenda le stesse righe.

LA SOLUZIONE
Un advisory lock di PostgreSQL (`pg_try_advisory_lock`), non un lockfile:
il DB e' l'unica cosa che `api` e `scheduler` condividono davvero, e il lock
viene rilasciato automaticamente se il processo che lo teneva muore - un
file su volume condiviso resterebbe invece "preso" per sempre dopo un kill,
bloccando il job fino a un intervento a mano. E' `try` e non bloccante: chi
non ottiene il lock rinuncia subito e lo dice al chiamante, invece di
accodarsi e partire quando il primo ha finito (sarebbe di nuovo lavoro
duplicato, solo in sequenza).

Il lock e' PER NOME: job diversi che scrivono tabelle diverse non si
bloccano tra loro. `data_daily_refresh`, `data_future_sync`,
`data_sync_today` e `data_settlement` condividono invece lo stesso nome
(`LOCK_IMPORT_MATCH`) perche' finiscono tutti in `download_import_matches`,
cioe' sulle stesse righe di `match`/`statistics`/`odds`/`odds_snapshot`.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from contextlib import contextmanager

from sqlalchemy import text

from src.repository.base.repository_db import engine

logger = logging.getLogger(__name__)

# Nome condiviso da tutti i job che passano da `download_import_matches`.
LOCK_IMPORT_MATCH = "import_match"

# Lock gia' tenuti da QUESTO thread, con il rispettivo livello di
# annidamento. Serve alla rientranza: `run_daily_refresh` prende il lock e
# poi chiama `run_manual_import`, che lo richiede di nuovo. Senza questo
# conteggio la chiamata interna aprirebbe una connessione NUOVA, per
# Postgres sarebbe un'altra sessione, `pg_try_advisory_lock` fallirebbe e il
# job salterebbe le proprie sotto-fasi - cioe' si bloccherebbe da solo.
#
# Deliberatamente per-THREAD e non per-processo: due job diversi eseguiti in
# thread APScheduler distinti devono continuare a escludersi (starebbero
# duplicando lo stesso lavoro), quindi il secondo deve arrivare fino a
# Postgres e ricevere un rifiuto.
_locali = threading.local()


def _annidati() -> dict[str, int]:
    if not hasattr(_locali, "tenuti"):
        _locali.tenuti = {}
    return _locali.tenuti


class JobLockNotAcquired(RuntimeError):
    """Un altro processo sta gia' eseguendo un job con lo stesso lock."""


def _lock_key(nome: str) -> int:
    """Nome -> intero a 64 bit con segno, che e' quello che
    `pg_try_advisory_lock(bigint)` si aspetta. Un hash stabile (sha1
    troncato) e non `hash()`, che in Python e' randomizzato per processo
    (PYTHONHASHSEED) e darebbe chiavi diverse in `api` e `scheduler` - cioe'
    due lock distinti, esattamente il bug che questo modulo deve evitare."""
    digest = hashlib.sha1(nome.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=True)


@contextmanager
def job_lock(nome: str, *, obbligatorio: bool = True):
    """Prende l'advisory lock `nome` per la durata del blocco.

    :param obbligatorio: se True (default) solleva `JobLockNotAcquired`
        quando il lock e' gia' preso. Con False il blocco viene eseguito
        comunque e si ottiene solo una riga di log - usato dove il doppio
        giro e' spreco ma non un rischio di correttezza.

    Su un DB che non sia PostgreSQL (SQLite dei test) il lock e' un no-op:
    li' non c'e' nessun secondo processo da cui difendersi.

    IMPORTANTE - perche' una connessione DEDICATA e non `SessionLocal()`:
    un advisory lock di sessione e' legato alla CONNESSIONE (il backend
    Postgres), non alla Session di SQLAlchemy. Prendendolo sulla Session
    "scoped" condivisa, `download_import_matches` - che a fine job chiama
    `SessionLocal.remove()` - restituirebbe al pool la connessione che
    tiene il lock: il lock resterebbe appeso su quella connessione e il
    successivo `pg_advisory_unlock` potrebbe finire su un'ALTRA connessione
    del pool, senza rilasciare niente. Risultato: un lock che non si libera
    piu' e blocca ogni import successivo.

    Il rilascio e' quindi l'`pg_advisory_unlock` esplicito sulla NOSTRA
    connessione. Se quella query fallisce si ricorre a `invalidate()`, che
    chiude davvero il socket verso Postgres e fa cadere con esso tutti i
    lock di sessione: `close()` da solo NON basterebbe, perche' con un pool
    restituisce la connessione senza chiuderla (il backend resta vivo, e un
    lock di sessione sopravvive anche al rollback che il pool esegue al
    rientro). Si paga una connessione nuova al giro successivo, ma solo nel
    caso di errore - un lock bloccato per sempre costerebbe molto di piu'.
    """
    if engine.dialect.name != "postgresql":
        yield True
        return

    tenuti = _annidati()
    if tenuti.get(nome):
        # Gia' nostro (chiamata annidata): nessuna query, nessuna nuova
        # connessione, e soprattutto nessun rilascio all'uscita di QUESTO
        # blocco - il lock deve restare fino all'uscita di quello esterno.
        tenuti[nome] += 1
        try:
            yield True
        finally:
            tenuti[nome] -= 1
        return

    chiave = _lock_key(nome)
    try:
        connessione = engine.connect()
    except Exception as exc:
        logger.warning("Job lock '%s' non verificabile (%s): si procede senza.", nome, exc)
        yield True
        return

    try:
        try:
            preso = bool(connessione.execute(text("select pg_try_advisory_lock(:k)"), {"k": chiave}).scalar())
        except Exception as exc:
            # Se non riusciamo nemmeno a CHIEDERE il lock (DB irraggiungibile,
            # connessione caduta) si procede comunque: questo lock e' una
            # protezione dal lavoro duplicato, non un prerequisito di
            # correttezza. Farlo diventare un motivo per NON far girare
            # l'import significherebbe trasformare un problema di rete in un
            # buco nei dati - e la doppia riga a DB e' ormai impedita a monte
            # dall'indice unico su `id_fixture`.
            logger.warning("Job lock '%s' non verificabile (%s): si procede senza.", nome, exc)
            yield True
            return

        if not preso:
            messaggio = (
                f"Job lock '{nome}' già preso da un altro processo "
                f"(api/scheduler): esecuzione saltata per non duplicare il lavoro."
            )
            if obbligatorio:
                raise JobLockNotAcquired(messaggio)
            logger.warning(messaggio)
            yield False
            return

        tenuti[nome] = 1
        try:
            yield True
        finally:
            tenuti.pop(nome, None)
            try:
                connessione.execute(text("select pg_advisory_unlock(:k)"), {"k": chiave})
            except Exception as exc:
                # L'unlock non e' passato: buttiamo via la connessione invece
                # di restituirla al pool ancora "titolare" del lock. Chiudere
                # il socket e' l'unico modo per essere certi che Postgres lo
                # rilasci (vedi docstring).
                logger.warning(
                    "Rilascio job lock '%s' fallito (%s): invalido la connessione per forzarlo.",
                    nome,
                    exc,
                )
                try:
                    connessione.invalidate()
                except Exception as exc_inv:
                    logger.error("Invalidazione connessione del job lock '%s' fallita: %s", nome, exc_inv)
    finally:
        try:
            connessione.close()
        except Exception as exc:
            logger.warning("Chiusura connessione del job lock '%s' fallita: %s", nome, exc)
