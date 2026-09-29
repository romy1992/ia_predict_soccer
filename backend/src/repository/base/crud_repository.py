import logging

from sqlalchemy import or_
from sqlalchemy.orm import RelationshipProperty

from src.repository.base.repository_db import SessionLocal

logging.basicConfig(level=logging.DEBUG)


def _is_relationship_attribute(col) -> bool:
    """True se `col` e' un attributo di RELAZIONE ORM (es. `Match.odds`/
    `Match.statistics`, one-to-many) invece di una colonna scalare (es.
    `Match.mean_statistics`, JSON).

    Bug fix: per le relazioni, SQLAlchemy NON supporta `is_not(None)`/
    `is_(None)` (solleva `NotImplementedError`) — serve invece verificare
    se la lista collegata e' vuota/non vuota con `.any()` (EXISTS subquery).
    Le colonne scalari (incluse quelle JSON) restano invariate su
    `is_not`/`is_` (nessun cambio di comportamento per i filtri gia'
    funzionanti oggi)."""
    return isinstance(getattr(col, "property", None), RelationshipProperty)


class CrudRepository:
    """
    Classe base per i CRUD delle entità
    """

    def __init__(self, entity):
        super().__init__()
        self.entity = entity  # Nome dell'entità
        self._session_override = None  # SOLO per i test: vedi setter di `session`

    @property
    def session(self):
        """Sessione SQLAlchemy per-thread (bug fix 2026-09-06): PRIMA veniva
        creata una sola volta in `__init__` e riusata per tutta la vita del
        repository. Repository come `repo_match`/`repo_snapshot` in
        `download_match_service.py` sono pero' singleton di MODULO condivisi
        tra job schedulati diversi (es. `data_sync_today` + `data_settlement`,
        che richiama internamente `download_import_matches`) eseguiti in
        thread APScheduler separati ma nello stesso processo: con una Session
        fissa condivisa, l'uso concorrente da thread diversi causava errori
        SQLAlchemy "concurrent operations are not permitted" / "Method
        close() can't be called here" a raffica (vedi jobs_history.jsonl).
        Ora ogni accesso richiama `SessionLocal()`, che con `scoped_session`
        (vedi repository_db.py) ritorna la Session del thread CORRENTE -
        mai piu' condivisa tra thread diversi.

        IMPORTANTE (bug fix 2026-09-06 bis): i metodi sotto NON chiudono piu'
        questa sessione a fine chiamata (vedi commento su `save`/`filter_by`)
        - la property quindi ritorna la STESSA istanza per tutte le chiamate
        fatte dallo stesso thread durante lo stesso job, invece di aprirne
        una nuova (round-trip di rete verso il DB remoto) ad ogni singola
        query. Il cleanup (`SessionLocal.remove()`) va fatto esplicitamente a
        fine job (vedi `download_import_matches`), non qui ad ogni get."""
        if self._session_override is not None:
            return self._session_override
        return SessionLocal()

    @session.setter
    def session(self, value):
        """SOLO per i test: permette di iniettare una Session dedicata (es.
        SQLite in-memory, vedi crud_repository_test.py/prediction_ledger_test.py)
        al posto dello scoped_session di produzione. Mai usato nel codice di
        produzione - il getter sopra continua a risolvere sempre la Session
        del thread corrente quando questo override e' `None` (default)."""
        self._session_override = value

    def save(self, obj):
        """
        Inserisce un nuovo record se NON esiste
        Aggiorna se esiste gestendo le relazioni

        Bug fix 2026-09-06 (performance): NON usa piu' `with self.session as
        session:` - con la Session ora "scoped" (vedi property sopra), quel
        context manager la CHIUDEVA ad ogni singola chiamata, costringendo
        SQLAlchemy a riaprire una connessione/transazione verso il DB remoto
        per la chiamata successiva. Su un loop di centinaia di fixture (es.
        "Aggiorna tutto" -> `download_import_matches`, 1 `filter_by` + 1
        `save` per fixture) questo si traduceva in centinaia di round-trip
        di RETE aggiuntivi, misurati fino a ~25s/fixture in produzione
        (Railway) - la causa principale per cui il job restava "in
        esecuzione" per decine di minuti/ore. Ora la sessione resta aperta e
        riusata per tutto il thread; il cleanup e' esplicito a fine job.
        :return: None
        """
        session = self.session
        try:
            session.merge(obj)
            session.commit()
        except Exception as e:
            logging.error(str(e))
            session.rollback()
            raise

    def save_all(self, list_obj: list):
        """
        Inserisce in maniera massiva le entità
        :return:
        """
        if len(list_obj) > 0:
            session = self.session
            try:
                session.add_all(list_obj)
                session.commit()
            except Exception as e:
                logging.error(str(e))
                session.rollback()
                raise

    def search_all(self):
        """
        Ritorna tutti i record senza query specifiche
        :return: lista di entità
        """
        session = self.session
        try:
            return session.query(self.entity).all()
        except Exception as e:
            logging.error(str(e))
            raise

    def filter_by(self, **kwargs):
        """
        Ricerca puntuale dell'entità
        :param kwargs: query da eseguire
        :return: entità o lista trovata/e

        Bug fix 2026-09-06: questo metodo ritorna una Query LAZY (il
        chiamante fa poi `.first()`/`.all()`) - chiudere la sessione qui
        DENTRO (col vecchio `with self.session as session:`) era gia' di
        per se' concettualmente sbagliato (la query viene eseguita DOPO,
        a sessione ormai chiusa) oltre che lento (vedi `save` sopra)."""
        session = self.session
        try:
            return session.query(self.entity).filter_by(**kwargs.get('dict_search'))
        except Exception as e:
            logging.error(str(e))
            raise

    def update(self, **kwargs):
        """
        Aggiorna l'entità
        :param kwargs: query da eseguire
        :return: entità aggiornata
        """
        session = self.session
        try:
            field_change = kwargs.get('field_change')
            value_change = kwargs.get('value_change')
            to_dict = kwargs.get('to_dict')
            filter_by = self.filter_by(**kwargs).first()
            if filter_by:
                setattr(filter_by, field_change, value_change)
                session.commit()
            return filter_by.to_dict() if to_dict else filter_by
        except Exception as e:
            logging.error(str(e))
            session.rollback()
            raise

    def delete(self, **kwargs):
        """
        Cancella in cascade
        :param kwargs: query da eseguire
        :return: None
        """
        session = self.session
        try:
            filter_by = self.filter_by(**kwargs).first()
            if filter_by:
                session.delete(filter_by)
                session.commit()
        except Exception as e:
            logging.error(str(e))
            session.rollback()
            raise

    def search_filter(self, filters: dict):
        """
        Ricerca valori passati in filters che creerò delle condizioni da applicare:
        :param filters: dizionario con i filtri da applicare
            - Per ricerche con 'not None' -> {'id': "not None"} (valore in stringa)
            - Per ricerche con 'None' -> {'id': "None"} (valore in stringa)
            - Per ricerche di uguaglianza -> {'name': "pippo"}
            - Per ricerche con OR -> {"OR": [("id_team_home", 135), ("id_team_away", 135)]}
            - Per ricerche con IN -> {'id':[123,145]} -> il valore può essere una lista o tuple
            - Per ricerche con >, <, >=, <=, = -> {'score': '> 10'}
        :return: array di Matches
        """

        conditions = []
        for k, v in filters.items():
            if k == "OR":
                # Costruisco la condizione OR
                or_conditions = []
                for field_name, field_value in v:
                    col = getattr(self.entity, field_name)
                    if isinstance(field_value, (list, tuple)):
                        or_conditions.append(col.in_(field_value))
                    else:
                        or_conditions.append(col == field_value)
                conditions.append(or_(*or_conditions))
            else:
                # Condizione normale AND
                col = getattr(self.entity, k)
                if isinstance(v, (list, tuple)):
                    conditions.append(col.in_(v))
                else:
                    is_relationship = _is_relationship_attribute(col)
                    if v == 'not None':
                        # Relazione (es. Match.odds/Match.statistics): "not
                        # None" significa "esiste almeno un record collegato"
                        # -> .any(), MAI is_not(None) (NotImplementedError).
                        conditions.append(col.any() if is_relationship else col.is_not(None))
                    elif v == 'None':
                        conditions.append(~col.any() if is_relationship else col.is_(None))
                    # TODO: Rivedere questa parte per gestire gli operatori di confronto
                    # elif isinstance(v, str) and '>=' in v:
                    #     v = v.replace('>=', '').strip()
                    #     num_val = float(v) if '.' in v else int(v)
                    #     conditions.append(col >= num_val)
                    # elif isinstance(v, str) and '<=' in v:
                    #     v = v.replace('<=', '').strip()
                    #     num_val = float(v) if '.' in v else int(v)
                    #     conditions.append(col <= num_val)
                    # elif isinstance(v, str) and '>' in v:
                    #     v = v.replace('>', '').strip()
                    #     num_val = float(v) if '.' in v else int(v)
                    #     conditions.append(col > num_val)
                    # elif isinstance(v, str) and '<' in v:
                    #     v = v.replace('<', '').strip()
                    #     num_val = float(v) if '.' in v else int(v)
                    #     conditions.append(col < num_val)
                    # elif isinstance(v, str) and '=' in v:
                    #     v = v.replace('=', '').strip()
                    #     num_val = float(v) if '.' in v else int(v)
                    #     conditions.append(col == num_val)
                    else:
                        conditions.append(col == v)

        # Bug fix 2026-09-07: usava ancora `with self.session as session:`,
        # il vecchio pattern gia' rimosso da TUTTI gli altri metodi di questa
        # classe (vedi commenti su `save`/`filter_by` sopra, bug fix
        # 2026-09-06) perche' chiude la Session "scoped" condivisa del thread
        # corrente a fine chiamata. Essendo `search_filter` il metodo usato
        # da OGNI dataset di training (`FilterMarketService._search_matches`,
        # `totals_market.py::run_totals_benchmark_from_db`, tutti i
        # market_*.py) ed eseguito ripetutamente nello stesso thread/job,
        # chiuderlo qui esponeva allo stesso rischio gia' diagnosticato per
        # `save`/`filter_by`: query successive sullo stesso thread possono
        # trovarsi con la Session scoped gia' chiusa ("This Session's
        # transaction has been rolled back / Session is closed"), oltre a
        # forzare una riconnessione di rete al DB remoto ad ogni chiamata.
        # Cleanup resta esplicito a fine job (`SessionLocal.remove()`), MAI
        # qui dentro - stesso principio ovunque in questo file.
        session = self.session
        try:
            return session.query(self.entity).filter(*conditions).all()
        except Exception as e:
            logging.error(str(e))
            raise

    def massive_update_bulk(self, list_obj: list):
        """
        Update in maniera massiva ma non verrà propagata ai figli
        :param list_obj: lista di chiave:valore = [{pk:1,campo:'valore'}]
        :return: None
        Esempio:
            rows = [{"id": 1, "status": "done"},
                {"id": 2, "status": "pending"}]

            Equivale a:
            UPDATE my_table SET status = 'done' WHERE id = 1;
            UPDATE my_table SET status = 'pending' WHERE id = 2;
        """
        if len(list_obj) > 0:
            session = self.session
            try:
                session.bulk_update_mappings(self.entity, list_obj)
                session.commit()
            except Exception as e:
                logging.error(str(e))
                session.rollback()
                raise
