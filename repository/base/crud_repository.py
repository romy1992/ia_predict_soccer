import logging

from repository.base.repository_db import SessionLocal

logging.basicConfig(level=logging.DEBUG)


class CrudRepository:
    """
    Classe base per i CRUD delle entità
    """

    def __init__(self, entity):
        super().__init__()
        self.session = SessionLocal()  # Apre la connessione al db
        self.entity = entity  # Nome dell'entità

    def insert(self, obj):
        """
        Inserisce un nuovo record
        :return: None
        """
        with self.session as session:
            try:
                # session.add(obj)
                session.merge(obj)
                session.commit()
            except Exception as e:
                logging.error(str(e))
                session.rollback()
                raise

    def insert_massive(self, list_obj: list):
        """
        Inserisce in maniera massiva le entità
        :return:
        """
        if len(list_obj) > 0:
            with self.session as session:
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
        with self.session as session:
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
        """
        with self.session as session:
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
        with self.session as session:
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
        with self.session as session:
            try:
                filter_by = self.filter_by(**kwargs).first()
                if filter_by:
                    session.delete(filter_by)
                    session.commit()
            except Exception as e:
                logging.error(str(e))
                session.rollback()
                raise

    from sqlalchemy.dialects.postgresql import insert

    # def upsert_massive_pg(session, Model, rows: list[dict], conflict_cols: list[str],
    #                       update_cols: list[str] | None = None):
    #     if not rows:
    #         return
    #     stmt = insert(Model.__table__).values(rows)
    #
    #     # colonne da aggiornare (di default tutte tranne le di conflitto e la PK)
    #     if update_cols is None:
    #         exclude = set(conflict_cols)
    #         update_cols = [c.name for c in Model.__table__.columns if c.name not in exclude]
    #
    #     stmt = stmt.on_conflict_do_update(
    #         index_elements=[Model.__table__.c[c] for c in conflict_cols],
    #         set_={c: stmt.excluded[c] for c in update_cols}
    #     )
    #     session.execute(stmt)
    #     session.commit()
