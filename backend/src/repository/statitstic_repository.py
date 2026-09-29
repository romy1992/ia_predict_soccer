from src.repository.base.crud_repository import CrudRepository
from src.service_ia.model.match import Statistics


class StatisticRepository(CrudRepository):  # Connessione base con i metodi crud

    def __init__(self):
        super().__init__(Statistics)
