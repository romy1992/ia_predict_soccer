"""Test per `CrudRepository.search_filter` (bug fix): `Match.odds` e
`Match.statistics` sono RELAZIONI ORM one-to-many (non colonne scalari) —
`is_not(None)`/`is_(None)` non sono supportati da SQLAlchemy su una
relazione e sollevano `NotImplementedError`. Il fix usa `.any()`/`~.any()`
per le relazioni, mantenendo invariato `is_not`/`is_` per le colonne
scalari (es. `Match.mean_statistics`, JSON).

Usa un vero engine SQLite in-memory (non un mock) per esercitare
DAVVERO la generazione della query SQLAlchemy: tutti gli altri test del
progetto mockano `search_filter` a un livello piu' alto, motivo per cui
questo bug non era mai stato scoperto prima di eseguire un training
contro un DB reale."""

import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.repository.base.crud_repository import CrudRepository, _is_relationship_attribute
from src.service_ia.model.match import Base, Match, Odds, Statistics


def _make_in_memory_repo() -> CrudRepository:
    """CrudRepository con sessione SQLite in-memory al posto del
    Postgres reale (stesso pattern di dependency override "manuale":
    l'attributo `session` e' pubblico, mai serializzato/nascosto)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    repo = CrudRepository(entity=Match)
    repo.session.close()  # chiude la sessione Postgres di default (mai usata)
    repo.session = session_factory()
    return repo


def _add_match(session, *, with_statistics: bool, with_odds: bool, with_mean_statistics: bool) -> str:
    match_id = str(uuid.uuid4())
    match = Match(
        id_match_fk=match_id,
        status="FT",
        mean_statistics={"goals": 1.5} if with_mean_statistics else None,
    )
    session.add(match)
    if with_statistics:
        session.add(Statistics(id_statistics_fk=str(uuid.uuid4()), id_match=match_id, statistics_team_id=1))
    if with_odds:
        session.add(Odds(id_odds_fk=str(uuid.uuid4()), id_match=match_id, odds_from="test"))
    session.commit()
    return match_id


class TestIsRelationshipAttribute(unittest.TestCase):
    def test_odds_and_statistics_are_relationships(self):
        self.assertTrue(_is_relationship_attribute(Match.odds))
        self.assertTrue(_is_relationship_attribute(Match.statistics))

    def test_mean_statistics_is_not_a_relationship(self):
        self.assertFalse(_is_relationship_attribute(Match.mean_statistics))

    def test_scalar_columns_are_not_relationships(self):
        self.assertFalse(_is_relationship_attribute(Match.status))
        self.assertFalse(_is_relationship_attribute(Match.season))


class TestSearchFilterNotNoneOnRelationship(unittest.TestCase):
    """Bug fix: prima di questo fix, questi filtri sollevavano
    `NotImplementedError` (mai un errore di dati, un errore di
    costruzione della query — QUALUNQUE fixture veniva esclusa/il
    training falliva sempre, indipendentemente dai dati presenti)."""

    def test_not_none_on_relationship_does_not_raise_and_filters_correctly(self):
        repo = _make_in_memory_repo()
        with_link = _add_match(repo.session, with_statistics=True, with_odds=True, with_mean_statistics=True)
        _add_match(repo.session, with_statistics=False, with_odds=False, with_mean_statistics=True)

        results = repo.search_filter(filters={"statistics": "not None", "odds": "not None"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id_match_fk, with_link)

    def test_none_on_relationship_does_not_raise_and_filters_correctly(self):
        repo = _make_in_memory_repo()
        _add_match(repo.session, with_statistics=True, with_odds=True, with_mean_statistics=True)
        without_link = _add_match(repo.session, with_statistics=False, with_odds=False, with_mean_statistics=True)

        results = repo.search_filter(filters={"statistics": "None"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id_match_fk, without_link)

    def test_combined_filters_matching_filter_market_service_usage(self):
        """Stesso identico shape di filtri di `FilterMarketService._search_matches`
        (`mean_statistics`/`odds`/`statistics`: "not None" + `status` in lista) —
        la combinazione che falliva sempre prima del fix."""
        repo = _make_in_memory_repo()
        complete = _add_match(repo.session, with_statistics=True, with_odds=True, with_mean_statistics=True)
        _add_match(repo.session, with_statistics=False, with_odds=True, with_mean_statistics=True)  # manca statistics
        _add_match(repo.session, with_statistics=True, with_odds=True, with_mean_statistics=False)  # manca mean_statistics

        results = repo.search_filter(
            filters={
                "mean_statistics": "not None",
                "odds": "not None",
                "statistics": "not None",
                "status": ["FT"],
            }
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id_match_fk, complete)


class TestSearchFilterOnScalarColumnUnchanged(unittest.TestCase):
    """Nessuna regressione: le colonne scalari (anche JSON) restano su
    `is_not`/`is_`, comportamento identico a prima del fix."""

    def test_not_none_on_scalar_json_column(self):
        repo = _make_in_memory_repo()
        with_stats = _add_match(repo.session, with_statistics=False, with_odds=False, with_mean_statistics=True)
        _add_match(repo.session, with_statistics=False, with_odds=False, with_mean_statistics=False)

        results = repo.search_filter(filters={"mean_statistics": "not None"})

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id_match_fk, with_stats)

    def test_equality_filter_on_scalar_column(self):
        repo = _make_in_memory_repo()
        _add_match(repo.session, with_statistics=False, with_odds=False, with_mean_statistics=False)

        results = repo.search_filter(filters={"status": ["FT"]})
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
