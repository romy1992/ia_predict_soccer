"""Fixture autouse: ogni test in questo pacchetto gira contro un bucket S3
fake in-process, non il bucket Railway reale - evita di dover ripetere
`bucket_store.set_client_for_tests(...)` in ogni singolo test file che
passa da `ModelRegistry`/`JobHistory`/altri moduli migrati al bucket."""

import pytest

from src.storage import bucket_store
from tests.support.fake_bucket import FakeS3Client


@pytest.fixture(autouse=True)
def _fake_bucket():
    bucket_store.set_client_for_tests(FakeS3Client(), "test-bucket")
    yield
    bucket_store.reset_client_for_tests()
