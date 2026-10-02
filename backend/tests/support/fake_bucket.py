"""Fake in-process del client S3 usato da `bucket_store.py` nei test.

Condiviso tra tutti i test che passano da `ModelRegistry`/`JobHistory`/
altri moduli migrati al bucket - niente `moto` (il bucket Railway e'
S3-compatible ma non AWS reale, un fake dedicato evita falsa sicurezza da
comportamenti AWS-specifici che Railway potrebbe non replicare)."""

from __future__ import annotations

import io

from botocore.exceptions import ClientError


def _not_found_error(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": key}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "GetObject",
    )


class FakeS3Client:
    def __init__(self):
        self._objects: dict[str, bytes] = {}

    def head_object(self, Bucket: str, Key: str):
        if Key not in self._objects:
            raise _not_found_error(Key)
        return {}

    def get_object(self, Bucket: str, Key: str):
        if Key not in self._objects:
            raise _not_found_error(Key)
        return {"Body": io.BytesIO(self._objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = None):
        self._objects[Key] = bytes(Body)

    def delete_object(self, Bucket: str, Key: str):
        self._objects.pop(Key, None)

    def get_paginator(self, operation_name: str):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self._objects)


class _FakePaginator:
    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def paginate(self, Bucket: str, Prefix: str = ""):
        matching = sorted(k for k in self._objects if k.startswith(Prefix))
        yield {"Contents": [{"Key": k} for k in matching]}
