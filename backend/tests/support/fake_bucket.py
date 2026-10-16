"""Fake in-process del client S3 usato da `bucket_store.py` nei test.

Condiviso tra tutti i test che passano da `ModelRegistry`/`JobHistory`/
altri moduli migrati al bucket - niente `moto` (il bucket Railway e'
S3-compatible ma non AWS reale, un fake dedicato evita falsa sicurezza da
comportamenti AWS-specifici che Railway potrebbe non replicare)."""

from __future__ import annotations

import datetime as dt
import io

from botocore.exceptions import ClientError


def _not_found_error(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": key}, "ResponseMetadata": {"HTTPStatusCode": 404}},
        "GetObject",
    )


def _access_denied_error(key: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": "AccessDenied", "Message": key}, "ResponseMetadata": {"HTTPStatusCode": 403}},
        "GetObject",
    )


class FakeS3Client:
    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._modified: dict[str, dt.datetime] = {}
        self._seq = 0
        # Chiavi che devono rispondere AccessDenied su get_object, pur
        # esistendo (`put_object` le scrive comunque) - simula il caso
        # reale osservato in produzione: un oggetto presente ma non
        # leggibile (permessi/credito), gli altri sotto lo stesso prefisso
        # restano leggibili normalmente.
        self.deny_read_keys: set[str] = set()
        self.gets = 0

    def head_object(self, Bucket: str, Key: str):
        if Key not in self._objects:
            raise _not_found_error(Key)
        return {"ContentLength": len(self._objects[Key])}

    def get_object(self, Bucket: str, Key: str, Range: str | None = None):
        self.gets += 1
        if Key in self.deny_read_keys:
            raise _access_denied_error(Key)
        if Key not in self._objects:
            raise _not_found_error(Key)
        data = self._objects[Key]
        if Range:
            spec = Range.split("=", 1)[1]
            start_s, end_s = spec.split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(data) - 1
            data = data[start : end + 1]
        return {"Body": io.BytesIO(data)}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str = None):
        self._seq += 1
        self._objects[Key] = bytes(Body)
        self._modified[Key] = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
            seconds=self._seq
        )

    def delete_object(self, Bucket: str, Key: str):
        self._objects.pop(Key, None)
        self._modified.pop(Key, None)

    def get_paginator(self, operation_name: str):
        assert operation_name == "list_objects_v2"
        return _FakePaginator(self)


class _FakePaginator:
    def __init__(self, client: FakeS3Client):
        self._client = client

    def paginate(self, Bucket: str, Prefix: str = ""):
        matching = sorted(k for k in self._client._objects if k.startswith(Prefix))
        yield {
            "Contents": [
                {
                    "Key": key,
                    "LastModified": self._client._modified.get(key),
                    "Size": len(self._client._objects[key]),
                }
                for key in matching
            ]
        }
