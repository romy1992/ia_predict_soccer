"""Primitiva di storage S3-compatible condivisa da `api` e `scheduler`.

Sostituisce `open()`/`os.path` su `best_models/` (montato come Volume SOLO
su `api`, mai su `scheduler` - ogni scrittura di `scheduler` sul filesystem
locale viene persa ad ogni redeploy e non e' mai visibile ad `api`). Un
Bucket Railway S3-compatible e' raggiungibile allo stesso modo da entrambi
i container, nessun Volume necessario.

Client `boto3` costruito pigro (alla prima chiamata, non a import-time):
le 5 variabili d'ambiente `ARCHIVE_BUCKET_*` sono reference variables
Railway, risolte solo a runtime dentro il container - leggerle a import
time romperebbe qualunque comando locale (test, script) che importa questo
modulo senza quelle variabili impostate."""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import boto3
import joblib
from botocore.config import Config
from botocore.exceptions import ClientError

from .env import require_bucket_env

# `ModelRegistry`/`JobHistory` leggono un prefisso intero con un
# `ThreadPoolExecutor(max_workers=16)` (vedi `list_json`) - il pool HTTP
# di default di botocore (10) e' piu' piccolo di 16, quindi ogni lettura
# di un prefisso scartava connessioni invece di riusarle ("Connection pool
# is full, discarding connection", osservato nei log di `scheduler`).
# Allineato al parallelismo massimo usato in questo modulo.
_MAX_POOL_CONNECTIONS = 20

logging.basicConfig(level=logging.INFO)


class BucketKeyNotFound(KeyError):
    """La chiave richiesta non esiste nel bucket."""


_client = None
_bucket_name: Optional[str] = None
_lock_initialized = False


def _get_client_and_bucket():
    global _client, _bucket_name, _lock_initialized
    if _lock_initialized:
        return _client, _bucket_name

    env = require_bucket_env()
    _client = boto3.client(
        "s3",
        endpoint_url=env.endpoint,
        aws_access_key_id=env.access_key_id,
        aws_secret_access_key=env.secret_access_key,
        region_name=env.region or "auto",
        config=Config(max_pool_connections=_MAX_POOL_CONNECTIONS),
    )
    _bucket_name = env.bucket_name
    _lock_initialized = True
    return _client, _bucket_name


def _is_not_found(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in ("404", "NoSuchKey", "NotFound") or status == 404


def exists(key: str) -> bool:
    client, bucket = _get_client_and_bucket()
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if _is_not_found(exc):
            return False
        raise


def get_bytes(key: str) -> bytes:
    client, bucket = _get_client_and_bucket()
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        if _is_not_found(exc):
            raise BucketKeyNotFound(key) from exc
        raise


def put_bytes(key: str, data: bytes, content_type: Optional[str] = None) -> None:
    client, bucket = _get_client_and_bucket()
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key, "Body": data}
    if content_type:
        kwargs["ContentType"] = content_type
    client.put_object(**kwargs)


def get_json(key: str, default: Any = None) -> Any:
    try:
        raw = get_bytes(key)
    except BucketKeyNotFound:
        return default
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logging.exception("Contenuto non valido per la chiave %s, uso il default", key)
        return default


def put_json(key: str, payload: Any) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    put_bytes(key, data, content_type="application/json")


def get_joblib(key: str) -> Any:
    raw = get_bytes(key)
    buffer = io.BytesIO(raw)
    return joblib.load(buffer)


def put_joblib(key: str, obj: Any) -> None:
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    buffer.seek(0)
    put_bytes(key, buffer.read())


def list_keys(prefix: str) -> list[str]:
    client, bucket = _get_client_and_bucket()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def list_json(prefix: str) -> list[Any]:
    """`list_keys` + `get_json` in parallelo per ogni chiave - un oggetto
    per record (job, run, evento di promozione...) sotto `prefix` invece
    di un file ad append, quindi N GET invece di una lettura sola. Le
    chiamate girano in un ThreadPoolExecutor cosi' il costo resta
    dominato dalla latenza di rete, non dalla somma delle N round-trip
    sequenziali."""
    keys = list_keys(prefix)
    if not keys:
        return []
    with ThreadPoolExecutor(max_workers=min(16, len(keys))) as pool:
        results = list(pool.map(get_json, keys))
    return [item for item in results if item]


def delete(key: str) -> None:
    client, bucket = _get_client_and_bucket()
    client.delete_object(Bucket=bucket, Key=key)


def reset_client_for_tests() -> None:
    """SOLO per i test: forza la ricostruzione del client al prossimo uso,
    cosi' ogni test puo' iniettare env var/mock diversi senza interferenze
    dall'ordine di esecuzione."""
    global _client, _bucket_name, _lock_initialized
    _client = None
    _bucket_name = None
    _lock_initialized = False


def set_client_for_tests(client: Any, bucket_name: str) -> None:
    """SOLO per i test: inietta un client (reale o fake) senza passare da
    `ARCHIVE_BUCKET_*`, cosi' i moduli che chiamano `bucket_store` a livello
    di modulo (non iniettabile via costruttore) restano testabili."""
    global _client, _bucket_name, _lock_initialized
    _client = client
    _bucket_name = bucket_name
    _lock_initialized = True
