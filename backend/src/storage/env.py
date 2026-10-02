"""Lettura delle variabili d'ambiente del Bucket Railway (`ARCHIVE_BUCKET_*`).

Reference variables Railway (`${{models-archive.*}}`), iniettate dal
servizio ad ogni deploy - mai hardcodate, mai lette a import-time (vedi
`bucket_store.py`)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

_REQUIRED_VARS = (
    "ARCHIVE_BUCKET_ENDPOINT",
    "ARCHIVE_BUCKET_ACCESS_KEY_ID",
    "ARCHIVE_BUCKET_SECRET_ACCESS_KEY",
    "ARCHIVE_BUCKET_NAME",
)


@dataclass(frozen=True)
class BucketEnv:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    region: Optional[str]


def require_bucket_env() -> BucketEnv:
    missing = [name for name in _REQUIRED_VARS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Variabili d'ambiente del bucket mancanti: "
            f"{', '.join(missing)}. Verifica che il servizio sia collegato "
            "al bucket Railway 'models-archive' (reference variables)."
        )
    return BucketEnv(
        endpoint=os.environ["ARCHIVE_BUCKET_ENDPOINT"],
        access_key_id=os.environ["ARCHIVE_BUCKET_ACCESS_KEY_ID"],
        secret_access_key=os.environ["ARCHIVE_BUCKET_SECRET_ACCESS_KEY"],
        bucket_name=os.environ["ARCHIVE_BUCKET_NAME"],
        region=os.environ.get("ARCHIVE_BUCKET_REGION") or None,
    )
