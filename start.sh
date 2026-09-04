#!/bin/bash
set -e
cd /app
alembic upgrade head
exec uvicorn src.api.main:app --host 0.0.0.0 --port 8000
