#!/bin/bash
set -e
cd /app
alembic upgrade head
exec uvicorn src.api.main:app --host :: --port 8000
