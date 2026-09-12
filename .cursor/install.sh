#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# L'immagine di base ha python3 ma non il modulo venv.
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3-venv
fi

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/pip install --quiet pytest

npm ci --prefix frontend
