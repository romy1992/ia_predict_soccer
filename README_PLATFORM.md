# Soccer ML Platform - Quick Start

Questa guida copre i nuovi moduli introdotti per:
- training multi-mercato,
- model registry,
- API backend,
- dashboard React,
- scheduler giornaliero alle 23:00.

## Nuovi moduli principali
- `src/service_ia/training/train_multi_market.py`
- `src/service_ia/training/market_service/filter_market_service.py`
- `src/service_ia/training/model_registry.py`
- `src/jobs/scheduler.py`
- `src/api/main.py`
- `frontend/src/App.jsx`
- `docker-compose.yml`
- `Dockerfile.api`

## Struttura runtime canonica
- runtime ufficiale backend/test/job: `src/`
- cartella legacy esperimenti: `service_ia/` (vedi `service_ia/README_LEGACY.md`)

## Configurazione
Nel file `properties/config.env` puoi aggiungere:

- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- `APP_LEAGUES=135,136,140,39`
- `APP_SEASONS=2025,2026`
- `SCHEDULER_HOUR=23`
- `SCHEDULER_MINUTE=0`
- `DATABASE_SCHEMA=public`

Se non li imposti, vengono usati i default del codice.

### Policy DATABASE_URL (dev/test/prod)
- `dev locale`: imposta `DATABASE_URL` verso l'istanza locale scelta (es. `localhost:5432`)
- `docker compose`: `api` e `scheduler` puntano entrambi a `postgresql://postgres:postgres@db:5432/match_db`
- `test`: usa un DB isolato tramite override env (`DATABASE_URL`) prima di lanciare i test
- verifica target attivo con `GET /health/database` (host/db/schema + conteggi tabelle)

Nel setup corrente locale la sorgente runtime e impostata su:
- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`

Per il frontend React puoi usare anche:
- `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env`

## Avvio con Docker (consigliato)
```powershell
docker compose up --build -d
docker compose ps
```

URL servizi:
- `http://localhost:3000` -> frontend React
- `http://localhost:8000/docs` -> API FastAPI
- `localhost:5433` -> Postgres container (porta host)
- `scheduler` -> job giornaliero automatico alle 23:00

Stop servizi:
```powershell
docker compose down
```

## Avvio locale senza Docker
### Backend
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend React
```powershell
Set-Location frontend
npm install
npm run dev
```

Apri nel browser:
- `http://127.0.0.1:3000/` (dashboard React)
- `http://127.0.0.1:8000/docs` (OpenAPI)

## Run training multi-mercato (manuale)
```powershell
python -m src.service_ia.training.train_multi_market
```

Output principali:
- `best_models/*.pkl`
- `best_models/registry/index.jsonl`
- `best_models/training_summary.json`

## Run scheduler (job giornaliero)
```powershell
python -m src.jobs.scheduler
```

Il job giornaliero esegue:
1. import match/statistiche/quote,
2. aggiornamento mean feature,
3. retrain multi-mercato,
4. log esito job in `best_models/jobs_history.jsonl`.

## Trigger manuale da API
- `POST /jobs/import`
- `POST /jobs/today-update`
- `POST /jobs/future-sync`
- `POST /jobs/settlement`
- `POST /jobs/retrain`

Entrambi supportano `async_run=true/false`.

## Nota operativa
Le predizioni vengono loggate in:
- `best_models/predictions_log.jsonl`

Le metriche/versioni modello sono consultabili in:
- `GET /metrics/{market}`

Endpoint utili Data Platform:
- `GET /health/database`
- `GET /odds/snapshots/{fixture_id}`
- `GET /settlement/overview`
- `GET /data/quality`
- `GET /jobs/history?limit=100&job_type=import&status=success`
- `GET /dashboard/match/{fixture_id}` include anche `bookmaker_baseline` (implied/fair probabilities)

Parametri utili:
- `POST /jobs/import` accetta `from_date`, `to_date`, `fixture_date`, `statuses`, `seasons`, `leagues`, `async_run`
- `GET /data/quality` accetta `top_n`, `seasons` (csv), `leagues` (csv)

Endpoint dashboard dedicati alla UI React:
- `GET /dashboard/overview?target_date=YYYY-MM-DD`
- `GET /dashboard/live?target_date=YYYY-MM-DD&limit=30`
- `GET /dashboard/day?target_date=YYYY-MM-DD&phase=all|to_play|live|finished&search=term`
- `GET /dashboard/match/{fixture_id}`

Nota dati dashboard:
- live e calendario giorno arrivano da API Sports (con cache di 60 secondi),
- se la fixture esiste anche nel DB locale e ci sono modelli disponibili, vengono aggiunte le previsioni,
- in assenza di feature/modelli, la partita e' comunque visibile ma senza prediction.

## Smoke test rapido API
```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_api.ps1
```












