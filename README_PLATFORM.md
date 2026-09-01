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

## Configurazione
Nel file `properties/config.env` puoi aggiungere:

- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- `APP_LEAGUES=135,136,140,39`
- `APP_SEASONS=2025,2026`
- `SCHEDULER_HOUR=23`
- `SCHEDULER_MINUTE=0`

Se non li imposti, vengono usati i default del codice.

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
- `POST /jobs/retrain`

Entrambi supportano `async_run=true/false`.

## Nota operativa
Le predizioni vengono loggate in:
- `best_models/predictions_log.jsonl`

Le metriche/versioni modello sono consultabili in:
- `GET /metrics/{market}`

Endpoint dashboard dedicati alla UI React:
- `GET /dashboard/overview?target_date=YYYY-MM-DD`
- `GET /dashboard/live?target_date=YYYY-MM-DD&limit=30`
- `GET /dashboard/day?target_date=YYYY-MM-DD&phase=all|to_play|live|finished&search=term`

## Smoke test rapido API
```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_api.ps1
```





