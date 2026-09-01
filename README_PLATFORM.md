# Soccer ML Platform - Quick Start

Questa guida copre i nuovi moduli introdotti per:
- training multi-mercato,
- model registry,
- API backend,
- dashboard,
- scheduler giornaliero alle 23:00.

## Nuovi moduli principali
- `src/service_ia/training/train_multi_market.py`
- `src/service_ia/training/market_service/filter_market_service.py`
- `src/service_ia/training/model_registry.py`
- `src/jobs/scheduler.py`
- `src/api/main.py`
- `frontend/dashboard.html`

## Configurazione
Nel file `properties/config.env` puoi aggiungere:

- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- `APP_LEAGUES=135,136,140,39`
- `APP_SEASONS=2025,2026`
- `SCHEDULER_HOUR=23`
- `SCHEDULER_MINUTE=0`

Se non li imposti, vengono usati i default del codice.

## Installazione dipendenze
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run API + dashboard
```powershell
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Apri nel browser:
- `http://127.0.0.1:8000/` (dashboard)
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

