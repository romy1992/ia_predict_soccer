# Soccer ML Platform - Quick Start

Questa guida copre i nuovi moduli introdotti per:
- training multi-mercato,
- model registry,
- API backend,
- dashboard React,
- scheduler con job separati (data sync frequenti + training giornaliero indipendente, OPS-01).

## Nuovi moduli principali
- `src/service_ia/training/train_multi_market.py`
- `src/service_ia/training/market_service/filter_market_service.py`
- `src/service_ia/training/model_registry.py`
- `src/ml/registry/promotion_policy.py` (OPS-02: gate metriche + confronto candidate/production)
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

- `DATABASE_URL` -> default gia' impostato sul DB dev remoto Railway (vedi sezione "Policy DATABASE_URL" subito sotto); vecchio valore locale ormai sostituito, mantenuto solo come riferimento storico: `postgresql://postgres:postgres@localhost:5432/match_db`
- `APP_LEAGUES=135,136,140,39`
- `APP_SEASONS=2025,2026`
- `DATABASE_SCHEMA=public`

Scheduler (OPS-01, job separati - vedi sezione dedicata piu' sotto):
- `DATA_SYNC_INTERVAL_MINUTES=30` (fixture odierne, IntervalTrigger)
- `SETTLEMENT_INTERVAL_MINUTES=60` (settlement, IntervalTrigger)
- `FUTURE_SYNC_HOUR=4` / `FUTURE_SYNC_MINUTE=30` (fixture future, CronTrigger giornaliero)
- `TRAINING_HOUR=23` / `TRAINING_MINUTE=0` (retrain ML, CronTrigger giornaliero INDIPENDENTE dai data job — `SCHEDULER_HOUR`/`SCHEDULER_MINUTE` restano supportati come fallback legacy)

Se non li imposti, vengono usati i default del codice.

### Policy DATABASE_URL (dev/test/prod)
- `dev` (default, d'ora in poi): `DATABASE_URL` punta SEMPRE al remoto Postgres "dev" ospitato su Railway (`sakura.proxy.rlwy.net:18862/railway`, richiesto esplicitamente 2026-09-04) - stessa istanza per esecuzione locale, Docker e training, nessuna dipendenza da un Postgres nativo dell'host
- `docker compose`: **nessun Postgres containerizzato**. `api` e `scheduler` ereditano ENTRAMBI `DATABASE_URL` da `properties/config.env` (via `env_file`, nessun override in `environment:` cosi' da avere un'unica sorgente di verita' e non rischiare che i due servizi divergano) - DB remoto Railway sopra (nessun `host.docker.internal` necessario per il DB; resta configurato solo per compatibilita' con eventuali altri usi locali)
- fallback nel codice (SOLO se `DATABASE_URL` non e' impostata affatto, es. nessun `config.env` caricato): `DEFAULT_DATABASE_URL` in `src/service_ia/config/app_config.py`, anch'esso allineato al DB dev Railway
- `test`: usa un DB isolato tramite override env (`DATABASE_URL`) prima di lanciare i test
- verifica target attivo con `GET /health/database` (host/db/schema + conteggi tabelle)

Nel setup corrente la sorgente runtime e impostata su:
- `DATABASE_URL=postgresql://postgres:postgres@sakura.proxy.rlwy.net:18862/railway` (DB dev Railway, vedi `properties/config.env` per le credenziali complete)

Un'unica sorgente dati remota (Postgres Railway, dataset storico reale) per locale, Docker e training: nessun DB duplicato/vuoto da mantenere allineato.

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
- `scheduler` -> 4 job APScheduler separati e indipendenti (OPS-01): data sync/settlement frequenti (minuti) + future sync/training giornalieri (orari indipendenti) — vedi sezione "Run scheduler" piu' sotto

Prerequisito: connessione di rete raggiungibile verso il Postgres dev remoto su Railway (`sakura.proxy.rlwy.net:18862/railway`, vedi policy sopra), con le migration Alembic allineate (`alembic upgrade head`). Nessun Postgres locale richiesto.


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

## Run scheduler (job separati, OPS-01)
```powershell
python -m src.jobs.scheduler
```

Lo scheduler registra 4 job APScheduler COMPLETAMENTE separati e indipendenti (`max_instances=1`, `coalesce=True` su ciascuno) — mai un retrain automatico legato al ciclo di import:

| Job id             | Funzione                     | Trigger                                  | Frequenza di default        |
|--------------------|-------------------------------|-------------------------------------------|------------------------------|
| `data_sync_today`  | `run_manual_today_update`     | `IntervalTrigger` (minuti)                 | ogni 30 minuti               |
| `data_settlement`  | `run_manual_settlement`       | `IntervalTrigger` (minuti)                 | ogni 60 minuti               |
| `data_future_sync` | `run_manual_future_sync`      | `CronTrigger` (orario giornaliero)         | 04:30                        |
| `ml_training`      | `run_manual_retrain`          | `CronTrigger` (orario giornaliero, INDIPENDENTE) | 23:00                  |
| `data_sync_live`   | `run_manual_live_sync`        | `IntervalTrigger` (SECONDI, LIVE-01)       | ogni 90 secondi              |

Ogni job logga il proprio esito in `best_models/jobs_history.jsonl` (`job_type`: `today_update`/`settlement`/`future_sync`/`retrain`/`live_sync`). `build_scheduler(cfg)` costruisce lo scheduler SENZA avviarlo (usato dai test); `start_scheduler()` lo avvia (entry point di `python -m src.jobs.scheduler`).

## Trigger manuale da API
- `POST /jobs/import`
- `POST /jobs/today-update`
- `POST /jobs/future-sync`
- `POST /jobs/settlement`
- `POST /jobs/retrain`
- `POST /jobs/live-sync` (LIVE-01)

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

Endpoint dataset LIVE distinto (LIVE-01, `src/data/live/`) — dato GREZZO della pipeline live, separato dal dataset pre-match e da `/dashboard/live` (che resta invariato):
- `GET /live/fixtures` (ultimo snapshot per fixture non in stato finale)
- `GET /live/fixtures/{fixture_id}/events` (eventi con timestamp, ordinati per minuto)
- `GET /live/fixtures/{fixture_id}/statistics` (ultimo snapshot statistiche per squadra)

Nota dati dashboard:
- live e calendario giorno arrivano da API Sports (con cache di 60 secondi),
- se la fixture esiste anche nel DB locale e ci sono modelli disponibili, vengono aggiunte le previsioni,
- in assenza di feature/modelli, la partita e' comunque visibile ma senza prediction.

## Smoke test rapido API
```powershell
powershell -ExecutionPolicy Bypass -File scripts/smoke_api.ps1
```











