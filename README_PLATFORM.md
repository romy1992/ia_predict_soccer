# Soccer ML Platform - Quick Start

Questa guida copre i nuovi moduli introdotti per:
- training multi-mercato,
- model registry,
- API backend,
- dashboard React,
- scheduler con job separati (data sync frequenti + training giornaliero indipendente, OPS-01).

## Struttura repo (monorepo)
- `backend/` - API FastAPI, scheduler, training ML, migrazioni Alembic (vedi sotto)
- `frontend/` - dashboard React
- `docker-compose.yml`, `docs/` - orchestrazione e documentazione a livello repo

## Nuovi moduli principali
- `backend/src/service_ia/training/train_multi_market.py`
- `backend/src/service_ia/training/market_service/filter_market_service.py`
- `backend/src/service_ia/training/model_registry.py`
- `backend/src/ml/registry/promotion_policy.py` (OPS-02: gate metriche + confronto candidate/production)
- `backend/src/jobs/scheduler.py`
- `backend/src/api/main.py`
- `frontend/src/App.jsx`
- `docker-compose.yml`
- `backend/Dockerfile`

## Struttura runtime canonica
- runtime ufficiale backend/test/job: `backend/src/`
- cartella legacy esperimenti: `backend/service_ia/` (vedi `backend/service_ia/README_LEGACY.md`)

## Configurazione
Nel file `backend/properties/config.env` puoi aggiungere:

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
- `docker compose`: **nessun Postgres containerizzato**. `api` e `scheduler` ereditano ENTRAMBI `DATABASE_URL` da `backend/properties/config.env` (via `env_file`, nessun override in `environment:` cosi' da avere un'unica sorgente di verita' e non rischiare che i due servizi divergano) - DB remoto Railway sopra (nessun `host.docker.internal` necessario per il DB; resta configurato solo per compatibilita' con eventuali altri usi locali)
- fallback nel codice (SOLO se `DATABASE_URL` non e' impostata affatto, es. nessun `config.env` caricato): `DEFAULT_DATABASE_URL` in `backend/src/service_ia/config/app_config.py`, anch'esso allineato al DB dev Railway
- `test`: usa un DB isolato tramite override env (`DATABASE_URL`) prima di lanciare i test
- verifica target attivo con `GET /health/database` (host/db/schema + conteggi tabelle)

Nel setup corrente la sorgente runtime e impostata su:
- `DATABASE_URL=postgresql://postgres:postgres@sakura.proxy.rlwy.net:18862/railway` (DB dev Railway, vedi `backend/properties/config.env` per le credenziali complete)

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
pip install -r backend\requirements.txt
Set-Location backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Nota: i comandi `python -m src...` e `uvicorn src...` vanno sempre lanciati con `backend/` come cartella corrente (cwd) - e' da li' che il codice risolve `properties/config.env`, `best_models/` e `logs/`.

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
Set-Location backend
python -m src.service_ia.training.train_multi_market
```

Output principali:
- `backend/best_models/*.pkl`
- `backend/best_models/registry/index.jsonl`
- `backend/best_models/training_summary.json`

## Run scheduler (job separati, OPS-01)
```powershell
Set-Location backend
python -m src.jobs.scheduler
```

Lo scheduler registra 6 job APScheduler COMPLETAMENTE separati e indipendenti (`max_instances=1`, `coalesce=True` su ciascuno) — mai un retrain automatico legato al ciclo di import:

| Job id               | Funzione                     | Trigger                                  | Frequenza di default        |
|-----------------------|-------------------------------|-------------------------------------------|------------------------------|
| `data_sync_today`    | `run_manual_today_update`     | `IntervalTrigger` (minuti)                 | ogni 30 minuti               |
| `data_settlement`    | `run_manual_settlement`       | `IntervalTrigger` (minuti)                 | ogni 60 minuti               |
| `data_future_sync`   | `run_manual_future_sync`      | `CronTrigger` (orario giornaliero)         | 04:30                        |
| `data_daily_refresh` | `run_daily_refresh`           | `CronTrigger` (orario giornaliero)         | 05:00 (ieri + prossimi 7gg)  |
| `ml_training`        | `run_manual_retrain`          | `CronTrigger` (orario giornaliero, INDIPENDENTE) | 23:00                  |
| `data_sync_live`     | `run_manual_live_sync`        | `IntervalTrigger` (SECONDI, LIVE-01)       | ogni 90 secondi              |

`data_daily_refresh` e' lo STESSO job invocato dal bottone "Aggiorna tutto" della Sidebar (sempre visibile, in ogni pagina del frontend): importa le partite di IERI (tutti i campionati censiti) + sincronizza il calendario prossimo (`DAILY_REFRESH_DAYS_AHEAD` giorni, default 7, con upsert sulle partite gia' presenti). Si sovrappone volutamente alla finestra di `data_future_sync` — disattivabile da Impostazioni se si vuole un solo giro/giorno.

Tutti i job che passano da `download_import_matches` (`data_daily_refresh`, `data_sync_today`, `data_future_sync`, `data_settlement`) e il bottone "Aggiorna tutto" condividono un advisory lock Postgres (`backend/src/jobs/job_lock.py`, 2026-09-15): `api` e `scheduler` sono container distinti e `max_instances=1` di APScheduler vale solo dentro il proprio processo, quindi senza lock il bottone premuto mentre girava il job schedulato produceva due import sovrapposti sulla stessa finestra di date — 116 partite finite a DB in doppia copia. Chi non ottiene il lock salta il giro e lo storico registra un `success` con `skipped_locked` (non un `failed`, che `_is_job_due` non potrebbe distinguere da un problema vero).

### Quote di chiusura per il CLV (`closing_odds_capture`)

Job "Quote di chiusura (CLV)" (`run_closing_odds_capture`, servizio `backend/src/oracle/ledger/closing_odds_capture_service.py`): rende misurabile il CLV ufficiale (`GET /predictions/official/clv`, campo `by_decision`) per PLAY, BORDERLINE e NO BET. Senza, l'ultimo snapshot quote prima del kickoff e' spesso quello usato per la giocata (catturata fino a 60 minuti prima) e il CLV resta `closing_not_after_bet`.

- **Quando**: ogni 5 minuti (`interval_minutes`, modificabile da Impostazioni o con `CLOSING_ODDS_CAPTURE_INTERVAL_MINUTES`; minimo 5). Abilitato di default, `calls_api_sports=True`: sospeso dall'auto-pausa a quota esaurita e ripristinato al reset. "Esegui ora": `POST /jobs/closing-odds-capture`.
- **Quali partite**: scelte dal DB, senza chiamate API: status `NS`, almeno una riga nel Prediction Ledger con cohort `official_paper` (qualunque decisione) e kickoff (`match.date_match`) nei prossimi 15 minuti.
- **Quante chiamate**: al massimo una `GET /odds?fixture=<id>` per momento, quindi al massimo 2 per partita. `early` fra 15 e 10 minuti prima del kickoff (recuperato fra 10 e 5 se la cadenza, 5 minuti piu' il battito di 30 secondi, salta quella finestra); `late` negli ultimi 5 minuti. Il momento viene prenotato prima della chiamata in `best_models/closing_odds_capture_state.json` sul bucket (voci rimosse dopo 2 giorni), cosi' un secondo giro o un "Esegui ora" non lo ripete. A circuit breaker aperto (nessuna richiesta partita) il momento resta libero per il giro successivo; dopo una chiamata fallita (provider non disponibile, quota esaurita) no, cosi' il limite di 2 chiamate resta rispettato.
- **Salvataggio**: `map_odds_snapshots` + `OddsSnapshotRepository.save_many` (upsert idempotente, righe in ordine di chiave). Non prende il lock degli import, che salterebbe ogni volta che gira `data_sync_today`: puo' girare in parallelo a un import.
- **`captured_at`**: e' il campo `update` della risposta API-Sports (ora in cui il provider ha aggiornato le quote), o l'ora del download se manca. Il download avviene prima del kickoff, quindi lo snapshot e' una chiusura valida (`captured_at <= kickoff`). Se il provider non ha aggiornato le quote dopo la giocata, `update` non cambia: lo snapshot coincide con quello gia' salvato (nessuna riga nuova) e il CLV di quella giocata resta `closing_not_after_bet`, correttamente. Il report lo conta in `provider_update_unchanged`.
- **Report** (storico job, `job_type=closing_odds_capture`): `candidates`, `calls`, `captured`, `snapshots_saved`, `empty_responses`, `provider_update_unchanged`, `skipped` (`already_captured`, `outside_window`, `quota_exhausted`), `errors` (`provider`, `quota_exhausted`, `save`, `registry`) con `error_details`, e `captures` per partita. Gli errori del provider non fanno fallire il job.
- **Costo**: al massimo 2 chiamate API-Sports per partita con giocate ufficiali al giorno, contro un limite giornaliero di 7.500 (`API_SPORTS_DAILY_LIMIT` in `backend/src/service_ia/config/app_config.py`). Con 100 partite ufficiali in un giorno sono al massimo 200 chiamate (2,7% del limite). Ogni chiamata passa da `ApiSportsProvider`, che su 429/5xx/errori di rete ritenta fino a 3 volte: solo in quel caso le richieste HTTP possono superare 2 per partita.

Ogni job logga il proprio esito in `backend/best_models/jobs_history.jsonl` (`job_type`: `today_update`/`settlement`/`future_sync`/`daily_refresh`/`retrain`/`live_sync`). `build_scheduler(cfg)` costruisce lo scheduler SENZA avviarlo (usato dai test); `start_scheduler()` lo avvia (entry point di `python -m src.jobs.scheduler`).

## Trigger manuale da API
- `POST /jobs/import`
- `POST /jobs/today-update`
- `POST /jobs/future-sync`
- `POST /jobs/daily-refresh` (ieri + prossimi N giorni - bottone "Aggiorna tutto")
- `POST /jobs/settlement`
- `POST /jobs/retrain`
- `POST /jobs/live-sync` (LIVE-01)
- `POST /jobs/closing-odds-capture` (quote di chiusura per il CLV, vedi sopra)

Tutti supportano `async_run=true/false`.

## Nota operativa
Le predizioni vengono loggate in:
- `backend/best_models/predictions_log.jsonl`

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
- `GET /dashboard/day?target_date=YYYY-MM-DD&phase=all|to_play|live|finished|unknown&search=term`
  - `unknown` (2026-09-15): partita con calcio d'inizio passato da oltre 4 ore ma ancora senza stato finale a DB, cioe' una riga che il sync non ha aggiornato. Prima queste righe venivano classificate `live` deducendo la fase dall'orario, e una partita di giorni prima restava "In diretta" per sempre (vedi `DashboardService._classify_phase`).
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
powershell -ExecutionPolicy Bypass -File backend/scripts/smoke_api.ps1
```











