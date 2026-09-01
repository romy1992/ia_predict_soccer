# Roadmap crescita progetto ML + Dashboard

## Obiettivo
Portare il progetto da script tecnici a piattaforma operativa con:
- training multi-mercato (un modello per ogni mercato),
- monitoraggio predizioni e performance,
- import feature giornaliero manuale e automatico,
- job pianificato ogni giorno alle 23:00.

## Stato attuale (confermato da codice)

### Data ingestion e DB
- ORM e mercati odds in `src/service_ia/model/match.py`.
- Import partite/statistiche/quote in `src/service_ia/pre_processing/download_match_service.py`.
- Mercati presenti a livello modello: `h2h`, `under_over_1_5`, `under_over_2_5`, `under_over_3_5`, `under_over_4_5`, `under_over_home_away`, `goal_no_goal`, `corners`, `cards`, `dc`.

### Training ML
- Hyperparameter search disponibile in `src/service_ia/training/utility_training/fit_search_best_model.py` (Grid, Random, Halving).
- Utility di valutazione e CV in `src/service_ia/training/utility_training/fit_utility.py`.
- Ensemble disponibili in `src/service_ia/training/utility_training/fit_ensemble_classifier.py` (Voting, Bagging, Stacking).
- Persistenza modelli in `src/service_ia/training/utility_training/save_load.py` (solo `.pkl`).

### Scope funzionale attuale
- Pipeline training principalmente orientata a under/over (`src/service_ia/training/under_over/...`).
- Nessuna API web per serving predizioni.
- Nessuna dashboard UI.
- Nessun scheduler applicativo per job giornalieri.

## Gap critici da risolvere prima della scalabilita

### Alta priorita
1. **Moduli con esecuzione automatica in import**
   - Esempi: `download_match_service.py`, `df_statistics_service.py`, `df_aggregate_service.py`, `stacking_events.py`.
   - Impatto: side effect, run non controllati, rischio in produzione/test.

2. **Filtri DB fragili su `None`/`not None`**
   - In `src/repository/base/crud_repository.py` si usano controlli Python (`col is None`) invece di operatori SQLAlchemy.
   - Impatto: query potenzialmente errate, dataset training incoerenti.

3. **Nessun model registry/versioning**
   - Solo file PKL, nessun metadata strutturato (feature, metriche, run_id, data training).

4. **Nessun layer API e orchestrazione job**
   - Mancano endpoint per predizione/metriche/stato pipeline e scheduler alle 23:00.

### Media priorita
5. **Configurazioni hardcoded** (`leagues`, `seasons`, DB URL in codice).
6. **Feature selection non formalizzata** (manca pipeline standard per selezione feature per mercato).
7. **Test coverage limitata** su pipeline end-to-end e job schedulati.

## Architettura target (proposta)

### Layer 1 - Data & Jobs
- Job giornaliero (23:00) con orchestrazione centralizzata:
  1) import nuovi match/statistiche/quote,
  2) calcolo feature aggregate,
  3) training/retraining modelli per mercati attivi,
  4) validazione e publish del modello migliore,
  5) logging esito job.

### Layer 2 - ML Platform
- Pipeline unica parametrica per mercato:
  - dataset builder,
  - feature engineering,
  - feature selection,
  - tuning (grid/halving/random),
  - confronto baseline vs ensemble,
  - salvataggio modello + metadata.

### Layer 3 - Serving API
- API backend (es. FastAPI):
  - `POST /predict/{market}`,
  - `GET /models/{market}/latest`,
  - `GET /metrics/{market}`,
  - `POST /jobs/import` (manuale),
  - `POST /jobs/retrain` (manuale).

### Layer 4 - Dashboard
- UI di controllo:
  - stato job giornaliero,
  - andamento metriche per mercato,
  - storico predizioni vs risultato reale,
  - trigger manuale import/retrain,
  - gestione mercati attivi/disattivi.

## Piano operativo in 4 fasi

## Fase 0 - Stabilizzazione tecnica (1-2 settimane)
**Obiettivo:** rendere il codice eseguibile in modo deterministico.

Task:
- Rimuovere esecuzioni a fondo file e introdurre `if __name__ == "__main__":`.
- Correggere `search_filter` con operatori SQLAlchemy (`is_(None)`, `is_not(None)`).
- Spostare config in env (DB URL, leagues, seasons, flags).
- Aggiungere logging strutturato base.

Done quando:
- import moduli non scatena job,
- test esistenti passano,
- query filtri restituiscono risultati attesi.

## Fase 1 - Core ML multi-mercato (2-4 settimane)
**Obiettivo:** creare framework unico per addestrare ogni mercato.

Task:
- Definire catalogo mercati (DB + mappatura quote + label logic).
- Generalizzare builder dataset oltre under/over (`h2h`, `goal_no_goal`, `corners`, `cards`, `dc`).
- Integrare feature selection (filtro + wrapper) per mercato.
- Implementare benchmark automatico:
  - modelli base,
  - tuning,
  - ensemble (voting/stacking).
- Salvare metriche comparabili per mercato.

Done quando:
- per ogni mercato attivo esiste un training report ripetibile,
- esiste un modello "champion" per mercato con metadata.

## Fase 2 - Jobs e automazione (1-2 settimane)
**Obiettivo:** import e retrain automatici alle 23:00 + trigger manuale.

Task:
- Introdurre scheduler (APScheduler o Celery beat + worker).
- Job quotidiano con lock anti-concorrenza.
- Endpoint/CLI per trigger manuale import e retrain.
- Notifiche esito job (log + eventuale Telegram).

Done quando:
- job 23:00 gira in automatico,
- manual trigger disponibile da API,
- storicizzazione esiti job disponibile.

## Fase 3 - Dashboard operativa (2-3 settimane)
**Obiettivo:** pannello unico di governo.

Task:
- Backend API per metriche/predizioni/job status.
- Frontend dashboard (React/Vue) con 4 viste:
  1) overview mercati,
  2) metriche modelli,
  3) prediction log,
  4) control panel import/retrain.
- Ruoli utente minimi (admin/read-only).

Done quando:
- puoi vedere e gestire tutto dal pannello,
- niente run manuali da script per operazioni ordinarie.

## KPI da monitorare
- Coverage mercati attivi (% mercati con modello champion valido).
- Hit-rate e ROI per mercato (oltre accuracy).
- Drift feature e drift performance settimanale.
- Tasso successo job 23:00.
- Lead time retrain -> deploy.

## Deliverable finali
1. Pipeline ML multi-mercato standardizzata.
2. Registry modelli con metadata e storico.
3. Scheduler giornaliero + trigger manuale.
4. API backend per predizione e monitoraggio.
5. Dashboard operativa per gestione end-to-end.

## Prossimo step consigliato (immediato)
Aprire il primo sprint sulla **Fase 0** e chiuderlo prima di aggiungere nuova UI:
- stabilita codice,
- quality gate minimo,
- basi solide per costruire dashboard e automazione senza regressioni.

