# Soccer Oracle V2 - Implementation Log

## Baseline (SOCCER-00)
- base branch verificato: `feature/ml-dashboard-platform`
- branch operativo attivo: `feature/soccer-oracle-v2`
- baseline SHA iniziale: `d12a661`
- data verifica: 2026-09-02

## Premessa comune prima di ogni task
Prima di ogni task viene applicata la premessa in `AI_MASTER_PROMPT.md`:
1. ispezione file coinvolti
2. no riscritture inutili
3. compatibilita DB storico
4. migration Alembic per cambi DB
5. no leakage temporale
6. no split random in pipeline production
7. latest model != production model
8. aggiornare test pertinenti

## Sequenza immediata (CURRENT_TASK)
1. SOCCER-00
2. SOCCER-01
3. SOCCER-02
4. DATA-01
5. DATA-02
6. DATA-03
7. DATA-04
8. DATA-06
9. ML-01
10. ML-02

## Stato implementazione
- [x] SOCCER-00
- [x] SOCCER-01
- [x] SOCCER-02
- [x] DATA-01
- [x] DATA-02
- [x] DATA-03
- [x] DATA-04
- [x] DATA-06
- [x] ML-01
- [x] ML-02
- [x] DATA-05
- [x] DATA-07
- [x] ML-03
- [x] DATA-08
- [x] FE-01
- [x] FE-02
- [x] FE-03
- [x] ML-04
- [x] ML-05
- [x] ML-06
- [x] ML-07
- [x] EXP-01
- [x] EXP-02

## Estensioni introdotte
- settlement job idempotente con completezza finale (`/jobs/settlement`)
- overview settlement (`/settlement/overview`)
- report Data Quality machine-readable (`/data/quality`)
- pipeline training con selector dentro CV fold (anti-leakage)
- migrazione Alembic per campi settlement su `match`
- job history evoluta con lifecycle queued/running/success/failed e filtri tipo/stato
- refactor frontend a feature components + router locale (`frontend/src/features/`)
- Data Center FE con import storico/oggi/future e settlement manuale
- Data Quality FE con coverage/anomalie/distribuzioni e filtri stagione/lega
- migration compatibilità schema `match` (`current_league`, `league_match`, `status`, `mean_statistics`)
- baseline bookmaker implied/fair probabilities nei report match detail
- framework metriche probabilistiche (LogLoss, Brier, ECE, AUC) e champion selection multi-metrica
- calibration framework (Platt/sigmoid + isotonic) con confronto pre/post e calibratore versionato per model run (`src/ml/calibration/calibration_service.py`)
- Model Registry con lifecycle completo candidate/champion/production/retired, promotion history, lookup produzione per mercato (`src/service_ia/training/model_registry.py`)
- Team Strength Expert: rating offensivo/difensivo, home advantage, rolling form point-in-time, backtest base (`src/ml/experts/team_strength/team_strength_expert.py`)
- Goal Distribution Expert: Poisson lambda regressor con validazione temporale (no train_test_split), soglie U/O 1.5/2.5/3.5/4.5, score distribution completa e confronto Poisson vs Binomiale Negativa (`src/ml/experts/goal_distribution/goal_distribution_expert.py`)

## Migrazioni applicate (locale + docker)
- locale: `alembic stamp 55bbb5f0a367` + `alembic upgrade head`
- docker: `docker compose exec api alembic upgrade head`

## Connessione DB locale
- runtime locale fissato su DB reale: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- verifica rapida: tabella `match` letta con volume storico (>47k righe)

Note: gli stati sopra sono riferiti all'implementazione tecnica nel branch corrente; la validazione finale dipende dall'esecuzione acceptance/test su ambiente dati reale.








