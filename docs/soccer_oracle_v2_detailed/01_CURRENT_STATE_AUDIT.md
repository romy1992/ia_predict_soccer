# 01 — Current State Audit

## KEEP
- importer API-Sports storico;
- mapper match/statistics/odds;
- repository ORM;
- dataset storico;
- React/FastAPI come stack;
- Model Registry come concetto;
- Prediction Logger come concetto;
- sperimentazioni U/O gerarchiche e Poisson;
- Docker come base locale.

## REFACTOR
- `src/api/dashboard_service.py`: troppe responsabilità;
- `src/service_ia/training/train_multi_market.py`: CV random + F1 come criterio dominante;
- `src/service_ia/training/model_registry.py`: manca lifecycle candidate/champion/production;
- `src/service_ia/training/prediction_logger.py`: da spostare verso DB/ledger;
- `src/service_ia/training/market_service/filter_market_service.py`: problemi semantici su 1X2/DC;
- `src/jobs/scheduler.py`: import + retrain giornaliero troppo accoppiati;
- frontend `App.jsx`: da spezzare in feature/componenti.

## REWRITE
- definizione ufficiale del mercato 1X2;
- Double Chance come derivazione coerente;
- validazione temporale production-grade;
- decision policy hardcoded.

## NEW
- DB canonico e policy ambiente;
- odds snapshots;
- point-in-time dataset builder;
- walk-forward validation;
- calibration framework;
- bookmaker fair baseline;
- betting backtesting;
- model lifecycle/promotion;
- paper betting;
- correlation engine;
- schedina Oracle.
