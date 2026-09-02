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
- [x] EXP-03
- [x] EXP-04
- [x] EXP-05
- [x] MARKET-01
- [x] MARKET-02
- [x] MARKET-03
- [x] MARKET-04
- [x] MARKET-05

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
- Statistics Expert: modello pre-match basato esclusivamente su `mean_statistics` (nessuna feature odds), validazione temporale con metriche probabilistiche ML-05, embedding numerico riusabile (`src/ml/experts/statistics/statistics_expert.py`)
- Market/Odds Expert: fair probabilities (riuso bookmaker_baseline), dispersione bookmaker, movement quote e opening/latest/closing point-in-time-safe (closing mai valorizzato prima del kickoff) (`src/ml/experts/market/market_odds_expert.py`)
- Direct Market Expert: interfaccia comune `predict_proba` sui champion model esistenti (h2h/dc/goal_no_goal/corners/cards/under_over_*), con metadati espliciti che impediscono di scambiare h2h binario per un 1X2 multiclasse (`src/ml/experts/direct/direct_market_expert.py`)
- **Fase ORACLE EXPERTS completata (EXP-01..05)**
- Vero mercato 1X2 multiclass (HOME/DRAW/AWAY, nessun mapping draw->away): dataset dedicato, GridSearchCV logistic/random_forest con validazione temporale, metriche multiclasse dedicate (log_loss/brier generalizzato/ECE su confidence/AUC OvR) e calibrazione multiclasse via `CalibratedClassifierCV` (`src/ml/markets/market_1x2.py`, `src/ml/evaluation/multiclass_probability_metrics.py`, `src/ml/calibration/multiclass_calibration_service.py`)
- Double Chance derivata da 1X2 coerente (nessun training proprio): P(1X)=P(HOME)+P(DRAW), P(12)=P(HOME)+P(AWAY), P(X2)=P(DRAW)+P(AWAY), fair odds e wrapper batch su `Market1x2Expert` (`src/ml/markets/market_double_chance.py`)
- BTTS consolidato (score distribution EXP-02 vs direct expert 'goal_no_goal' EXP-05): benchmark comparativo (score_distribution/direct_expert/ensemble) con selezione via `champion_probability_score`, calibrazione OOF temporale (Platt/isotonic) del solo approccio vincente, P(Yes)+P(No)=1 garantito per costruzione (No=1-Yes); dataset builder end-to-end che riusa rating point-in-time (`TeamStrengthExpert`, EXP-01) + score distribution Poisson (`GoalDistributionExpert`, EXP-02) + feature/target 'goal_no_goal' (`FilterMarketService`, EXP-05); orchestratore `run_btts_benchmark(_from_db)` che registra il calibratore vincente come 'candidate' (mai 'production' automatica) (`src/ml/markets/btts/btts_market.py`)
- U/O 1.5-4.5 multi-linea consolidato: confronto sullo STESSO walk-forward tra binary_independent (4 classificatori scorrelati), hierarchical (1 solo classificatore multiclasse sui bin di gol totali 0/1/2/3/4, confini esattamente sulle soglie, P(Over t) = cumulata dall'alto) e goal_distribution (Poisson EXP-02 da rating EXP-01); selezione per metriche probabilistiche+betting (`champion_probability_score`) aggregate sulle 4 soglie; monotonicità P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5) resa OBBLIGATORIA sull'output finale via proiezione isotonica (`enforce_monotonic_over_probabilities`), indipendentemente dall'approccio vincente; orchestratore `run_totals_benchmark(_from_db)` che registra il modello vincente come 'candidate' solo se introduce un nuovo estimator (hierarchical/binary_independent), nessun salvataggio per goal_distribution (deterministico) (`src/ml/markets/totals/totals_market.py`)
- Corners O/U specializzato con linea configurabile (8.5/9.5/10.5/11.5 come parametro, non più soglia fissa a 9.5): feature dedicate (media storica corner fatti/concessi per squadra, totale atteso, differenziale, da `mean_statistics['Corner Kicks']`) oltre alle generiche odds/mean_stats; calibrazione (Platt/isotonic) via `CalibrationService` (ML-06, riusata) per il modello di ciascuna linea; report con metriche per linea; `CornersExpert` parametrico sulla linea, ogni linea registrata come mercato indipendente nel registry (es. `corners_line_9_5`) (`src/ml/markets/corners/corners_market.py`)



## Migrazioni applicate (locale + docker)
- locale: `alembic stamp 55bbb5f0a367` + `alembic upgrade head`
- docker: `docker compose exec api alembic upgrade head`

## Connessione DB locale
- runtime locale fissato su DB reale: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- verifica rapida: tabella `match` letta con volume storico (>47k righe)

Note: gli stati sopra sono riferiti all'implementazione tecnica nel branch corrente; la validazione finale dipende dall'esecuzione acceptance/test su ambiente dati reale.
























