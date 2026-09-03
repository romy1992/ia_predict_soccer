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
- [x] MARKET-06
- [x] ORACLE-01
- [x] ORACLE-02
- [x] ORACLE-03
- [x] ORACLE-04
- [x] BET-01
- [x] BET-02
- [x] BET-03
- [x] BET-04
- [x] BET-05
- [x] BET-06

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
- Cards O/U specializzato con linea configurabile (3.5/4.5/5.5/6.5 come parametro): feature "team/style" già disponibili genericamente da `mean_statistics` (Yellow/Red Cards, Fouls) più feature ARBITRO nuove e point-in-time (`build_referee_features_dataset`: media storica cartellini elargiti nelle partite precedenti arbitrate dallo stesso arbitro, nessun leakage, fallback a prior neutro senza storico); calibrazione via `CalibrationService` (ML-06) per ciascuna linea; `CardsExpert` parametrico sulla linea, stesso pattern registry di `CornersExpert` (`src/ml/markets/cards/cards_market.py`)
- **Fase MARKETS completata (MARKET-01..06)**
- Schema di output standardizzato per gli Oracle Experts (dataclass `ExpertOutput`: expert_name/expert_version, probability_vector con validazione [0,1], model_run_id, stage, feature_timestamp, confidence con default automatico, metadata, raw_output) e `combine_expert_outputs` per assemblare la riga di feature multi-esperto senza collisioni di naming (`src/ml/ensemble/expert_output.py`); adapter dedicati per ciascun esperto esistente SENZA modificarli (EXP-01 team_strength: probability_vector vuoto per costruzione, solo rating/feature; EXP-02 goal_distribution: over/under da Poisson o da score matrix; EXP-03 statistics: vettore binario complementare dalla probabilità calibrata; EXP-04 market_odds: fair probabilities da `compute_market_baseline`; adapter generico `from_predict_proba_expert` per EXP-05 `DirectMarketExpert` + MARKET-05/06 `CornersExpert`/`CardsExpert`, che condividono la stessa interfaccia `predict_proba`/`predict_proba_dict`) (`src/ml/ensemble/adapters.py`)
- **ORACLE-01 completato**: tutti gli esperti sono consumabili da un futuro meta-model tramite lo schema comune (acceptance criteria verificato dal test end-to-end `test_outputs_from_different_experts_are_consumable_by_meta_model`)
- Meta Model / Stacker per mercato (ORACLE-02): confronto, sulle STESSE meta-feature (probabilita' dei singoli Oracle Expert per lo stesso mercato, tipicamente da `combine_expert_outputs`/ORACLE-01), tra `weighted_blend` (`WeightedBlendClassifier`, pesi non negativi che sommano a 1, ottimizzati via softmax+minimizzazione log loss) e `learned_stacker` (default `LogisticRegression`); ENTRAMBI validati con lo stesso meccanismo di OOF walk-forward gia' in uso nel progetto (`temporal_oof_probabilities`, ML-05, mai fit sul fold di validazione), selezione del vincitore via `champion_probability_score` e meta-model finale rifittato sull'intero storico; `build_meta_features_from_expert_outputs` collega esplicitamente lo schema ORACLE-01 alle meta-feature; orchestratore `run_stacking_benchmark` che registra il meta-model vincente come 'candidate' (mai 'production' automatica) (`src/ml/ensemble/stacking.py`)
- Calibrazione finale Oracle Ensemble (ORACLE-03): applica `CalibrationService` (ML-06, gia' riusata da corners/cards, non modificata) al meta-model vincitore di ORACLE-02 (`benchmark_stacking_approaches`), stessi fold walk-forward gia' usati per selezionare weighted_blend/learned_stacker (nessun nuovo meccanismo di split); report pre/post metrics (log_loss/brier/ece) esposto in `OracleEnsembleCalibrationReport`; fallback esplicito e non bloccante — nessuna eccezione propagata al chiamante — quando il campione e' sotto `DEFAULT_MIN_CALIBRATION_SAMPLES` (60 righe) o la calibrazione fallisce per qualunque motivo: in quel caso il meta-model raw (non calibrato) di ORACLE-02 viene usato cosi' com'e', con il motivo riportato in `fallback_reason`; orchestratore `run_oracle_ensemble_calibration` che registra il modello finale (calibrato o raw fallback) come 'candidate' (mai 'production' automatica) (`src/ml/ensemble/oracle_calibration.py`)
- Model Consensus per spiegabilita' (ORACLE-04): per una singola fixture/mercato, assembla l'output di Direct Expert (EXP-05, modello registrato per quel mercato) e Market/Odds Expert (EXP-04, fair probability dalle quote gia' salvate in `odds_snapshot`) — GENERICAMENTE applicabili a QUALUNQUE mercato di `FilterMarketService.SUPPORTED_MARKETS`, a differenza di team_strength/goal_distribution/statistics volutamente esclusi (rispettivamente: nessun probability_vector per costruzione, non generico oltre i mercati "gol", nessuna persistenza su registry); l'Oracle finale usa il meta-model registrato (ORACLE-02/03, cercato ESPLICITAMENTE per prefisso `model_name` — mai un `get_latest(market)` generico che confonderebbe meta-model e modello diretto sullo stesso mercato) quando disponibile, altrimenti la media semplice delle probabilita' comparabili (fallback esplicito, mai un'eccezione); dispersione = deviazione standard tra le probabilita' comparabili, con etichetta di accordo (high/medium/low); nessuna spiegazione testuale generata, solo numeri/metadati calcolati (`src/ml/ensemble/model_consensus.py`); esposto tramite `GET /dashboard/match/{fixture_id}/consensus?market=...` (`src/api/main.py`, `src/api/schemas.py::ModelConsensusResponse`)
- **Fase ENSEMBLE completata (ORACLE-01..04)**
- Fair Odds Engine (BET-01): standardizza, per ciascun outcome di un mercato, `p_market_raw`/`p_market_fair` (overround rimosso, RIUSA `compute_market_baseline` di ML-04 senza duplicarlo) + `fair_odd = 1/p_market_fair` + `p_model` (Oracle, quando fornito dal chiamante — tipicamente da `model_consensus.py`, ORACLE-04) affiancati per lo stesso outcome ("Confronto Oracle vs market"); scope DELIBERATAMENTE limitato alle sole probabilita'/quote (NESSUN calcolo di `prob_edge`/`EV`/decision: quello e' BET-02, dipendenza tecnica esplicita, "REWRITE" di `DashboardService._value_decision`); outcome con `p_model` noto ma senza quota di mercato (o viceversa) restano comunque nell'output invece di sparire silenziosamente; nuovo package `src/oracle/` (primo modulo della Fase BETTING, distinto da `src/ml/`) (`src/oracle/fair_odds/fair_odds_engine.py`)
- Value Engine (BET-02, REWRITE): sostituisce il precedente calcolo che chiamava impropriamente "edge" un Expected Value; distingue esplicitamente `prob_edge = p_model - p_market_fair` (differenza di probabilita', indipendente dalla quota — invariante se la quota cambia a parita' di probabilita') da `ev = p_model*odd-1` (valore atteso in termini di quota, cio' che il codice precedente calcolava sotto il nome "edge"); soglie di decisione PLAY/BORDERLINE/NO BET VERSIONATE (`ValueDecisionPolicy`, mai hardcoded inline nel corpo della funzione, stessi valori numerici gia' in uso per non introdurre un cambio di policy non richiesto da questo task); consuma direttamente `FairOddsOutcome` (BET-01) garantendo per costruzione che market/outcome/quota si riferiscano allo stesso outcome; corretto in `DashboardService._pick_and_odd_for_prediction` il bug per cui il pick "Away" sul mercato h2h poteva usare la quota "Draw" come fallback (outcome scorretto: ora resta `None` se manca la quota "Away", gestito esplicitamente dal Value Engine) (`src/oracle/value_engine/value_engine.py`, wiring in `src/api/dashboard_service.py::_build_decision_cards`)
- Betting Backtester (BET-03): per un singolo mercato di `FilterMarketService.SUPPORTED_MARKETS`, genera `p_model` SOLO out-of-sample riusando lo STESSO walk-forward gia' validato altrove (ML-05: `expanding_window_splits` + `temporal_oof_probabilities`, nessuna nuova logica di split/leakage), lo abbina alle quote storiche REALI per lo stesso outcome (fair odds via `compute_market_baseline`/BET-01: per corners/cards viene usata ESCLUSIVAMENTE la linea che coincide con la soglia target di `FilterMarketService`, 9.5/4.5 — una linea non quotata per quella fixture resta esplicitamente NO BET invece di un fallback silenzioso su una quota di una linea diversa che distorcerebbe ROI/EV) e valuta ciascuna bet con il Value Engine (BET-02, non modificato); calcola, con stake flat, ROI/yield/profit, hit rate, avg odds, max drawdown (peak-to-trough sulla curva di equity cumulata) e la scomposizione per edge bucket/mercato/lega, con ordinamento cronologico deterministico che rende il report indipendente dall'ordine di input passato dal chiamante ("Report riproducibile"); parte "pura" testabile con numeri fissi e zero dipendenze DB/ML (`BacktestBet`/`compute_backtest_report`, `src/oracle/backtest/betting_backtester.py`) separata dall'orchestratore che collega dataset reale/OOF/fair-odds/value-engine (`run_market_backtest(_from_db)`, `src/oracle/backtest/market_backtest.py`)
- Decision Policy versionata (BET-04, REWRITE): estende `ValueDecisionPolicy` (BET-02, un'unica istanza GLOBALE identica per ogni mercato/outcome) con soglie che possono variare ESPLICITAMENTE per mercato e per singolo outcome di un mercato (`DecisionPolicy.thresholds_for`, priorita' outcome > mercato > default), piu' due nuovi filtri richiesti dall'acceptance criteria: `min_samples` (numero minimo di bookmaker che quotano l'outcome, da `FairOddsOutcome.bookmakers` gia' calcolato da BET-01/ML-04 — sotto soglia, NO BET indipendentemente da edge/EV) e `min_odd`/`max_odd` opzionali (range di quota accettabile, NO BET se fuori range); riusa DIRETTAMENTE `compute_prob_edge`/`compute_expected_value` (BET-02, mai duplicati); `DEFAULT_DECISION_POLICY` (`decision_policy_v1`) riproduce ESATTAMENTE lo stesso comportamento di `DEFAULT_POLICY` (BET-02) quando i nuovi filtri non sono vincolanti (`min_samples=0`, `min/max odd=None`) — nessun override per-mercato precaricato in produzione, la capacita' e' disponibile ma non attivata finche' una taratura specifica non sia esplicitamente richiesta (nessun cambio di policy non richiesto da questo task); `dashboard_service.py::_build_decision_cards` aggiornato per usare `evaluate_decision_from_fair_odds_outcome` (passando `bookmakers` come `samples`) al posto della chiamata diretta al Value Engine — nessuna soglia numerica hardcoded nel dashboard service (acceptance criteria) (`src/oracle/decision_engine/decision_policy.py`)
- Closing Line Value — CLV (BET-05, NEW): calcola quanto la quota presa al momento della decisione (`odd_at_bet`) e' stata migliore/peggiore della vera quota di CHIUSURA (`closing_odd`), metrica di valutazione EX-POST ("quando disponibile": SOLO per le fixture con `OddsSnapshot`/DATA-06 registrati fino al kickoff — mai per il backtest storico su `match.odds`, singolo valore senza serie temporale, per cui il CLV resta esplicitamente non disponibile); riusa `build_opening_latest_closing` (EXP-04) chiamato con `as_of=kickoff_at` per ottenere per costruzione l'ultimo snapshot PRIMA del kickoff per bookmaker/outcome (mai una quota "live" post-kickoff spacciata per chiusura) e `compute_market_baseline`/`get_market_outcome_baseline` (ML-04/BET-01) per la fair probability aggregata al closing; espone due metriche distinte — `clv_odd_pct = odd_at_bet/closing_odd - 1` (quota) e `clv_prob = p_fair_at_bet - p_fair_closing` (probabilita' fair) — mai calcolate come feature di training (acceptance criteria "Nessun uso del closing price come feature pre-match illegittima": modulo puramente descrittivo/ex-post); nessuna migration DB necessaria (schema `odds_snapshot` gia' introdotto da DATA-06); report aggregato per mercato/modello (`compute_clv_report`) con conteggio esplicito disponibili/non disponibili, mai nascosto (`src/oracle/backtest/clv.py`)
- **Fix critico CrudRepository.search_filter** (scoperto durante un retrain manuale multi-mercato che falliva SEMPRE, per qualunque mercato, indipendentemente dai dati presenti — `retrain_run.log`): (1) `col.is_not(None)`/`is_(None)` su relazioni ORM one-to-many (`Match.odds`/`Match.statistics`) solleva `NotImplementedError` in SQLAlchemy — fix con `_is_relationship_attribute` che usa `.any()`/`~.any()` per le relazioni, invariato per le colonne scalari; (2) `Match.mean_statistics` (colonna JSON) salvava un Python `None` come letterale JSON `'null'` anziche' SQL `NULL` (default SQLAlchemy `JSON(none_as_null=False)`), rendendo il filtro `mean_statistics: not None` un NO-OP silenzioso su TUTTA la pipeline multi-mercato (`filter_market_service.py`, `point_in_time_builder.py`, `corners_market.py`, `totals_market.py`, `cards_market.py`, `market_backtest.py`) — fix con `JSON(none_as_null=True)` (`src/service_ia/model/match.py`); verificato via query diretta sul DB locale che non ci sono righe storiche da bonificare con una migration di data-fix (0 righe con `'null'` letterale su 3 totali). Test con vero SQLite in-memory (non mock, per esercitare davvero la query SQLAlchemy generata) in `tests/service/crud_repository_test.py`; suite completa rieseguita dopo il fix: 395 passed, 0 failed (`src/repository/base/crud_repository.py`, `src/service_ia/model/match.py`)
- Prediction Ledger / Paper Betting (BET-06): nuova tabella `prediction_ledger` (migration `f3a9c1d8e2b7`, validata upgrade/downgrade/upgrade sul DB reale) che separa ESPLICITAMENTE i campi "originali" scritti UNA SOLA VOLTA prima del kickoff (fixture_id/market/outcome/model_run_id/model_name/policy_version/p_model/p_market_fair/odd/fair_odd/prob_edge/ev/decision/stake/kickoff_at/created_at) dai campi di SETTLEMENT scritti SOLO dopo il risultato reale (is_settled/settled_at/settlement_status/actual_outcome/won/pnl) — "immutabilita' logica della prediction originale" garantita dalla service layer (`settle_prediction_record` usa `dataclasses.replace`, mai una mutazione in-place) e verificata da un test dedicato che confronta il record pre/post settlement; modulo puro (`src/oracle/ledger/prediction_ledger.py`, nessun DB, stesso stile BET-01/02/03) con `build_prediction_record` (riusa DIRETTAMENTE `Decision`/BET-04: market/outcome/p_model/p_market_fair/odd/prob_edge/ev/decision/policy_version provengono TUTTI dalla stessa valutazione, garanzia gia' vista in BET-01/02/04 che si riferiscano allo stesso outcome), `resolve_actual_outcome` (esito REALE riusando `FilterMarketService._label_by_market` + `_canonical_outcome_for_prediction`/BET-03, mai una nuova logica di labeling) e `compute_paper_pnl` (stessa formula di `bet_profit`/BET-03, duplicata deliberatamente per mantenere il ledger indipendente dal backtest storico, coerenza numerica garantita da test); orchestrazione DB in `src/oracle/ledger/ledger_service.py` (`PredictionLedgerService`): `log_prediction` idempotente per (fixture/market/outcome/model_run_id), `settle_pending`/`settle_prediction` settlano SOLO le prediction il cui match e' effettivamente concluso (`FINAL_STATUSES`, riusato da `settlement_service.py`) — le altre restano pending, mai forzate; nessun esito inventato quando le statistiche non sono disponibili (`settlement_status='void_no_result'`) o quando manca la quota (`'void_missing_odd'`); report PnL doppio: `raw_pnl_summary` (somma diretta dei `pnl` gia' salvati, sempre affidabile) e `paper_pnl_report` (RIUSA `compute_backtest_report`/BET-03 costruendo `BacktestBet` dalle righe settled, per ROI/hit-rate/max drawdown/edge bucket); endpoint `POST/GET /predictions/ledger`, `GET /predictions/ledger/fixture/{fixture_id}`, `POST /predictions/ledger/settle`, `GET /predictions/ledger/pnl` (`src/api/main.py`, `src/api/schemas.py`); repository dedicato stesso pattern di `OddsSnapshotRepository` (`src/repository/prediction_ledger_repository.py`); 26 nuovi test (pure + service con vero SQLite in-memory, stesso principio di `crud_repository_test.py`) in `tests/service/prediction_ledger_test.py`; `src/service_ia/training/prediction_logger.py` (JSONL) NON modificato/rimosso (compatibilita', endpoint `/predict/{market}` e `/predictions/log` invariati) — il Ledger e' un componente NUOVO e distinto, non ancora agganciato all'endpoint `/predict/{market}` (nessuna riscrittura non richiesta di un endpoint funzionante: il chiamante che ha gia' calcolato p_model/odd puo' loggare esplicitamente via `POST /predictions/ledger`)
- **Fase BETTING completata (BET-01..06)**



## Migrazioni applicate (locale + docker)
- locale: `alembic stamp 55bbb5f0a367` + `alembic upgrade head`
- docker: `docker compose exec api alembic upgrade head`

## Connessione DB locale
- runtime locale fissato su DB reale: `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/match_db`
- verifica rapida: tabella `match` letta con volume storico (>47k righe)

**ATTENZIONE (2026-09-03):** il container Docker `soccer_db` di questo progetto espone attualmente la porta `5433` (non `5432`, occupata su questa macchina da un progetto Docker diverso — `fantappero-postgres` — e/o da un Postgres nativo Windows con messaggi in lingua italiana), e il volume `ia_predict_soccer_postgres_data` risulta creato il 2026-09-01 con solo 3 righe nella tabella `match` — NON le >47k righe storiche annotate sopra. Nessuna azione distruttiva e' stata effettuata sul DB/volume in questa sessione. La migration `f3a9c1d8e2b7` (BET-06, `prediction_ledger`) e' stata comunque validata con successo (upgrade/downgrade/upgrade) eseguendo `alembic` DENTRO il container `soccer_api` (dove `DATABASE_URL` punta correttamente a `db:5432` sulla rete Docker interna, bypassando il conflitto di porta sull'host). Da verificare con l'utente se i dati storici risiedono altrove (altro host/volume/backup) prima di considerare il dataset locale attendibile per un retrain reale.

Note: gli stati sopra sono riferiti all'implementazione tecnica nel branch corrente; la validazione finale dipende dall'esecuzione acceptance/test su ambiente dati reale.








































