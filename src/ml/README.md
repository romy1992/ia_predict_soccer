# ML Runtime (`src/ml`)

Questo package contiene componenti production-oriented introdotti in Soccer Oracle V2.

## Moduli
- `datasets/point_in_time_builder.py`: dataset builder point-in-time con controllo anti-leakage.
- `validation/temporal_split.py`: split expanding/rolling/final holdout senza shuffle.
- `baselines/bookmaker_baseline.py`: implied/fair probabilities con rimozione overround.
- `evaluation/probability_metrics.py`: LogLoss, Brier, ECE, reliability e report grouped (binario).
- `evaluation/multiclass_probability_metrics.py`: equivalente multiclasse (log-loss nativo, Brier generalizzato, ECE su confidence, AUC OvR) usato dal mercato 1X2.
- `calibration/calibration_service.py`: calibrazione sigmoid/isotonic con confronto pre/post (binario).
- `calibration/multiclass_calibration_service.py`: calibrazione multiclasse via `CalibratedClassifierCV`, stesso principio pre/post su OOF temporali.
- `experts/`: Team Strength, Goal Distribution, Statistics, Market/Odds, Direct Market (EXP-01..05).
- `markets/market_1x2.py`: vero mercato 1X2 multiclass (HOME/DRAW/AWAY), nessun mapping draw->away (MARKET-01).
- `markets/market_double_chance.py`: Double Chance derivata da 1X2 coerente (nessun training proprio, solo derivazione aritmetica) (MARKET-02).
- `markets/btts/btts_market.py`: BTTS consolidato, benchmark score_distribution (EXP-02) vs direct_expert 'goal_no_goal' (EXP-05) vs ensemble, selezione via `champion_probability_score` e calibrazione OOF temporale del solo vincitore (MARKET-03).
- `markets/totals/totals_market.py`: U/O 1.5-4.5 multi-linea, confronto binary_independent vs hierarchical (bin ordinali su gol totali) vs goal_distribution (Poisson EXP-02) sullo stesso walk-forward, monotonicità P(O1.5)>=...>=P(O4.5) obbligatoria via proiezione isotonica (MARKET-04).
- `markets/corners/corners_market.py`: Corners O/U con linea configurabile (parametro, non più soglia fissa), feature dedicate corner (media/differenziale da `mean_statistics`) e calibrazione via `CalibrationService` (ML-06) per ciascuna linea (MARKET-05).
- `markets/cards/cards_market.py`: Cards O/U con linea configurabile, feature "team/style" (mean_statistics) + feature ARBITRO nuove point-in-time (`build_referee_features_dataset`, media storica cartellini per arbitro, nessun leakage) e calibrazione per ciascuna linea (MARKET-06).
- `ensemble/expert_output.py`: schema di output comune (`ExpertOutput`) per tutti gli esperti — probability vector, model_run_id, feature timestamp, confidence/metadata — e `combine_expert_outputs` per assemblare la riga di feature per un futuro meta-model (ORACLE-01).
- `ensemble/adapters.py`: adapter (zero modifiche agli esperti esistenti) da output nativo di EXP-01..05 e dei market expert (corners/cards) verso `ExpertOutput` (ORACLE-01).
- `ensemble/stacking.py`: Meta Model / Stacker per mercato, confronto `weighted_blend` (`WeightedBlendClassifier`) vs `learned_stacker` (`LogisticRegression`) sulle meta-feature ORACLE-01, entrambi validati con lo stesso OOF walk-forward (ORACLE-02).
- `ensemble/oracle_calibration.py`: calibrazione finale (Platt/isotonic via `CalibrationService`, ML-06) del meta-model vincitore di ORACLE-02, con report pre/post metrics e fallback esplicito (meta-model raw) quando il campione è insufficiente o la calibrazione fallisce (ORACLE-03).
- `ensemble/model_consensus.py`: Model Consensus per spiegabilità — output per-fixture di Direct Expert (EXP-05) e Market/Odds Expert (EXP-04), Oracle finale (meta-model ORACLE-02/03 se registrato, altrimenti media semplice come fallback esplicito) e dispersione tra esperti; esposto via `GET /dashboard/match/{fixture_id}/consensus` (ORACLE-04).
- `live/live_feature_store.py`: feature store LIVE point-in-time (minute, scoreline, cards da eventi, shots/xG se disponibili dalle statistiche grezze, pre-match prior dal Prediction Ledger) per fixture/istante `as_of`, costruito sul dataset live distinto di LIVE-01 — ogni feature timestamped, mai leakage rispetto ad `as_of` (LIVE-02).
- `live/live_match_outcome_dataset.py`: dataset multi-riga-per-fixture (una riga per ogni snapshot di ogni fixture GIA' conclusa, `LiveFeatureStore.build_feature_history`) con target 1X2 (`label_1x2`, risultato reale finale) costante per fixture — base per l'addestramento di modelli LIVE (LIVE-03).
- `live/live_match_outcome_model.py`: addestramento/validazione modelli LIVE 1X2 (`market='1x2_live'`, pipeline SEMPRE separata da quella pre-match `market_1x2.py`) — split `match_level_temporal_splits` (mai riga-per-riga: ogni fixture resta interamente su un solo lato train/valid), confronto logistic/random_forest sullo stesso OOF walk-forward, report metriche per finestra di minuto (`minute_window_report`) (LIVE-03).

## Esecuzione rapida
```powershell
python -m src.ml.datasets.build_dataset_runner --market under_over_2_5 --seasons 2025,2026 --save-snapshot
```

Output:
- summary JSON su stdout
- opzionale snapshot CSV+metadata in `best_models/datasets/`










