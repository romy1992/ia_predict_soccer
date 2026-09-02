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

## Esecuzione rapida
```powershell
python -m src.ml.datasets.build_dataset_runner --market under_over_2_5 --seasons 2025,2026 --save-snapshot
```

Output:
- summary JSON su stdout
- opzionale snapshot CSV+metadata in `best_models/datasets/`





