# Cards — Step A/B training (2026-09-19)

Dataset da DB reale via `FilterMarketService`, features solo-quota (6 colonne, stesso pattern goal/no-goal), righe filtrate su `odds_count_line_X` non nullo. CV temporale (`_build_temporal_cv`).

## Step A — Curva soglia/ROI (calibrazione isotonic, OOF)

Nota: ROI calcolato sulla quota media (`odds_mean_over/under`), non sulla massima. Soglie con n<30 omesse dalla tabella (riportate come "poche" nei log).

### cards_line_3_5 (n=2706, ECE 0.042→0.041)

| Dir | Soglia | n | Precisione | Quota media | ROI |
|---|---|---|---|---|---|
| OVER | 0.55 | 723 | 62.0% | 1.462 | -10.5% |
| OVER | 0.60 | 550 | 65.1% | 1.419 | -8.5% |
| OVER | 0.65 | 379 | 68.1% | 1.371 | -7.1% |
| OVER | 0.70 | 176 | 71.0% | 1.319 | -6.3% |
| OVER | 0.75 | 34 | 67.6% | 1.247 | -15.5% |
| UNDER | 0.45 | 201 | 68.7% | 1.685 | **+14.4%** |
| UNDER | 0.40 | 139 | 71.2% | 1.642 | **+16.2%** |
| UNDER | 0.35 | 47 | 76.6% | 1.586 | +20.3% (n<50) |

### cards_line_4_5 (n=3141, ECE 0.108→0.035)

| Dir | Soglia | n | Precisione | Quota media | ROI |
|---|---|---|---|---|---|
| OVER | 0.55 | 134 | 53.7% | 1.517 | -18.9% |
| OVER | 0.60 | 49 | 46.9% | 1.430 | -34.1% (n<50) |
| UNDER | 0.45 | 883 | 70.9% | 1.521 | **+6.2%** |
| UNDER | 0.40 | 625 | 75.5% | 1.459 | **+9.1%** |
| UNDER | 0.35 | 416 | 78.6% | 1.406 | **+9.8%** |
| UNDER | 0.30 | 303 | 83.2% | 1.365 | **+13.3%** |
| UNDER | 0.25 | 165 | 87.9% | 1.322 | **+16.2%** |
| UNDER | 0.20 | 107 | 86.0% | 1.293 | **+11.0%** |

### cards_line_5_5 (n=2843, ECE 0.245→0.025)

| Dir | Soglia | n | Precisione | Quota media | ROI |
|---|---|---|---|---|---|
| OVER | 0.55 | 37 | 48.6% | 1.655 | -20.3% (n<50) |
| UNDER | 0.45 | 1327 | 76.2% | 1.382 | **+3.4%** |
| UNDER | 0.40 | 1280 | 77.2% | 1.369 | **+4.1%** |
| UNDER | 0.35 | 1144 | 79.2% | 1.331 | **+4.3%** |
| UNDER | 0.30 | 922 | 81.6% | 1.289 | **+4.4%** |
| UNDER | 0.25 | 712 | 83.4% | 1.258 | **+4.3%** |
| UNDER | 0.20 | 357 | 89.4% | 1.204 | **+7.4%** |

### cards_line_6_5 (n=2106, ECE 0.314→0.014)

| Dir | Soglia | n | Precisione | Quota media | ROI |
|---|---|---|---|---|---|
| OVER | tutte | ≤9 | — | — | poche (n<30, nessun dato utilizzabile) |
| UNDER | 0.45 | 1040 | 85.5% | 1.210 | **+2.3%** |
| UNDER | 0.40 | 1031 | 85.6% | 1.206 | **+2.2%** |
| UNDER | 0.35 | 1012 | 86.0% | 1.200 | **+2.2%** |
| UNDER | 0.30 | 967 | 86.9% | 1.187 | **+2.3%** |
| UNDER | 0.25 | 857 | 88.9% | 1.164 | **+3.1%** |
| UNDER | 0.20 | 773 | 90.4% | 1.151 | **+3.9%** |

**Decisione:** in tutte e 4 le linee la direzione OVER non supera mai il criterio (ROI negativo o n<50 a tutte le soglie). La direzione UNDER supera il criterio (ROI positivo, n≥50) su tutte e 4 le linee, con margini via via più larghi su 3_5 e più stretti su 6_5. Il modello (`train_market`) allena un unico classificatore per mercato (probabilità "over"), quindi si procede con lo Step B su tutte e 4 le linee — l'edge reale in produzione andrà sfruttato filtrando le predizioni lato UNDER (probabilità bassa) sopra soglia, non lato OVER.

## Step B — Training pipeline completa (grid search + ensemble, `save_model=True`)

Nota tecnica: la prima esecuzione con i 4 training in parallelo ha saturato la memoria di sistema (OpenBLAS/joblib, file di paging insufficiente su Windows). Rieseguiti in sequenza con `LOKY_MAX_CPU_COUNT=2`/`OMP_NUM_THREADS=2`/`OPENBLAS_NUM_THREADS=2`/`MKL_NUM_THREADS=2`, senza toccare il codice (nessuna modifica agli `n_jobs=-1` nella pipeline).

| Mercato | Champion | AUC (post-cal) | ECE pre→post | File salvato |
|---|---|---|---|---|
| cards_line_3_5 | stacking | 0.650 | 0.044 → 0.042 | `best_models/cards/cards_line_3_5/cards_line_3_5_champion.pkl` |
| cards_line_4_5 | logistic | 0.652 | 0.108 → 0.036 | `best_models/cards/cards_line_4_5/cards_line_4_5_champion.pkl` |
| cards_line_5_5 | random_forest_smote | 0.622 | 0.173 → 0.018 | `best_models/cards/cards_line_5_5/cards_line_5_5_champion.pkl` |
| cards_line_6_5 | random_forest_smote | 0.674 | 0.189 → 0.016 | `best_models/cards/cards_line_6_5/cards_line_6_5_champion.pkl` |

Tutti e 4 i modelli salvati come **candidati** nel registry (`best_models/registry/cards_line_X_*.json`), sotto `best_models/cards/<mercato>/` come da fix `destination_subdir()`. Calibratore isotonic salvato accanto a ciascun champion (`*_champion_calibrator.pkl`).

`promote_with_policy()` **non è stato chiamato** per nessuna linea — nessun modello è stato promosso a production.
