# Promozione goal_no_goal (rifacimento solo-quote) — 2026-09-16

## Comando
```
python scripts/analysis/promuovi_goal_no_goal.py
```
Eseguito su branch `feature/soccer-oracle-v2` (worktree `ia_predict_soccer_export`), dopo `git pull` (ce10980 -> 58e84c4).

## Dataset
- CSV: `scripts/analysis/_export/goal_no_goal_raw.csv`
- Righe dopo filtro NaN sulle 6 feature di quota e scarto sospetti (overround < 1.0 o rapporto max/mean quota > 3.0): **1.013** righe
- Base rate: 0.5311
- Feature (solo quote): `prob_norm_goal`, `odds_mean_goal`, `odds_mean_no_goal`, `odds_count`, `odds_std_goal`, `overround`

## Training
- Champion selezionato: **voting** (base models: random_forest_smote, logistic)
- Selection score: **0.6767**
- Calibrazione: isotonic (ECE pre 0.0393 -> post 0.0252)

## Gate / comparison
```
esito promozione: allowed=True
gate passed: True
comparison: method=selection_score, candidate_score=0.6767073692985085,
production_score=0.6678008913849377, delta=+0.0089068,
reason="candidate >= production (oltre il margine richiesto dalla policy)"
```
**Promozione a production eseguita.**

## File salvati
- `best_models/goal_no_goal/goal_no_goal_champion_20260916.pkl`
- `best_models/goal_no_goal/goal_no_goal_champion_20260916_calibrator.pkl`
- Registry riscritto: `model_path -> /app/best_models/goal_no_goal/goal_no_goal_champion_20260916.pkl`

## Verifiche post-promozione
- File modello presente su disco (4.2 MB) e caricabile con `joblib.load` (CalibratedClassifierCV) — OK
- File calibratore presente e caricabile — OK
- `ModelRegistry().get_production(market="goal_no_goal")`:
  - `run_id`: `goal_no_goal_20260916T125326272848Z`
  - `model_path`: `/app/best_models/goal_no_goal/goal_no_goal_champion_20260916.pkl`
  - `current_stage`: `production`
  - Confermato path container corretto sotto `best_models/goal_no_goal/`

## Container
- `soccer_api` e `soccer_scheduler` riavviati — entrambi `Up` e `soccer_api` in stato `healthy` dopo il riavvio.

## Note
- `best_models/` non e' tracciato in git: nessun file di modello committato, solo questo report.
