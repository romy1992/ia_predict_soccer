# Fix: goal_no_goal mancante nel worktree reale (Docker-mounted) — 2026-09-16

## Problema
La sessione precedente aveva addestrato/promosso `goal_no_goal` (run_id
`goal_no_goal_20260916T125326272848Z`) nel worktree `ia_predict_soccer_export`,
non in `ia_predict_soccer` (quello montato da Docker via `./best_models:/app/best_models`,
dove i container leggono il registry reale). Il modello non era visibile
all'operatore perché non era mai arrivato in quel worktree.

## Interventi

1. **File modello**: copiati (non spostati) da `ia_predict_soccer_export/best_models/goal_no_goal/`
   a `ia_predict_soccer/best_models/goal_no_goal/` (cartella creata, non esisteva):
   - `goal_no_goal_champion_20260916.pkl`
   - `goal_no_goal_champion_20260916_calibrator.pkl`
   - Copiato anche il metadata JSON `registry/goal_no_goal_20260916T125326272848Z.json`,
     con `metadata_path` corretto per puntare al worktree reale (era ancora impostato
     sul path dell'export).

2. **Registry reale** (`best_models/registry/`): il run_id NON era presente.
   Backup creati prima della scrittura (`index.jsonl.bak_20260916T150827`,
   `promotion_history.jsonl.bak_20260916T150827`), poi append (mai sovrascrittura):
   - `index.jsonl`: riga del run `goal_no_goal_20260916T125326272848Z`, con
     `model_path` già in forma container (`/app/best_models/goal_no_goal/...`).
   - `promotion_history.jsonl`: 2 eventi copiati dall'export, necessari per
     coerenza dello stage (`ModelRegistry.promote()` genera sempre la coppia
     supersede+promote in un'unica transazione):
     - `goal_no_goal_20260908T150042264590Z`: production → champion (superseded by 20260916)
     - `goal_no_goal_20260916T125326272848Z`: candidate → production

   Nota: il registry reale aveva altri run `goal_no_goal` (20260913, 20260914) mai
   promossi — restano `candidate`, invariati. La production precedente era
   `goal_no_goal_20260908T150042264590Z` (unica promozione trovata nello storico
   reale), correttamente retrocessa a `champion` dai due eventi sopra.

## Verifiche
- `ModelRegistry().get_production(market="goal_no_goal")` nel worktree reale:
  `run_id=goal_no_goal_20260916T125326272848Z`, `current_stage=production`,
  `model_path=/app/best_models/goal_no_goal/goal_no_goal_champion_20260916.pkl` — OK
- Modello e calibratore caricabili con `joblib.load` (`CalibratedClassifierCV`) — OK

## Container
- `docker compose restart api scheduler` eseguito nel worktree reale.
- `soccer_api`: Up, `healthy`. `soccer_scheduler`: Up.

## Note
- `best_models/` non è tracciato in git: nessun file di modello o di registry
  committato, solo questo report.
