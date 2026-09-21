# Report: ritiro modelli mercato corner (2026-09-21)

## Contesto

I 4 modelli `corners_line_8_5/9_5/10_5/11_5` erano in `production` dal 2026-09-12,
addestrati su 73 feature (incluse le quote legacy aggregate) su dati che precedono il
fix del 16/09 della contaminazione bookmaker (BetMGM/BetRivers/Bovada con quote
placeholder costanti su ~60% delle partite). AUC registrate 0.51-0.54.

Il 2026-09-21 e' stata condotta una verifica ROI con IC bootstrap 95% sui dati puliti
(post-esclusione bookmaker, righe filtrate su quota reale, quote-only per esito, logistic
+ random forest, tutte e 4 le linee, entrambe le direzioni): zero soglie con edge robusto,
e molte robustamente negative (ROI fra -3% e -9%, cioe' il margine del bookmaker; es. 8.5
OVER>=0.55 ROI -7.7% IC [-10.3%, -4.9%]). Dettaglio completo in
`docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md` §3-bis.

Decisione esplicita dell'operatore: chiudere il mercato corner e togliere quei modelli
dalla produzione.

## Merge

Fast-forward di `origin/feature/soccer-oracle-v2` su `main` (nessun conflitto), portando
tra gli altri i commit `4bd2c45` (report prototipo Poisson/Dixon-Coles) e `fdd99c6`
(chiusura mercato corner, verifica ROI). Incluso anche l'aggiornamento del commento in
`src/oracle/decision_engine/line_market_signal_policy.py` che documenta la chiusura (nessuna
modifica funzionale alle soglie, restano come riferimento storico).

## Esito ritiro modelli (via `ModelRegistry.promote`, stage `retired`, `actor="operator_request"`)

| Mercato | run_id | Esito |
|---|---|---|
| corners_line_8_5 | corners_line_8_5_20260912T194952985622Z | retired |
| corners_line_9_5 | corners_line_9_5_20260912T194953056739Z | retired |
| corners_line_10_5 | corners_line_10_5_20260912T194953282007Z | retired |
| corners_line_11_5 | corners_line_11_5_20260912T194953535594Z | retired |

Nessun file `.pkl` rimosso, nessuna riga di registry sovrascritta: transizione tracciata
come evento in `best_models/registry/promotion_history.jsonl` (append-only, reason completo
riportato per ciascun evento).

## Verifica post-ritiro

`ModelRegistry().get_production(market=...)` ritorna `None` per tutte e 4 le linee
(verificato subito dopo il ritiro).

## Container Docker

**Non riavviati.** `docker compose restart api scheduler` e' stato bloccato dal classificatore
di Claude Code auto mode (azione su infrastruttura condivisa). I container `soccer_api`,
`soccer_web`, `soccer_scheduler` risultavano gia' `Exited` da ~19h prima di questo intervento
(non collegato al ritiro dei modelli). I log dell'ultimo run di `soccer_api` mostrano uno
shutdown pulito (nessun errore); i log di `soccer_scheduler` non mostrano errori legati ai
mercati corner. Non e' stato quindi possibile verificare "a caldo" dentro un container in
esecuzione che l'assenza di production per le 4 linee non causi errori nel serving. **Azione
richiesta all'utente**: avviare/riavviare manualmente i container e controllare `/health` e i
log del servizio corner.

## Note

- Lavorato esclusivamente nel worktree `ia_predict_soccer`.
- Nessun file cancellato o registry sovrascritto.
