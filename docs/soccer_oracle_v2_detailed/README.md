# Soccer Oracle V2 — Pacchetto operativo dettagliato

Questo pacchetto è pensato per essere inserito direttamente nel repository `romy1992/ia_predict_soccer`.

Base consigliata:
- branch di partenza: `feature/ml-dashboard-platform`
- nuovo branch di lavoro: `feature/soccer-oracle-v2`

## Come usarlo
1. Leggi `00_PROJECT_OVERVIEW.md`.
2. Leggi `01_CURRENT_STATE_AUDIT.md`.
3. Usa `ROADMAP_OPERATIVA.md` come sequenza principale.
4. Apri `CURRENT_TASK.md` per sapere qual è il prossimo task.
5. Copia il prompt del task e passalo a Codex / Claude / Cursor.
6. Dopo il completamento, valida gli acceptance criteria prima di passare al task successivo.

## Regola principale
Non saltare direttamente alla Schedina Oracle o al Live ML. Prima servono:
- Data Platform affidabile
- DB canonico
- dataset point-in-time
- validazione temporale
- calibrazione
- backtesting
- model promotion
- paper betting
