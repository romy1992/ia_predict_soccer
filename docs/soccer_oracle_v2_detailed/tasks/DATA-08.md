# DATA-08 — Evolvere Job History per import, settlement, training e backtest.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** DATA-03

## Obiettivo
Evolvere Job History per import, settlement, training e backtest.

## Cosa deve fare
1. Stato queued/running/success/failed.
2. started_at/finished_at/duration.
3. Parametri input.
4. summary risultati.
5. error payload.

## File / aree da ispezionare
- `src/jobs/job_history.py`
- `src/api/`

## Acceptance criteria
- [ ] Ogni job importante tracciabile.
- [ ] Storico filtrabile per tipo/stato.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-08
Fase: DATA PLATFORM
Stato: REFACTOR
Priorità: P1
Dipendenze: DATA-03

Obiettivo:
Evolvere Job History per import, settlement, training e backtest.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Stato queued/running/success/failed.
- started_at/finished_at/duration.
- Parametri input.
- summary risultati.
- error payload.

File/aree da ispezionare:
- src/jobs/job_history.py
- src/api/

Acceptance criteria:
- Ogni job importante tracciabile.
- Storico filtrabile per tipo/stato.

Vincoli generali:
- niente leakage temporale;
- niente split random per pipeline ML production;
- niente logica scientifica nel frontend;
- latest model non equivale automaticamente a production model;
- aggiungi/aggiorna test pertinenti;
- non anticipare task successivi salvo dipendenze strettamente tecniche.

A fine lavoro restituisci:
1. file modificati;
2. sintesi implementazione;
3. test eseguiti e risultato;
4. eventuali rischi/TODO;
5. verifica uno per uno degli acceptance criteria.
```
