# OPS-01 — Separare scheduler dati da scheduler ML.

## Metadata
- **Fase:** OPERATIONS
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** DATA-03, DATA-04, DATA-05

## Obiettivo
Separare scheduler dati da scheduler ML.

## Cosa deve fare
1. Data jobs frequenti.
2. Training job indipendente.
3. No retrain automatico ad ogni import.

## File / aree da ispezionare
- `src/jobs/scheduler.py`

## Acceptance criteria
- [ ] Scheduler leggibile e configurabile.
- [ ] max_instances/coalesce corretti.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: OPS-01
Fase: OPERATIONS
Stato: REFACTOR
Priorità: P1
Dipendenze: DATA-03, DATA-04, DATA-05

Obiettivo:
Separare scheduler dati da scheduler ML.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Data jobs frequenti.
- Training job indipendente.
- No retrain automatico ad ogni import.

File/aree da ispezionare:
- src/jobs/scheduler.py

Acceptance criteria:
- Scheduler leggibile e configurabile.
- max_instances/coalesce corretti.

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
