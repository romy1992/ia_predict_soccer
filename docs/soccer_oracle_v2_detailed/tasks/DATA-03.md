# DATA-03 — Creare il job/endpoint 'Aggiorna oggi'.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-02

## Obiettivo
Creare il job/endpoint 'Aggiorna oggi'.

## Cosa deve fare
1. Sincronizzare fixture odierne.
2. Aggiornare stato, score, statistiche e quote quando disponibili.
3. Persistenza idempotente.
4. Esporre stato job.

## File / aree da ispezionare
- `src/jobs/`
- `src/api/`
- `src/service_ia/pre_processing/`

## Acceptance criteria
- [ ] Endpoint/job manuale funzionante.
- [ ] Statistiche job registrate.
- [ ] Errori per fixture non bloccano tutto il batch.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-03
Fase: DATA PLATFORM
Stato: NEW
Priorità: P0
Dipendenze: DATA-02

Obiettivo:
Creare il job/endpoint 'Aggiorna oggi'.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Sincronizzare fixture odierne.
- Aggiornare stato, score, statistiche e quote quando disponibili.
- Persistenza idempotente.
- Esporre stato job.

File/aree da ispezionare:
- src/jobs/
- src/api/
- src/service_ia/pre_processing/

Acceptance criteria:
- Endpoint/job manuale funzionante.
- Statistiche job registrate.
- Errori per fixture non bloccano tutto il batch.

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
