# DATA-04 — Creare import delle partite future per una finestra configurabile.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-02

## Obiettivo
Creare import delle partite future per una finestra configurabile.

## Cosa deve fare
1. Supportare `days_ahead`.
2. Importare fixture NS.
3. Salvare quote pre-match disponibili.
4. Calcolare/aggiornare feature pre-match senza usare dati futuri.

## File / aree da ispezionare
- `src/jobs/`
- `src/service_ia/pre_processing/`

## Acceptance criteria
- [ ] Finestra futura parametrica.
- [ ] Nessun target/result leakage.
- [ ] Riesecuzione aggiorna senza duplicare.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-04
Fase: DATA PLATFORM
Stato: NEW
Priorità: P0
Dipendenze: DATA-02

Obiettivo:
Creare import delle partite future per una finestra configurabile.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Supportare `days_ahead`.
- Importare fixture NS.
- Salvare quote pre-match disponibili.
- Calcolare/aggiornare feature pre-match senza usare dati futuri.

File/aree da ispezionare:
- src/jobs/
- src/service_ia/pre_processing/

Acceptance criteria:
- Finestra futura parametrica.
- Nessun target/result leakage.
- Riesecuzione aggiorna senza duplicare.

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
