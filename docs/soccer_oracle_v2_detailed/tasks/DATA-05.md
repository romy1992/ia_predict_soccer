# DATA-05 — Creare settlement/finalizzazione delle partite concluse.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-03

## Obiettivo
Creare settlement/finalizzazione delle partite concluse.

## Cosa deve fare
1. Rilevare match terminati.
2. Aggiornare score/statistiche finali.
3. Marcare dati completi/incompleti.
4. Preparare hook per settlement prediction.

## File / aree da ispezionare
- `src/jobs/`
- `src/service_ia/model/`

## Acceptance criteria
- [ ] Partite finali riconciliate.
- [ ] Stato completezza visibile.
- [ ] Job idempotente.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-05
Fase: DATA PLATFORM
Stato: NEW
Priorità: P1
Dipendenze: DATA-03

Obiettivo:
Creare settlement/finalizzazione delle partite concluse.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Rilevare match terminati.
- Aggiornare score/statistiche finali.
- Marcare dati completi/incompleti.
- Preparare hook per settlement prediction.

File/aree da ispezionare:
- src/jobs/
- src/service_ia/model/

Acceptance criteria:
- Partite finali riconciliate.
- Stato completezza visibile.
- Job idempotente.

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
