# DATA-07 — Creare Data Quality report.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-02, DATA-03, DATA-04

## Obiettivo
Creare Data Quality report.

## Cosa deve fare
1. Coverage fixture/statistics/odds per mercato.
2. Null/duplicati/orfani.
3. Distribuzione per lega/stagione.
4. Controlli temporali.
5. Endpoint/servizio per dashboard.

## File / aree da ispezionare
- `src/data/`
- `src/api/`
- `tests/`

## Acceptance criteria
- [ ] Report riproducibile.
- [ ] Anomalie conteggiate.
- [ ] Output machine-readable.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-07
Fase: DATA PLATFORM
Stato: NEW
Priorità: P1
Dipendenze: DATA-02, DATA-03, DATA-04

Obiettivo:
Creare Data Quality report.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Coverage fixture/statistics/odds per mercato.
- Null/duplicati/orfani.
- Distribuzione per lega/stagione.
- Controlli temporali.
- Endpoint/servizio per dashboard.

File/aree da ispezionare:
- src/data/
- src/api/
- tests/

Acceptance criteria:
- Report riproducibile.
- Anomalie conteggiate.
- Output machine-readable.

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
