# DATA-06 — Introdurre odds snapshot normalizzati nel tempo.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-01, SOCCER-02

## Obiettivo
Introdurre odds snapshot normalizzati nel tempo.

## Cosa deve fare
1. Creare schema `odds_snapshot`.
2. Campi: fixture, bookmaker, market, period, line, outcome, odd, captured_at, source.
3. Migration Alembic.
4. Non eliminare subito JSON odds legacy.
5. Backfill dove possibile.

## File / aree da ispezionare
- `alembic/`
- `src/service_ia/model/`
- `src/repository/`

## Acceptance criteria
- [ ] Snapshot multipli per fixture supportati.
- [ ] Timestamp obbligatorio.
- [ ] Query per opening/latest/closing possibili.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-06
Fase: DATA PLATFORM
Stato: NEW
Priorità: P0
Dipendenze: DATA-01, SOCCER-02

Obiettivo:
Introdurre odds snapshot normalizzati nel tempo.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Creare schema `odds_snapshot`.
- Campi: fixture, bookmaker, market, period, line, outcome, odd, captured_at, source.
- Migration Alembic.
- Non eliminare subito JSON odds legacy.
- Backfill dove possibile.

File/aree da ispezionare:
- alembic/
- src/service_ia/model/
- src/repository/

Acceptance criteria:
- Snapshot multipli per fixture supportati.
- Timestamp obbligatorio.
- Query per opening/latest/closing possibili.

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
