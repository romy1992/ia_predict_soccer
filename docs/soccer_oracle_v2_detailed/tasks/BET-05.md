# BET-05 — Calcolare CLV quando disponibile.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-06, BET-03

## Obiettivo
Calcolare CLV quando disponibile.

## Cosa deve fare
1. Closing odds snapshot.
2. CLV per prediction.
3. Report per modello/mercato.

## File / aree da ispezionare
- `src/oracle/backtest/`
- `src/repository/`

## Acceptance criteria
- [ ] Nessun uso del closing price come feature pre-match illegittima.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-05
Fase: BETTING
Stato: NEW
Priorità: P1
Dipendenze: DATA-06, BET-03

Obiettivo:
Calcolare CLV quando disponibile.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Closing odds snapshot.
- CLV per prediction.
- Report per modello/mercato.

File/aree da ispezionare:
- src/oracle/backtest/
- src/repository/

Acceptance criteria:
- Nessun uso del closing price come feature pre-match illegittima.

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
