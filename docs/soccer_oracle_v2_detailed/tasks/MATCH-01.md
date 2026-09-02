# MATCH-01 — Rifare Match Center usando output Oracle ufficiali.

## Metadata
- **Fase:** MATCH CENTER
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** MARKET-01, MARKET-04, BET-04, FE-01

## Obiettivo
Rifare Match Center usando output Oracle ufficiali.

## Cosa deve fare
1. Probabilità mercato.
2. Quote.
3. fair market.
4. edge/EV.
5. badge decision.

## File / aree da ispezionare
- `frontend/src/`
- `src/api/`

## Acceptance criteria
- [ ] FE non ricalcola logica betting.
- [ ] Ogni dato ha fonte API backend.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MATCH-01
Fase: MATCH CENTER
Stato: REFACTOR
Priorità: P1
Dipendenze: MARKET-01, MARKET-04, BET-04, FE-01

Obiettivo:
Rifare Match Center usando output Oracle ufficiali.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Probabilità mercato.
- Quote.
- fair market.
- edge/EV.
- badge decision.

File/aree da ispezionare:
- frontend/src/
- src/api/

Acceptance criteria:
- FE non ricalcola logica betting.
- Ogni dato ha fonte API backend.

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
