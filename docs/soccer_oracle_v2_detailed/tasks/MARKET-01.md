# MARKET-01 — Implementare vero 1X2 multiclass.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REWRITE
- **Priorità:** P0
- **Dipendenze:** ML-01, EXP-05

## Obiettivo
Implementare vero 1X2 multiclass.

## Cosa deve fare
1. Classi HOME/DRAW/AWAY.
2. Probabilità sommano a 1.
3. Calibrazione multiclass.
4. Bookmaker baseline.

## File / aree da ispezionare
- `src/service_ia/training/market_service/filter_market_service.py`
- `src/ml/markets/`

## Acceptance criteria
- [ ] Nessun mapping draw->away.
- [ ] API espone tre probabilità.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-01
Fase: MARKETS
Stato: REWRITE
Priorità: P0
Dipendenze: ML-01, EXP-05

Obiettivo:
Implementare vero 1X2 multiclass.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Classi HOME/DRAW/AWAY.
- Probabilità sommano a 1.
- Calibrazione multiclass.
- Bookmaker baseline.

File/aree da ispezionare:
- src/service_ia/training/market_service/filter_market_service.py
- src/ml/markets/

Acceptance criteria:
- Nessun mapping draw->away.
- API espone tre probabilità.

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
