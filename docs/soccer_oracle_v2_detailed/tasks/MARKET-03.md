# MARKET-03 — Consolidare BTTS.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** EXP-02, EXP-05

## Obiettivo
Consolidare BTTS.

## Cosa deve fare
1. Confrontare derivazione score distribution vs direct expert.
2. Calibrare finale.
3. Usare ensemble se migliore.

## File / aree da ispezionare
- `src/ml/markets/btts/`

## Acceptance criteria
- [ ] P(Yes)+P(No)=1.
- [ ] Report benchmark.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-03
Fase: MARKETS
Stato: REFACTOR
Priorità: P1
Dipendenze: EXP-02, EXP-05

Obiettivo:
Consolidare BTTS.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Confrontare derivazione score distribution vs direct expert.
- Calibrare finale.
- Usare ensemble se migliore.

File/aree da ispezionare:
- src/ml/markets/btts/

Acceptance criteria:
- P(Yes)+P(No)=1.
- Report benchmark.

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
