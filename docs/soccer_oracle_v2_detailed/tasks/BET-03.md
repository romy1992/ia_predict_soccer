# BET-03 — Creare Betting Backtester.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** BET-02, ML-05

## Obiettivo
Creare Betting Backtester.

## Cosa deve fare
1. Stake flat iniziale.
2. ROI/yield/profit.
3. Hit rate.
4. Avg odds.
5. Max drawdown.
6. Performance per edge bucket/mercato/lega.

## File / aree da ispezionare
- `src/oracle/backtest/`

## Acceptance criteria
- [ ] Backtest solo out-of-sample.
- [ ] Report riproducibile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-03
Fase: BETTING
Stato: NEW
Priorità: P0
Dipendenze: BET-02, ML-05

Obiettivo:
Creare Betting Backtester.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Stake flat iniziale.
- ROI/yield/profit.
- Hit rate.
- Avg odds.
- Max drawdown.
- Performance per edge bucket/mercato/lega.

File/aree da ispezionare:
- src/oracle/backtest/

Acceptance criteria:
- Backtest solo out-of-sample.
- Report riproducibile.

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
