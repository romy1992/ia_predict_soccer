# BET-01 — Creare Fair Odds Engine.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-04, ORACLE-03

## Obiettivo
Creare Fair Odds Engine.

## Cosa deve fare
1. p_fair bookmaker.
2. fair odd = 1/p.
3. Confronto Oracle vs market.

## File / aree da ispezionare
- `src/oracle/fair_odds/`

## Acceptance criteria
- [ ] Output standard per outcome.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-01
Fase: BETTING
Stato: NEW
Priorità: P0
Dipendenze: ML-04, ORACLE-03

Obiettivo:
Creare Fair Odds Engine.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- p_fair bookmaker.
- fair odd = 1/p.
- Confronto Oracle vs market.

File/aree da ispezionare:
- src/oracle/fair_odds/

Acceptance criteria:
- Output standard per outcome.

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
