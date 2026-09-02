# BET-06 — Creare Prediction Ledger / Paper Betting.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** BET-04, ML-07

## Obiettivo
Creare Prediction Ledger / Paper Betting.

## Cosa deve fare
1. Salvare prediction prima del kickoff.
2. model_run_id.
3. p_model, market fair, odd, edge, EV, decision.
4. Settlement dopo risultato.
5. Immutabilità logica della prediction originale.

## File / aree da ispezionare
- `src/service_ia/training/prediction_logger.py`
- `alembic/`
- `src/oracle/`

## Acceptance criteria
- [ ] Prediction storiche ricostruibili.
- [ ] PnL paper calcolabile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-06
Fase: BETTING
Stato: NEW
Priorità: P1
Dipendenze: BET-04, ML-07

Obiettivo:
Creare Prediction Ledger / Paper Betting.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Salvare prediction prima del kickoff.
- model_run_id.
- p_model, market fair, odd, edge, EV, decision.
- Settlement dopo risultato.
- Immutabilità logica della prediction originale.

File/aree da ispezionare:
- src/service_ia/training/prediction_logger.py
- alembic/
- src/oracle/

Acceptance criteria:
- Prediction storiche ricostruibili.
- PnL paper calcolabile.

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
