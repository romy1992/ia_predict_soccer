# MATCH-02 — Creare Oracle Match Detail.

## Metadata
- **Fase:** MATCH CENTER
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** MATCH-01, ORACLE-04

## Obiettivo
Creare Oracle Match Detail.

## Cosa deve fare
1. Overview.
2. Probabilities.
3. Value Bets.
4. Team Strength.
5. Expected Goals.
6. Score Matrix.
7. Odds Movement.
8. Model Consensus.

## File / aree da ispezionare
- `frontend/src/`
- `src/api/`

## Acceptance criteria
- [ ] Dettaglio navigabile per fixture.
- [ ] Dati mancanti gestiti.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MATCH-02
Fase: MATCH CENTER
Stato: NEW
Priorità: P1
Dipendenze: MATCH-01, ORACLE-04

Obiettivo:
Creare Oracle Match Detail.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Overview.
- Probabilities.
- Value Bets.
- Team Strength.
- Expected Goals.
- Score Matrix.
- Odds Movement.
- Model Consensus.

File/aree da ispezionare:
- frontend/src/
- src/api/

Acceptance criteria:
- Dettaglio navigabile per fixture.
- Dati mancanti gestiti.

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
