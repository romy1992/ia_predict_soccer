# OPS-03 — Creare monitoring performance modello e data drift.

## Metadata
- **Fase:** OPERATIONS
- **Stato iniziale:** NEW
- **Priorità:** P2
- **Dipendenze:** BET-06

## Obiettivo
Creare monitoring performance modello e data drift.

## Cosa deve fare
1. Prediction volume.
2. Calibration drift.
3. ROI rolling solo diagnostico.
4. Coverage feature.
5. Alert base.

## File / aree da ispezionare
- `src/ml/monitoring/`
- `frontend/src/`

## Acceptance criteria
- [ ] Dashboard/endpoint monitoring disponibile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: OPS-03
Fase: OPERATIONS
Stato: NEW
Priorità: P2
Dipendenze: BET-06

Obiettivo:
Creare monitoring performance modello e data drift.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Prediction volume.
- Calibration drift.
- ROI rolling solo diagnostico.
- Coverage feature.
- Alert base.

File/aree da ispezionare:
- src/ml/monitoring/
- frontend/src/

Acceptance criteria:
- Dashboard/endpoint monitoring disponibile.

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
