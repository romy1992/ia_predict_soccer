# ORACLE-01 — Standardizzare output degli esperti.

## Metadata
- **Fase:** ENSEMBLE
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** EXP-01, EXP-02, EXP-03, EXP-04, EXP-05

## Obiettivo
Standardizzare output degli esperti.

## Cosa deve fare
1. Schema expert output comune.
2. probability vector.
3. model_run_id.
4. feature timestamp.
5. confidence/metadata.

## File / aree da ispezionare
- `src/ml/ensemble/`

## Acceptance criteria
- [ ] Tutti gli esperti consumabili da meta-model.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: ORACLE-01
Fase: ENSEMBLE
Stato: NEW
Priorità: P1
Dipendenze: EXP-01, EXP-02, EXP-03, EXP-04, EXP-05

Obiettivo:
Standardizzare output degli esperti.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Schema expert output comune.
- probability vector.
- model_run_id.
- feature timestamp.
- confidence/metadata.

File/aree da ispezionare:
- src/ml/ensemble/

Acceptance criteria:
- Tutti gli esperti consumabili da meta-model.

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
