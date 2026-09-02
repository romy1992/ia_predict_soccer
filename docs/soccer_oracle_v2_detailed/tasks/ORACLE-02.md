# ORACLE-02 — Creare Meta Model / Stacker per mercato.

## Metadata
- **Fase:** ENSEMBLE
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** ORACLE-01

## Obiettivo
Creare Meta Model / Stacker per mercato.

## Cosa deve fare
1. OOF predictions solo temporali.
2. Meta features dagli expert.
3. Niente leakage stacking.
4. Confronto weighted blend vs learned stacker.

## File / aree da ispezionare
- `src/ml/ensemble/`

## Acceptance criteria
- [ ] OOF temporalmente corrette.
- [ ] Meta-model versionato.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: ORACLE-02
Fase: ENSEMBLE
Stato: NEW
Priorità: P1
Dipendenze: ORACLE-01

Obiettivo:
Creare Meta Model / Stacker per mercato.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- OOF predictions solo temporali.
- Meta features dagli expert.
- Niente leakage stacking.
- Confronto weighted blend vs learned stacker.

File/aree da ispezionare:
- src/ml/ensemble/

Acceptance criteria:
- OOF temporalmente corrette.
- Meta-model versionato.

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
