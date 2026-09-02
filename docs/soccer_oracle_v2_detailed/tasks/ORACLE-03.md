# ORACLE-03 — Applicare calibrazione finale all'Oracle Ensemble.

## Metadata
- **Fase:** ENSEMBLE
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** ORACLE-02, ML-06

## Obiettivo
Applicare calibrazione finale all'Oracle Ensemble.

## Cosa deve fare
1. Calibration per mercato/outcome.
2. Report pre/post.
3. Fallback quando campione insufficiente.

## File / aree da ispezionare
- `src/ml/calibration/`
- `src/ml/ensemble/`

## Acceptance criteria
- [ ] Final probabilities calibrate e versionate.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: ORACLE-03
Fase: ENSEMBLE
Stato: NEW
Priorità: P1
Dipendenze: ORACLE-02, ML-06

Obiettivo:
Applicare calibrazione finale all'Oracle Ensemble.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Calibration per mercato/outcome.
- Report pre/post.
- Fallback quando campione insufficiente.

File/aree da ispezionare:
- src/ml/calibration/
- src/ml/ensemble/

Acceptance criteria:
- Final probabilities calibrate e versionate.

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
