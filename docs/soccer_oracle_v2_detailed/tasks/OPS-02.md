# OPS-02 — Creare Candidate -> Champion -> Production promotion.

## Metadata
- **Fase:** OPERATIONS
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-07, BET-03

## Obiettivo
Creare Candidate -> Champion -> Production promotion.

## Cosa deve fare
1. Gate metriche.
2. Confronto production/candidate.
3. Promozione manuale o policy controllata.
4. Rollback.

## File / aree da ispezionare
- `src/service_ia/training/model_registry.py`
- `src/ml/`

## Acceptance criteria
- [ ] Ultimo training non diventa automaticamente production.
- [ ] Audit promotion.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: OPS-02
Fase: OPERATIONS
Stato: NEW
Priorità: P1
Dipendenze: ML-07, BET-03

Obiettivo:
Creare Candidate -> Champion -> Production promotion.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Gate metriche.
- Confronto production/candidate.
- Promozione manuale o policy controllata.
- Rollback.

File/aree da ispezionare:
- src/service_ia/training/model_registry.py
- src/ml/

Acceptance criteria:
- Ultimo training non diventa automaticamente production.
- Audit promotion.

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
