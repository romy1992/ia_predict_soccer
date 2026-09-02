# EXP-02 — Portare il Goal Distribution Expert in pipeline ufficiale.

## Metadata
- **Fase:** ORACLE EXPERTS
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-02

## Obiettivo
Portare il Goal Distribution Expert in pipeline ufficiale.

## Cosa deve fare
1. Recuperare Poisson/goal distribution esistente.
2. Supportare 1.5/2.5/3.5/4.5.
3. Validazione temporale.
4. Confronto con alternative (es. negative binomial se utile).

## File / aree da ispezionare
- `src/service_ia/training/under_over/consistent/total_goals/`
- `src/ml/experts/goal_distribution/`

## Acceptance criteria
- [ ] Probabilità U/O monotone per costruzione.
- [ ] Score distribution disponibile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: EXP-02
Fase: ORACLE EXPERTS
Stato: REFACTOR
Priorità: P1
Dipendenze: ML-01, ML-02

Obiettivo:
Portare il Goal Distribution Expert in pipeline ufficiale.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Recuperare Poisson/goal distribution esistente.
- Supportare 1.5/2.5/3.5/4.5.
- Validazione temporale.
- Confronto con alternative (es. negative binomial se utile).

File/aree da ispezionare:
- src/service_ia/training/under_over/consistent/total_goals/
- src/ml/experts/goal_distribution/

Acceptance criteria:
- Probabilità U/O monotone per costruzione.
- Score distribution disponibile.

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
