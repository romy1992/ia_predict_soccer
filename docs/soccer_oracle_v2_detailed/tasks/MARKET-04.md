# MARKET-04 — Consolidare U/O 1.5-4.5 multi-linea.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** EXP-02

## Obiettivo
Consolidare U/O 1.5-4.5 multi-linea.

## Cosa deve fare
1. Confrontare binary indipendenti, hierarchical, goal distribution.
2. Stesso walk-forward.
3. Selezione per metriche probabilistiche e betting.
4. Monotonicità obbligatoria.

## File / aree da ispezionare
- `src/service_ia/training/under_over/`
- `src/ml/markets/totals/`

## Acceptance criteria
- [ ] P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5).
- [ ] Report comparativo.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-04
Fase: MARKETS
Stato: REFACTOR
Priorità: P1
Dipendenze: EXP-02

Obiettivo:
Consolidare U/O 1.5-4.5 multi-linea.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Confrontare binary indipendenti, hierarchical, goal distribution.
- Stesso walk-forward.
- Selezione per metriche probabilistiche e betting.
- Monotonicità obbligatoria.

File/aree da ispezionare:
- src/service_ia/training/under_over/
- src/ml/markets/totals/

Acceptance criteria:
- P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5).
- Report comparativo.

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
