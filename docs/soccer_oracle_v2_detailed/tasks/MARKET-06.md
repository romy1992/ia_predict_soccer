# MARKET-06 — Consolidare Cards O/U come mercato specializzato.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REFACTOR
- **Priorità:** P2
- **Dipendenze:** ML-01

## Obiettivo
Consolidare Cards O/U come mercato specializzato.

## Cosa deve fare
1. Line configurabile.
2. Feature arbitro/team/style.
3. Calibrazione.

## File / aree da ispezionare
- `src/ml/markets/cards/`

## Acceptance criteria
- [ ] Linee configurabili.
- [ ] Metriche per linea.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-06
Fase: MARKETS
Stato: REFACTOR
Priorità: P2
Dipendenze: ML-01

Obiettivo:
Consolidare Cards O/U come mercato specializzato.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Line configurabile.
- Feature arbitro/team/style.
- Calibrazione.

File/aree da ispezionare:
- src/ml/markets/cards/

Acceptance criteria:
- Linee configurabili.
- Metriche per linea.

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
