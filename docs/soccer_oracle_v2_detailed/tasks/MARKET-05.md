# MARKET-05 — Consolidare Corners O/U come mercato specializzato.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REFACTOR
- **Priorità:** P2
- **Dipendenze:** ML-01

## Obiettivo
Consolidare Corners O/U come mercato specializzato.

## Cosa deve fare
1. Definire line come parametro, non soglia hardcoded unica.
2. Feature dedicate.
3. Calibrazione.

## File / aree da ispezionare
- `src/ml/markets/corners/`

## Acceptance criteria
- [ ] Linee configurabili.
- [ ] Metriche per linea.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-05
Fase: MARKETS
Stato: REFACTOR
Priorità: P2
Dipendenze: ML-01

Obiettivo:
Consolidare Corners O/U come mercato specializzato.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Definire line come parametro, non soglia hardcoded unica.
- Feature dedicate.
- Calibrazione.

File/aree da ispezionare:
- src/ml/markets/corners/

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
