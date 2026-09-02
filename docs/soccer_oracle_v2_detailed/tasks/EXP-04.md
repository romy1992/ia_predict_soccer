# EXP-04 — Creare Market/Odds Expert.

## Metadata
- **Fase:** ORACLE EXPERTS
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-06, ML-04

## Obiettivo
Creare Market/Odds Expert.

## Cosa deve fare
1. Fair probabilities.
2. Dispersione bookmaker.
3. Movement quote.
4. Opening/latest/closing solo se temporalmente lecito.

## File / aree da ispezionare
- `src/ml/experts/market/`

## Acceptance criteria
- [ ] No closing odds in prediction pre-match se non disponibili al timestamp.
- [ ] Output probabilistico.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: EXP-04
Fase: ORACLE EXPERTS
Stato: NEW
Priorità: P1
Dipendenze: DATA-06, ML-04

Obiettivo:
Creare Market/Odds Expert.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Fair probabilities.
- Dispersione bookmaker.
- Movement quote.
- Opening/latest/closing solo se temporalmente lecito.

File/aree da ispezionare:
- src/ml/experts/market/

Acceptance criteria:
- No closing odds in prediction pre-match se non disponibili al timestamp.
- Output probabilistico.

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
