# EXP-05 — Trasformare i direct market models in esperti diretti.

## Metadata
- **Fase:** ORACLE EXPERTS
- **Stato iniziale:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** ML-03

## Obiettivo
Trasformare i direct market models in esperti diretti.

## Cosa deve fare
1. Binary/multiclass corretti.
2. Training temporale.
3. Calibrazione.
4. Output standardizzato.

## File / aree da ispezionare
- `src/service_ia/training/train_multi_market.py`
- `src/ml/experts/direct/`

## Acceptance criteria
- [ ] Interfaccia comune expert.predict_proba.
- [ ] Niente H2H binario travestito da 1X2.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: EXP-05
Fase: ORACLE EXPERTS
Stato: REFACTOR
Priorità: P1
Dipendenze: ML-03

Obiettivo:
Trasformare i direct market models in esperti diretti.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Binary/multiclass corretti.
- Training temporale.
- Calibrazione.
- Output standardizzato.

File/aree da ispezionare:
- src/service_ia/training/train_multi_market.py
- src/ml/experts/direct/

Acceptance criteria:
- Interfaccia comune expert.predict_proba.
- Niente H2H binario travestito da 1X2.

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
