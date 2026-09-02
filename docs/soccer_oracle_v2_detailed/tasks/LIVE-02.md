# LIVE-02 — Definire feature store live.

## Metadata
- **Fase:** LIVE ORACLE
- **Stato iniziale:** NEW
- **Priorità:** P3
- **Dipendenze:** LIVE-01, ML-01

## Obiettivo
Definire feature store live.

## Cosa deve fare
1. minute.
2. scoreline.
3. cards.
4. shots/xG se disponibili.
5. pre-match prior.

## File / aree da ispezionare
- `src/ml/live/`

## Acceptance criteria
- [ ] Ogni feature live timestamped.
- [ ] Schema documentato.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: LIVE-02
Fase: LIVE ORACLE
Stato: NEW
Priorità: P3
Dipendenze: LIVE-01, ML-01

Obiettivo:
Definire feature store live.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- minute.
- scoreline.
- cards.
- shots/xG se disponibili.
- pre-match prior.

File/aree da ispezionare:
- src/ml/live/

Acceptance criteria:
- Ogni feature live timestamped.
- Schema documentato.

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
