# BET-02 — Creare Value Engine e sostituire l'edge hardcoded attuale.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** REWRITE
- **Priorità:** P0
- **Dipendenze:** BET-01

## Obiettivo
Creare Value Engine e sostituire l'edge hardcoded attuale.

## Cosa deve fare
1. prob_edge = p_model-p_market_fair.
2. ev = p_model*odd-1.
3. Gestire quota mancante.
4. Usare outcome corretto.

## File / aree da ispezionare
- `src/api/dashboard_service.py`
- `src/oracle/value_engine/`

## Acceptance criteria
- [ ] Edge ed EV distinti.
- [ ] Test numerici.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-02
Fase: BETTING
Stato: REWRITE
Priorità: P0
Dipendenze: BET-01

Obiettivo:
Creare Value Engine e sostituire l'edge hardcoded attuale.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- prob_edge = p_model-p_market_fair.
- ev = p_model*odd-1.
- Gestire quota mancante.
- Usare outcome corretto.

File/aree da ispezionare:
- src/api/dashboard_service.py
- src/oracle/value_engine/

Acceptance criteria:
- Edge ed EV distinti.
- Test numerici.

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
