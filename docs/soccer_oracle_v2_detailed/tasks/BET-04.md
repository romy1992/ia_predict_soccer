# BET-04 — Creare Decision Policy versionata.

## Metadata
- **Fase:** BETTING
- **Stato iniziale:** REWRITE
- **Priorità:** P1
- **Dipendenze:** BET-03

## Obiettivo
Creare Decision Policy versionata.

## Cosa deve fare
1. Soglie per mercato/outcome.
2. Min samples.
3. Min edge/EV.
4. Min/max odd opzionali.
5. PLAY/BORDERLINE/NO BET.

## File / aree da ispezionare
- `src/oracle/decision_engine/`
- `src/api/dashboard_service.py`

## Acceptance criteria
- [ ] Niente soglie globali hardcoded nel dashboard service.
- [ ] policy_version registrata.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: BET-04
Fase: BETTING
Stato: REWRITE
Priorità: P1
Dipendenze: BET-03

Obiettivo:
Creare Decision Policy versionata.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Soglie per mercato/outcome.
- Min samples.
- Min edge/EV.
- Min/max odd opzionali.
- PLAY/BORDERLINE/NO BET.

File/aree da ispezionare:
- src/oracle/decision_engine/
- src/api/dashboard_service.py

Acceptance criteria:
- Niente soglie globali hardcoded nel dashboard service.
- policy_version registrata.

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
