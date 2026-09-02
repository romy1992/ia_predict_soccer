# LIVE-01 — Creare pipeline dati live separata.

## Metadata
- **Fase:** LIVE ORACLE
- **Stato iniziale:** NEW
- **Priorità:** P3
- **Dipendenze:** DATA-03, MATCH-02

## Obiettivo
Creare pipeline dati live separata.

## Cosa deve fare
1. Live fixtures/events/stats.
2. Timestamp eventi.
3. Cache/polling controllato.
4. Non mischiare training pre-match e live.

## File / aree da ispezionare
- `src/data/live/`
- `src/api/`

## Acceptance criteria
- [ ] Dataset live distinto.
- [ ] Provider errors isolati.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: LIVE-01
Fase: LIVE ORACLE
Stato: NEW
Priorità: P3
Dipendenze: DATA-03, MATCH-02

Obiettivo:
Creare pipeline dati live separata.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Live fixtures/events/stats.
- Timestamp eventi.
- Cache/polling controllato.
- Non mischiare training pre-match e live.

File/aree da ispezionare:
- src/data/live/
- src/api/

Acceptance criteria:
- Dataset live distinto.
- Provider errors isolati.

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
