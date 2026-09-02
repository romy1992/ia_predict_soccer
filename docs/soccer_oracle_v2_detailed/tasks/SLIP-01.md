# SLIP-01 — Creare pool pick candidati per Schedina Oracle.

## Metadata
- **Fase:** SCHEDINA
- **Stato iniziale:** NEW
- **Priorità:** P2
- **Dipendenze:** BET-04, BET-06

## Obiettivo
Creare pool pick candidati per Schedina Oracle.

## Cosa deve fare
1. Solo PLAY e opzionalmente borderline configurato.
2. Vincoli quota/EV.
3. Una selezione per outcome/mercato.

## File / aree da ispezionare
- `src/oracle/betslip/`

## Acceptance criteria
- [ ] Pool deterministico e tracciabile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SLIP-01
Fase: SCHEDINA
Stato: NEW
Priorità: P2
Dipendenze: BET-04, BET-06

Obiettivo:
Creare pool pick candidati per Schedina Oracle.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Solo PLAY e opzionalmente borderline configurato.
- Vincoli quota/EV.
- Una selezione per outcome/mercato.

File/aree da ispezionare:
- src/oracle/betslip/

Acceptance criteria:
- Pool deterministico e tracciabile.

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
