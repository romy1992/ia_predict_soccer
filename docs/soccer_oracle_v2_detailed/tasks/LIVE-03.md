# LIVE-03 — Addestrare e validare modelli live separati.

## Metadata
- **Fase:** LIVE ORACLE
- **Stato iniziale:** NEW
- **Priorità:** P3
- **Dipendenze:** LIVE-02, ML-02

## Obiettivo
Addestrare e validare modelli live separati.

## Cosa deve fare
1. Split per match/time.
2. No leakage eventi futuri.
3. Probabilità aggiornate.

## File / aree da ispezionare
- `src/ml/live/`

## Acceptance criteria
- [ ] Pipeline separata da pre-match.
- [ ] Metriche live per minuto/finestra.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: LIVE-03
Fase: LIVE ORACLE
Stato: NEW
Priorità: P3
Dipendenze: LIVE-02, ML-02

Obiettivo:
Addestrare e validare modelli live separati.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Split per match/time.
- No leakage eventi futuri.
- Probabilità aggiornate.

File/aree da ispezionare:
- src/ml/live/

Acceptance criteria:
- Pipeline separata da pre-match.
- Metriche live per minuto/finestra.

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
