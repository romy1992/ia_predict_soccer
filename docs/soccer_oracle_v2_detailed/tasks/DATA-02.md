# DATA-02 — Rendere l'import storico parametrico per intervallo date, stagioni e leghe.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** DATA-01

## Obiettivo
Rendere l'import storico parametrico per intervallo date, stagioni e leghe.

## Cosa deve fare
1. Rimuovere date manuali dal flusso operativo.
2. Accettare `from_date`, `to_date`, `leagues`, `seasons`.
3. Import idempotente.
4. Restituire report inserted/updated/skipped/failed.

## File / aree da ispezionare
- `src/service_ia/pre_processing/download_match_service.py`
- `src/api/`

## Acceptance criteria
- [ ] Nessuna data manuale necessaria nel codice.
- [ ] Riesecuzione non duplica fixture.
- [ ] Report import disponibile.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-02
Fase: DATA PLATFORM
Stato: REFACTOR
Priorità: P0
Dipendenze: DATA-01

Obiettivo:
Rendere l'import storico parametrico per intervallo date, stagioni e leghe.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Rimuovere date manuali dal flusso operativo.
- Accettare `from_date`, `to_date`, `leagues`, `seasons`.
- Import idempotente.
- Restituire report inserted/updated/skipped/failed.

File/aree da ispezionare:
- src/service_ia/pre_processing/download_match_service.py
- src/api/

Acceptance criteria:
- Nessuna data manuale necessaria nel codice.
- Riesecuzione non duplica fixture.
- Report import disponibile.

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
