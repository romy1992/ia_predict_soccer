# DATA-01 — Estrarre un provider API-Sports pulito dal downloader storico.

## Metadata
- **Fase:** DATA PLATFORM
- **Stato iniziale:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** SOCCER-02

## Obiettivo
Estrarre un provider API-Sports pulito dal downloader storico.

## Cosa deve fare
1. Separare chiamate HTTP da mapping/persistenza.
2. Creare metodi per fixtures, statistics, odds, events.
3. Centralizzare retry/error handling/API quota.
4. Mantenere compatibilità con importer attuale.

## File / aree da ispezionare
- `src/service_ia/utility/request_api.py`
- `src/service_ia/pre_processing/download_match_service.py`

## Acceptance criteria
- [ ] Provider testabile isolatamente.
- [ ] Nessun endpoint hardcoded duplicato.
- [ ] Errori provider tracciati.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: DATA-01
Fase: DATA PLATFORM
Stato: REFACTOR
Priorità: P0
Dipendenze: SOCCER-02

Obiettivo:
Estrarre un provider API-Sports pulito dal downloader storico.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Separare chiamate HTTP da mapping/persistenza.
- Creare metodi per fixtures, statistics, odds, events.
- Centralizzare retry/error handling/API quota.
- Mantenere compatibilità con importer attuale.

File/aree da ispezionare:
- src/service_ia/utility/request_api.py
- src/service_ia/pre_processing/download_match_service.py

Acceptance criteria:
- Provider testabile isolatamente.
- Nessun endpoint hardcoded duplicato.
- Errori provider tracciati.

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
