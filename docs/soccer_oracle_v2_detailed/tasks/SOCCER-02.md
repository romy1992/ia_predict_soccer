# SOCCER-02 — Definire il database canonico usato da API, scheduler, training e dashboard.

## Metadata
- **Fase:** FOUNDATION
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** SOCCER-00

## Obiettivo
Definire il database canonico usato da API, scheduler, training e dashboard.

## Cosa deve fare
1. Eliminare ambiguità tra PostgreSQL locale e Docker.
2. Introdurre configurazione ambiente unica per `DATABASE_URL`.
3. Aggiungere health/audit endpoint o comando per mostrare DB target e conteggi principali.
4. Documentare dev/test/prod.

## File / aree da ispezionare
- `docker-compose.yml`
- `src/repository/base/repository_db.py`
- `src/service_ia/config/app_config.py`

## Acceptance criteria
- [ ] API e scheduler leggono la stessa configurazione.
- [ ] È possibile verificare host/db/schema attivi.
- [ ] Conteggi base documentabili.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SOCCER-02
Fase: FOUNDATION
Stato: NEW
Priorità: P0
Dipendenze: SOCCER-00

Obiettivo:
Definire il database canonico usato da API, scheduler, training e dashboard.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Eliminare ambiguità tra PostgreSQL locale e Docker.
- Introdurre configurazione ambiente unica per `DATABASE_URL`.
- Aggiungere health/audit endpoint o comando per mostrare DB target e conteggi principali.
- Documentare dev/test/prod.

File/aree da ispezionare:
- docker-compose.yml
- src/repository/base/repository_db.py
- src/service_ia/config/app_config.py

Acceptance criteria:
- API e scheduler leggono la stessa configurazione.
- È possibile verificare host/db/schema attivi.
- Conteggi base documentabili.

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
