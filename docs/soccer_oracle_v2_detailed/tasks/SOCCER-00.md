# SOCCER-00 — Creare il branch operativo Soccer Oracle V2 e fissare la baseline tecnica.

## Metadata
- **Fase:** FOUNDATION
- **Stato iniziale:** NEW
- **Priorità:** P0
- **Dipendenze:** Nessuna

## Obiettivo
Creare il branch operativo Soccer Oracle V2 e fissare la baseline tecnica.

## Cosa deve fare
1. Verificare che `feature/ml-dashboard-platform` sia la base.
2. Creare/usarе `feature/soccer-oracle-v2`.
3. Registrare commit SHA iniziale nella documentazione.
4. Non modificare logica applicativa.

## File / aree da ispezionare
- `README.md`
- `docs/`

## Acceptance criteria
- [ ] Branch corretto attivo.
- [ ] Baseline documentata.
- [ ] Working tree pulita.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SOCCER-00
Fase: FOUNDATION
Stato: NEW
Priorità: P0
Dipendenze: Nessuna

Obiettivo:
Creare il branch operativo Soccer Oracle V2 e fissare la baseline tecnica.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Verificare che `feature/ml-dashboard-platform` sia la base.
- Creare/usarе `feature/soccer-oracle-v2`.
- Registrare commit SHA iniziale nella documentazione.
- Non modificare logica applicativa.

File/aree da ispezionare:
- README.md
- docs/

Acceptance criteria:
- Branch corretto attivo.
- Baseline documentata.
- Working tree pulita.

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
