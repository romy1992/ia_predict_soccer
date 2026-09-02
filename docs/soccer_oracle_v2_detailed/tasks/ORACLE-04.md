# ORACLE-04 — Esporre Model Consensus per spiegabilità.

## Metadata
- **Fase:** ENSEMBLE
- **Stato iniziale:** NEW
- **Priorità:** P2
- **Dipendenze:** ORACLE-02

## Obiettivo
Esporre Model Consensus per spiegabilità.

## Cosa deve fare
1. Output expert per match.
2. Oracle finale.
3. Dispersione consensus.
4. Nessuna spiegazione inventata.

## File / aree da ispezionare
- `src/api/`
- `src/ml/ensemble/`

## Acceptance criteria
- [ ] API restituisce consensus strutturato.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: ORACLE-04
Fase: ENSEMBLE
Stato: NEW
Priorità: P2
Dipendenze: ORACLE-02

Obiettivo:
Esporre Model Consensus per spiegabilità.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Output expert per match.
- Oracle finale.
- Dispersione consensus.
- Nessuna spiegazione inventata.

File/aree da ispezionare:
- src/api/
- src/ml/ensemble/

Acceptance criteria:
- API restituisce consensus strutturato.

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
