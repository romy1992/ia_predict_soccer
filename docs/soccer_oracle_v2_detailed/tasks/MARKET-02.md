# MARKET-02 — Derivare Double Chance da 1X2 coerente.

## Metadata
- **Fase:** MARKETS
- **Stato iniziale:** REWRITE
- **Priorità:** P0
- **Dipendenze:** MARKET-01

## Obiettivo
Derivare Double Chance da 1X2 coerente.

## Cosa deve fare
1. P1X=P1+PX.
2. P12=P1+P2.
3. PX2=PX+P2.
4. Quote/fair value per outcome.

## File / aree da ispezionare
- `src/ml/markets/`
- `src/api/`

## Acceptance criteria
- [ ] Tre outcome DC disponibili.
- [ ] Coerenza matematica testata.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: MARKET-02
Fase: MARKETS
Stato: REWRITE
Priorità: P0
Dipendenze: MARKET-01

Obiettivo:
Derivare Double Chance da 1X2 coerente.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- P1X=P1+PX.
- P12=P1+P2.
- PX2=PX+P2.
- Quote/fair value per outcome.

File/aree da ispezionare:
- src/ml/markets/
- src/api/

Acceptance criteria:
- Tre outcome DC disponibili.
- Coerenza matematica testata.

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
