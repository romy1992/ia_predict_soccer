# SLIP-02 — Creare Correlation Engine.

## Metadata
- **Fase:** SCHEDINA
- **Stato iniziale:** NEW
- **Priorità:** P2
- **Dipendenze:** SLIP-01

## Obiettivo
Creare Correlation Engine.

## Cosa deve fare
1. Regole same-match.
2. Dipendenze goal/BTTS/1X2.
3. Penalità o esclusione.
4. Matrice/regole versionate.

## File / aree da ispezionare
- `src/oracle/betslip/`

## Acceptance criteria
- [ ] Combinazioni fortemente correlate non passano come indipendenti.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SLIP-02
Fase: SCHEDINA
Stato: NEW
Priorità: P2
Dipendenze: SLIP-01

Obiettivo:
Creare Correlation Engine.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Regole same-match.
- Dipendenze goal/BTTS/1X2.
- Penalità o esclusione.
- Matrice/regole versionate.

File/aree da ispezionare:
- src/oracle/betslip/

Acceptance criteria:
- Combinazioni fortemente correlate non passano come indipendenti.

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
