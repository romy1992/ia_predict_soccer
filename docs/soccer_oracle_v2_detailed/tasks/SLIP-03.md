# SLIP-03 — Generare schedine 2/3/4 eventi con profili Safe/Balanced/Aggressive.

## Metadata
- **Fase:** SCHEDINA
- **Stato iniziale:** NEW
- **Priorità:** P2
- **Dipendenze:** SLIP-02

## Obiettivo
Generare schedine 2/3/4 eventi con profili Safe/Balanced/Aggressive.

## Cosa deve fare
1. Ranking probability/EV/risk.
2. Limiti correlazione.
3. Output spiegabile.

## File / aree da ispezionare
- `src/oracle/betslip/`
- `frontend/src/`

## Acceptance criteria
- [ ] Tre profili distinti.
- [ ] Quote combinate e probabilità dichiarate con metodo esplicito.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SLIP-03
Fase: SCHEDINA
Stato: NEW
Priorità: P2
Dipendenze: SLIP-02

Obiettivo:
Generare schedine 2/3/4 eventi con profili Safe/Balanced/Aggressive.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Ranking probability/EV/risk.
- Limiti correlazione.
- Output spiegabile.

File/aree da ispezionare:
- src/oracle/betslip/
- frontend/src/

Acceptance criteria:
- Tre profili distinti.
- Quote combinate e probabilità dichiarate con metodo esplicito.

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
