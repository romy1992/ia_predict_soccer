# EXP-01 — Creare Team Strength Expert.

## Metadata
- **Fase:** ORACLE EXPERTS
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-02

## Obiettivo
Creare Team Strength Expert.

## Cosa deve fare
1. Rating offensivo/difensivo.
2. Home advantage.
3. Rolling form solo passato.
4. Output numerici usabili dagli altri modelli.

## File / aree da ispezionare
- `src/ml/experts/team_strength/`

## Acceptance criteria
- [ ] Feature point-in-time.
- [ ] Output versionato.
- [ ] Backtest base.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: EXP-01
Fase: ORACLE EXPERTS
Stato: NEW
Priorità: P1
Dipendenze: ML-01, ML-02

Obiettivo:
Creare Team Strength Expert.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Rating offensivo/difensivo.
- Home advantage.
- Rolling form solo passato.
- Output numerici usabili dagli altri modelli.

File/aree da ispezionare:
- src/ml/experts/team_strength/

Acceptance criteria:
- Feature point-in-time.
- Output versionato.
- Backtest base.

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
