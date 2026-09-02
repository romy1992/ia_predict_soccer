# EXP-03 — Creare Statistics Expert.

## Metadata
- **Fase:** ORACLE EXPERTS
- **Stato iniziale:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-03

## Obiettivo
Creare Statistics Expert.

## Cosa deve fare
1. Modello su forma/statistiche pre-match.
2. Niente odds nel modello puro statistics.
3. Output probabilistico/embedding features.

## File / aree da ispezionare
- `src/ml/experts/statistics/`

## Acceptance criteria
- [ ] Separazione netta da market expert.
- [ ] Metriche temporali disponibili.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: EXP-03
Fase: ORACLE EXPERTS
Stato: NEW
Priorità: P1
Dipendenze: ML-01, ML-03

Obiettivo:
Creare Statistics Expert.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Modello su forma/statistiche pre-match.
- Niente odds nel modello puro statistics.
- Output probabilistico/embedding features.

File/aree da ispezionare:
- src/ml/experts/statistics/

Acceptance criteria:
- Separazione netta da market expert.
- Metriche temporali disponibili.

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
