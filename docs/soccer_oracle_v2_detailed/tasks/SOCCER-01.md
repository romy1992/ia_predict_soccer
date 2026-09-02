# SOCCER-01 — Ridurre duplicazioni e definire la struttura canonica del codice senza cancellare esperimenti utili.

## Metadata
- **Fase:** FOUNDATION
- **Stato iniziale:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** SOCCER-00

## Obiettivo
Ridurre duplicazioni e definire la struttura canonica del codice senza cancellare esperimenti utili.

## Cosa deve fare
1. Mappare `service_ia/` vs `src/service_ia/`.
2. Marcare codice legacy/deprecated.
3. Definire `src/` come package runtime ufficiale.
4. Non eliminare dataset/modelli senza inventario.
5. Aggiornare import se necessario.

## File / aree da ispezionare
- `service_ia/`
- `src/service_ia/`
- `tests/`

## Acceptance criteria
- [ ] Esiste una sola struttura runtime ufficiale.
- [ ] Legacy chiaramente identificato.
- [ ] Test esistenti continuano a passare.

## Prompt pronto per l'AI

```text
Sei nel progetto Soccer Oracle V2.

Task: SOCCER-01
Fase: FOUNDATION
Stato: REFACTOR
Priorità: P0
Dipendenze: SOCCER-00

Obiettivo:
Ridurre duplicazioni e definire la struttura canonica del codice senza cancellare esperimenti utili.

Prima di intervenire:
- esplora i file coinvolti;
- verifica cosa esiste già;
- evita riscritture non necessarie;
- preserva compatibilità e test esistenti;
- applica migration Alembic per ogni modifica DB.

Implementa:
- Mappare `service_ia/` vs `src/service_ia/`.
- Marcare codice legacy/deprecated.
- Definire `src/` come package runtime ufficiale.
- Non eliminare dataset/modelli senza inventario.
- Aggiornare import se necessario.

File/aree da ispezionare:
- service_ia/
- src/service_ia/
- tests/

Acceptance criteria:
- Esiste una sola struttura runtime ufficiale.
- Legacy chiaramente identificato.
- Test esistenti continuano a passare.

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
