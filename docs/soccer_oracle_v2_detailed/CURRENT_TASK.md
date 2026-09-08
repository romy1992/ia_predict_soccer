# CURRENT TASK

## Task corrente
**COMPLETATO (2026-09-08)**: addestrati modelli ML REALI (fit completo, non solo benchmark) per Under/Over 1.5/2.5/3.5/4.5 e REGISTRATI come 'candidate' in `ModelRegistry` (`best_models/registry/index.jsonl`). Dettaglio completo in `IMPLEMENTATION_LOG.md` (voce "Training reale Under/Over 1.5/2.5/3.5/4.5", 2026-09-08).

Fix propedeutici (`src/service_ia/training/train_multi_market.py`): (1) griglia `RandomForestClassifier` alleggerita (`n_estimators` 500->200, combinazioni max_depth/min_samples_split/min_samples_leaf 36->8 per selector_k) per evitare il MemoryError che aveva bloccato il pilot; (2) **fix critico ulteriore**: `StackingClassifier(cv=cv_splits)` riusato dentro `cross_val_score`/`temporal_oof_probabilities`/`CalibrationService` su sottoinsiemi piu' piccoli di `X` causava sistematicamente `ValueError: cross_val_predict only works for partitions` — MAI stato eseguito con successo finora. Fix: `cv=5` (intero).

**Risultati reali (dataset da DB Railway, trasferito via export CSV - vedi sotto):**

| Soglia | Champion | selection_score | rows |
|---|---|---|---|
| under_over_1_5 | random_forest | 0.7131 | 8349 |
| under_over_2_5 | voting | 0.6727 | 15243 |
| under_over_3_5 | random_forest | 0.6906 | 8335 |
| under_over_4_5 | random_forest | 0.7820 | 8286 |

Tutti e 4 registrati come `candidate`, gate di promozione (`evaluate_promotion`) verificato su ciascuno: tutte le metriche standard lette correttamente, `allowed=true` (nessuna production precedente per questi mercati, prima promozione consentita) - restano `candidate`, nessuna promozione automatica eseguita.

`scripts/analysis/phase3b_sequential_stacking_thresholds.py` (idea "stacking sequenziale a cascata tra soglie", anti-leakage verificato esplicitamente anche in negativo) eseguito su dati reali: **SCARTATO** - delta selection_score vs baseline non-cascata sullo stesso sottoinsieme di righe: over_2_5 -0.0030, over_3_5 +0.0009, over_4_5 -0.0014, tutti sotto la soglia di adozione (0.01). Emerso anche un limite strutturale intrinseco: ogni livello di cascata consuma un fold aggiuntivo di dati per l'anti-leakage (n_eval_rows 7620 -> 6096 -> 4572 -> 3048). Risultato in `best_models/phase3b_sequential_stacking_thresholds_result.json`.

**Come e' stato risolto il blocco di rete** (la sessione cloud `anthropic_cloud` non puo' raggiungere il Postgres remoto Railway via TCP diretto, confermato non aggirabile a nessun livello di network access - vedi doc ufficiale ambienti cloud): nuovo script `scripts/analysis/export_datasets_for_cloud_training.py`, da eseguire in un ambiente CON accesso DB reale (sessione bridge sulla macchina locale dell'operatore), esporta le feature GIA' ELABORATE (non i JSON grezzi, troppo pesanti) per le 4 soglie + il frame 'totals' in CSV compatti (`scripts/analysis/_export/`, ~52MB totali, non gitignored di proposito per poter essere trasferiti via git). `scripts/analysis/train_from_export.py`/`phase3b_from_export.py` eseguono la logica di produzione INVARIATA (`train_market()`/`run_sequential_cascade()`) sostituendo solo la sorgente dati (CSV invece di query DB dirette). I modelli addestrati (file `.pkl` + registry, gitignored) sono stati consegnati all'operatore per essere posizionati nel `best_models/` della sua macchina locale (dove gira l'app reale). I CSV di export sono stati rimossi dal repo dopo l'uso (workaround temporaneo, non uno strumento di produzione permanente).

Nota: due tentativi precedenti di training diretto su una sessione bridge locale sono falliti per disconnessioni ripetute della sessione (macchina/rete dell'operatore, non un bug del codice) - da cui la scelta di spostare il training pesante in cloud e limitare la sessione bridge alla sola (breve) esportazione dati.

---

Roadmap operativa di base **COMPLETA** (53/53 task, 12/12 fasi — vedi `MANIFEST.json` e `IMPLEMENTATION_LOG.md`).

`LIVE-03` (ultimo task, Fase 11 LIVE ORACLE) è stato completato e validato.

Dopo LIVE-03 sono state completate e validate ulteriori estensioni extra-roadmap (2026-09-05/07, vedi `IMPLEMENTATION_LOG.md` sezione "Estensioni introdotte" per il dettaglio completo): bugfix critico gestione quota API-Sports, job "Aggiorna tutto"/Daily Refresh (bottone Sidebar sempre visibile), pagina Impostazioni (enable/disable job + quota API-Sports), consolidamento frontend Dashboard (rimosse le pagine separate Live/Today/Match Center, `PhaseTabs`/`MarketTabs` + select data accumulata), fix critici di performance/concorrenza DB (`scoped_session`, sessione persistente per job, bulk upsert `odds_snapshot`), auto-pausa job + disabilitazione bottoni a quota esaurita, fix critico performance cambio-giorno Dashboard (query multi-mercato consolidata, `get_overview` senza predizioni, skip API-Sports per date storiche gia' a DB, `force_refresh`, filtro Fase/Mercato client-side). Suite completa rieseguita: 712 passed. Import `src.api.main` verificato end-to-end (51 route). Build frontend verificata senza errori.

Il **07/09** e' stata inoltre completata un'estensione mirata su Under/Over 1.5/2.5/3.5/4.5 (MARKET-04), a fasi (0/1+2/3/4), dettaglio completo in `IMPLEMENTATION_LOG.md`: Fase 0 nuovo report Data Quality per soglia (`DataQualityService.build_under_over_threshold_report`); Fase 1+2 nuovo campo `best_approach_by_threshold` sul benchmark totals (rivela che `over_2_5` preferisce `binary_independent` mentre le altre 3 soglie preferiscono `goal_distribution`); **fix critico**: le metriche registrate dal benchmark totals non esponevano le chiavi standard richieste dal gate di promozione (OPS-02), che quindi falliva sempre per 'totals' — risolto; Fase 3 A/B test cascading sulla soglia 2.5 con esito negativo documentato (feature scartata, delta insufficiente). Script di analisi in `scripts/analysis/`, risultati in `best_models/*.json`. Test mirati verificati: 28/28 passed.

## Regola
Completare e validare questo task prima di aggiornare il file al task successivo.

## Task completati (vedi IMPLEMENTATION_LOG.md)
SOCCER-00, SOCCER-01, SOCCER-02, DATA-01..08, ML-01..07, FE-01..03, EXP-01..05 (fase ORACLE EXPERTS completata), MARKET-01..06 (fase MARKETS completata), ORACLE-01..04 (fase ENSEMBLE completata), BET-01..06 (fase BETTING completata), MATCH-01..02 (fase MATCH CENTER completata), SLIP-01..03 (fase SCHEDINA completata), OPS-01..03 (fase OPERATIONS completata), LIVE-01..03 (fase LIVE ORACLE completata)

## Prossimi in coda
Nessuno pianificato nella roadmap operativa attuale. Possibili prossimi passi (da concordare, non ancora un task formale):
- Promozione controllata (`ModelRegistry.promote_with_policy`) dei modelli `1x2_live` (LIVE-03) da `candidate` a `production` una volta raccolto un volume sufficiente di fixture concluse nel dataset live reale.
- Esposizione API/frontend delle probabilità live aggiornate (oggi solo `src/ml/live/`, nessun endpoint dedicato).
- Eventuale nuova roadmap di crescita (vedi `ROADMAP_CRESCITA_ML_DASHBOARD.md`, documento di contesto più datato, in gran parte già superato dalle 12 fasi completate).

La roadmap completa è in `ROADMAP_OPERATIVA.md`.

















