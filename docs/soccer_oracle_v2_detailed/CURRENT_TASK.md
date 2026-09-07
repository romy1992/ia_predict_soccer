# CURRENT TASK

## Task corrente
Nessuno: la roadmap operativa è **COMPLETA** (53/53 task, 12/12 fasi — vedi `MANIFEST.json` e `IMPLEMENTATION_LOG.md`).

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

















