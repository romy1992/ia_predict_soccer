# ROADMAP OPERATIVA — Soccer Oracle V2

Questa è la sequenza operativa principale. Ogni task ha un file dedicato in `tasks/` con prompt completo.

## Regole di esecuzione
- Eseguire i task in ordine salvo dipendenze esplicite.
- Non considerare completato un task finché gli acceptance criteria non sono verificati.
- I task P0 vengono prima dei P1/P2/P3 della stessa area quando bloccanti.
- Il Live Oracle viene dopo la stabilizzazione del pre-match.

## Fase 00 — FOUNDATION

### SOCCER-00 — Creare il branch operativo Soccer Oracle V2 e fissare la baseline tecnica.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** —
- **Task dettagliato:** `tasks/SOCCER-00.md`

**Da fare:**
- Verificare che `feature/ml-dashboard-platform` sia la base.
- Creare/usarе `feature/soccer-oracle-v2`.
- Registrare commit SHA iniziale nella documentazione.
- Non modificare logica applicativa.

**Acceptance criteria:**
- [ ] Branch corretto attivo.
- [ ] Baseline documentata.
- [ ] Working tree pulita.

**Prompt:** vedi il file task dedicato.

### SOCCER-01 — Ridurre duplicazioni e definire la struttura canonica del codice senza cancellare esperimenti utili.
- **Stato:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** SOCCER-00
- **Task dettagliato:** `tasks/SOCCER-01.md`

**Da fare:**
- Mappare `service_ia/` vs `src/service_ia/`.
- Marcare codice legacy/deprecated.
- Definire `src/` come package runtime ufficiale.
- Non eliminare dataset/modelli senza inventario.
- Aggiornare import se necessario.

**Acceptance criteria:**
- [ ] Esiste una sola struttura runtime ufficiale.
- [ ] Legacy chiaramente identificato.
- [ ] Test esistenti continuano a passare.

**Prompt:** vedi il file task dedicato.

### SOCCER-02 — Definire il database canonico usato da API, scheduler, training e dashboard.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** SOCCER-00
- **Task dettagliato:** `tasks/SOCCER-02.md`

**Da fare:**
- Eliminare ambiguità tra PostgreSQL locale e Docker.
- Introdurre configurazione ambiente unica per `DATABASE_URL`.
- Aggiungere health/audit endpoint o comando per mostrare DB target e conteggi principali.
- Documentare dev/test/prod.

**Acceptance criteria:**
- [ ] API e scheduler leggono la stessa configurazione.
- [ ] È possibile verificare host/db/schema attivi.
- [ ] Conteggi base documentabili.

**Prompt:** vedi il file task dedicato.

## Fase 01 — DATA PLATFORM

### DATA-01 — Estrarre un provider API-Sports pulito dal downloader storico.
- **Stato:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** SOCCER-02
- **Task dettagliato:** `tasks/DATA-01.md`

**Da fare:**
- Separare chiamate HTTP da mapping/persistenza.
- Creare metodi per fixtures, statistics, odds, events.
- Centralizzare retry/error handling/API quota.
- Mantenere compatibilità con importer attuale.

**Acceptance criteria:**
- [ ] Provider testabile isolatamente.
- [ ] Nessun endpoint hardcoded duplicato.
- [ ] Errori provider tracciati.

**Prompt:** vedi il file task dedicato.

### DATA-02 — Rendere l'import storico parametrico per intervallo date, stagioni e leghe.
- **Stato:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** DATA-01
- **Task dettagliato:** `tasks/DATA-02.md`

**Da fare:**
- Rimuovere date manuali dal flusso operativo.
- Accettare `from_date`, `to_date`, `leagues`, `seasons`.
- Import idempotente.
- Restituire report inserted/updated/skipped/failed.

**Acceptance criteria:**
- [ ] Nessuna data manuale necessaria nel codice.
- [ ] Riesecuzione non duplica fixture.
- [ ] Report import disponibile.

**Prompt:** vedi il file task dedicato.

### DATA-03 — Creare il job/endpoint 'Aggiorna oggi'.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-02
- **Task dettagliato:** `tasks/DATA-03.md`

**Da fare:**
- Sincronizzare fixture odierne.
- Aggiornare stato, score, statistiche e quote quando disponibili.
- Persistenza idempotente.
- Esporre stato job.

**Acceptance criteria:**
- [ ] Endpoint/job manuale funzionante.
- [ ] Statistiche job registrate.
- [ ] Errori per fixture non bloccano tutto il batch.

**Prompt:** vedi il file task dedicato.

### DATA-04 — Creare import delle partite future per una finestra configurabile.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-02
- **Task dettagliato:** `tasks/DATA-04.md`

**Da fare:**
- Supportare `days_ahead`.
- Importare fixture NS.
- Salvare quote pre-match disponibili.
- Calcolare/aggiornare feature pre-match senza usare dati futuri.

**Acceptance criteria:**
- [ ] Finestra futura parametrica.
- [ ] Nessun target/result leakage.
- [ ] Riesecuzione aggiorna senza duplicare.

**Prompt:** vedi il file task dedicato.

### DATA-05 — Creare settlement/finalizzazione delle partite concluse.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-03
- **Task dettagliato:** `tasks/DATA-05.md`

**Da fare:**
- Rilevare match terminati.
- Aggiornare score/statistiche finali.
- Marcare dati completi/incompleti.
- Preparare hook per settlement prediction.

**Acceptance criteria:**
- [ ] Partite finali riconciliate.
- [ ] Stato completezza visibile.
- [ ] Job idempotente.

**Prompt:** vedi il file task dedicato.

### DATA-06 — Introdurre odds snapshot normalizzati nel tempo.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-01, SOCCER-02
- **Task dettagliato:** `tasks/DATA-06.md`

**Da fare:**
- Creare schema `odds_snapshot`.
- Campi: fixture, bookmaker, market, period, line, outcome, odd, captured_at, source.
- Migration Alembic.
- Non eliminare subito JSON odds legacy.
- Backfill dove possibile.

**Acceptance criteria:**
- [ ] Snapshot multipli per fixture supportati.
- [ ] Timestamp obbligatorio.
- [ ] Query per opening/latest/closing possibili.

**Prompt:** vedi il file task dedicato.

### DATA-07 — Creare Data Quality report.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-02, DATA-03, DATA-04
- **Task dettagliato:** `tasks/DATA-07.md`

**Da fare:**
- Coverage fixture/statistics/odds per mercato.
- Null/duplicati/orfani.
- Distribuzione per lega/stagione.
- Controlli temporali.
- Endpoint/servizio per dashboard.

**Acceptance criteria:**
- [ ] Report riproducibile.
- [ ] Anomalie conteggiate.
- [ ] Output machine-readable.

**Prompt:** vedi il file task dedicato.

### DATA-08 — Evolvere Job History per import, settlement, training e backtest.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** DATA-03
- **Task dettagliato:** `tasks/DATA-08.md`

**Da fare:**
- Stato queued/running/success/failed.
- started_at/finished_at/duration.
- Parametri input.
- summary risultati.
- error payload.

**Acceptance criteria:**
- [ ] Ogni job importante tracciabile.
- [ ] Storico filtrabile per tipo/stato.

**Prompt:** vedi il file task dedicato.

## Fase 02 — FRONTEND

### FE-01 — Spezzare `App.jsx` in una struttura React a feature.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** SOCCER-01
- **Task dettagliato:** `tasks/FE-01.md`

**Da fare:**
- Creare layout/router/componenti.
- Feature: dashboard, matches, data-center, ml-lab, oracle.
- Mantenere funzionalità esistente durante refactor.

**Acceptance criteria:**
- [ ] App.jsx non contiene tutta la UI.
- [ ] Nessuna regressione API principale.

**Prompt:** vedi il file task dedicato.

### FE-02 — Creare Data Center per import manuale.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-02, DATA-03, DATA-04, FE-01
- **Task dettagliato:** `tasks/FE-02.md`

**Da fare:**
- Selettore date.
- Leghe/stagioni.
- Azioni: storico, oggi, future.
- Mostrare job status e riepilogo.

**Acceptance criteria:**
- [ ] Utente può lanciare import senza modificare codice.
- [ ] Errori mostrati chiaramente.
- [ ] Azioni disabilitate durante job se necessario.

**Prompt:** vedi il file task dedicato.

### FE-03 — Creare Data Quality dashboard.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-07, FE-01
- **Task dettagliato:** `tasks/FE-03.md`

**Da fare:**
- Coverage per mercato.
- Fixture incomplete.
- Odds/statistics mancanti.
- Filtri lega/stagione.

**Acceptance criteria:**
- [ ] Dati leggibili e verificabili.
- [ ] Nessuna logica qualità implementata nel FE.

**Prompt:** vedi il file task dedicato.

## Fase 03 — ML FOUNDATION

### ML-01 — Creare dataset builder point-in-time ufficiale.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** DATA-06, DATA-07
- **Task dettagliato:** `tasks/ML-01.md`

**Da fare:**
- Ogni feature deve avere available_at <= prediction_at.
- Separare target da feature.
- Supportare market/period/line/outcome.
- Snapshot dataset versionabile.

**Acceptance criteria:**
- [ ] Test anti-leakage.
- [ ] Dataset riproducibile.
- [ ] Timestamp prediction esplicito.

**Prompt:** vedi il file task dedicato.

### ML-02 — Creare split temporali production-grade.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-01
- **Task dettagliato:** `tasks/ML-02.md`

**Da fare:**
- Expanding window.
- Rolling window.
- Holdout finale.
- No shuffle.
- Configurazione per date/stagioni.

**Acceptance criteria:**
- [ ] Nessun random split nella pipeline production.
- [ ] Test di ordinamento temporale.

**Prompt:** vedi il file task dedicato.

### ML-03 — Portare feature selection e preprocessing dentro le pipeline CV.
- **Stato:** REFACTOR
- **Priorità:** P0
- **Dipendenze:** ML-02
- **Task dettagliato:** `tasks/ML-03.md`

**Da fare:**
- Imputer/selector/scaler/model nella stessa pipeline.
- Evitare selection leakage.
- Supportare modelli tree e linear senza preprocessing inutile.

**Acceptance criteria:**
- [ ] Feature selection eseguita per fold.
- [ ] Test pipeline.

**Prompt:** vedi il file task dedicato.

### ML-04 — Creare bookmaker baseline con rimozione overround.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-01, DATA-06
- **Task dettagliato:** `tasks/ML-04.md`

**Da fare:**
- Calcolare implied raw.
- Normalizzare fair probability.
- Supportare 1X2, BTTS, U/O.
- Persist/test helper.

**Acceptance criteria:**
- [ ] Somma fair probability ≈1 per mercati esclusivi.
- [ ] Baseline disponibile nei report.

**Prompt:** vedi il file task dedicato.

### ML-05 — Creare framework metriche probabilistiche.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-02
- **Task dettagliato:** `tasks/ML-05.md`

**Da fare:**
- LogLoss.
- Brier.
- ECE/reliability.
- AUC secondaria.
- Report per market/league/season.

**Acceptance criteria:**
- [ ] Champion non selezionato solo via F1.
- [ ] Metriche salvate nel registry.

**Prompt:** vedi il file task dedicato.

### ML-06 — Creare calibration framework.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-02, ML-05
- **Task dettagliato:** `tasks/ML-06.md`

**Da fare:**
- Platt/sigmoid.
- Isotonic dove campione sufficiente.
- Confronto pre/post calibration.
- Calibratore versionato.

**Acceptance criteria:**
- [ ] Brier/LogLoss pre-post disponibili.
- [ ] Calibratore associato al model run.

**Prompt:** vedi il file task dedicato.

### ML-07 — Evolvere Model Registry a lifecycle completo.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** ML-05, ML-06
- **Task dettagliato:** `tasks/ML-07.md`

**Da fare:**
- dataset_version.
- feature_version.
- train/validation/test windows.
- git SHA.
- candidate/champion/production/retired.
- promotion history.

**Acceptance criteria:**
- [ ] Latest != production.
- [ ] Lookup production per mercato.
- [ ] Test lifecycle.

**Prompt:** vedi il file task dedicato.

## Fase 04 — ORACLE EXPERTS

### EXP-01 — Creare Team Strength Expert.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-02
- **Task dettagliato:** `tasks/EXP-01.md`

**Da fare:**
- Rating offensivo/difensivo.
- Home advantage.
- Rolling form solo passato.
- Output numerici usabili dagli altri modelli.

**Acceptance criteria:**
- [ ] Feature point-in-time.
- [ ] Output versionato.
- [ ] Backtest base.

**Prompt:** vedi il file task dedicato.

### EXP-02 — Portare il Goal Distribution Expert in pipeline ufficiale.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-02
- **Task dettagliato:** `tasks/EXP-02.md`

**Da fare:**
- Recuperare Poisson/goal distribution esistente.
- Supportare 1.5/2.5/3.5/4.5.
- Validazione temporale.
- Confronto con alternative (es. negative binomial se utile).

**Acceptance criteria:**
- [ ] Probabilità U/O monotone per costruzione.
- [ ] Score distribution disponibile.

**Prompt:** vedi il file task dedicato.

### EXP-03 — Creare Statistics Expert.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-01, ML-03
- **Task dettagliato:** `tasks/EXP-03.md`

**Da fare:**
- Modello su forma/statistiche pre-match.
- Niente odds nel modello puro statistics.
- Output probabilistico/embedding features.

**Acceptance criteria:**
- [ ] Separazione netta da market expert.
- [ ] Metriche temporali disponibili.

**Prompt:** vedi il file task dedicato.

### EXP-04 — Creare Market/Odds Expert.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-06, ML-04
- **Task dettagliato:** `tasks/EXP-04.md`

**Da fare:**
- Fair probabilities.
- Dispersione bookmaker.
- Movement quote.
- Opening/latest/closing solo se temporalmente lecito.

**Acceptance criteria:**
- [ ] No closing odds in prediction pre-match se non disponibili al timestamp.
- [ ] Output probabilistico.

**Prompt:** vedi il file task dedicato.

### EXP-05 — Trasformare i direct market models in esperti diretti.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** ML-03
- **Task dettagliato:** `tasks/EXP-05.md`

**Da fare:**
- Binary/multiclass corretti.
- Training temporale.
- Calibrazione.
- Output standardizzato.

**Acceptance criteria:**
- [ ] Interfaccia comune expert.predict_proba.
- [ ] Niente H2H binario travestito da 1X2.

**Prompt:** vedi il file task dedicato.

## Fase 05 — MARKETS

### MARKET-01 — Implementare vero 1X2 multiclass.
- **Stato:** REWRITE
- **Priorità:** P0
- **Dipendenze:** ML-01, EXP-05
- **Task dettagliato:** `tasks/MARKET-01.md`

**Da fare:**
- Classi HOME/DRAW/AWAY.
- Probabilità sommano a 1.
- Calibrazione multiclass.
- Bookmaker baseline.

**Acceptance criteria:**
- [ ] Nessun mapping draw->away.
- [ ] API espone tre probabilità.

**Prompt:** vedi il file task dedicato.

### MARKET-02 — Derivare Double Chance da 1X2 coerente.
- **Stato:** REWRITE
- **Priorità:** P0
- **Dipendenze:** MARKET-01
- **Task dettagliato:** `tasks/MARKET-02.md`

**Da fare:**
- P1X=P1+PX.
- P12=P1+P2.
- PX2=PX+P2.
- Quote/fair value per outcome.

**Acceptance criteria:**
- [ ] Tre outcome DC disponibili.
- [ ] Coerenza matematica testata.

**Prompt:** vedi il file task dedicato.

### MARKET-03 — Consolidare BTTS.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** EXP-02, EXP-05
- **Task dettagliato:** `tasks/MARKET-03.md`

**Da fare:**
- Confrontare derivazione score distribution vs direct expert.
- Calibrare finale.
- Usare ensemble se migliore.

**Acceptance criteria:**
- [ ] P(Yes)+P(No)=1.
- [ ] Report benchmark.

**Prompt:** vedi il file task dedicato.

### MARKET-04 — Consolidare U/O 1.5-4.5 multi-linea.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** EXP-02
- **Task dettagliato:** `tasks/MARKET-04.md`

**Da fare:**
- Confrontare binary indipendenti, hierarchical, goal distribution.
- Stesso walk-forward.
- Selezione per metriche probabilistiche e betting.
- Monotonicità obbligatoria.

**Acceptance criteria:**
- [ ] P(O1.5)>=P(O2.5)>=P(O3.5)>=P(O4.5).
- [ ] Report comparativo.

**Prompt:** vedi il file task dedicato.

### MARKET-05 — Consolidare Corners O/U come mercato specializzato.
- **Stato:** REFACTOR
- **Priorità:** P2
- **Dipendenze:** ML-01
- **Task dettagliato:** `tasks/MARKET-05.md`

**Da fare:**
- Definire line come parametro, non soglia hardcoded unica.
- Feature dedicate.
- Calibrazione.

**Acceptance criteria:**
- [ ] Linee configurabili.
- [ ] Metriche per linea.

**Prompt:** vedi il file task dedicato.

### MARKET-06 — Consolidare Cards O/U come mercato specializzato.
- **Stato:** REFACTOR
- **Priorità:** P2
- **Dipendenze:** ML-01
- **Task dettagliato:** `tasks/MARKET-06.md`

**Da fare:**
- Line configurabile.
- Feature arbitro/team/style.
- Calibrazione.

**Acceptance criteria:**
- [ ] Linee configurabili.
- [ ] Metriche per linea.

**Prompt:** vedi il file task dedicato.

## Fase 06 — ENSEMBLE

### ORACLE-01 — Standardizzare output degli esperti.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** EXP-01, EXP-02, EXP-03, EXP-04, EXP-05
- **Task dettagliato:** `tasks/ORACLE-01.md`

**Da fare:**
- Schema expert output comune.
- probability vector.
- model_run_id.
- feature timestamp.
- confidence/metadata.

**Acceptance criteria:**
- [ ] Tutti gli esperti consumabili da meta-model.

**Prompt:** vedi il file task dedicato.

### ORACLE-02 — Creare Meta Model / Stacker per mercato.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** ORACLE-01
- **Task dettagliato:** `tasks/ORACLE-02.md`

**Da fare:**
- OOF predictions solo temporali.
- Meta features dagli expert.
- Niente leakage stacking.
- Confronto weighted blend vs learned stacker.

**Acceptance criteria:**
- [ ] OOF temporalmente corrette.
- [ ] Meta-model versionato.

**Prompt:** vedi il file task dedicato.

### ORACLE-03 — Applicare calibrazione finale all'Oracle Ensemble.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** ORACLE-02, ML-06
- **Task dettagliato:** `tasks/ORACLE-03.md`

**Da fare:**
- Calibration per mercato/outcome.
- Report pre/post.
- Fallback quando campione insufficiente.

**Acceptance criteria:**
- [ ] Final probabilities calibrate e versionate.

**Prompt:** vedi il file task dedicato.

### ORACLE-04 — Esporre Model Consensus per spiegabilità.
- **Stato:** NEW
- **Priorità:** P2
- **Dipendenze:** ORACLE-02
- **Task dettagliato:** `tasks/ORACLE-04.md`

**Da fare:**
- Output expert per match.
- Oracle finale.
- Dispersione consensus.
- Nessuna spiegazione inventata.

**Acceptance criteria:**
- [ ] API restituisce consensus strutturato.

**Prompt:** vedi il file task dedicato.

## Fase 07 — BETTING

### BET-01 — Creare Fair Odds Engine.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** ML-04, ORACLE-03
- **Task dettagliato:** `tasks/BET-01.md`

**Da fare:**
- p_fair bookmaker.
- fair odd = 1/p.
- Confronto Oracle vs market.

**Acceptance criteria:**
- [ ] Output standard per outcome.

**Prompt:** vedi il file task dedicato.

### BET-02 — Creare Value Engine e sostituire l'edge hardcoded attuale.
- **Stato:** REWRITE
- **Priorità:** P0
- **Dipendenze:** BET-01
- **Task dettagliato:** `tasks/BET-02.md`

**Da fare:**
- prob_edge = p_model-p_market_fair.
- ev = p_model*odd-1.
- Gestire quota mancante.
- Usare outcome corretto.

**Acceptance criteria:**
- [ ] Edge ed EV distinti.
- [ ] Test numerici.

**Prompt:** vedi il file task dedicato.

### BET-03 — Creare Betting Backtester.
- **Stato:** NEW
- **Priorità:** P0
- **Dipendenze:** BET-02, ML-05
- **Task dettagliato:** `tasks/BET-03.md`

**Da fare:**
- Stake flat iniziale.
- ROI/yield/profit.
- Hit rate.
- Avg odds.
- Max drawdown.
- Performance per edge bucket/mercato/lega.

**Acceptance criteria:**
- [ ] Backtest solo out-of-sample.
- [ ] Report riproducibile.

**Prompt:** vedi il file task dedicato.

### BET-04 — Creare Decision Policy versionata.
- **Stato:** REWRITE
- **Priorità:** P1
- **Dipendenze:** BET-03
- **Task dettagliato:** `tasks/BET-04.md`

**Da fare:**
- Soglie per mercato/outcome.
- Min samples.
- Min edge/EV.
- Min/max odd opzionali.
- PLAY/BORDERLINE/NO BET.

**Acceptance criteria:**
- [ ] Niente soglie globali hardcoded nel dashboard service.
- [ ] policy_version registrata.

**Prompt:** vedi il file task dedicato.

### BET-05 — Calcolare CLV quando disponibile.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** DATA-06, BET-03
- **Task dettagliato:** `tasks/BET-05.md`

**Da fare:**
- Closing odds snapshot.
- CLV per prediction.
- Report per modello/mercato.

**Acceptance criteria:**
- [ ] Nessun uso del closing price come feature pre-match illegittima.

**Prompt:** vedi il file task dedicato.

### BET-06 — Creare Prediction Ledger / Paper Betting.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** BET-04, ML-07
- **Task dettagliato:** `tasks/BET-06.md`

**Da fare:**
- Salvare prediction prima del kickoff.
- model_run_id.
- p_model, market fair, odd, edge, EV, decision.
- Settlement dopo risultato.
- Immutabilità logica della prediction originale.

**Acceptance criteria:**
- [ ] Prediction storiche ricostruibili.
- [ ] PnL paper calcolabile.

**Prompt:** vedi il file task dedicato.

## Fase 08 — MATCH CENTER

### MATCH-01 — Rifare Match Center usando output Oracle ufficiali.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** MARKET-01, MARKET-04, BET-04, FE-01
- **Task dettagliato:** `tasks/MATCH-01.md`

**Da fare:**
- Probabilità mercato.
- Quote.
- fair market.
- edge/EV.
- badge decision.

**Acceptance criteria:**
- [ ] FE non ricalcola logica betting.
- [ ] Ogni dato ha fonte API backend.

**Prompt:** vedi il file task dedicato.

### MATCH-02 — Creare Oracle Match Detail.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** MATCH-01, ORACLE-04
- **Task dettagliato:** `tasks/MATCH-02.md`

**Da fare:**
- Overview.
- Probabilities.
- Value Bets.
- Team Strength.
- Expected Goals.
- Score Matrix.
- Odds Movement.
- Model Consensus.

**Acceptance criteria:**
- [ ] Dettaglio navigabile per fixture.
- [ ] Dati mancanti gestiti.

**Prompt:** vedi il file task dedicato.

## Fase 09 — SCHEDINA

### SLIP-01 — Creare pool pick candidati per Schedina Oracle.
- **Stato:** NEW
- **Priorità:** P2
- **Dipendenze:** BET-04, BET-06
- **Task dettagliato:** `tasks/SLIP-01.md`

**Da fare:**
- Solo PLAY e opzionalmente borderline configurato.
- Vincoli quota/EV.
- Una selezione per outcome/mercato.

**Acceptance criteria:**
- [ ] Pool deterministico e tracciabile.

**Prompt:** vedi il file task dedicato.

### SLIP-02 — Creare Correlation Engine.
- **Stato:** NEW
- **Priorità:** P2
- **Dipendenze:** SLIP-01
- **Task dettagliato:** `tasks/SLIP-02.md`

**Da fare:**
- Regole same-match.
- Dipendenze goal/BTTS/1X2.
- Penalità o esclusione.
- Matrice/regole versionate.

**Acceptance criteria:**
- [ ] Combinazioni fortemente correlate non passano come indipendenti.

**Prompt:** vedi il file task dedicato.

### SLIP-03 — Generare schedine 2/3/4 eventi con profili Safe/Balanced/Aggressive.
- **Stato:** NEW
- **Priorità:** P2
- **Dipendenze:** SLIP-02
- **Task dettagliato:** `tasks/SLIP-03.md`

**Da fare:**
- Ranking probability/EV/risk.
- Limiti correlazione.
- Output spiegabile.

**Acceptance criteria:**
- [ ] Tre profili distinti.
- [ ] Quote combinate e probabilità dichiarate con metodo esplicito.

**Prompt:** vedi il file task dedicato.

## Fase 10 — OPERATIONS

### OPS-01 — Separare scheduler dati da scheduler ML.
- **Stato:** REFACTOR
- **Priorità:** P1
- **Dipendenze:** DATA-03, DATA-04, DATA-05
- **Task dettagliato:** `tasks/OPS-01.md`

**Da fare:**
- Data jobs frequenti.
- Training job indipendente.
- No retrain automatico ad ogni import.

**Acceptance criteria:**
- [ ] Scheduler leggibile e configurabile.
- [ ] max_instances/coalesce corretti.

**Prompt:** vedi il file task dedicato.

### OPS-02 — Creare Candidate -> Champion -> Production promotion.
- **Stato:** NEW
- **Priorità:** P1
- **Dipendenze:** ML-07, BET-03
- **Task dettagliato:** `tasks/OPS-02.md`

**Da fare:**
- Gate metriche.
- Confronto production/candidate.
- Promozione manuale o policy controllata.
- Rollback.

**Acceptance criteria:**
- [ ] Ultimo training non diventa automaticamente production.
- [ ] Audit promotion.

**Prompt:** vedi il file task dedicato.

### OPS-03 — Creare monitoring performance modello e data drift.
- **Stato:** NEW
- **Priorità:** P2
- **Dipendenze:** BET-06
- **Task dettagliato:** `tasks/OPS-03.md`

**Da fare:**
- Prediction volume.
- Calibration drift.
- ROI rolling solo diagnostico.
- Coverage feature.
- Alert base.

**Acceptance criteria:**
- [ ] Dashboard/endpoint monitoring disponibile.

**Prompt:** vedi il file task dedicato.

## Fase 11 — LIVE ORACLE

### LIVE-01 — Creare pipeline dati live separata.
- **Stato:** NEW
- **Priorità:** P3
- **Dipendenze:** DATA-03, MATCH-02
- **Task dettagliato:** `tasks/LIVE-01.md`

**Da fare:**
- Live fixtures/events/stats.
- Timestamp eventi.
- Cache/polling controllato.
- Non mischiare training pre-match e live.

**Acceptance criteria:**
- [ ] Dataset live distinto.
- [ ] Provider errors isolati.

**Prompt:** vedi il file task dedicato.

### LIVE-02 — Definire feature store live.
- **Stato:** NEW
- **Priorità:** P3
- **Dipendenze:** LIVE-01, ML-01
- **Task dettagliato:** `tasks/LIVE-02.md`

**Da fare:**
- minute.
- scoreline.
- cards.
- shots/xG se disponibili.
- pre-match prior.

**Acceptance criteria:**
- [ ] Ogni feature live timestamped.
- [ ] Schema documentato.

**Prompt:** vedi il file task dedicato.

### LIVE-03 — Addestrare e validare modelli live separati.
- **Stato:** NEW
- **Priorità:** P3
- **Dipendenze:** LIVE-02, ML-02
- **Task dettagliato:** `tasks/LIVE-03.md`

**Da fare:**
- Split per match/time.
- No leakage eventi futuri.
- Probabilità aggiornate.

**Acceptance criteria:**
- [ ] Pipeline separata da pre-match.
- [ ] Metriche live per minuto/finestra.

**Prompt:** vedi il file task dedicato.
