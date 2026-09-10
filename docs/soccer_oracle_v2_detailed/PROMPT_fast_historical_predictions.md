# Prompt: vista storica sempre veloce + copertura completa banca dati

Richiesto esplicitamente dall'operatore (2026-09-10), dopo aver segnalato che
il cambio data in Dashboard resta lento **anche al secondo giro sulla stessa
data** (non solo la prima volta) — la banca dati predizioni
(`match_prediction_snapshot`) copre solo le partite viste in Dashboard o
intercettate dal job mentre erano ancora `NS`, mai lo storico gia' passato
ne' le partite appena finite fuori da quella finestra.

## Stato: 4/4 completati, MERGIATO in main e deployato (2026-09-10) - punto 3 (backfill storico completo) deciso di NON completarlo, non serve per l'uso reale del progetto (vedi punto 3 per il dettaglio)

## 1. Niente ricalcolo al volo per le date storiche (vista lista) — ☑ Fatto (commit `ad50648`)

**Obiettivo**: la vista lista (`GET /dashboard/day`) per una data passata
(`target_date < oggi`) non deve MAI ricalcolare una predizione al volo — se
manca la riga salvata, mostra il mercato come non disponibile invece di
aspettare. Cosi' la vista storica e' SEMPRE veloce, mai "quasi sempre".

- `PredictionSnapshotService.resolve_predictions` (`src/ml/serving/prediction_snapshot_service.py`):
  nuovo parametro `allow_compute: bool = True`. Quando `False` e lo status e'
  finale senza una riga salvata, il mercato viene saltato (nessun frame,
  nessun modello caricato) invece di calcolare+salvare.
- `DashboardService._predict_fixture`/`_serialize_api_fixture`/`_serialize_match`
  (`src/api/dashboard_service.py`): thread `allow_compute` fino al servizio.
- `DashboardService.get_day_matches`: passa `allow_compute=False` SOLO
  quando `is_historical_date` e' vero (variabile gia' presente nel metodo) -
  MAI per oggi/date future (li' la predizione deve poter essere calcolata la
  prima volta). `get_match_detail` (vista dettaglio, un click deliberato su
  UNA fixture) resta sempre `allow_compute=True` - la lentezza li' e'
  accettabile, e' un'azione singola non un rendering di lista.
- Frontend: gestire il caso "mercato assente dal payload" nelle celle di
  previsione (`PredictionBadges.jsx`) con un placeholder chiaro (es. "N/D" o
  "in coda") invece di un crash o una cella vuota ambigua.
- Test: nuovi casi in `prediction_snapshot_service_test.py`
  (`allow_compute=False` con/senza snapshot esistente) e
  `dashboard_service_test.py` (`get_day_matches` passa `allow_compute`
  corretto per date storiche vs oggi/future).

**Fatto**: `allow_compute` implementato e threaded fino a
`get_day_matches` (attivo SOLO per date storiche, MAI oggi/future, MAI
`get_match_detail` - verificato leggendo anche `get_live_matches`, che
resta correttamente sempre `allow_compute=True` di default). Frontend:
`PredictionBadges.jsx` mostra ora un chip placeholder "In coda" (nuova
classe `.prediction-pending`) per ogni mercato registrato (`model_markets`,
gia' presente nella risposta di `/dashboard/day`) assente dal payload di
previsioni - propagato da `DashboardPage` a `MatchTable` a
`PredictionBadges`. 11 nuovi/aggiornati casi di test, suite completa
(813 test) verde, build frontend verificata. Nessuna decisione ancora
presa sul colore "corretta/sbagliata" del badge (feature separata,
richiesta il 2026-09-10 ma non parte di questo piano a 4 punti).

## 2. Job esteso per le partite appena finite — ☑ Fatto (commit `caf12bb`)

**Obiettivo**: da questo momento in poi, copertura automatica — nessuna
manutenzione manuale richiesta per le partite nuove.

- `run_prediction_snapshot_refresh` (`src/jobs/scheduler.py`): estendere la
  query oltre a `status == "NS"` per includere anche le fixture con status
  finale (`FINAL_STATUSES`) negli ultimi N giorni (finestra piccola, es. 2-3
  giorni - non l'intero storico, quello e' lo script di backfill al punto 3)
  che NON hanno ancora nessuna riga in `match_prediction_snapshot`
  (serve un modo per individuarle: query anti-join o controllo via
  `MatchPredictionSnapshotRepository.get_latest_bulk` sui fixture_id del
  batch, scartando quelli gia' coperti).
- Aggiornare `JOB_DEFINITIONS["prediction_snapshot_refresh"]["description"]`
  (`src/jobs/job_settings.py`) per riflettere il nuovo scope (non piu' solo
  "partite non ancora disputate").
- Aggiornare la docstring del job e la sezione corrispondente in
  `build_scheduler`.
- Test: nuovo caso in `prediction_snapshot_refresh_job_test.py` (fixture
  FINAL recente senza snapshot viene raccolta; fixture FINAL gia' coperta
  NON viene ricalcolata; fixture FINAL vecchia fuori dalla piccola finestra
  NON viene toccata da questo job - quella e' lo scope dello script punto 3).

**Fatto**: `run_prediction_snapshot_refresh` ora interroga ANCHE le fixture
con status finale concluse negli ultimi 3 giorni (`_RECENTLY_FINISHED_WINDOW_DAYS`,
nuovo parametro opzionale `recently_finished_days`), con un anti-join via
`MatchPredictionSnapshotRepository.get_latest_bulk` per saltare quelle gia'
completamente coperte (nessuna chiamata a `resolve_predictions` sprecata).
Aggiornate anche la description del job in Impostazioni e la docstring di
`build_scheduler`. 5 nuovi test, suite completa (817 test) verde.

## 3. Script di backfill storico (una tantum) — ☑ Fatto (commit `d72b4f5`), DA LANCIARE

**Obiettivo**: chiudere il buco su tutto lo storico gia' in DB. Lanciato UNA
VOLTA dall'operatore (o da me per suo conto), non schedulato.

- Nuovo `scripts/backfill_prediction_snapshots.py`: itera tutte le fixture
  con status finale che non hanno ancora nessuna riga in
  `match_prediction_snapshot` (stessa query "anti-join" del punto 2 ma senza
  finestra temporale - tutto lo storico), in batch (evitare di caricare
  tutto in memoria in un colpo solo), chiamando
  `PredictionSnapshotService.resolve_predictions(allow_compute=True)` per
  ciascuna. Log di progresso (quante fixture processate/coperte/errori).
- Richiede accesso reale al DB - da lanciare dalla sessione bridge locale
  (stesso meccanismo gia' usato per backup/migration), NON eseguibile da
  questa sessione cloud.

**Fatto (codice)**: `scripts/backfill_prediction_snapshots.py` scritto e
testato (6 test con SQLite in-memory: raccolta, anti-join, isolamento
errori, paginazione multi-batch via keyset su `id_fixture`, `--limit`).
Suite completa (823 test) verde.

**Esecuzione (2026-09-10)**: lanciato una prima volta (giro completo) dalla
sessione bridge locale, poi interrotto manualmente dall'operatore prima
del completamento. Nessun danno/stato inconsistente (script idempotente,
anti-join per fixture+mercato, mai un overwrite) - quanto processato prima
dell'interruzione resta comunque salvato correttamente.

**Deciso di NON completarlo (2026-09-10, chiarimento esplicito
dell'operatore)**: "il progetto e' nuovo... tutte le partite storiche in
realta' dovevano servire solo per addestrare... da oggi in poi mi
interessano che le partite vengano salvate con predizioni". Chiarimento
importante: lo storico profondo (usato per il TRAINING dei modelli) legge
direttamente da `match`/`statistics`/`odds`, MAI da
`match_prediction_snapshot` - quella tabella e' solo un log/cache delle
predizioni SERVITE in Dashboard, zero impatto sul training presente o
futuro. Cio' che l'operatore vuole davvero ("qualche giorno indietro" +
"da oggi in poi") e' ESATTAMENTE cio' che il Punto 2 (job in background)
gia' fa in automatico, per sempre, senza alcun intervento manuale: copre
le fixture concluse negli ultimi `_RECENTLY_FINISHED_WINDOW_DAYS` (3)
giorni ad ogni giro schedulato. **Il giro completo dello script di
backfill (tutto lo storico pre-esistente) NON e' quindi necessario per
l'uso reale del progetto** - resta disponibile (codice pronto e testato)
solo per un eventuale uso futuro one-off (es. un'analisi retrospettiva
sull'intero storico), non per l'operativita' quotidiana. Le fixture
storiche vecchie mai coperte continueranno a mostrare il placeholder
"In coda" in Dashboard - accettato come comportamento finale, non un
difetto da risolvere.

## 4. Bottone "Ricalcola previsione" nel dettaglio partita — ☑ Fatto (commit `5ceba8b`)

**Obiettivo**: forzatura puntuale, manuale, su una singola fixture - utile
se il backfill ha saltato qualcosa o serve un refresh mirato.

- `PredictionSnapshotService.resolve_predictions`: nuovo parametro
  `force: bool = False` - quando `True`, ignora qualunque riga esistente
  (anche per partite finali "congelate") e ricalcola+salva sempre una riga
  nuova. Uso ESPLICITO e manuale, mai automatico.
- Nuovo endpoint `POST /dashboard/matches/{fixture_id}/recompute-predictions`
  (`src/api/main.py`, schema dedicato in `schemas.py`) che chiama
  `resolve_predictions(fixture_id=..., markets=..., force=True)` per tutti i
  mercati con un modello registrato.
- Frontend: bottone "Ricalcola previsione" dentro il pannello di dettaglio
  partita (`MatchDetailPanel.jsx` o dove si aprono "Apri"/"Oracle"), NON
  nella lista - chiama il nuovo endpoint e ricarica le previsioni della
  fixture.
- Test: nuovo caso in `prediction_snapshot_service_test.py` (`force=True` su
  una fixture finale gia' congelata produce una riga nuova, non riusa quella
  vecchia).

**Fatto**: `force=True` implementato in `resolve_predictions` (bypassa sia
il regime "congelato" per le finali sia il fingerprint-reuse per le NS, ha
priorita' su `allow_compute`). Nuovo `DashboardService.recompute_predictions`
+ endpoint `POST /dashboard/matches/{fixture_id}/recompute-predictions` +
schemi dedicati. Bottone "Ricalcola previsione" nel pannello di dettaglio
partita (`MatchDetailPanel.jsx`), con stato di caricamento ed errore
dedicati - ricarica il dettaglio dopo il ricalcolo. 7 nuovi test, suite
completa (829 test) verde, build frontend verificata.

## Riepilogo finale

Tutti e 4 i punti sono COMPLETATI dal lato codice (commit `ad50648`,
`caf12bb`, `d72b4f5`, `5ceba8b`, tutti su `feature/soccer-oracle-v2`),
**mergiati in `main` e deployati** (merge commit `06b1405`, 2026-09-10) -
in produzione da questo momento.

Il giro completo di `scripts/backfill_prediction_snapshots.py` (punto 3)
e' stato lanciato una volta, interrotto volontariamente a meta', e
**deciso di NON completarlo**: l'operatore ha chiarito che lo storico
profondo serve solo per il training (che non dipende da questa tabella) e
che l'esigenza reale e' "qualche giorno indietro + da oggi in poi", gia'
coperta interamente e per sempre dal job automatico del punto 2, ora
attivo in produzione. Nessuna azione manuale ulteriore richiesta.

## Note trasversali

- Nessuna modifica alla logica di invalidazione per fingerprint gia'
  esistente (partite NS/future) - questo prompt riguarda SOLO le partite
  gia' concluse.
- Suite completa da rieseguire prima di ogni commit, come sempre in questo
  progetto.
- Aggiornare `IMPLEMENTATION_LOG.md`/`CURRENT_TASK.md` a lavoro concluso
  (tutti e 4 i punti), non ad ogni singolo commit intermedio.
