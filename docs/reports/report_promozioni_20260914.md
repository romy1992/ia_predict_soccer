# Report promozioni 2026-09-14 — Under/Over 1.5, 2.5, 3.5

## Premessa operativa: disallineamento tra worktree

Il registry reale dei modelli (`best_models/registry`, montato dai container `soccer_api` e `soccer_scheduler` in produzione) vive fisicamente nel worktree `C:/Users/trott/git/ia_predict_soccer` (branch `main`), mentre `scripts/maintenance/promuovi_e_archivia.py` è arrivato con il pull sul worktree `C:/Users/trott/git/ia_predict_soccer_export` (branch `feature/soccer-oracle-v2`), la cui `best_models/` era vuota (cartella esclusa da `.gitignore`, quindi non condivisa tra worktree).

Il codice necessario alla ricostruzione deterministica dei modelli (feature engineering delle quote separate per esito, dataset `_export/*.csv`, `best_params`) esiste solo su `feature/soccer-oracle-v2`. Per eseguire lo script con codice e dati corretti insieme, si è proceduto così:
1. copiati in `ia_predict_soccer_export/best_models/` il registry reale (`registry/index.jsonl`, `registry/promotion_history.jsonl`, snapshot `.json`) e i 6 `.pkl` (champion + calibratore) dei 3 mercati da rifare, prelevati da `ia_predict_soccer/best_models/`;
2. eseguito lo script nel worktree con il codice;
3. ricopiati in `ia_predict_soccer/best_models/` (il registry reale, quello montato dai container) solo i file effettivamente prodotti: i 6 nuovi `.pkl`, `registry/index.jsonl`, `registry/promotion_history.jsonl`, i 3 nuovi snapshot `.json`, la cartella di backup `_backup_20260914/`.

## Stato prima (STATO ATTUALE)

```
mercato            feature  run_id                   file
----------------------------------------------------------------------------------
under_over_1_5          69  20260908T133703530786Z   under_over_1_5_champion.pkl
under_over_2_5          69  20260908T134618537938Z   under_over_2_5_champion.pkl
under_over_3_5          69  20260908T135451858813Z   under_over_3_5_champion.pkl
under_over_4_5          69  20260908T140412146780Z   under_over_4_5_champion.pkl
corners                  -  (nessuna production)
cards                    -  (nessuna production)
goal_no_goal            69  20260908T150042264590Z   goal_no_goal_champion.pkl
h2h                      -  (nessuna production)
```
Mercati da rifare: `under_over_1_5`, `under_over_2_5`, `under_over_3_5`. Gli altri (inclusi gli intoccabili) non sono stati toccati.

## Backup del registry

Eseguito prima di ogni scrittura, in `best_models/registry/_backup_20260914/`:
- `index.jsonl`: 20 righe (originali)
- `promotion_history.jsonl`: 13 righe (originali)

## Archiviazione

`file spostati: 0   righe di registry aggiornate: 0`

Nessun file è stato spostato in questa esecuzione. Motivo: la funzione di archiviazione gira prima della promozione dei nuovi modelli, quando i vecchi campioni (69 feature) sono ancora quelli in produzione — e per progetto lo script non archivia mai un modello che al momento è ancora in produzione. I vecchi `.pkl` restano quindi al loro posto in `best_models/` (nessun file cancellato, coerente con la regola "mai cancellare"); verranno spostati in `best_models/archivio/` alla prossima esecuzione dello script, quando risulteranno definitivamente superati.

## Riaddestramento e promozione

### under_over_1_5 — champion `random_forest`
- righe: 8.316, feature: 33, fold CV: 5, base rate: 0.7736
- calibrazione isotonic — prima: ece 0.1451, logloss 0.5925, brier 0.2024, auc 0.5817
- calibrazione isotonic — dopo: ece 0.0154, logloss 0.5384, brier 0.1775, auc 0.5830
- run registrato: `under_over_1_5_20260914T192131033741Z`
- esito gate: **promosso normalmente** (nessuna forzatura)

### under_over_2_5 — candidato imposto `logistic` (non il champion del gate)
- righe: 14.816, feature: 33, fold CV: 5, base rate: 0.5275
- calibrazione isotonic — prima: ece 0.0224, logloss 0.6723, brier 0.2399, auc 0.6105
- calibrazione isotonic — dopo: ece 0.0217, logloss 0.6828, brier 0.2399, auc 0.6097
- run registrato: `under_over_2_5_20260914T192135064272Z`
- esito gate: **promosso normalmente** (candidato imposto in partenza dallo script, non dal gate — vedi motivazione nello script: la logistica batte l'ensemble su AUC/log-loss/Brier/ECE, perdeva solo il punteggio composito)

### under_over_3_5 — champion `random_forest`
- righe: 8.315, feature: 33, fold CV: 5, base rate: 0.3157
- calibrazione isotonic — prima: ece 0.1321, logloss 0.6367, brier 0.2227, auc 0.5959
- calibrazione isotonic — dopo: ece 0.0147, logloss 0.5974, brier 0.2043, auc 0.5981
- run registrato: `under_over_3_5_20260914T192150036858Z`
- esito gate: **rifiutato dalla policy** → **forzatura applicata**

## Forzatura su under_over_3_5

Log dello script:
```
under_over_3_5: la policy ha rifiutato. Forzatura su decisione dell'operatore.
   run: under_over_3_5_20260914T192150036858Z
   forzata
```

Motivazione registrata nel registry (`actor: operator_request`), come da script:
- confronto con la production: selection_score candidato 0,6961 contro production 0,7008 (delta -0,0047) — la policy blocca su questo confronto, pur passando il gate di qualità su tutte e cinque le soglie (log_loss, brier, ece, auc, sample_size).
- **a favore della forzatura**: la production usa le quote LEGACY mescolate (media unica su Over e Under insieme); il candidato le separa per esito. Il candidato vince sull'AUC in tutte e tre le configurazioni provate (0,5974 contro 0,5867) e, dopo calibrazione, ha ECE molto migliore (0,0146 contro 0,1070 della production).
- **contro la forzatura**: il candidato perde su log-loss (0,6363 contro 0,6287) e Brier (0,2225 contro 0,2191) — proprio le metriche su cui pesa di più il selection_score.
- Decisione: forzatura consapevole dell'operatore, non correzione di un errore del gate. Entrambe le evidenze (pro e contro) sono registrate nel metadato della promozione per audit futuro.

## Stato dopo (STATO FINALE)

```
mercato            feature  run_id                   file
----------------------------------------------------------------------------------
under_over_1_5          33  20260914T192131033741Z   under_over_1_5_champion_20260914.pkl
under_over_2_5          33  20260914T192135064272Z   under_over_2_5_champion_20260914.pkl
under_over_3_5          33  20260914T192150036858Z   under_over_3_5_champion_20260914.pkl
under_over_4_5          69  20260908T140412146780Z   under_over_4_5_champion.pkl        (invariato)
corners                  -  (nessuna production)                                        (invariato)
cards                    -  (nessuna production)                                        (invariato)
goal_no_goal            69  20260908T150042264590Z   goal_no_goal_champion.pkl          (invariato)
h2h                      -  (nessuna production)                                        (invariato)
```

`index.jsonl`: 20 → 23 righe (+3). `promotion_history.jsonl`: 13 → 20 righe (+7, incluse le voci di promozione e quelle di audit della forzatura).

## Anomalia riscontrata e corretta: percorsi errati nel registry

La verifica finale eseguita nel worktree di lavoro (con solo i file dei 3 mercati copiati) segnalava 13 righe con file "inesistente", tutte relative a mercati non toccati (`under_over_4_5`, `goal_no_goal`, `corners`, `cards`): falso allarme dovuto al fatto che in quel worktree non era stata copiata l'intera `best_models/`. Verificato che tutti e 13 i file esistono realmente nel registry di produzione (`ia_predict_soccer/best_models`).

Riscontrato invece un problema reale: per le 3 righe appena promosse, `model_path` (e `calibrator_path`, `metadata_path`) erano stati scritti come percorso assoluto Windows del worktree di lavoro (es. `C:\Users\trott\git\ia_predict_soccer_export\best_models\under_over_1_5_champion_20260914.pkl`) invece del percorso container `/app/best_models/...` usato da tutte le altre righe. Se non corretto, al riavvio i container non avrebbero trovato il file per questi 3 mercati. Corretto manualmente `model_path` sul registry reale (quello in `ia_predict_soccer/best_models/registry/index.jsonl`) impostandolo a `/app/best_models/<nome-file>`, coerente con le righe già esistenti; `calibrator_path` e `metadata_path` sono stati allineati al percorso reale sulla macchina (questi due campi non sono usati dalla serving logic, solo `model_path` lo è).

## Verifica finale (sul registry reale, dopo la correzione)

```
righe che puntano a file inesistente: 0
```

Caricamento di prova dei 3 modelli promossi (dal registry reale):
```
under_over_1_5 -> /app/best_models/under_over_1_5_champion_20260914.pkl   feature: 33   caricato OK (CalibratedClassifierCV)
under_over_2_5 -> /app/best_models/under_over_2_5_champion_20260914.pkl   feature: 33   caricato OK (CalibratedClassifierCV)
under_over_3_5 -> /app/best_models/under_over_3_5_champion_20260914.pkl   feature: 33   caricato OK (CalibratedClassifierCV)
```

## Riavvio container (primo giro)

`docker restart soccer_api soccer_scheduler` eseguito. Entrambi ripartiti (`soccer_api` healthy, `soccer_scheduler` up). Nessun errore relativo a modelli/under_over nei log successivi al riavvio. Presente un errore preesistente e non correlato nei log di `soccer_api` (`alembic: Can't locate revision identified by 'head'`, oltre a un warning di riga CRLF in `start.sh`): non toccato in questo intervento, non impedisce l'avvio del servizio (health check 200 OK).

## Archiviazione manuale dei vecchi modelli (post-report)

Verifica successiva: rilanciando lo script, `archivia()` non sarebbe mai stata raggiunta, perché con i 3 mercati già a 33 feature la funzione `main()` esce subito su "Niente da fare" prima del passo di archiviazione — quindi i vecchi `.pkl` a 69 feature non si sarebbero mai spostati automaticamente. Su richiesta dell'operatore, replicata a mano la stessa logica di `archivia()` (stesso criterio: sposta il file, riscrive `model_path` come `/app/best_models/archivio/<nome>`) sul registry reale:

- spostati in `best_models/archivio/` i 6 file (3 champion + 3 calibratori a 69 feature) di `under_over_1_5`, `2_5`, `3_5`.
- **Trovata un'anomalia preesistente** durante l'operazione: nel registry esistevano altre 4 righe storiche (run del 2026-09-08 e 2026-09-13, precedenti all'introduzione dei nomi file con suffisso data) che puntavano agli **stessi nomi file condivisi** (es. più run diversi tutti con `model_path: under_over_2_5_champion.pkl`, perché prima d'ora ogni riaddestramento sovrascriveva il file invece di crearne uno nuovo). Spostando il file, queste 4 righe si sarebbero ritrovate rotte. Corrette anch'esse, reindirizzandole allo stesso percorso in `archivio/`.
- Nessun file cancellato, solo spostato.

Verifica finale ripetuta sul registry reale dopo l'archiviazione: **23 righe totali, 0 che puntano a file inesistente.**

## Allineamento del sito con i nuovi modelli

Controllo della dashboard (`/dashboard/bundle?target_date=2026-09-14`) dopo il primo riavvio: tutte le fixture non ancora giocate mostravano ancora i **vecchi `run_id`** (`..._20260908T...`) per i 3 mercati promossi. Causa: l'immagine Docker di `soccer_api`/`soccer_scheduler`/`soccer_web` era stata compilata il 2026-09-13 09:25, **prima** del commit `bb8a48d` (10:09 dello stesso giorno) che introduce il meccanismo di ricalcolo delle predizioni per la vista giornaliera. Il codice non è montato in bind nei container: un semplice restart non basta a far girare il codice nuovo.

Azioni eseguite, su autorizzazione esplicita dell'operatore:
1. `docker compose build api scheduler web` — rebuild delle 3 immagini dal codice attuale di `main` (build riuscita per tutte e 3).
2. `docker compose up -d --no-deps api scheduler web` — ricreati i container dalle nuove immagini. Tutti e 3 ripartiti sani (`soccer_api` healthy, `soccer_scheduler` up, `soccer_web` up).
3. `POST /jobs/prediction-snapshot-refresh` (endpoint "Ricalcola previsioni del giorno", ora presente) con `async_run: false` — ricalcolo sincrono sulla finestra di default. Esito: `fixtures_considered: 438`, `fixtures_upcoming: 318`, `fixtures_recently_finished: 120`, `predictions_resolved: 2500`, `errors: []`, `duration_seconds: 7053.99` (~1h 57m).
4. Verifica: interrogata di nuovo la dashboard di oggi — le 9 fixture non concluse mostrano ora, per tutti e 3 i mercati, il `run_id` nuovo (`..._20260914T192131033741Z`, `..._20260914T192135064272Z`, `..._20260914T192150036858Z`): 27/27 predizioni allineate, 0 con run_id vecchio.

Nota: le partite già concluse (`status` finale) mantengono la predizione storica "congelata" per scelta di progetto esplicita e documentata nel codice (`PredictionSnapshotService`) — non vengono e non devono essere ricalcolate nemmeno dopo una promozione, perché rappresentano "cosa prediceva il modello in quel momento".

## Riepilogo

| Mercato | Prima | Dopo | Esito gate |
|---|---|---|---|
| under_over_1_5 | 69 feature (legacy) | 33 feature | promosso |
| under_over_2_5 | 69 feature (legacy) | 33 feature | promosso (candidato imposto: logistic) |
| under_over_3_5 | 69 feature (legacy) | 33 feature | **rifiutato dal gate, forzato su decisione dell'operatore** |
| under_over_4_5, corners, cards, goal_no_goal, h2h | invariati | invariati | non toccati |

Nessun file cancellato in tutto il processo, solo spostato in `best_models/archivio/`. Registry reale verificato a 0 righe rotte. Immagini Docker di `api`, `scheduler`, `web` ricostruite dal codice attuale di `main` e ricreate. Dashboard verificata allineata ai nuovi modelli per tutte le fixture non concluse di oggi.
