# Documentazione progetto Soccer Oracle

## 1) Obiettivo di prodotto
Costruire una dashboard unica per il calcio che unisce:
- tutte le partite del giorno
- live match center
- previsioni multi-mercato per ogni match
- decisione pick con badge `PLAY / BORDERLINE / NO BET`
- suggerimenti schedina automatici
- training modelli per mercato con metriche e versioning

Questa documentazione descrive lo stato attuale (as-is), il target (to-be) e il piano operativo.

---

## 2) Stato attuale del repository (as-is)

### Backend
- `src/api/main.py`: avvio FastAPI, routing API, CORS.
- `src/api/dashboard_service.py`: logica dashboard (overview/live/day/match detail), aggregazioni e predizioni.
- `src/api/schemas.py`: schemi risposta/request API.
- `src/repository/`: accesso dati DB e query helper.
- `src/jobs/`: job operativi (`import`, `retrain`, scheduler, history).

### Training/ML
- Pipeline e utility in:
  - `service_ia/training/`
  - `src/service_ia/training/`
- Presenza di pipeline storiche e script di fit, ma da consolidare in una pipeline unica per mercato.

### Frontend
- `frontend/src/App.jsx`: app principale a pagina unica con menu laterale e sezioni.
- `frontend/src/api.js`: client API frontend.
- `frontend/src/styles.css`: tema UI e layout.

### Ops e setup
- Docker/API: `docker-compose.yml`, `Dockerfile.api`.
- Migrazioni DB: `alembic/`.
- Smoke API: `scripts/smoke_api.ps1`.

### 2.1 Audit DB reale (controllo approfondito)
Il progetto attualmente convive con piu istanze PostgreSQL locali. Il controllo completo ha evidenziato:

- `localhost:5432` -> PostgreSQL 17.5 (`C:/Program Files/PostgreSQL/17/data`), DB storico reale.
- `localhost:5433` -> PostgreSQL 18.1 (`C:/Program Files/PostgreSQL/18/data`), solo DB `postgres` (non usato dal progetto).
- Docker `api`/`scheduler` -> **nessun Postgres containerizzato** (rimosso da `docker-compose.yml`). Usano `DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/match_db`, cioe' la STESSA istanza storica dell'host raggiunta tramite l'hostname speciale di Docker Desktop.

Conclusione (aggiornata 2026-09-03): risolto il mismatch. Non esiste piu' un DB Docker separato: sia l'esecuzione locale sia i container `api`/`scheduler` leggono/scrivono sullo stesso Postgres storico dell'host (`localhost:5432`, >47k righe `match`). Verificato via `GET /health/database` con l'API in esecuzione dentro Docker.

### 2.2 Volumi reali del DB storico (`localhost:5432`, `match_db`)

Tabelle `public`:
- `match`: `47,228` righe
- `odds`: `20,402` righe
- `statistics`: `81,244` righe

Copertura relazioni:
- `odds` distinct `id_match`: `20,402`
- `statistics` distinct `id_match`: `43,230`
- record orfani `odds -> match`: `0`
- record orfani `statistics -> match`: `0`

Finestra temporale osservata in `match.date_match`:
- minimo campione: `2015-01-01T12:45:00+00:00`
- massimo campione: `2026-08-31T19:30:00+00:00`

Distribuzione stato (top):
- `FT`: `47,177`
- `PEN`: `28`
- `AET`: `22`
- `ABD`: `1`

### 2.3 Schema reale utile al training (DB storico)

`match` (19 colonne)
- PK: `id_match_fk`
- campi chiave aggiuntivi presenti: `current_league`, `league_match`, `mean_statistics`, `status`
- altri campi: `id_fixture`, squadre, data, lega, stagione, round, referee

`odds` (13 colonne)
- PK: `id_odds_fk`
- FK: `id_match -> match.id_match_fk`
- mercati JSON: `h2h`, `under_over_1_5`, `under_over_2_5`, `under_over_3_5`, `under_over_4_5`, `under_over_home_away`, `goal_no_goal`, `corners`, `cards`, `dc`

`statistics` (21 colonne)
- PK: `id_statistics_fk`
- FK: `id_match -> match.id_match_fk`
- campi risultato e eventi + JSON estesi (`shots`, `passes`, `form`, `for_`, `against`, `preview_matches`, `comparison`, `generic_statistics`, `predict`)

### 2.4 Copertura mercati odds (righe non-null in `odds`)
- `h2h`: `20,374`
- `under_over_1_5`: `20,368`
- `under_over_2_5`: `20,374`
- `under_over_3_5`: `20,368`
- `under_over_4_5`: `20,368`
- `goal_no_goal`: `20,368`
- `dc`: `20,374`
- `corners`: `20,304`
- `cards`: `18,675`

Questo e un segnale molto buono per training mercato-specifico (inclusi U/O 1.5-4.5).

### 2.5 Incoerenza tra ambienti (RISOLTO 2026-09-03)
In precedenza esisteva un mismatch operativo tra:
- schema/volumi del DB storico locale (`localhost:5432`),
- schema del DB usato dal compose Docker (`db:5432`), separato e con struttura differente (dataset quasi vuoto).

Fix applicato:
- rimosso il servizio Postgres containerizzato da `docker-compose.yml`;
- `api`/`scheduler` ora puntano a `postgresql://postgres:postgres@host.docker.internal:5432/match_db` (stesso host storico usato in locale);
- applicata su `localhost:5432` la migration mancante (`f3a9c1d8e2b7`, `prediction_ledger`), gia' allineata alla head del codice;
- verificato con `GET /health/database` dal container `soccer_api`: `host=host.docker.internal`, `match=47228`, `statistics=81244`, `odds=20402`.

Impatto: API/dashboard in Docker e training locale condividono ora la stessa sorgente dati, nessuna divergenza di metriche/output.

### 2.6 Backup disponibili nel repo
In `src/backup_db/` sono presenti dump storici `.backup` e `.sql` (circa 33-40 MB), utili per restore/replica controllata in ambiente Docker quando serve uniformare runtime e training.

### 2.7 Profiling dettagliato DB storico (`localhost:5432`)

Completezza tabella `match` (su `47,228` righe):
- `id_fixture` valorizzato: `44,682` (`94.61%`)
- `season` valorizzata: `44,682` (`94.61%`)
- `current_league` valorizzata: `39,466` (`83.56%`)
- `mean_statistics` valorizzato: `41,793` (`88.49%`)

Distribuzione `status` in `match`:
- `FT`: `47,177` (`99.89%`)
- `PEN`: `28` (`0.06%`)
- `AET`: `22` (`0.05%`)
- `ABD`: `1` (`0.00%` arrotondato)

Distribuzione stagioni (`match.season`):
- copertura principale: `2014` -> `2026`
- righe con `season = NULL`: `2,546`
- picchi volume: `2025` (`5,276`), `2023` (`4,816`), `2024` (`4,253`), `2021` (`4,203`)

Copertura mercati in `odds` (su `20,402` righe):
- `h2h`: `20,374` (`99.86%`)
- `under_over_1_5`: `20,368` (`99.83%`)
- `under_over_2_5`: `20,374` (`99.86%`)
- `under_over_3_5`: `20,368` (`99.83%`)
- `under_over_4_5`: `20,368` (`99.83%`)
- `goal_no_goal`: `20,368` (`99.83%`)
- `dc`: `20,374` (`99.86%`)
- `corners`: `20,304` (`99.52%`)
- `cards`: `18,675` (`91.54%`)

Copertura tabella `statistics`:
- principali blocchi (`score_ft`, `shots`, `passes`, `form`, `generic_statistics`, `predict`) risultano popolati su tutte le `81,244` righe.

Top leghe per volume (snapshot):
- id `94`, `140`, `39`, `136`, `135`, `203`, `61`, `88`, `78` con migliaia di record ciascuna
- presenti anche record con metadati lega mancanti/parziali (`current_league` o `title_league` non sempre coerenti)

Valutazione training readiness:
- dataset storico adatto a training multi-mercato, incluso U/O `1.5`, `2.5`, `3.5`, `4.5`
- prima del training serve un passaggio di normalizzazione anagrafica lega/stagione e gestione `NULL`

### 2.8 Piano di allineamento ambienti dati (ATTUATO 2026-09-03)
Per evitare mismatch tra API, dashboard e training:

1. ~~scegliere una sola sorgente ufficiale~~ -> scelto `localhost:5432` (Postgres storico host, >47k match)
2. ~~far puntare `api` e `scheduler` alla stessa sorgente~~ -> fatto, `DATABASE_URL` unica via `host.docker.internal:5432` in Docker
3. ~~validare schema condiviso~~ -> fatto, `alembic upgrade head` allineato su `localhost:5432` (revisione `f3a9c1d8e2b7`)
4. bloccare i job ML finche i conteggi base non coincidono tra ambiente runtime e ambiente training -> verificato via `/health/database`
5. documentare il profilo con un report di controllo versionato -> vedi `IMPLEMENTATION_LOG.md` (voce 2026-09-03, disattivazione DB Docker)

---

## 3) Frontend attuale (immagini reali)
Le immagini sotto sono screenshot del frontend corrente.

### 3.1 Dashboard live
![Dashboard live](images/frontend-dashboard.png)

### 3.2 Partite in diretta
![Partite in diretta](images/frontend-live.png)

### 3.3 Partite del giorno
![Partite del giorno](images/frontend-today.png)

### 3.4 Storico previsioni
![Storico previsioni](images/frontend-predictions.png)

### 3.5 Operazioni ML
![Operazioni ML](images/frontend-ops-ml.png)

---

## 4) Cosa esiste gia a FE (lettura funzionale)
Menu laterale attuale:
- Dashboard live
- Partite in diretta
- Partite del giorno
- Storico previsioni
- Operazioni ML

Funzioni gia presenti:
- filtro data/mercato/ricerca
- refresh periodico dashboard
- tabella partite con previsioni per mercato
- dettaglio match (quando disponibile) con timeline + quote medie + decision card
- pagina operativa per import/retrain e storico jobs

Nota: il FE e ancora organizzato per sezioni tecniche. Il target e una UX centrata su decisione betting e schedina.

---

## 5) Target funzionale richiesto (to-be)

### 5.1 Match Center
- tutte le partite del giorno, con live in evidenza
- prediction per mercato su ogni match
- `p_model`, quota media, `p_implied`, `edge`
- badge: `PLAY / BORDERLINE / NO BET`

### 5.2 Mercati MVP+
- `1X2`
- `Double Chance`
- `BTTS` (Goal/No Goal)
- `Under/Over 1.5`
- `Under/Over 2.5`
- `Under/Over 3.5`
- `Under/Over 4.5`

### 5.3 Schedina Oracle
- pool pick automatico da badge
- combinazioni 2/3/4 eventi
- filtri anti-correlazione
- ranking per EV, probabilita stimata e rischio

### 5.4 ML Lab
- training separato per mercato
- calibrazione probabilita
- report metriche e versioning modello

---

## 6) Requisito chiave Under/Over multi-linea
Serve coerenza tra linee goal:
- `P(Over 1.5) >= P(Over 2.5) >= P(Over 3.5) >= P(Over 4.5)`

Strategia consigliata:
1. modelli separati per linea (`1.5`, `2.5`, `3.5`, `4.5`)
2. calibrazione probabilita per ogni linea
3. monotonicity correction post-predizione
4. report violazioni prima/dopo correzione

Confrontare anche una seconda opzione (modello ordinale/constrained) e scegliere in base a metriche reali su DB.

---

## 7) Dati e training: cosa serve per fare bene

### 7.1 Audit dati iniziale (obbligatorio)
- mappa tabelle e campi reali
- coverage odds per mercato/outcome
- qualita dati (null, duplicati, incoerenze)
- controllo leakage temporale

### 7.2 Feature set
- pre-match: forma, performance casa/trasferta, trend goal, forza relativa
- odds: implied probability normalize, dispersione bookmaker, drift quote
- live (pipeline separata): minuto, scoreline, eventi chiave (red card, ecc.)

### 7.3 Metriche
- LogLoss, Brier, AUC (dove applicabile)
- calibrazione (reliability/ECE)
- betting KPI: hit-rate e ROI simulato per label

---

## 8) Gap principali attuali
- pipeline ML da unificare e rendere robusta per tutti i mercati richiesti
- policy soglie per badge da tarare per mercato (non globale)
- motore schedina da completare (combinazioni + ranking)
- monitoraggio continuo performance prediction/pick

---

## 9) Piano implementativo consigliato

### Sprint 1 - Audit e fondazioni
- audit DB completo
- mapping robusto odds -> mercati/outcome
- dataset builder unico per mercato

### Sprint 2 - Training core mercati
- training `1X2`, `BTTS`, `U/O 1.5/2.5/3.5/4.5`
- calibrazione e benchmark
- monotonicity check/correction U/O

### Sprint 3 - Decision Engine e API
- calcolo edge e badge per mercato
- endpoint consolidati dashboard + dettaglio
- storico predizioni coerente

### Sprint 4 - Schedina Oracle
- generazione pool pick
- builder combinazioni 2/3/4
- ranking EV/prob/rischio

### Sprint 5 - Hardening
- tuning soglie finale
- test end-to-end
- report performance e limiti noti

---

## 10) Definition of Done
Il progetto puo essere considerato pronto quando:
- Match Center mostra tutte le partite del giorno con prediction mercati target (se dati presenti)
- probabilita U/O finali rispettano monotonicita
- ogni pick ha motivazione numerica (`p_model`, quota, `edge`, badge)
- Schedina Oracle genera combinazioni con filtri rischio/correlazione
- ML Lab espone metriche/versioni e job history verificabile
- test unit/integration/smoke passano

---

## 11) File chiave da presidiare
- Backend:
  - `src/api/main.py`
  - `src/api/dashboard_service.py`
  - `src/api/schemas.py`
  - `src/repository/`
  - `src/jobs/`
- Frontend:
  - `frontend/src/App.jsx`
  - `frontend/src/api.js`
  - `frontend/src/styles.css`
- ML:
  - `service_ia/training/`
  - `src/service_ia/training/`
- Test:
  - `tests/`

---

## 12) Come sono state generate le immagini FE
Script usato:
- `scripts/capture_frontend_screenshots.mjs`

Cartella output:
- `docs/images/`

Comando usato:
```powershell
node "C:\Users\trott\.claude\skills\browser-automation\browser.mjs" "http://localhost:3000/" --script "C:\Users\trott\git\ia_predict_soccer\scripts\capture_frontend_screenshots.mjs"
```

---

## 13) Prossimo passo operativo suggerito
Aprire uno sprint tecnico dedicato a:
- allineamento sorgente dati unica tra API Docker e DB storico
- audit DB + dataset builder per mercati richiesti
- training U/O multi-linea con vincolo di coerenza
- validazione soglie badge per mercato







