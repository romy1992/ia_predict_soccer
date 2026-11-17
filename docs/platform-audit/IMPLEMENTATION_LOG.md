# Implementation Log — Soccer Oracle (platform audit)

Branch `cursor/platform-audit-evolution-e436`, base `origin/main` = `18eebfe`.
Nessun deploy, nessuna PR, nessun merge in `main`. Nessuna chiamata ai provider a
pagamento (API-Sports, The Odds API, RapidAPI/FotMob): tutto verificato con mock,
fixture e, per i log di urllib3, un server HTTP su loopback.

Questo file descrive lo stato **verificato il 2026-11-17**. I numeri dei test sono in
[TEST_REPORT.md](./TEST_REPORT.md).

## Storia del branch

| Data | Commit | Contenuto | Stato dopo la verifica del 2026-11-17 |
|---|---|---|---|
| 2026-11-11 | `226d03c` | Resilienza API-Sports (backoff esponenziale + jitter, circuit breaker), pausa quota collegata allo scheduler, settlement senza import a quota esaurita, trasparenza dashboard, benchmark 1X2, segnaposto in `config.env.example` | verificato, corretto dai commit sotto |
| 2026-11-11 | `eff8175` | Primi documenti platform-audit | sostituiti da questa versione (citavano log locali non piu' disponibili) |
| 2026-11-14 | `d395c34` | Pausa quota: baseline conservata, toggle dell'operatore ricordati | verificato, corretto da `fc0444d` |
| 2026-11-14 | `3b9d7e5` | Campo additivo `partially_calls_api_sports` | verificato, OK |
| 2026-11-14 | `39d7cf2` | The Odds API: errori di rete senza `apiKey` nel messaggio | verificato, completato da `4196864` |
| 2026-11-14 | `b514fd2` | Test di concorrenza/backoff delle primitive di resilienza | verificato, OK |
| 2026-11-14 | `d28d486` | Motivo esplicito per mercati senza modello, feature riempite a 0 dichiarate | verificato, corretto da `1b08cf4`, `903d3f9`, `56d8770` |
| 2026-11-17 | `fc0444d` | Pausa quota: letture strette, sincronizzazione serializzata | nuovo |
| 2026-11-17 | `3643636` | Esito dell'import dal provider dichiarato in settlement e aggiornamento puntuale | nuovo |
| 2026-11-17 | `572b535` | Test del contratto di `ApiSportsProvider` verso i job | nuovo (solo test) |
| 2026-11-17 | `4196864` | The Odds API: `apiKey` mascherata nei log di urllib3 | nuovo |
| 2026-11-17 | `1b08cf4` | `xg_available` falso quando la media xG e' zero | nuovo |
| 2026-11-17 | `903d3f9` | Il motivo "nessun modello" viene prima di quote e partita | nuovo |
| 2026-11-17 | `56d8770` | `features_defaulted` anche sugli snapshot riusati; test sulle righe servite | nuovo |
| 2026-11-17 | `ce7d797` | Timeout The Odds API da ambiente letto senza far fallire l'import | nuovo |
| 2026-11-17 | `3d8ea22` | Commento dello scheduler con i numeri reali | nuovo (solo commento) |

## 1. Pausa quota collegata allo scheduler

- `src/jobs/scheduler.py::_run_if_due` chiama `_sync_quota_pause_if_needed(job_id)` prima di
  `is_job_enabled`, solo per i 6 job di `QUOTA_SENSITIVE_JOB_IDS` (`data_daily_refresh`,
  `data_future_sync`, `data_sync_live`, `data_sync_today`, `player_cards_predictions`,
  `scorer_predictions`), al massimo una volta ogni 60 s per processo
  (`_QUOTA_SYNC_MIN_INTERVAL_SECONDS`, heartbeat 30 s).
- Lo stato quota e' quello salvato sul bucket (`best_models/api_quota_state.json`) dalle
  chiamate dati: nessuna chiamata al provider. Test:
  `quota_pause_test.py::test_la_sincronizzazione_legge_solo_il_bucket` (3 letture dal bucket,
  0 richieste HTTP).
- Difetti trovati e corretti (`fc0444d`):
  - una lettura fallita del bucket diventava un valore vuoto e lo stato veniva riscritto da li':
    Impostazioni illeggibili -> default salvati (toggle dell'operatore azzerati), pausa
    illeggibile -> baseline sovrascritta con lo stato gia' spento, quota illeggibile -> pausa
    tolta. Ora le letture che precedono una scrittura sono strette e, se falliscono, non si
    scrive nulla (vale anche per `update_job_settings`, cioe' `POST /settings/jobs`: errore
    esplicito invece dei default salvati in silenzio);
  - la baseline viene salvata prima di spegnere i job; se la scrittura fallisce i job non
    vengono toccati;
  - i job dello stesso tick girano su thread diversi: il secondo job sensibile saltava il
    riallineamento in corso e partiva con lo stato pre-pausa. Ora aspetta (`_quota_sync_lock`
    nello scheduler, `_QUOTA_SYNC_LOCK` in `job_settings.py`);
  - il ripristino ignora job della baseline non piu' esistenti (prima bloccavano il
    ripristino per sempre).
- I toggle espliciti dell'operatore durante la pausa (`remember_operator_toggles_during_pause`,
  chiamata da `POST /settings/jobs`) aggiornano la baseline: un job spento a mano non viene
  riacceso al reset.

## 2. Settlement e aggiornamento puntuale

- `data_settlement` resta `calls_api_sports=False` (mai in pausa automatica) con il campo
  additivo `partially_calls_api_sports=True` (`GET/POST /settings/jobs`, `jobs[]`); lo schema
  `JobSettingRow` ha default `False`, il frontend legge solo `calls_api_sports`.
- `SettlementService.run_settlement`: a quota esaurita oggi salta l'import
  (`import_report = {"skipped": true, "reason": "api_sports_quota_exhausted_today"}`,
  `provider_import_skipped: true`) e riconcilia comunque dal DB.
- Nuovo (`3643636`): `download_import_matches` non solleva a quota esaurita o a provider giu'
  (segna `quota_exceeded` / `failed_leagues`). Campo additivo `provider_status`
  (`ok` / `quota_exhausted` / `unavailable`) nel report del settlement e in quello di
  `run_single_fixture_refresh` (`POST /dashboard/matches/{fixture_id}/refresh`), che riceve
  anche `errors["import"]`. L'aggiornamento puntuale non passa da `_run_if_due`: l'auto-pausa
  non lo disattiva; a quota esaurita le fasi locali (settlement, predizioni) girano comunque.

## 3. Provider

- `ApiSportsProvider` (`src/service_ia/pre_processing/api_sports_provider.py`): default
  `timeout_seconds=20`, `max_retries=3`, `retry_backoff_seconds=1.5`,
  `retry_backoff_max_seconds=30`, `circuit_failure_threshold=5`, `circuit_recovery_seconds=60`,
  `Retry-After` limitato a 120 s. Backoff "equal jitter" (`resilience.py::backoff_delay`).
  Quota giornaliera esaurita: 1 sola chiamata, nessun retry, `ApiSportsQuotaExceededError`.
  Errori di richiesta e 4xx non aprono il breaker. A retry esauriti resta
  `ApiSportsUnavailableError` (`ApiSportsCircuitOpenError` ne e' una sottoclasse). La chiave
  viaggia solo nell'header `x-apisports-key`. Test aggiunti in `572b535`.
- The Odds API (`src/service_ia/utility/request_api.py` e copia storica
  `backend/service_ia/utility/request_api.py`): timeout 30 s, messaggio d'errore senza `apiKey`.
  Nuovo (`4196864`): urllib3 a DEBUG logga la riga di richiesta con la query string; con il
  root logger a DEBUG la chiave finiva nei log (verificato su loopback). Filtro sul logger
  `urllib3.connectionpool` che sostituisce il valore con `***`.

## 4. Trasparenza del serving (campi additivi, probabilita' invariate)

| Campo | Dove | Endpoint |
|---|---|---|
| `markets_missing_reasons` (mercato -> motivo) | riga partita | `GET /dashboard/day`, `GET /dashboard/live`, `GET /dashboard/bundle` (`day.rows[]`, `live.rows[]`), `GET /dashboard/overview` (`day_highlights[]`, `live_preview[]`), `GET /dashboard/match/{fixture_id}` (`fixture`) |
| `data_quality` (`odds_available`, `team_statistics_available`, `xg_available`; `null` se la partita non e' nel DB) | riga partita | come sopra |
| `computed_at` (ISO UTC dello snapshot) | `predictions[<mercato>]` | come sopra e `POST /dashboard/matches/{fixture_id}/recompute-predictions` (`predictions`) |
| `features_defaulted`, `features_defaulted_count` | `predictions[<mercato>]`, solo se il frame e' stato costruito | come sopra |
| `partially_calls_api_sports` | `jobs[]` | `GET/POST /settings/jobs` |
| `provider_status`, `provider_import_skipped` | report del job | settlement (`/jobs/settlement`, job history), `POST /dashboard/matches/{fixture_id}/refresh` |

Valori di `markets_missing_reasons`, in ordine di precedenza: `no_model_registered`,
`model_retired`, `model_file_missing`, `match_not_in_db`, `no_odds_row`, `no_odds_for_market`,
`not_computed`. `markets_unavailable` (badge "In coda") e' invariato.

Correzioni del 2026-11-17: `xg_available` era vero anche con media xG = 0, cioe' xG mai
rilevato (`calculate_mean` azzera l'xG assente prima della media); il motivo del modello veniva
nascosto da `no_odds_for_market`; `features_defaulted` spariva al secondo accesso.

## 5. Benchmark 1X2 offline

Vedi la sezione "Benchmark" di [TEST_REPORT.md](./TEST_REPORT.md).

## 6. Segreti

- `backend/properties/config.env.example`: la password dell'URL di esempio e' un segnaposto
  (`<PASSWORD_LOCALE>`, riga 32). Su `main` era gia' `CHANGE_ME`: il file non ha mai contenuto
  una password reale (`git log --follow`: due soli commit, `5601ad2` e `226d03c`).
- Nessun segreto dall'aspetto reale nei file tracciati (`*.example`, doc, test) al commit
  `3d8ea22` (scansione per assegnazioni/URL con credenziali/token, valori mai stampati).
  `docs/soccer_oracle_v2_detailed/IMPLEMENTATION_LOG.md:432` riporta host e porta del DB
  Railway (password oscurata).
- Nella storia git raggiungibile da `main` resta `properties/config.env` (presente almeno in
  `f59f14f` del 2026-09-19, primo commit del clone locale superficiale; rimosso in `560d4c9`;
  `git log --all` lo trova in 11 commit dal 2025-07-17) con valori non vuoti per
  `API_SPORTS_KEY`, `API_ODDS_KEY_BASE`, `API_ODDS_KEY`, `TELEGRAM_TOKEN`, `STRIPE_SECRET_KEY` e
  un `DATABASE_URL` Railway con password di 32 caratteri; sul ramo remoto
  `feature/config_drive` (`cddb54b`) restano `properties/client_secret.json` e un JSON di
  service account. Toglierli dal repository non li toglie dalla storia.
  **Proposta: ruotare tutte queste credenziali.** (Valori mai stampati: verificati solo nomi e
  lunghezze.)

## 7. Indizi

Vedi [TEST_REPORT.md](./TEST_REPORT.md), sezione "Indizi".

## Dati versionati

- `backend/src/backup_db`: 14 file tracciati = 12 dump PostgreSQL (11 `.backup` + 1 `.sql`
  da 35.750.929 byte) + 2 file `.sql` di query (3.874 e 2.005 byte); 434.438.069 byte in tutto
  (`du -sb backend/src/backup_db`, 415M con `du -sh`; comandi del 2026-11-17). Non toccati.
- Aggiunti dal branch: `backend/scripts/analysis/_export/top5_matches_for_benchmark.csv.gz`
  (635.277 byte, 19.683 partite, necessario per rieseguire il benchmark senza un Postgres) e
  `backend/scripts/analysis/walk_forward_1x2_benchmark_result.json`.

## Rischi residui (non risolti)

- Pausa quota: il lock e' per processo; fra i container `api` e `scheduler` resta una finestra
  di pochi istanti senza esclusione reciproca. Con il bucket illeggibile la pausa non viene
  applicata ne' tolta fino alla lettura successiva riuscita.
- `POST /settings/jobs` con bucket illeggibile ora risponde con errore invece di salvare i
  default: comportamento diverso solo nel caso di guasto.
- `ApiSportsProvider`: una risposta 4xx senza campo `errors` restituisce `[]`, indistinguibile
  da "nessun dato" (preesistente; non apre il breaker e non viene ritentata). Il breaker e' per
  istanza: `live_data_service.py:186` crea un'istanza propria. A rate-limit al minuto esaurito
  negli header resta uno `sleep(60)` per risposta (`api_sports_provider.py:388`, preesistente).
  Caso peggiore per una singola richiesta con i default: 3 x 20 s di timeout + 2 attese fino a
  120 s (`Retry-After`) + eventuali `sleep(60)`.
- `features_defaulted` non e' salvato sullo snapshot: assente quando lo snapshot e' servito
  senza ricostruire il frame (partite concluse, liste storiche). `xg_available` non riconosce
  una media xG positiva ma calcolata con alcune partite senza xG.
- h2h, dc e 1X2 non riconciliati nel dettaglio partita (vedi indizio (b) in TEST_REPORT.md).
- Credenziali nella storia git: vanno ruotate (sezione 6).
