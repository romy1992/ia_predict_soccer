# Test Report — Soccer Oracle (platform audit)

Tutti i numeri sotto vengono da comandi eseguiti il **2026-11-17** in questa sessione.
Nessuna rete verso i provider: test con mock/fixture.

## Suite completa: main e branch

Comando (dalla cartella `backend`):

```
LOKY_MAX_CPU_COUNT=2 OMP_NUM_THREADS=2 nohup timeout 7200 \
  python3 -m pytest tests -q -p no:cacheprovider -o addopts="" --durations=25 > <log> 2>&1 &
```

Le due suite girano in parallelo sulla stessa macchina (4 CPU): le due variabili d'ambiente
dividono i core fra le due esecuzioni e `--durations` stampa solo i tempi. Il primo tentativo
della mattina (`timeout 3000`, senza divisione dei core, load average 14-20 su 4 CPU) non e'
arrivato in fondo: la run del branch su `d28d486` e' stata interrotta dopo 26 minuti con 293
superati e 0 falliti, quella di main era ancora sui test Corners quando la macchina e' stata
riavviata. I risultati sotto sono della seconda esecuzione, avviata alle 13:03 UTC. Dopo
`3d8ea22` il branch cambia solo questi due file `.md`, che nessun test legge.

| | commit | test raccolti | superati | falliti | durata |
|---|---|---|---|---|---|
| `origin/main` | `18eebfe` | 1.375 | 1.358 | 17 | 2.388,82 s |
| branch | `3d8ea22` | 1.498 | 1.481 | 17 | 2.404,52 s |

**Stessi 17 test falliti** su main e sul branch (confronto con `comm` delle due liste):
nessun fallimento nuovo introdotto dal branch; +123 test, tutti verdi.

Falliti (preesistenti su `main`, non modificati):

- `dashboard_service_test.py::TestDashboardService::` `test_get_day_matches_from_api_feed`,
  `test_get_day_matches_loads_snapshots_once_for_all_fixtures`, `test_get_live_matches_filter`,
  `test_get_match_detail_adds_correct_for_finished_match`,
  `test_recompute_predictions_calls_predict_fixture_with_force_true`
- `download_match_test.py::` `TestDownloadMatch::test_download_import_matches_continue_on_fixture_error`,
  `TestDownloadMatch::test_download_import_matches_insert_report`,
  `TestDownloadMatch::test_download_import_matches_update_report`,
  `TestDownloadMatch::test_snapshot_agganciati_all_id_restituito_dall_upsert`,
  `TestDownloadImportMatchesScoreWithoutStatistics::test_score_saved_even_when_statistics_endpoint_returns_nothing`,
  `TestDownloadImportMatchesLegaNonDisponibile::test_lega_non_disponibile_registrata_e_giro_continua`
- `frontend_match_center_test.py::test_mobile_view_keeps_void_odd_and_situation_visible`
- `job_status_endpoint_test.py::TestJobStatusEndpoint::test_get_job_returns_progress_summary`
- `markets_endpoint_test.py::TestMarketsEndpoint::test_excludes_markets_whose_production_is_archived`
- `model_legend_endpoint_test.py::TestModelLegendEndpoint::` `test_covers_over_signal_line_market_signal_and_generic_bucket`,
  `test_reflects_real_production_state_per_market`
- `model_paths_test.py::TestDestinationSubdir::test_gli_altri_mercati_restano_nella_radice`

Le cause sono eterogenee e non sono state corrette (non era nel perimetro); prime righe
d'errore, per esempio: `'h2h' not found in {}` (il test simula `registry.list_markets`, il
codice usa `list_active_markets`), `0 != 1` sui contatori di `download_import_matches`,
`401 != 404`, `'goal_no_goal' != ''`, testo `Situazione` non trovato nel sorgente del frontend.
Non verificato se alcuni richiedano un Postgres locale.

I documenti dell'11/11 dichiaravano 17 falliti / 1.358 superati su main e 17 / 1.420 sul branch
di allora (`226d03c`), citando log locali non piu' disponibili: il dato di main e'
confermato; quello del branch e' superato (oggi 1.481 superati a `3d8ea22`).

Test piu' lenti (identici su main e branch): `cards_market_test.py` e `corners_market_test.py`,
sei test fra 335 e 386 s ciascuno (`TestTrain*AllLines::test_report_has_metrics_per_line`,
`TestRun*Benchmark::test_end_to_end_benchmark_without_saving`,
`TestRun*Benchmark::test_uses_provided_frame_directly_ignoring_matches`).

## Test aggiunti o modificati dal branch

`quota_pause_test.py`, `scheduler_test.py`, `settlement_service_test.py`,
`api_sports_provider_test.py`, `resilience_test.py`, `request_api_odds_test.py`,
`serving_transparency_test.py`, `tests/analysis/walk_forward_1x2_benchmark_test.py`.
Prima di correggere il codice il 2026-11-17, i nuovi test della pausa quota e dello scheduler
fallivano sul codice di `d28d486` (7 falliti su 63 in quei due file): dimostrano i difetti
corretti in `fc0444d`.

Verifica su loopback (nessun provider) della fuga della chiave nei log di urllib3: prima della
correzione la riga era `"GET /v4/sports?apiKey=<chiave> HTTP/1.1" 200`, dopo e'
`"GET /v4/sports?regions=eu&apiKey=*** HTTP/1.1" 200`.

## Benchmark walk-forward 1X2 (offline, top 5 campionati)

Riproducibilita': `python scripts/analysis/walk_forward_1x2_benchmark.py --output <file>`
(dalla cartella `backend`, il comando documentato nello script, con `--output` verso un file
temporaneo per non sovrascrivere quello committato), rieseguito il 2026-11-17 (2.541,5 s su
macchina condivisa). Confrontato campo per campo con
`backend/scripts/analysis/walk_forward_1x2_benchmark_result.json`: **0 differenze** esclusi
`generated_at`, `elapsed_seconds` ed `environment`, con librerie diverse da quelle dell'11/11
(oggi Python 3.13.18, numpy 2.5.3, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.18.1; allora
3.12.3, 2.5.4, 3.1.1, 1.9.1, 1.18.1). Il file committato non e' stato rigenerato.

Dati: `backend/scripts/analysis/_export/top5_matches_for_benchmark.csv.gz` (635.277 byte,
sha256 `c0735cf7...0539da`), 19.682 partite di campionato concluse con risultato dal
2015-01-01 al 2026-05-26 (Premier League, La Liga, Serie A, Bundesliga, Ligue 1; Ligue 1 2022
assente dal dump), estratte da `dump-match_db-20260803093803-08-26.backup` con
`export_top5_matches_for_benchmark.py`. 9.918 partite con quote 1X2, dalla 2020-06-11.

Protocollo (verificato leggendo il codice e da `tests/analysis/walk_forward_1x2_benchmark_test.py`,
classe `TestNoFutureInformation`): ordine cronologico; Elo e forma calcolati con le sole
partite dei giorni precedenti; Poisson e Dixon-Coles rifittati a inizio mese per campionato
sulle sole partite precedenti; modelli supervisionati su finestra espandibile, 20 fold
trimestrali dal 2021Q3 al 2026Q2; iperparametri scelti solo su 2018-07-01..2021-07-01 (Elo
`k15_hfa40_gd`, decadimento `xi=0.002`/giorno; `rho` di Dixon-Coles medio -0,0633, fra -0,1934
e 0,0491). Mercato no-vig: per ogni bookmaker con tutti e tre gli esiti si normalizzano le
probabilita' implicite, poi media fra bookmaker (5,34 bookmaker e overround 4,32% in media sulle
partite di test).

Quali quote: una sola rilevazione per partita, istante non registrato, quindi **non garantito
che siano quote di chiusura**. Sulle 8.118 partite di test quotate, 6.410 hanno quote importate
da The Odds API e 1.708 da API-Sports. Per The Odds API il codice di import storico (oggi
configurato per altri campionati) interroga snapshot nei giorni 1, 11 e 21 del mese alle 10:05
UTC (`backend/src/service_ia/pre_processing/df_odds_service.py:67-91`, `:198-201`): se le quote
dei top 5 sono state importate cosi', possono precedere il calcio d'inizio anche di giorni
(dedotto dal codice, non verificato sui dati). Il mercato di riferimento e' quindi probabilmente piu' debole della chiusura.

### 1X2 sulle partite di test quotate (n = 8.118, riferimento: mercato no-vig)

| modello | n | log loss | Brier (multiclasse) | RPS | accuratezza | ECE top-label | delta log loss vs mercato [IC95 bootstrap appaiato] | fold migliori del mercato |
|---|---|---|---|---|---|---|---|---|
| market_recalibrated | 8118 | 0.97388 | 0.57960 | 0.19607 | 0.5346 | 0.0072 | -0.00021 [-0.00144, +0.00105] | 11/20 |
| market_no_vig | 8118 | 0.97408 | 0.57968 | 0.19615 | 0.5350 | 0.0126 | riferimento | - |
| rf_with_odds | 8118 | 0.97934 | 0.58253 | 0.19705 | 0.5312 | 0.0051 | +0.00525 [+0.00256, +0.00793] | 5/20 |
| lr_with_odds | 8118 | 0.98054 | 0.58384 | 0.19758 | 0.5297 | 0.0118 | +0.00646 [+0.00389, +0.00912] | 5/20 |
| lr_no_odds | 8118 | 0.98659 | 0.58814 | 0.19990 | 0.5227 | 0.0140 | +0.01251 [+0.00924, +0.01593] | 0/20 |
| rf_no_odds | 8118 | 0.98772 | 0.58870 | 0.19996 | 0.5277 | 0.0050 | +0.01364 [+0.01011, +0.01732] | 0/20 |
| elo_multinomial | 8118 | 0.98859 | 0.58939 | 0.20039 | 0.5241 | 0.0145 | +0.01451 [+0.01097, +0.01784] | 0/20 |
| dixon_coles | 8118 | 0.98950 | 0.59011 | 0.20091 | 0.5228 | 0.0104 | +0.01542 [+0.01169, +0.01918] | 0/20 |
| poisson_independent | 8118 | 0.99039 | 0.59055 | 0.20099 | 0.5228 | 0.0111 | +0.01630 [+0.01286, +0.01954] | 0/20 |
| hgb_no_odds | 8118 | 0.99411 | 0.59196 | 0.20134 | 0.5200 | 0.0213 | +0.02002 [+0.01482, +0.02499] | 1/20 |
| hgb_with_odds | 8118 | 1.00319 | 0.59655 | 0.20210 | 0.5191 | 0.0335 | +0.02911 [+0.02367, +0.03524] | 1/20 |
| base_rate | 8118 | 1.07390 | 0.64983 | 0.22985 | 0.4350 | 0.0078 | +0.09982 [+0.09212, +0.10733] | 0/20 |

Lettura: nessun modello batte il mercato no-vig in modo significativo; la ricalibrazione del
mercato e' indistinguibile dal mercato (IC che include lo 0). Fra i modelli senza quote,
Elo (multinomiale sulla differenza Elo), Dixon-Coles e Poisson indipendente sono vicini fra loro
(log loss 0,98859 / 0,98950 / 0,99039) e tutti peggiori del mercato in 20 fold su 20.
Calibrazione del pareggio del mercato: 3 partite con probabilita' di pareggio fra 0,7 e 0,9
(quote anomale nei dati), per il resto scarto massimo 0,0152 nei bin con almeno 600 partite.

### Mercati binari (stesso protocollo)

Over 2.5, tutte le partite di test (n = 8.502, riferimento: frequenza storica):
lr_no_odds 0.67855 (delta -0.01282 [-0.01661, -0.00908]); Poisson e Dixon-Coles 0.68217;
base_rate 0.69137.
Over 2.5 contro il mercato, solo quote API-Sports (n = 1.708; sulle righe The Odds API il 49,8%
delle partite ha quote Over 2.5 dei bookmaker che differiscono oltre il 15%, contro il 2,63% di
API-Sports): mercato 0.67407; miglior modello lr_no_odds 0.68269 (+0.00862 [+0.00343, +0.01389]).
Goal/No Goal, tutte le partite di test (n = 8.502): lr_no_odds 0.68707 contro base_rate 0.68952
(-0.00245 [-0.00438, -0.00036]). Contro il mercato (n = 746): mercato 0.69034, nessuna
differenza significativa.

## Indizi

| Indizio | Esito | Evidenza |
|---|---|---|
| (a) Nessuna baseline Elo o Dixon-Coles nel serving di produzione | **VERIFICATO** | `grep -rn -i "\belo\b\|dixon\|elo_rating\|elo_diff" --include=*.py backend/src` (esclusi i test): 0 righe. Esiste solo un Poisson indipendente (`backend/src/ml/experts/goal_distribution/goal_distribution_expert.py:62`), usato da `GET /dashboard/match/{fixture_id}/oracle-detail` (`backend/src/api/oracle_match_detail_service.py:62`, `backend/src/api/main.py:626`) per la spiegabilita', non per le previsioni servite ne' per il ledger. Elo e Dixon-Coles esistono solo nel benchmark offline (`backend/scripts/analysis/walk_forward_1x2_benchmark.py`). |
| (b) h2h binario, dc binario e 1X2 multiclasse non riconciliati nel serving | **VERIFICATO** (nel codice) | h2h = vittoria casa e dc = 1X sono modelli binari separati (`backend/src/service_ia/training/market_service/filter_market_service.py:119-124`). `1x2` multiclasse e' aggiunto "accanto, mai in sostituzione" e solo nel dettaglio partita (`backend/src/api/dashboard_service.py:1652-1687`, chiamato a `:2240`); le liste `/dashboard/day`/`live` non lo hanno. Nelle decision card del dettaglio il `1x2` produce anche card `dc` derivate (casa+pareggio, ecc., `:899-934`) che finiscono nello stesso gruppo `dc` delle card del modello binario (`:936` e seguenti, `_mark_best_cards_by_market` a `:1133`): nessun vincolo P(dc 1X) = P(1) + P(X) ne' P(h2h) = P(1). La cattura ufficiale invece e' coerente: usa solo il `1x2` e la doppia chance derivata (`backend/src/oracle/ledger/official_capture_service.py:241`, `:312-316` esclude h2h/dc/1x2 dai mercati "standard"). Se h2h e dc abbiano oggi un modello in production: NON VERIFICABILE offline (registry sul bucket). |
| (c) Mercati senza modello saltati in silenzio (prima) e dichiarati (ora) | **VERIFICATO** | Su `origin/main`, `backend/src/ml/serving/prediction_snapshot_service.py:235-236` (`if not model_meta: continue`), `:245-246`, `:249-250`, `:276-277`: nessun log, nessun campo; la dashboard dichiarava solo l'assenza di quote (`markets_unavailable`). Ora: log per ogni salto (`prediction_snapshot_service.py:287`, `:298`, `:303`, `:341`) e campo `markets_missing_reasons` (`backend/src/api/dashboard_service.py:255`, righe a `:1554`, `:1805`, `:2246`). Nota: di default i mercati richiesti sono quelli con un modello in production (`list_active_markets`), quindi "nessun modello" compare soprattutto quando i mercati sono chiesti esplicitamente o il registry cambia fra la lista e il calcolo. |
