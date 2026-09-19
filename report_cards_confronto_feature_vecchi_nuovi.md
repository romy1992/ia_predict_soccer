# Confronto registry: modelli cards vecchi (production, 2026-09-12) vs candidati nuovi (2026-09-19)

Fonte: `best_models/registry/index.jsonl` (nessun file dedicato `<run_id>.json` esiste per i run del 12/09; per i run del 19/09 esiste anche il metadata file separato, contenuto identico). Nessuna modifica al registry, nessun training eseguito — solo lettura.

Verifica incrociata con `best_models/registry/promotion_history.jsonl`: gli eventi di promozione del 2026-09-19T08:19:5x confermano `production_score` e `candidate_score` esattamente uguali ai `selection_score` letti da `index.jsonl` per gli stessi run_id.

## Tabella compatta (8 righe)

| Mercato | Stage | Run ID | Creato (UTC) | model_family | rows/sample_size | n_feature | tipo feature | selection_score | log_loss | brier | ece | auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cards_line_3_5 | **production (12/09)** | cards_line_3_5_20260912T201046855184Z | 2026-09-12 20:10:46.855 | calibrated_random_forest (champion_family=random_forest) | 8355 | 73 | quote (11) + stats squadra (58) + referee (4) | 0.6651116475 | pre 0.66647 / post 0.65872 | pre 0.23696 / post 0.23290 | pre 0.07752 / post 0.04407 | pre 0.62710 / post 0.62568 |
| cards_line_4_5 | **production (12/09)** | cards_line_4_5_20260912T201047076924Z | 2026-09-12 20:10:47.077 | calibrated_random_forest (champion_family=random_forest) | 8355 | 73 | quote (11) + stats squadra (58) + referee (4) | 0.6870682299 | pre 0.66591 / post 0.65838 | pre 0.23677 / post 0.23331 | pre 0.07297 / post 0.03915 | pre 0.62661 / post 0.62356 |
| cards_line_5_5 | **production (12/09)** | cards_line_5_5_20260912T201047367977Z | 2026-09-12 20:10:47.368 | calibrated_random_forest (champion_family=random_forest) | 8355 | 73 | quote (11) + stats squadra (58) + referee (4) | 0.7241183718 | pre 0.59805 / post 0.55465 | pre 0.20520 / post 0.18526 | pre 0.14234 / post 0.03229 | pre 0.62498 / post 0.61879 |
| cards_line_6_5 | **production (12/09)** | cards_line_6_5_20260912T201047628969Z | 2026-09-12 20:10:47.629 | calibrated_random_forest (champion_family=random_forest) | 8355 | 73 | quote (11) + stats squadra (58) + referee (4) | 0.8043754027 | pre 0.51653 / post 0.41735 | pre 0.16787 / post 0.12738 | pre 0.19639 / post 0.01816 | pre 0.64859 / post 0.64407 |
| cards_line_3_5 | candidate (19/09, **vincitore → promosso a production**) | cards_line_3_5_20260919T064238116277Z | 2026-09-19 06:42:38.116 | stacking (base: logistic + random_forest_smote) | 2706 | 6 | SOLO quote | 0.6894405542 | 0.66055 (pre-cal; post-cal 0.68051) | 0.23414 (pre-cal; post-cal 0.23447) | 0.04405 (pre-cal; post-cal 0.04159) | 0.65080 |
| cards_line_4_5 | candidate (19/09, **respinto dal gate**) | cards_line_4_5_20260919T064401589205Z | 2026-09-19 06:44:01.589 | logistic | 3141 | 6 | SOLO quote | 0.6863854482 | 0.65159 (pre-cal; post-cal 0.62909) | 0.23036 (pre-cal; post-cal 0.22035) | 0.10771 (pre-cal; post-cal 0.03592) | 0.65424 |
| cards_line_5_5 | candidate (19/09, **respinto dal gate**) | cards_line_5_5_20260919T064559240432Z | 2026-09-19 06:45:59.240 | random_forest_smote | 2843 | 6 | SOLO quote | 0.6960345446 | 0.63951 (pre-cal; post-cal 0.54203) | 0.22167 (pre-cal; post-cal 0.17971) | 0.17336 (pre-cal) | 0.63373 |
| cards_line_6_5 | candidate (19/09, **respinto dal gate**) | cards_line_6_5_20260919T064753521140Z | 2026-09-19 06:47:53.521 | random_forest_smote | 2106 | 6 | SOLO quote | 0.7453318585 | 0.54278 (pre-cal; post-cal 0.39954) | 0.18058 (pre-cal; post-cal 0.12059) | 0.18906 (pre-cal) | 0.67200 |

Note tabella:
- Per i 4 modelli "production (12/09)" le metriche in `index.jsonl` sono salvate come `pre_*`/`post_*` (pre/post calibrazione isotonica); riportate entrambe. `selection_score` non specifica quale delle due venga usata internamente (campo unico, nessuna doppia versione).
- Per i 4 candidati "19/09" le metriche `log_loss`/`brier`/`ece`/`auc` in `index.jsonl` sono le versioni **pre-calibrazione**; sono presenti separatamente anche `pre_calibration_*`/`post_calibration_*` (identiche o quasi alle prime, più le versioni post-isotonica), riportate in parentesi. `selection_score` = 0.6894.../0.6863.../ecc. corrisponde ai valori pre-calibrazione.
- Per `cards_line_3_5` il 19/09 esistono **due** run candidate: `...T064008282946Z` (primo tentativo) e `...T064238116277Z` (poi promosso). Metriche, feature, rows e model_family sono identiche tra i due (unica differenza: nel secondo la calibrazione isotonica risulta effettivamente applicata/salvata, `calibrator_path` presente; nel primo il campo `calibration.error` segnala un errore di serializzazione del calibratore). Il secondo è quello di fatto promosso (vedi `promotion_history.jsonl`, evento `promotion_cards_line_3_5_20260919T064238116277Z_...`).

## Punto 5 — fonte dataset / fill_missing / search_strategy

Nessuno dei metadata (né vecchi né nuovi) contiene un campo esplicito `fill_missing`. Elementi osservati che supportano/confutano l'ipotesi dell'operatore:

- **Modelli vecchi (12/09)**: `dataset_version: "dataset:unspecified"`, nessun flag di filtro esplicito. `extra.model_search_results` mostra la ricerca comparativa su 5 famiglie (`logistic`, `random_forest`, `random_forest_smote`, `voting`, `stacking`) con `champion_family: "random_forest"` poi calibrato → `calibrated_random_forest`. `sample_size = 8355` per **tutte e 4** le linee (identico), a fronte di un pool di feature molto più ampio (73, incluse feature di squadra sempre disponibili anche senza quota reale). Il fatto che `sample_size` sia identico su tutte le linee mentre il numero di partite con quota reale per linea cambia (vedi sotto) è coerente con un dataset che **non filtra per presenza di quota reale su quella linea specifica** — cioè usa tutte le righe disponibili (con feature quota eventualmente a zero/imputate) invece di solo le righe con quota reale.
- **Candidati nuovi (19/09)**: `extra.selection_method: "kbest"` (selezione feature dichiarata, non un search_strategy generico), `extra.selection_in_pipeline: true`. `rows` varia per linea (2706 / 3141 / 2843 / 2106) — numeri sensibilmente più bassi e diversi tra loro, coerenti con un filtro sulle sole righe che hanno quota reale per quella specifica linea (linee più "esotiche" come 6.5 hanno meno quote disponibili → 2106 righe).
- Nessun campo registry conferma esplicitamente la parola "zero-riempite"/imputazione per i vecchi modelli: la conclusione sopra è un'inferenza dai numeri (stesso sample_size=8355 su 4 linee diverse + 73 feature sempre popolate) e andrebbe confermata leggendo il codice di costruzione dataset (fuori perimetro di questo controllo, che è solo lettura registry).
- `git_sha` dei vecchi run: `da3de75`; dei nuovi: `3fb9803` (commit più recente, verificabile con `git log`).
- `feature_version` cambia da `features:73:<hash>` (vecchi) a `features:6:<hash>` (nuovi) — conferma diretta che il pool di feature disponibili al training è stato ridotto da 73 a 6 tra le due generazioni di modelli.

## Conclusione sintetica

L'ipotesi dell'operatore è confermata dai dati di registry:
1. I modelli in production (12/09) usano **73 feature** (11 legate alle quote + 58 statistiche di squadra: tiri, passaggi, possesso, falli, cartellini, xG, corner, parate + 4 feature arbitro), non solo le quote.
2. Sono addestrati su **8355 righe**, identiche su tutte e 4 le linee — numero che non varia con la disponibilità di quota reale per linea, a differenza dei candidati nuovi.
3. I candidati del 19/09 usano **solo 6 feature, tutte derivate dalle quote** (`prob_norm_over_X`, `odds_mean_over_X`, `odds_mean_under_X`, `odds_count`, `odds_std_over_X`, `overround`), e un numero di righe molto più basso e variabile per linea (2106–3141), verosimilmente le sole righe con quota reale.
4. `selection_score` più alto nei vecchi modelli (specialmente 6.5: 0.804 vs 0.745) è quindi plausibilmente gonfiato da un training set più ampio/eterogeneo con feature aggiuntive che il gate di promozione confronta 1:1 contro un candidato più conservativo, senza che ciò implichi maggiore affidabilità out-of-sample del modello vecchio.

## Appendice — liste feature complete per run_id

### Vecchi (production, 12/09) — 73 feature, identiche su tutte le 4 linee (cambia solo il suffisso `_line_X_5` nei nomi quota)

```
mean_shots_off_goal_home_stat, mean_shots_off_goal_away_stat, mean_shots_off_goal_diff_stat,
mean_total_passes_home_stat, mean_total_passes_away_stat, mean_total_passes_diff_stat,
mean_goals_prevented_home_stat, mean_goals_prevented_away_stat, mean_goals_prevented_diff_stat,
mean_passes_accurate_home_stat, mean_passes_accurate_away_stat, mean_passes_accurate_diff_stat,
mean_passes_home_stat, mean_passes_away_stat, mean_passes_diff_stat,
mean_offsides_home_stat, mean_offsides_away_stat, mean_offsides_diff_stat,
mean_shots_outsidebox_home_stat, mean_shots_outsidebox_away_stat, mean_shots_outsidebox_diff_stat,
mean_shots_insidebox_home_stat, mean_shots_insidebox_away_stat, mean_shots_insidebox_diff_stat,
mean_blocked_shots_home_stat, mean_blocked_shots_away_stat, mean_blocked_shots_diff_stat,
mean_yellow_cards_home_stat, mean_yellow_cards_away_stat, mean_yellow_cards_diff_stat,
mean_total_shots_home_stat, mean_total_shots_away_stat, mean_total_shots_diff_stat,
mean_red_cards_home_stat, mean_red_cards_away_stat, mean_red_cards_diff_stat,
mean_expected_goals_home_stat, mean_expected_goals_away_stat, mean_expected_goals_diff_stat,
mean_shots_on_goal_home_stat, mean_shots_on_goal_away_stat, mean_shots_on_goal_diff_stat,
mean_ball_possession_home_stat, mean_ball_possession_away_stat, mean_ball_possession_diff_stat,
mean_corner_kicks_home_stat, mean_corner_kicks_away_stat, mean_corner_kicks_diff_stat,
mean_goalkeeper_saves_home_stat, mean_goalkeeper_saves_away_stat, mean_goalkeeper_saves_diff_stat,
mean_fouls_home_stat, mean_fouls_away_stat, mean_fouls_diff_stat,
odds_count_line_X_5, odds_mean_line_X_5, odds_std_line_X_5, odds_min_line_X_5, odds_max_line_X_5,
odds_slot_1_line_X_5, odds_slot_2_line_X_5, odds_slot_3_line_X_5, odds_slot_4_line_X_5, odds_slot_5_line_X_5,
odds_slot_6_line_X_5, odds_slot_7_line_X_5, odds_slot_8_line_X_5, odds_slot_9_line_X_5, odds_slot_10_line_X_5,
referee_avg_cards_prior, referee_severity_index_prior, referee_matches_officiated_prior, referee_has_history
```
(`X_5` = `3_5`, `4_5`, `5_5` o `6_5` a seconda del run_id; conteggio: 58 feature statistiche squadra + 11 feature quota + 4 feature arbitro = 73)

### Nuovi (candidate, 19/09) — 6 feature per run, tutte quote

- `cards_line_3_5_20260919T064008282946Z` e `cards_line_3_5_20260919T064238116277Z` (stesse feature):
  `prob_norm_over_3_5_line_3_5, odds_mean_over_3_5_line_3_5, odds_mean_under_3_5_line_3_5, odds_count_line_3_5, odds_std_over_3_5_line_3_5, overround_line_3_5`

- `cards_line_4_5_20260919T064401589205Z`:
  `prob_norm_over_4_5_line_4_5, odds_mean_over_4_5_line_4_5, odds_mean_under_4_5_line_4_5, odds_count_line_4_5, odds_std_over_4_5_line_4_5, overround_line_4_5`

- `cards_line_5_5_20260919T064559240432Z`:
  `prob_norm_over_5_5_line_5_5, odds_mean_over_5_5_line_5_5, odds_mean_under_5_5_line_5_5, odds_count_line_5_5, odds_std_over_5_5_line_5_5, overround_line_5_5`

- `cards_line_6_5_20260919T064753521140Z`:
  `prob_norm_over_6_5_line_6_5, odds_mean_over_6_5_line_6_5, odds_mean_under_6_5_line_6_5, odds_count_line_6_5, odds_std_over_6_5_line_6_5, overround_line_6_5`
