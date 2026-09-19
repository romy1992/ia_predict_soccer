# Cards — Promozione FORZATA delle 3 linee rifiutate dal gate (2026-09-19)

## Contesto

Nel precedente report (`report_cards_promozione.md`) il gate aveva promosso `cards_line_3_5` ma
respinto `cards_line_4_5`, `cards_line_5_5`, `cards_line_6_5` per `selection_score` piu' basso
rispetto ai modelli in production del 12/09 (comparison_failed).

Verifica di merito fatta prima di forzare: i modelli in production del 12/09 usano 73 feature, di
cui le "quote" sono in realta' le feature LEGACY aggregate (`odds_mean/min/max/std/slot_1..10`,
Over e Under mescolati insieme) — il pattern che questo progetto esclude deliberatamente dal
training nuovo (`LEGACY_ODDS_FEATURES` in `filter_market_service.py`) — piu' 58 feature
statistiche squadra e 4 arbitro, addestrati su 8355 righe **identiche** su tutte e 4 le linee
(segno di zero-riempimento delle righe senza quota reale, non filtro sulla quota vera). Test
locale: aggiungere le stesse 58 feature statistiche ai candidati nuovi **peggiora** l'AUC su
tutte e 4 le linee (es. 6.5: AUC 0.690 con sole quote vs 0.661 con statistiche aggiunte). Il
punteggio piu' alto dei vecchi modelli e' quindi un artefatto di training su dati piu' numerosi
ma meno puliti, non un modello piu' affidabile.

**Decisione esplicita dell'operatore**: forzare comunque la promozione dei 3 candidati, per avere
lo stesso principio metodologico (quote separate per esito, niente zero-fill, niente feature
legacy) su tutte e 4 le linee cards, coerentemente con `cards_line_3_5` (gia' promossa senza
forzare) e con gli altri mercati rifatti nel progetto (goal_no_goal, under_over 1.5/2.5/3.5).

## Meccanismo di forzatura

Verificata la firma di `ModelRegistry.promote_with_policy` in
`src/service_ia/training/model_registry.py:406-484`: il parametro `force: bool = False`, se
`True` e il gate blocca (`evaluation.allowed=False`), esegue comunque `promote()` e marca
l'evento con `metadata.event_type = "promotion_forced"` — **automaticamente**, senza bisogno di
annotazioni manuali aggiuntive per renderlo distinguibile da una promozione normale
(`promotion_approved`) o da un tentativo bloccato (`promotion_blocked`).

Script usato: `scripts/analysis/promuovi_cards_forzato.py` (chiama `promote_with_policy` con
`force=True` per i 3 run_id, poi riscrive i path in forma container con `to_container_path()`).

## Esito promozione per linea

| Linea | Run promosso | selection_score candidato | selection_score production 12/09 | Esito |
|---|---|---|---|---|
| cards_line_4_5 | `cards_line_4_5_20260919T064401589205Z` | 0.6864 | 0.6871 | ✅ **PROMOSSA (forzata)** |
| cards_line_5_5 | `cards_line_5_5_20260919T064559240432Z` | 0.6960 | 0.7241 | ✅ **PROMOSSA (forzata)** |
| cards_line_6_5 | `cards_line_6_5_20260919T064753521140Z` | 0.7453 | 0.8044 | ✅ **PROMOSSA (forzata)** |

## Audit trail

Verificato in `best_models/registry/promotion_history.jsonl`: i 3 nuovi eventi hanno
`from_stage: candidate -> to_stage: production` e `metadata.event_type: "promotion_forced"` per
tutte e 3 le linee — chiaramente distinguibile dagli eventi `promotion_approved` (cards_line_3_5)
e dai precedenti `promotion_blocked` (stessi run_id, tentativo del 19/09 senza force). Il
`reason` registrato riporta per intero il contesto sopra (score piu' basso, artefatto zero-fill/
feature legacy, decisione esplicita operatore).

## Correzione path (stesso problema gia' risolto per cards_line_3_5)

Il training Step A/B e' girato sull'host Windows: `ModelRegistry.register()` aveva scritto
`model_path` come path Windows assoluto (`C:\Users\...\best_models\cards\cards_line_X\...`), non
utilizzabile dentro il container Linux. Lo script ha applicato `to_container_path()` a tutte le
righe dell'index non ancora in forma container (19 campi riscritti in totale, incluse righe di
altri mercati/run gia' pendenti da una conversione precedente).

## Restart e verifica post-promozione

`docker compose restart api scheduler` → entrambi i container ripartiti correttamente.

Verifica **dentro il container** `soccer_api`:

```
cards_line_4_5 -> run_id: cards_line_4_5_20260919T064401589205Z  path: /app/best_models/cards/cards_line_4_5/cards_line_4_5_champion.pkl
cards_line_5_5 -> run_id: cards_line_5_5_20260919T064559240432Z  path: /app/best_models/cards/cards_line_5_5/cards_line_5_5_champion.pkl
cards_line_6_5 -> run_id: cards_line_6_5_20260919T064753521140Z  path: /app/best_models/cards/cards_line_6_5/cards_line_6_5_champion.pkl
```

`CardsExpert.load_production(line=X)` eseguito dentro il container per le 3 linee ha caricato
correttamente ciascun modello:

```
line=4.5 run_id=cards_line_4_5_20260919T064401589205Z stage=production n_features=6
line=5.5 run_id=cards_line_5_5_20260919T064559240432Z stage=production n_features=6
line=6.5 run_id=cards_line_6_5_20260919T064753521140Z stage=production n_features=6
```

## Stato finale registry cards

Tutte e 4 le linee (`cards_line_3_5/4_5/5_5/6_5`) sono ora in production con i modelli del
19/09/2026 (6 feature-quota, per-esito, righe filtrate su quota reale) — nessuna linea punta piu'
ai modelli legacy dell'archivio del 12/09.

## File modificati

- `best_models/registry/index.jsonl` — 3 righe promosse a `production` + correzione `model_path`
  (path host → path container) su 19 campi totali.
- `best_models/registry/cards_line_4_5_20260919T064401589205Z.json`,
  `cards_line_5_5_20260919T064559240432Z.json`, `cards_line_6_5_20260919T064753521140Z.json` —
  stage e metadata aggiornati per coerenza con l'index.
- `best_models/registry/promotion_history.jsonl` — 3 nuovi eventi `promotion_forced`.
- `scripts/analysis/promuovi_cards_forzato.py` — nuovo script di promozione forzata (documentato
  con la motivazione completa, riusabile se serve ripetere il pattern).
