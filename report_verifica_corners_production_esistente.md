# Verifica: production esistente per corners_line_8_5/9_5/10_5/11_5

**Data verifica:** 2026-09-19
**Ambito:** solo lettura, nessuna modifica al registry o ai file modello.

## Esito

Confermato: esiste una `production` reale, registrata nel `ModelRegistry` di
questa macchina (non una copia disallineata), per tutte e 4 le linee corners.
Questo **contraddice** quanto scritto in
`docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md` (sezione che
descrive la riverifica del 2026-09-16 come conclusa con "NON promuovere alcun
modello corners") e i commenti in
`src/oracle/decision_engine/line_market_signal_policy.py:19-29,67-68`
("NESSUN modello corners mai promosso a production" / "inerti in pratica").

## 1. Registry — `ModelRegistry.get_production()`

| Mercato | run_id | model_path (registrato) |
|---|---|---|
| corners_line_8_5 | corners_line_8_5_20260912T194952985622Z | /app/best_models/archivio/corners_line_8_5/corners_line_8_5_champion.pkl |
| corners_line_9_5 | corners_line_9_5_20260912T194953056739Z | /app/best_models/archivio/corners_line_9_5/corners_line_9_5_champion.pkl |
| corners_line_10_5 | corners_line_10_5_20260912T194953282007Z | /app/best_models/archivio/corners_line_10_5/corners_line_10_5_champion.pkl |
| corners_line_11_5 | corners_line_11_5_20260912T194953535594Z | /app/best_models/archivio/corners_line_11_5/corners_line_11_5_champion.pkl |

Stesso pattern temporale dei modelli cards promossi lo stesso giorno: `run_id`
creato il 2026-09-12T19:49:5x, evento di promozione
(`promotion_history[0]`) datato **2026-09-12T20:25:2x**, `actor:
operator_request`, motivo identico per le 4 linee:

> "Promozione post training reale con quote line-specific + monotonicita
> diagnostica + random search, richiesta esplicita operatore"

Gate di promozione (`promotion_policy_v1`) risultato `passed: true` per tutti
i check (log_loss, brier, ece, auc, sample_size) su tutte e 4 le linee;
confronto con `method: no_production_baseline` ("nessuna production esistente
per questo mercato: prima promozione consentita" — coerente con l'assenza di
promozioni precedenti, non con promozioni successive mai avvenute).

## 2. Esistenza file `.pkl` e `resolve_model_path()`

A differenza del problema riscontrato su goal_no_goal e cards (path Windows
non convertito), qui **non c'e' il problema**: il valore registrato e' gia' in
forma container (`/app/best_models/archivio/...`), e
`resolve_model_path()` lo risolve correttamente al percorso locale reale, che
esiste su disco per tutte e 4 le linee:

| Mercato | resolved path | esiste |
|---|---|---|
| corners_line_8_5 | `best_models\archivio\corners_line_8_5\corners_line_8_5_champion.pkl` | si |
| corners_line_9_5 | `best_models\archivio\corners_line_9_5\corners_line_9_5_champion.pkl` | si |
| corners_line_10_5 | `best_models\archivio\corners_line_10_5\corners_line_10_5_champion.pkl` | si |
| corners_line_11_5 | `best_models\archivio\corners_line_11_5\corners_line_11_5_champion.pkl` | si |

Conclusione: il codice di serving che usa `resolve_model_path()` **riesce a
caricare** questi 4 modelli production — non c'e' il bug di path che
affliggeva altri mercati.

## 3. Metriche registrate (run in production, `metrics` del run)

| Mercato | model_name | sample_size | post_log_loss | post_brier | post_ece | post_auc | post_accuracy | selection_score |
|---|---|---|---|---|---|---|---|---|
| corners_line_8_5 | calibrated_random_forest | 8804 | 0.6719 | 0.2394 | 0.0147 | 0.5282 | 0.6009 | 0.6391 |
| corners_line_9_5 | calibrated_logistic | 8804 | 0.7046 | 0.2524 | 0.0246 | 0.5341 | 0.5273 | 0.6526 |
| corners_line_10_5 | calibrated_random_forest | 8804 | 0.6588 | 0.2330 | 0.0235 | 0.5357 | 0.6268 | 0.6593 |
| corners_line_11_5 | calibrated_random_forest | 8804 | 0.5803 | 0.1955 | 0.0152 | 0.5106 | 0.7352 | 0.7166 |

Tutte con `dataset_version: dataset:unspecified`, `git_sha: da3de75`,
`feature_version: features:73:*` (73 feature, stesso schema visto per cards
production del 12/09 — probabile stessa contaminazione bookmaker
BetMGM/BetRivers/Bovada descritta in
`docs/soccer_oracle_v2_detailed/PROMPT_mercato_corners.md:68-71` come "fix
applicato solo a corners" ma **non verificato se questi 4 modelli in
production la includano o la precedano**, essendo stati creati lo stesso
istante 19:49:5x del 12/09).

AUC 0.51-0.54 su tutte le linee: segnale debole, coerente con quanto
descritto nel PROMPT come motivo del rifiuto — ma qui il gate
(`promotion_policy_v1`, soglia AUC 0.5) li ha comunque fatti passare, e sono
stati promossi manualmente (`operator_request`) lo stesso giorno.

## Nota

Non e' stata effettuata alcuna modifica al registry, ai file `.pkl` o alla
documentazione. La discrepanza tra codice/documentazione (che assumono
"nessuna promozione corners mai avvenuta") e lo stato reale del registry va
risolta a livello di decisione umana: i commenti in
`line_market_signal_policy.py` e il PROMPT vanno aggiornati, e va deciso se
`evaluate_line_market_signal` deve iniziare a considerare queste production
reali (oggi ignorate perche' il codice assume che non esistano).
