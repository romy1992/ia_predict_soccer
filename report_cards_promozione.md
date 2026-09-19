# Cards — Promozione a production (2026-09-19)

## Passo 0 — merge `feature/soccer-oracle-v2`

- `git fetch origin feature/soccer-oracle-v2`: confermato un solo commit avanti a main (`4e859a9`).
- Diff ri-verificato prima del merge: solo `line_market_signal_policy.py` + `MatchDetailPanel.jsx` + i suoi test. Grep su `expected_precision_over|expected_recall_over` in tutto il repo: nessun consumer fuori da quel file.
- `git merge origin/feature/soccer-oracle-v2 --no-edit` → **fast-forward pulito** (`3ae550e..4e859a9`).
- `git push origin main` → ok.
- Test: **9/9 passati** (`tests/service/line_market_signal_policy_test.py`). Nota: nessun `.venv/` presente in questo worktree Windows — eseguiti con `python` di sistema (Python 3.14.5), non `.venv/bin/python` come indicato nella richiesta.

## Scoperta preliminare importante: l'assunto "mercati nuovi" era sbagliato

Prima di promuovere ho verificato lo stato reale del registry. **Tutti e 4 i mercati cards hanno già una production**, promossa il 2026-09-12 (`"Promozione post training reale con quote line-specific + monotonicita diagnostica + random search, richiesta esplicita operatore"`). Non è quindi un bootstrap "prima promozione mai avvenuta" come indicato nella richiesta: `promote_with_policy` ha eseguito il confronto reale candidate-vs-production (`compare_candidate_to_production`, `min_improvement_over_production=0.0`, cioè basta non peggiorare sullo `selection_score`).

Nota comunque sul comportamento del gate per una prima promozione (rilevante in generale, anche se qui non si è applicato): `PromotionPolicy.allow_promotion_without_production_baseline=True` di default — se non esiste production, il confronto passa automaticamente, ma il **gate sulle metriche resta comunque attivo** (log_loss ≤0.75, brier ≤0.30, ece ≤0.25, auc ≥0.50, sample_size ≥30).

## Esito promozione per linea

Run candidato usato: il più recente per `created_at` con `stage="candidate"` in ciascun mercato.

| Linea | Run candidato (2026-09-19) | Gate metriche | Confronto vs production (2026-09-12) | Esito |
|---|---|---|---|---|
| cards_line_3_5 | `cards_line_3_5_20260919T064238116277Z` | passed | candidate 0.6894 vs production 0.6651 → **delta +0.0243** | ✅ **PROMOSSA** |
| cards_line_4_5 | `cards_line_4_5_20260919T064401589205Z` | passed | candidate 0.6864 vs production 0.6871 → delta **-0.00068** | ❌ **RIFIUTATA** (comparison_failed) |
| cards_line_5_5 | `cards_line_5_5_20260919T064559240432Z` | passed | candidate 0.6960 vs production 0.7241 → delta **-0.0281** | ❌ **RIFIUTATA** (comparison_failed) |
| cards_line_6_5 | `cards_line_6_5_20260919T064753521140Z` | passed | candidate 0.7453 vs production 0.8044 → delta **-0.0590** | ❌ **RIFIUTATA** (comparison_failed) |

Per tutte e 4 le linee il gate metriche (log_loss/brier/ece/auc/sample_size) è passato senza problemi. Il confronto usa `selection_score` in modo omogeneo su entrambi i lati (candidate e production espongono entrambi quella chiave, nessun mismatch di metodo). **Nessun bypass/force è stato usato**: le 3 rejection sono state chiamate con `force=False` (default) e sono tracciate nell'audit trail come `promotion_blocked`, senza alterare lo stage corrente dei run.

Interpretazione: il retraining Step A/B del 2026-09-19 ha rifatto i modelli con un pattern di feature diverso (solo-quota, 6 colonne, stesso approccio di goal/no-goal) e ha verificato la direzione Under/ROI positivo — ma su 3 linee su 4 il nuovo classificatore ha uno `selection_score` (probabilità composita) più basso del modello 2026-09-12 già in produzione. Il gate ha fatto esattamente il suo lavoro: la sostituzione non è automatica solo perché il training è "più recente" o perché la logica di soglia ROI è stata rifatta — serve reggere il confronto sullo score. Su `cards_line_4_5` il margine è minuscolo (-0.00068): potrebbe valere la pena rivedere se il nuovo pattern a 6 feature-quota è davvero comparabile allo score del vecchio modello, ma questa è una decisione di merito che non spetta al gate né a me forzare qui.

## Scoperta critica non prevista nella richiesta: path del modello promosso

Il training Step A/B (report `report_cards_training_step_a_b.md`) è girato **sull'host Windows**, non dentro Docker (per un problema di memoria OpenBLAS/joblib con i 4 training in parallelo, documentato nello stesso report). Di conseguenza `ModelRegistry.register()` ha calcolato `model_path` con `os.path.abspath()` risolto sull'host, cioè un path Windows assoluto (`C:\Users\trott\git\ia_predict_soccer\best_models\cards\cards_line_3_5\...`), non un path `/app/best_models/...`.

Il codice che serve le predizioni cards (`CardsExpert._from_run` in `src/ml/markets/cards/cards_market.py:934-939`) legge `model_path` **letteralmente** (`os.path.exists()` + `joblib.load()`), **senza** passare per `resolve_model_path()` — a differenza di `api/main.py`, `prediction_snapshot_service.py` e `model_diagnostics_service.py`, che invece lo usano proprio per gestire questo scenario (path registrato su un'altra macchina). Se avessi fatto `docker compose restart` senza intervenire, ogni richiesta di predizione su `cards_line_3_5` avrebbe fallito con `FileNotFoundError` dentro il container Linux (il path Windows non esiste lì, anche se il file fisico è presente via volume `./best_models:/app/best_models`).

**Ho chiesto conferma all'utente** su come procedere; scelta: riscrivere `model_path` usando `to_container_path()` — funzione già presente nel codebase (`src/service_ia/training/model_paths.py`) documentata esplicitamente per questo caso, ma mai collegata al flusso di registrazione. Ho applicato la conversione:
- `best_models/registry/index.jsonl`: aggiornato SOLO il campo `model_path` della riga `cards_line_3_5_20260919T064238116277Z`, da path Windows a `/app/best_models/cards/cards_line_3_5/cards_line_3_5_champion.pkl`.
- `best_models/registry/cards_line_3_5_20260919T064238116277Z.json` (metadata di audit del singolo run, non letto a runtime ma tenuto coerente): stesso campo aggiornato.
- Nessun altro campo toccato (stage, metrics, audit trail delle promozioni invariati).

Nota: esiste anche un `calibrator_path` annidato in `extra.calibration.calibrator_path` con lo stesso problema di path Windows, ma non risulta letto da nessun consumer a runtime (il modello calibrato viene già salvato per intero come `model_path`) — lasciato invariato, correggerlo non era necessario per la serving path.

## Restart e verifica post-promozione

`docker compose restart api scheduler` → entrambi i container ripartiti, `soccer_api` healthy.

Verifica **dentro il container** (non solo dall'host, per essere sicuri che il fix del path funzioni davvero nell'ambiente che serve le predizioni):

```
cards_line_3_5 -> run_id: cards_line_3_5_20260919T064238116277Z  path: /app/best_models/cards/cards_line_3_5/cards_line_3_5_champion.pkl
cards_line_4_5 -> run_id: cards_line_4_5_20260912T201047076924Z  path: /app/best_models/archivio/cards_line_4_5/cards_line_4_5_champion.pkl
cards_line_5_5 -> run_id: cards_line_5_5_20260912T201047367977Z  path: /app/best_models/archivio/cards_line_5_5/cards_line_5_5_champion.pkl
cards_line_6_5 -> run_id: cards_line_6_5_20260912T201047628969Z  path: /app/best_models/archivio/cards_line_6_5/cards_line_6_5_champion.pkl
```

Inoltre, `CardsExpert.load_production(line=3.5)` eseguito **dentro** `soccer_api` ha caricato correttamente il modello (`run_id=cards_line_3_5_20260919T064238116277Z`, `stage=production`), confermando che il fix del path risolve il problema e la promozione è realmente operativa.

Nota collaterale osservata nei log di avvio di `soccer_api` (preesistente, non causata da questa attività): `alembic` segnala `Can't locate revision identified by 'head'`; l'app parte comunque e risponde `/health` 200. Segnalato per trasparenza, non indagato oltre perché fuori dallo scope di questo task.

Le production di `cards_line_4_5`/`cards_line_5_5`/`cards_line_6_5` puntano a `/app/best_models/archivio/...`: sono modelli della "vecchia procedura" (pre-riorganizzazione best_models del 2026-09-15). Restano validi e caricabili (`archivio/` è solo escluso da `list_active_markets()` per la Dashboard, non dal serving), ma è utile sapere che i 3 candidati di oggi non li hanno scalzati.

## File modificati

- `best_models/registry/index.jsonl` — correzione `model_path` per il run promosso (path host → path container).
- `best_models/registry/cards_line_3_5_20260919T064238116277Z.json` — stesso campo, per coerenza con l'index.
- `best_models/registry/promotion_history.jsonl` — nuovi eventi di audit: 1 `promotion_approved` (cards_line_3_5) + 3 `promotion_blocked` (cards_line_4_5/5_5/6_5), scritti da `promote_with_policy`.
