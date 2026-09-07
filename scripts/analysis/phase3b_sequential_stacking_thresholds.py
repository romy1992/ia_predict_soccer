"""FASE 3B (idea aggiuntiva, vedi task): "stacking sequenziale a cascata tra
soglie" per Under/Over 1.5/2.5/3.5/4.5.

Allena un classificatore RandomForest per over_1.5 (stesso `_RF_KWARGS` di
`totals_market.py`, stesso walk-forward gia' in uso), usa la sua probabilita'
OOF (out-of-fold) come feature aggiuntiva per allenare over_2.5, la cui OOF
diventa feature per over_3.5, e cosi' via fino a over_4.5.

DA NON CONFONDERE con:
- "hierarchical" (gia' in `totals_market.py`): UN SOLO modello multiclasse
  sui bin di gol, non una cascata di modelli separati.
- La feature `gd_over_1_5` gia' testata in Fase 3 (`phase3_cascading_ab_test.py`,
  ESITO: scartato, delta +0.0008): quella e' la probabilita' DETERMINISTICA
  Poisson (Goal Distribution Expert/EXP-02, nessun training). Qui invece la
  feature aggiuntiva e' la probabilita' PREDETTA dal classificatore allenato
  per la soglia precedente.

Anti-leakage temporale tra soglie (stesso principio di
`PointInTimeDatasetBuilder.assert_no_leakage`/ML-01, verificato ESPLICITAMENTE
qui sotto, non solo per costruzione):
`expanding_window_splits` produce, per ogni fold, un train set che e' SEMPRE
il prefisso [0, train_end) del dataframe ordinato per `prediction_at`; l'OOF
di una soglia e' definito solo per le righe che sono state validation in
QUALCHE fold (`oof_index_all`, un intervallo contiguo
[min_train, min_train + n_splits*min_valid)). Quando quell'OOF diventa
feature per la soglia successiva, ogni fold di training viene ristretto a
`oof_index_all` (le righe del blocco iniziale "anchor", mai state validation
per nessuna soglia, vengono escluse dal training della feature a cascata:
non esiste per loro un valore OOF non soggetto a leakage). Il validation set
di ogni fold resta invariato (e' gia' sempre sottoinsieme di `oof_index_all`
per costruzione). `_assert_cascade_no_future_leakage` verifica esplicitamente,
fold per fold, che il `prediction_at` piu' recente del training set ristretto
sia <= al `prediction_at` piu' vecchio del validation set dello stesso fold.

Per isolare l'effetto della sola feature a cascata (e non un effetto
collaterale del dataset di training ridotto da `oof_index_all`), ogni soglia
da 2.5 in poi viene confrontata contro un baseline "binary_independent"
allenato sullo STESSO training set ristretto ma SENZA la feature a cascata
(stesso `_RF_KWARGS`, stessi fold). Il verdetto per soglia e quello
aggregato vengono anche confrontati contro i valori gia' noti (Fase 1+2):
goal_distribution aggregato 0.7300; over_2_5/binary_independent 0.6683 (su
dataset NON ristretto - riportato solo come riferimento esterno).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

logging.basicConfig(level=logging.WARNING)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.ml.evaluation.probability_metrics import champion_probability_score, compute_probability_metrics
from src.ml.markets.totals.totals_market import (
    _RF_KWARGS,
    THRESHOLD_LABELS,
    THRESHOLDS,
    _class1_probability,
    _threshold_label,
    build_totals_evaluation_frame,
)
from src.ml.validation.temporal_split import expanding_window_splits
from src.repository.match_repository import MatchRepository
from src.service_ia.utility.utils import convert_orm_match_to_dict

OUTPUT_PATH = os.path.abspath(os.path.join("best_models", "phase3b_sequential_stacking_thresholds_result.json"))

# Riferimenti esterni gia' noti (Fase 1+2/MARKET-04-EXT, dataset reale 15235
# righe, NON ristretto a oof_index_all - riportati solo come termine di
# paragone, non ricalcolati qui).
GOAL_DISTRIBUTION_AGGREGATE_SCORE_REFERENCE = 0.7300
BINARY_INDEPENDENT_OVER_2_5_SCORE_REFERENCE = 0.6683
ADOPTION_DELTA_THRESHOLD = 0.01


def _assert_cascade_no_future_leakage(
    frame: pd.DataFrame,
    cv_splits: list[tuple[list[int], list[int]]],
    oof_index_all: set[int],
    time_col: str = "prediction_at",
) -> None:
    """ML-01: per ogni fold, il training set RISTRETTO a `oof_index_all` deve
    restare interamente precedente (in `prediction_at`) al validation set
    dello stesso fold. Solleva AssertionError se una qualsiasi riga di
    training ristretto ha `prediction_at` >= alla riga di validation piu'
    vecchia dello stesso fold (leakage temporale)."""
    timestamps = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
        restricted_train = [i for i in train_idx if i in oof_index_all]
        if not restricted_train or not valid_idx:
            continue
        latest_train_ts = timestamps.iloc[restricted_train].max()
        earliest_valid_ts = timestamps.iloc[valid_idx].min()
        if not (latest_train_ts <= earliest_valid_ts):
            raise AssertionError(
                f"Leakage temporale rilevato nel fold {fold_idx}: "
                f"training ristretto piu' recente ({latest_train_ts}) >= "
                f"validation piu' vecchio ({earliest_valid_ts})"
            )


def _cascade_oof(
    frame: pd.DataFrame,
    feature_columns: list[str],
    cv_splits: list[tuple[list[int], list[int]]],
    y_col: str,
    restrict_train_to: Optional[set[int]] = None,
) -> np.ndarray:
    """OOF walk-forward per una soglia, con RandomForest (`_RF_KWARGS`,
    stesso di `totals_market.py`). Se `restrict_train_to` e' fornito, il
    training set di ogni fold viene ristretto a quell'insieme di indici
    (usato per le soglie a cascata: solo righe con una feature OOF di soglia
    precedente non soggetta a leakage)."""
    n = len(frame)
    result = np.full(n, np.nan)
    X = frame[feature_columns]
    y = frame[y_col].astype(int)

    for train_idx, valid_idx in cv_splits:
        effective_train_idx = train_idx if restrict_train_to is None else [i for i in train_idx if i in restrict_train_to]
        if not effective_train_idx or not valid_idx:
            continue
        if y.iloc[effective_train_idx].nunique() < 2:
            continue
        model = RandomForestClassifier(**_RF_KWARGS)
        model.fit(X.iloc[effective_train_idx], y.iloc[effective_train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])
        result[valid_idx] = _class1_probability(proba, model.classes_)

    return result


def _score(frame: pd.DataFrame, y_col: str, probs: np.ndarray, oof_index: list[int]) -> dict[str, Any]:
    y_true = frame[y_col].astype(int).to_numpy()[oof_index]
    p = probs[oof_index]
    metrics = compute_probability_metrics(y_true=y_true, probabilities=p, n_bins=10)
    predicted = (p >= 0.5).astype(int)
    f1_weighted = float(f1_score(y_true, predicted, average="weighted", zero_division=0))
    selection_score = champion_probability_score(metrics=metrics, f1_weighted=f1_weighted)
    return {**metrics, "f1_weighted": f1_weighted, "selection_score": selection_score}


def run_sequential_cascade(frame: pd.DataFrame, feature_columns: list[str], cv_splits: list[tuple[list[int], list[int]]]) -> dict[str, Any]:
    oof_index_all = sorted({idx for _, valid_idx in cv_splits for idx in valid_idx})
    if not oof_index_all:
        raise ValueError("Nessun indice OOF disponibile: walk-forward non valido")
    oof_index_set = set(oof_index_all)

    _assert_cascade_no_future_leakage(frame=frame, cv_splits=cv_splits, oof_index_all=oof_index_set)

    per_threshold: dict[str, dict[str, Any]] = {}
    cascade_scores: list[float] = []
    prev_oof: Optional[np.ndarray] = None
    prev_label: Optional[str] = None
    working_frame = frame.copy()

    for threshold in sorted(set(float(t) for t in THRESHOLDS)):
        label = _threshold_label(threshold)
        y_col = f"y_{label}"

        if prev_oof is None:
            # Soglia radice (1.5): nessuna feature a cascata disponibile,
            # nessuna restrizione del training set - identico a
            # 'binary_independent', ma valutato SOLO su oof_index_all per
            # restare confrontabile con le soglie successive (che sono
            # necessariamente ristrette a quell'insieme).
            cascade_probs = _cascade_oof(working_frame, feature_columns, cv_splits, y_col, restrict_train_to=None)
            eval_index = oof_index_all
            cascade_metrics = _score(working_frame, y_col, cascade_probs, eval_index)
            per_threshold[label] = {
                "cascade": cascade_metrics,
                "baseline_same_rows": None,
                "delta_selection_score": None,
                "n_eval_rows": len(eval_index),
            }
            cascade_scores.append(cascade_metrics["selection_score"])
        else:
            # Dominio disponibile per il training a cascata di QUESTA soglia:
            # solo le righe dove la feature OOF della soglia PRECEDENTE e'
            # definita (non-NaN) - cioe' sono state, a loro volta, validation
            # in un fold il cui training era interamente disponibile alla
            # soglia precedente. Ogni livello di cascata puo' quindi
            # "consumare" un fold iniziale in piu' (nessuna riga con feature
            # a cascata mancante entra MAI nel training, ne' qui ne' nella
            # baseline sotto - stesso identico train set per un confronto
            # onesto): dimensione del set valutabile riportata esplicitamente
            # in `n_eval_rows` per trasparenza.
            eligible_train = {int(i) for i in oof_index_all if not np.isnan(prev_oof[i])}

            cascade_feature_col = f"cascade_oof_{prev_label}"
            working_frame[cascade_feature_col] = prev_oof

            baseline_probs = _cascade_oof(working_frame, feature_columns, cv_splits, y_col, restrict_train_to=eligible_train)
            cascading_feature_columns = feature_columns + [cascade_feature_col]
            cascade_probs = _cascade_oof(working_frame, cascading_feature_columns, cv_splits, y_col, restrict_train_to=eligible_train)

            eval_index = sorted(
                i for i in eligible_train if not np.isnan(baseline_probs[i]) and not np.isnan(cascade_probs[i])
            )
            if not eval_index:
                raise ValueError(
                    f"Nessuna riga valutabile per '{label}' dopo la cascata: il walk-forward "
                    f"(n_splits/min_valid_size correnti) non lascia dati sufficienti per "
                    f"{len(THRESHOLDS)} soglie in cascata mantenendo la garanzia anti-leakage."
                )

            baseline_metrics = _score(working_frame, y_col, baseline_probs, eval_index)
            cascade_metrics = _score(working_frame, y_col, cascade_probs, eval_index)

            delta = cascade_metrics["selection_score"] - baseline_metrics["selection_score"]
            per_threshold[label] = {
                "cascade": cascade_metrics,
                "baseline_same_rows": baseline_metrics,
                "delta_selection_score": delta,
                "n_eval_rows": len(eval_index),
            }
            cascade_scores.append(cascade_metrics["selection_score"])

        prev_oof = cascade_probs
        prev_label = label

    aggregate_cascade_score = float(np.mean(cascade_scores))
    deltas = [payload["delta_selection_score"] for payload in per_threshold.values() if payload["delta_selection_score"] is not None]
    adopted_thresholds = [label for label, payload in per_threshold.items() if (payload["delta_selection_score"] or 0.0) > ADOPTION_DELTA_THRESHOLD]

    verdict = (
        "ADOTTATO: la cascata sequenziale migliora selection_score di oltre "
        f"{ADOPTION_DELTA_THRESHOLD} su tutte le soglie a valle di 1.5."
        if deltas and all(d > ADOPTION_DELTA_THRESHOLD for d in deltas)
        else "SCARTATO/PARZIALE: la cascata sequenziale non migliora selection_score in modo netto e consistente su tutte le soglie a valle di 1.5."
    )

    return {
        "n_oof": len(oof_index_all),
        "per_threshold": per_threshold,
        "aggregate_cascade_score": aggregate_cascade_score,
        "adopted_thresholds": adopted_thresholds,
        "verdict": verdict,
        "reference_goal_distribution_aggregate_score": GOAL_DISTRIBUTION_AGGREGATE_SCORE_REFERENCE,
        "reference_binary_independent_over_2_5_score": BINARY_INDEPENDENT_OVER_2_5_SCORE_REFERENCE,
    }


def main() -> None:
    t0 = time.time()
    match_repo = MatchRepository()
    filters = {"statistics": "not None", "mean_statistics": "not None", "odds": "not None", "status": ["FT"]}
    matches = convert_orm_match_to_dict(match_repo.search_filter(filters=filters))
    print(f"matches caricati: {len(matches)} ({time.time() - t0:.1f}s)")

    frame, feature_columns = build_totals_evaluation_frame(matches=matches)
    if frame.empty:
        print("Dataset vuoto, esperimento non eseguibile.")
        return
    print(f"rows={len(frame)} | feature_columns={len(feature_columns)}")

    min_train = max(30, int(len(frame) * 0.45))
    min_valid = max(10, int(len(frame) * 0.1))
    cv_splits = expanding_window_splits(frame=frame, time_col="prediction_at", n_splits=5, min_train_size=min_train, min_valid_size=min_valid)
    if not cv_splits:
        print("CV insufficiente, esperimento non eseguibile.")
        return

    t1 = time.time()
    result = run_sequential_cascade(frame=frame, feature_columns=feature_columns, cv_splits=cv_splits)
    print(f"cascata sequenziale calcolata ({time.time() - t1:.1f}s)")

    print(f"\n=== Stacking sequenziale a cascata tra soglie (n_oof={result['n_oof']}) ===")
    for label in THRESHOLD_LABELS:
        payload = result["per_threshold"][label]
        cascade = payload["cascade"]
        line = (
            f"{label:>10s}: n_eval_rows={payload['n_eval_rows']} selection_score={cascade['selection_score']:.4f} "
            f"log_loss={cascade['log_loss']:.4f} brier={cascade['brier']:.4f}"
        )
        if payload["baseline_same_rows"] is not None:
            line += (
                f" | baseline(no-cascade, same rows)={payload['baseline_same_rows']['selection_score']:.4f}"
                f" | delta={payload['delta_selection_score']:+.4f}"
            )
        print(line)

    print(f"\naggregate_cascade_score (media 4 soglie) = {result['aggregate_cascade_score']:.4f}")
    print(f"riferimento goal_distribution aggregato    = {GOAL_DISTRIBUTION_AGGREGATE_SCORE_REFERENCE:.4f}")
    print(f"riferimento binary_independent over_2_5    = {BINARY_INDEPENDENT_OVER_2_5_SCORE_REFERENCE:.4f}")
    print(f"\nVERDETTO: {result['verdict']}")

    payload = {**result, "rows": len(frame), "elapsed_seconds": round(time.time() - t0, 2)}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSalvato in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
